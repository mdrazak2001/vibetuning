"""Base whisper-small vs M3 LoRA adapter. CPU first; GPU fallback is flagged and capped."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from agent import ROOT
from agent.flags import status

ADAPTER = ROOT / "artifacts" / "m3" / "adapter"
BASE = "openai/whisper-small"

_PROCESSOR = None
_LOAD_ERROR = ""


def _to_16k_mono(audio, sr: int | None) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr and sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def _processor():
    global _PROCESSOR, _LOAD_ERROR
    if _PROCESSOR is not None:
        return _PROCESSOR
    from transformers import AutoProcessor

    _PROCESSOR = AutoProcessor.from_pretrained(BASE)
    return _PROCESSOR


def _transcribe_model(model, wav) -> str:
    import torch

    processor = _processor()
    feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
    device = next(model.parameters()).device
    feats = feats.to(device)
    prompt = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    with torch.no_grad():
        ids = model.generate(input_features=feats, forced_decoder_ids=prompt, max_new_tokens=128)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _load_model(kind: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(BASE, torch_dtype=dtype)
    if kind == "tuned":
        from peft import PeftModel

        if not (ADAPTER / "adapter_config.json").exists():
            raise FileNotFoundError(f"missing adapter at {ADAPTER}")
        model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.to(device)
    model.eval()
    return model, device


def transcribe_pair(audio) -> dict:
    """audio is (sr, np.ndarray) from Gradio or a filepath. Loads one model at a time."""
    import gc

    flags = status()
    t0 = time.perf_counter()
    if audio is None:
        return {"base": "", "tuned": "", "sec": 0, "device": "n/a", "gpu_fallback": False}
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, wav = audio
        wav = _to_16k_mono(wav, int(sr))
    else:
        wav = audio
    try:
        base_m, device = _load_model("base")
        base_text = _transcribe_model(base_m, wav)
        del base_m
        gc.collect()
        tuned_m, _ = _load_model("tuned")
        tuned_text = _transcribe_model(tuned_m, wav)
        del tuned_m
        gc.collect()
    except Exception as exc:
        msg = f"{exc.__class__.__name__}: {exc}"
        status().notes["whisper"] = msg
        return {"base": msg, "tuned": msg, "sec": round(time.perf_counter() - t0, 2), "device": "none", "gpu_fallback": False}
    sec = time.perf_counter() - t0
    slow = sec > 15
    note = ""
    if slow and flags.whisper_gpu:
        note = " CPU >15s/clip — WHISPER_GPU_FALLBACK is on; resume paused L4 from vibetuner with ₹20 cap (not auto-created here)."
    elif slow:
        note = " CPU >15s/clip. GPU fallback stays off unless WHISPER_GPU_FALLBACK=1 (₹20 cap)."
    return {
        "base": base_text.strip(),
        "tuned": tuned_text.strip(),
        "sec": round(sec, 2),
        "device": device,
        "gpu_fallback": bool(slow and flags.whisper_gpu),
        "slow": slow,
        "note": note,
        "clip_sec": round(sec, 2),
    }
