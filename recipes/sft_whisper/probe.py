#!/usr/bin/env python3
"""M3 probe: whisper-small + LoRA on a real Expresso subset.

50 train steps + generate-eval on 8 val clips. Logs time-to-first-step,
sec/train-step, and sec/eval-clip. Self-installs pip deps. No datasets.map
after CUDA (M1 deadlock).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

T0 = time.time()
DEPS = [
    "transformers==4.46.3",
    "peft==0.13.2",
    "evaluate",
    "jiwer",
    "librosa",
    "soundfile",
    "accelerate",
    "datasets==3.1.0",
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("M3_OUTPUT_DIR", "/home/outputs/m3_probe"))
DATA_DIR = Path(os.environ.get("M3_DATA_DIR", str(SCRIPT_DIR / "data")))
BASE_MODEL = os.environ.get("M3_BASE_MODEL", "openai/whisper-small")
MAX_STEPS = int(os.environ.get("M3_MAX_STEPS", "50"))
BATCH_SIZE = int(os.environ.get("M3_BATCH_SIZE", "8"))
EVAL_CLIPS = int(os.environ.get("M3_EVAL_CLIPS", "8"))
LORA_R = int(os.environ.get("M3_LORA_R", "8"))


def log(msg: str) -> None:
    print(msg, flush=True)


def elapsed() -> float:
    return round(time.time() - T0, 3)


def install_deps() -> None:
    log(f"[probe] pip install: {' '.join(DEPS)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *DEPS])
    log(f"[probe] pip install done t={elapsed()}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve(p: str, root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_wav_16k(path: Path):
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if int(sr) != 16000:
        n_out = int(round(len(y) * 16000 / float(sr)))
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = 16000
    return y, int(sr)


def normalize_en(text: str, normalizer) -> str:
    return normalizer(text or "").strip()


def transcribe_one(model, processor, audio, device) -> str:
    import torch

    y, sr = audio["array"], audio["sampling_rate"]
    feats = processor.feature_extractor(
        y, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device)
    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    with torch.no_grad():
        pred_ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=128)
    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict = {"t0_process": T0, "max_steps": MAX_STEPS, "batch_size": BATCH_SIZE}

    install_deps()
    timings["sec_deps"] = elapsed()

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    import evaluate

    t_imports = elapsed()
    timings["sec_imports"] = t_imports
    log(f"[probe] imports done t={t_imports}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[probe] device={device} cuda={torch.cuda.is_available()}")
    if device == "cuda":
        log(f"[probe] gpu={torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.to(device)
    timings["sec_model_loaded"] = elapsed()
    log(f"[probe] model loaded {BASE_MODEL} t={timings['sec_model_loaded']}")

    train_rows = load_jsonl(DATA_DIR / "manifest_train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "manifest_val.jsonl")
    log(f"[probe] manifests train={len(train_rows)} val={len(val_rows)}")

    t_data0 = time.time()
    prepared = []
    for i, row in enumerate(train_rows):
        wav = resolve(row["audio_path"], DATA_DIR)
        y, sr = load_wav_16k(wav)
        feats = processor.feature_extractor(y, sampling_rate=sr).input_features[0]
        labels = processor.tokenizer(row["text"]).input_ids
        prepared.append({"input_features": feats, "labels": labels})
        if (i + 1) % 50 == 0:
            log(f"[probe] prepared {i + 1}/{len(train_rows)}")
    from datasets import Dataset

    train_ds = Dataset.from_list(prepared)
    timings["sec_data_ready"] = elapsed()
    timings["sec_data_prep_only"] = round(time.time() - t_data0, 3)
    log(f"[probe] train_ds n={len(train_ds)} t={timings['sec_data_ready']}")

    eval_pack = []
    for row in val_rows[:EVAL_CLIPS]:
        wav = resolve(row["audio_path"], DATA_DIR)
        y, sr = load_wav_16k(wav)
        eval_pack.append({"audio": {"array": y, "sampling_rate": sr}, "text": row["text"], "path": str(wav)})
    log(f"[probe] eval_pack n={len(eval_pack)}")

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: WhisperProcessor

        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    step_times: list[float] = []
    first_step_at: list[float] = []

    class StepTimer(TrainerCallback):
        def __init__(self):
            self._last = None

        def on_step_begin(self, args, state, control, **kwargs):
            self._last = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self._last is None:
                return
            dt = time.time() - self._last
            step_times.append(dt)
            if len(first_step_at) == 0:
                first_step_at.append(elapsed())
                log(f"[probe] FIRST_STEP t={first_step_at[0]:.3f} step_dt={dt:.3f}s")
            if state.global_step % 10 == 0 or state.global_step == 1:
                log(f"[probe] step={state.global_step} dt={dt:.3f}s")

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    work_dir = OUTPUT_DIR / "trainer_work"
    args = Seq2SeqTrainingArguments(
        output_dir=str(work_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        fp16=device == "cuda",
        logging_steps=5,
        save_steps=200,
        save_total_limit=1,
        report_to=[],
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        generation_max_length=128,
    )
    trainer_kwargs = dict(
        args=args,
        model=model,
        train_dataset=train_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        callbacks=[StepTimer()],
    )
    try:
        trainer = Seq2SeqTrainer(processing_class=processor, **trainer_kwargs)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=processor.tokenizer, **trainer_kwargs)

    log(f"[probe] training max_steps={MAX_STEPS} batch={BATCH_SIZE}")
    trainer.train()
    timings["sec_train_done"] = elapsed()
    log("[probe] training done")

    adapter_dir = OUTPUT_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    normalizer = BasicTextNormalizer()
    wer_metric = evaluate.load("wer")
    eval_dts: list[float] = []
    preds: list[str] = []
    refs: list[str] = []
    samples = []
    log(f"[probe] generate-eval n={len(eval_pack)}")
    trainer.model.eval()
    for i, row in enumerate(eval_pack):
        t1 = time.time()
        hyp = transcribe_one(trainer.model, processor, row["audio"], device)
        dt = time.time() - t1
        eval_dts.append(dt)
        hyp_n = normalize_en(hyp, normalizer)
        ref_n = normalize_en(row["text"], normalizer)
        preds.append(hyp_n)
        refs.append(ref_n)
        samples.append(
            {
                "path": row["path"],
                "ref": row["text"],
                "hyp": hyp,
                "ref_norm": ref_n,
                "hyp_norm": hyp_n,
                "sec": round(dt, 3),
            }
        )
        log(f"[probe] eval_clip {i + 1}/{len(eval_pack)} {dt:.3f}s hyp={hyp!r}")

    wer = float(wer_metric.compute(predictions=preds, references=refs)) if preds else None
    log(f"[probe] WER_PROBE_VAL={wer}")

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    # Skip step 0/1 warmup for the steady-state rate if we have enough.
    steady = step_times[2:] if len(step_times) > 5 else step_times
    metrics = {
        "base_model": BASE_MODEL,
        "method": "lora_sft",
        "lora_r": LORA_R,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "n_train": len(train_ds),
        "n_eval_clips": len(eval_pack),
        "device": device,
        "fp16": device == "cuda",
        "sec_deps": timings.get("sec_deps"),
        "sec_model_loaded": timings.get("sec_model_loaded"),
        "sec_data_ready": timings.get("sec_data_ready"),
        "sec_to_first_step_in_process": first_step_at[0] if first_step_at else None,
        "sec_train_done": timings.get("sec_train_done"),
        "sec_per_train_step_all": _mean(step_times),
        "sec_per_train_step_steady": _mean(steady),
        "step_times": [round(x, 4) for x in step_times],
        "sec_per_eval_clip": _mean(eval_dts),
        "eval_clip_times": [round(x, 4) for x in eval_dts],
        "wer_probe_val_normalized": wer,
        "samples": samples,
        "elapsed_sec": elapsed(),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log("========== M3 PROBE METRICS ==========")
    log(json.dumps({k: v for k, v in metrics.items() if k not in {"samples", "step_times"}}, indent=2))
    log(f"M3_SEC_TO_FIRST_STEP_IN_PROCESS={metrics['sec_to_first_step_in_process']}")
    log(f"M3_SEC_PER_TRAIN_STEP={metrics['sec_per_train_step_steady']}")
    log(f"M3_SEC_PER_EVAL_CLIP={metrics['sec_per_eval_clip']}")
    log(f"M3_WER_PROBE_VAL={wer}")
    log("[probe] complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"[probe] FATAL: {exc!r}")
        raise
