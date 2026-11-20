#!/usr/bin/env python3
"""M3 full run: whisper-small LoRA SFT on Expresso whisper.

3 epochs, early stop on val WER, keep best. Base vs tuned WER with English
normalization on refs and hyps. No datasets.map after CUDA.
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
OUTPUT_DIR = Path(os.environ.get("M3_OUTPUT_DIR", "/home/outputs/m3"))
DATA_DIR = Path(os.environ.get("M3_DATA_DIR", "/home/data"))
EVAL_WHISPER_DIR = Path(os.environ.get("EVAL_WHISPER_DIR", "/home/eval_whisper"))
EVAL_NORMAL_DIR = Path(os.environ.get("EVAL_NORMAL_DIR", "/home/eval_normal"))
BASE_MODEL = os.environ.get("M3_BASE_MODEL", "openai/whisper-small")
EPOCHS = int(os.environ.get("M3_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("M3_BATCH_SIZE", "8"))
LORA_R = int(os.environ.get("M3_LORA_R", "8"))
LR = float(os.environ.get("M3_LR", "1e-4"))


def log(msg: str) -> None:
    print(msg, flush=True)


def elapsed() -> float:
    return round(time.time() - T0, 3)


def install_deps() -> None:
    log(f"[m3] pip install: {' '.join(DEPS)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *DEPS])
    log(f"[m3] pip install done t={elapsed()}")


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
    return path if path.is_absolute() else (root / path).resolve()


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


def transcribe_one(model, processor, y, sr, device) -> str:
    import torch

    feats = processor.feature_extractor(
        y, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        pred_ids = model.generate(
            feats,
            language="en",
            task="transcribe",
            max_new_tokens=128,
        )
    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]


def wer_on_rows(model, processor, rows, data_root: Path, device, normalizer, wer_metric, name: str) -> dict:
    preds, refs, times = [], [], []
    model.eval()
    for i, row in enumerate(rows):
        wav = resolve(row["audio_path"], data_root)
        y, sr = load_wav_16k(wav)
        t1 = time.time()
        hyp = transcribe_one(model, processor, y, sr, device)
        times.append(time.time() - t1)
        preds.append(normalizer(hyp).strip())
        refs.append(normalizer(row["text"]).strip())
        if (i + 1) % 20 == 0 or (i + 1) == len(rows):
            log(f"[m3] {name} {i + 1}/{len(rows)}")
    score = float(wer_metric.compute(predictions=preds, references=refs)) if preds else None
    log(f"[m3] WER {name}={score} n={len(rows)} sec/clip={sum(times) / len(times) if times else None:.3f}")
    return {
        "wer": score,
        "n": len(rows),
        "sec_per_clip": round(sum(times) / len(times), 4) if times else None,
        "samples": [
            {"ref": r["text"], "hyp_norm": p, "ref_norm": rf}
            for r, p, rf in list(zip(rows, preds, refs))[:5]
        ],
    }


def heldout_rows(directory: Path) -> list[dict]:
    """Minimal sentences.txt loader (m4a/mp3/wav). Empty if user has not dropped clips."""
    if not directory.is_dir():
        return []
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".webm"}
    files = sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in audio_exts],
        key=lambda p: p.name.lower(),
    )
    sent = directory / "sentences.txt"
    by_name, ordered = {}, []
    if sent.exists():
        for raw in sent.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                left, right = line.split("\t", 1)
            elif "|" in line:
                left, right = line.split("|", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) == 2 and Path(parts[0]).suffix.lower() in audio_exts:
                    left, right = parts
                else:
                    ordered.append(line)
                    continue
            by_name[Path(left.strip()).name] = right.strip()
            by_name[Path(left.strip()).stem] = right.strip()
            ordered.append(right.strip())
    rows = []
    for i, wav in enumerate(files):
        text = by_name.get(wav.name) or by_name.get(wav.stem)
        if text is None and i < len(ordered) and not by_name:
            text = ordered[i]
        if text:
            rows.append({"audio_path": str(wav.resolve()), "text": text})
    return rows


def prepare_ds(rows, processor, root: Path, label: str):
    from datasets import Dataset

    prepared = []
    for i, row in enumerate(rows):
        y, sr = load_wav_16k(resolve(row["audio_path"], root))
        feats = processor.feature_extractor(y, sampling_rate=sr).input_features[0]
        labels = processor.tokenizer(row["text"]).input_ids
        prepared.append({"input_features": feats, "labels": labels})
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            log(f"[m3] prepared {label} {i + 1}/{len(rows)}")
    return Dataset.from_list(prepared)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    install_deps()

    import evaluate
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[m3] device={device} t={elapsed()}")
    if device == "cuda":
        log(f"[m3] gpu={torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    log(f"[m3] model loaded {BASE_MODEL} t={elapsed()}")

    train_rows = load_jsonl(DATA_DIR / "manifest_train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "manifest_val.jsonl")
    log(f"[m3] train={len(train_rows)} val={len(val_rows)}")
    train_ds = prepare_ds(train_rows, processor, DATA_DIR, "train")
    val_ds = prepare_ds(val_rows, processor, DATA_DIR, "val")
    log(f"[m3] datasets ready t={elapsed()}")

    normalizer = BasicTextNormalizer()
    wer_metric = evaluate.load("wer")

    whisper_held = heldout_rows(EVAL_WHISPER_DIR)
    normal_held = heldout_rows(EVAL_NORMAL_DIR)
    log(f"[m3] heldout whisper={len(whisper_held)} normal={len(normal_held)}")

    log("[m3] WER before (base whisper-small, no LoRA)")
    wer_before = {
        "expresso_val": wer_on_rows(model, processor, val_rows, DATA_DIR, device, normalizer, wer_metric, "base/expresso_val"),
    }
    if whisper_held:
        wer_before["eval_whisper"] = wer_on_rows(
            model, processor, whisper_held, Path("/"), device, normalizer, wer_metric, "base/eval_whisper"
        )
    else:
        log("[m3] eval_whisper empty — skipped")
    if normal_held:
        wer_before["eval_normal"] = wer_on_rows(
            model, processor, normal_held, Path("/"), device, normalizer, wer_metric, "base/eval_normal"
        )
    else:
        log("[m3] eval_normal empty — skipped (forgetting check needs dropped clips)")

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

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalizer(s).strip() for s in pred_str]
        label_str = [normalizer(s).strip() for s in label_str]
        return {"wer": float(wer_metric.compute(predictions=pred_str, references=label_str))}

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
    common = dict(
        output_dir=str(work_dir),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        warmup_steps=20,
        num_train_epochs=EPOCHS,
        fp16=device == "cuda",
        logging_steps=10,
        save_total_limit=2,
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=128,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )
    try:
        args = Seq2SeqTrainingArguments(
            eval_strategy="epoch",
            save_strategy="epoch",
            **common,
        )
    except TypeError:
        args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            **common,
        )

    trainer_kwargs = dict(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    try:
        trainer = Seq2SeqTrainer(processing_class=processor, **trainer_kwargs)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=processor.tokenizer, **trainer_kwargs)

    log(f"[m3] training epochs={EPOCHS} batch={BATCH_SIZE} n_train={len(train_ds)}")
    trainer.train()
    log(f"[m3] training done t={elapsed()}")

    adapter_dir = OUTPUT_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    log(f"[m3] adapter -> {adapter_dir}")

    log("[m3] WER after (LoRA, best checkpoint)")
    wer_after = {
        "expresso_val": wer_on_rows(
            trainer.model, processor, val_rows, DATA_DIR, device, normalizer, wer_metric, "tuned/expresso_val"
        ),
    }
    if whisper_held:
        wer_after["eval_whisper"] = wer_on_rows(
            trainer.model, processor, whisper_held, Path("/"), device, normalizer, wer_metric, "tuned/eval_whisper"
        )
    if normal_held:
        wer_after["eval_normal"] = wer_on_rows(
            trainer.model, processor, normal_held, Path("/"), device, normalizer, wer_metric, "tuned/eval_normal"
        )

    payload = {
        "base_model": BASE_MODEL,
        "method": "lora_sft",
        "lora_r": LORA_R,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "fp16": device == "cuda",
        "wer_before": {k: v.get("wer") for k, v in wer_before.items()},
        "wer_after": {k: v.get("wer") for k, v in wer_after.items()},
        "wer_before_detail": wer_before,
        "wer_after_detail": wer_after,
        "heldout_whisper_n": len(whisper_held),
        "heldout_normal_n": len(normal_held),
        "elapsed_sec": elapsed(),
        "adapter_dir": str(adapter_dir),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "global_step": trainer.state.global_step,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("========== M3 FULL METRICS ==========")
    log(json.dumps({k: payload[k] for k in ("wer_before", "wer_after", "global_step", "elapsed_sec", "best_metric")}, indent=2))
    log(f"M3_WER_BEFORE_VAL={payload['wer_before'].get('expresso_val')}")
    log(f"M3_WER_AFTER_VAL={payload['wer_after'].get('expresso_val')}")
    log("[m3] complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"[m3] FATAL: {exc!r}")
        raise
