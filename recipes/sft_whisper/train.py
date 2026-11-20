#!/usr/bin/env python3
"""M1 plumbing: whisper-tiny + LoRA SFT on a tiny public ASR dummy set.

Installs its own pip deps. Prints WER before/after. Does not try to improve WER.
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

DEPS = [
    "transformers",
    "peft",
    "evaluate",
    "jiwer",
    "librosa",
    "soundfile",
    "accelerate",
    "datasets[audio]",
]

OUTPUT_DIR = Path(os.environ.get("M1_OUTPUT_DIR", "/home/outputs/m1"))
ADAPTER_DIR = OUTPUT_DIR / "adapter"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
BASE_MODEL = os.environ.get("M1_BASE_MODEL", "openai/whisper-tiny")
DATASET_ID = os.environ.get("M1_DATASET", "hf-internal-testing/librispeech_asr_dummy")
MAX_STEPS = int(os.environ.get("M1_MAX_STEPS", "150"))
BATCH_SIZE = int(os.environ.get("M1_BATCH_SIZE", "4"))
RECIPE_DIR = Path(__file__).resolve().parent
if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))


def log(msg: str) -> None:
    print(msg, flush=True)


def install_deps() -> None:
    log(f"[m1] pip install: {' '.join(DEPS)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *DEPS])
    log("[m1] pip install done")


def run_wer(model, processor, dataset, device) -> float:
    import evaluate
    import torch

    wer_metric = evaluate.load("wer")
    model.eval()
    preds: list[str] = []
    refs: list[str] = []
    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    for i, row in enumerate(dataset):
        audio = row["audio"]
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            pred_ids = model.generate(
                features,
                forced_decoder_ids=forced,
                max_new_tokens=128,
            )
        pred = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        preds.append(pred.strip().lower())
        refs.append(str(row["text"]).strip().lower())
        if (i + 1) % 10 == 0:
            log(f"[m1] wer eval {i + 1}/{len(dataset)}")
    score = float(wer_metric.compute(predictions=preds, references=refs))
    model.train()
    return score


def write_metrics(payload: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dataset_from_eval(items: list) -> list:
    """Load eval_whisper clips in-process. Do not datasets.map after CUDA."""
    import numpy as np
    import soundfile as sf

    rows = []
    for item in items:
        y, sr = sf.read(item["audio_path"], dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        rows.append({"audio": {"array": y, "sampling_rate": int(sr)}, "text": item["text"]})
    return rows


def main() -> int:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    install_deps()

    import torch
    from datasets import Audio, load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[m1] device={device} cuda={torch.cuda.is_available()}")
    if device == "cuda":
        log(f"[m1] gpu={torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.to(device)

    log(f"[m1] loading {DATASET_ID} clean/validation")
    raw = load_dataset(DATASET_ID, "clean", split="validation")
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))
    log(f"[m1] n_samples={len(raw)}")

    wer_set = raw
    eval_n = 0
    try:
        from eval_loader import is_eval_ready, load_eval_whisper

        if is_eval_ready():
            eval_items = load_eval_whisper()
            eval_n = len(eval_items)
            log(f"[m1] eval_whisper ready n={eval_n} -> WER uses held-out recorded clips")
            wer_set = _dataset_from_eval(eval_items)
        else:
            log("[m1] eval_whisper empty; drop wavs + sentences.txt there. WER uses dummy split")
    except Exception as exc:  # noqa: BLE001
        log(f"[m1] eval_loader skipped: {exc!r}")

    log("[m1] WER before (base whisper-tiny, no LoRA)")
    wer_before = run_wer(model, processor, wer_set, device)
    log(f"M1_WER_BEFORE={wer_before:.6f}")
    write_metrics(
        {
            "wer_before": wer_before,
            "wer_after": None,
            "base_model": BASE_MODEL,
            "dataset": DATASET_ID,
            "n_samples": len(raw),
            "max_steps": MAX_STEPS,
            "elapsed_sec": round(time.time() - started, 1),
        }
    )

    # Do NOT use datasets.map after CUDA init — it forks and deadlocks.
    from datasets import Dataset

    rows = []
    for i, row in enumerate(raw):
        audio = row["audio"]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(row["text"]).input_ids
        rows.append({"input_features": feats, "labels": labels})
        if (i + 1) % 10 == 0:
            log(f"[m1] prepared {i + 1}/{len(raw)}")
    train_ds = Dataset.from_list(rows)
    log(f"[m1] train_ds ready n={len(train_ds)}")

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

    lora = LoraConfig(
        r=8,
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
        logging_steps=10,
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
    )
    try:
        trainer = Seq2SeqTrainer(processing_class=processor, **trainer_kwargs)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=processor.tokenizer, **trainer_kwargs)

    log(f"[m1] training max_steps={MAX_STEPS}")
    trainer.train()
    log("[m1] training done")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))
    log(f"[m1] adapter saved -> {ADAPTER_DIR}")

    log("[m1] WER after (LoRA adapter)")
    wer_after = run_wer(trainer.model, processor, wer_set, device)
    log(f"M1_WER_AFTER={wer_after:.6f}")
    payload = {
        "wer_before": wer_before,
        "wer_after": wer_after,
        "base_model": BASE_MODEL,
        "dataset": DATASET_ID,
        "n_samples": len(raw),
        "n_eval_whisper": eval_n,
        "max_steps": MAX_STEPS,
        "elapsed_sec": round(time.time() - started, 1),
        "adapter_dir": str(ADAPTER_DIR),
    }
    write_metrics(payload)
    log(f"[m1] metrics={json.dumps(payload)}")
    log("[m1] complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"[m1] FATAL: {exc!r}")
        raise
