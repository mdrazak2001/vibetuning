"""Train/val loader for the M2 Expresso whisper manifests."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "data" / "expresso_whisper"


def load_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"audio_path": obj["audio_path"], "text": obj["text"]})
    return rows


def resolve_audio(audio_path: str, root: Path) -> Path:
    p = Path(audio_path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def load_train_val(data_dir: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    d = Path(data_dir) if data_dir else DEFAULT_DIR
    train = load_manifest(d / "manifest_train.jsonl")
    val = load_manifest(d / "manifest_val.jsonl")
    for row in train + val:
        row["audio_path"] = str(resolve_audio(row["audio_path"], d))
    return train, val
