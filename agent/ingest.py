"""Airbyte (PyAirbyte source-file) → training_rows. Pandas fallback if the box expires."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from agent import ROOT
from agent.audit import store
from agent.events import emit
from agent.flags import status
from agent.replay import M3_RUN

TRAIN = ROOT / "data" / "expresso_whisper" / "manifest_train.jsonl"
VAL = ROOT / "data" / "expresso_whisper" / "manifest_val.jsonl"
DATASET = "ylacombe/expresso"
LICENSE = "CC BY-NC 4.0"
SOURCE_URL = "https://huggingface.co/datasets/ylacombe/expresso"


def _rows_from_jsonl(path: Path, split: str, sync_id: str, connector: str) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.append(
            {
                "run_id": M3_RUN,
                "dataset": DATASET,
                "source_url": SOURCE_URL,
                "license": LICENSE,
                "split": split,
                "audio_path": rec.get("audio_path"),
                "text": rec.get("text"),
                "connector_version": connector,
                "sync_id": sync_id,
            }
        )
    return out


def _pyairbyte_rows(sync_id: str) -> list[dict]:
    import airbyte as ab

    combined = ROOT / "data" / "expresso_whisper" / "manifest_all.jsonl"
    parts = []
    for src, split in ((TRAIN, "train"), (VAL, "val")):
        if src.exists():
            for line in src.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec["_split"] = split
                    parts.append(json.dumps(rec, ensure_ascii=False))
    combined.write_text("\n".join(parts) + "\n", encoding="utf-8")

    source = ab.get_source("source-file", use_python=True)
    spec = {}
    try:
        spec = source.get_config_spec() if hasattr(source, "get_config_spec") else {}
    except Exception:
        spec = {}
    url = str(combined)
    # Docker source-file wants /local/; PyAirbyte python often accepts a real path or HTTPS.
    config = {
        "dataset_name": "expresso_whisper_manifest",
        "format": "jsonl",
        "url": url,
        "provider": {"storage": "local"},
    }
    # If spec forbids local, caller should retry HTTPS — we try local first.
    source.set_config(config)
    check = source.check()
    if check is False:
        raise RuntimeError(f"source.check failed spec_keys={list(spec)[:12] if isinstance(spec, dict) else spec}")
    if hasattr(source, "select_all_streams"):
        source.select_all_streams()
    result = source.read()
    frames = []
    if isinstance(result, dict):
        frames = list(result.values())
    elif hasattr(result, "values"):
        frames = list(result.values())
    else:
        frames = [result]
    rows = []
    for frame in frames:
        df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
        for rec in df.to_dict(orient="records"):
            rows.append(
                {
                    "run_id": M3_RUN,
                    "dataset": DATASET,
                    "source_url": SOURCE_URL,
                    "license": LICENSE,
                    "split": rec.get("_split") or rec.get("split") or "train",
                    "audio_path": rec.get("audio_path"),
                    "text": rec.get("text"),
                    "connector_version": "pyairbyte:source-file",
                    "sync_id": sync_id,
                }
            )
    if not rows:
        raise RuntimeError("airbyte read returned 0 rows")
    return rows


def ingest_manifests() -> dict:
    flags = status()
    db = store()
    existing = db.lineage(M3_RUN)
    if existing.get("n", 0) >= 1000:
        return {"ok": True, "skipped": True, **existing, "mode": "already"}

    sync_id = str(uuid.uuid4())
    mode = "pandas"
    connector = "pandas-jsonl"
    rows: list[dict] = []
    if flags.airbyte:
        try:
            rows = _pyairbyte_rows(sync_id)
            mode = "airbyte"
            connector = "pyairbyte:source-file"
            flags.notes["airbyte"] = f"synced {len(rows)} rows"
        except Exception as exc:
            flags.notes["airbyte"] = f"failed {exc.__class__.__name__}: {str(exc)[:140]}"
            flags.airbyte = False
            rows = []

    if not rows:
        rows = _rows_from_jsonl(TRAIN, "train", sync_id, connector) + _rows_from_jsonl(
            VAL, "val", sync_id, connector
        )
        mode = "pandas" if mode != "airbyte" else mode

    n = db.insert_training_rows(rows)
    emit(
        M3_RUN,
        "airbyte_sync_completed" if mode == "airbyte" else "manifest_ingested",
        {
            "n": n,
            "sync_id": sync_id,
            "connector": connector,
            "dataset": DATASET,
            "license": LICENSE,
        },
    )
    try:
        from agent import memory

        seed = ROOT / "data" / "memory_seed.md"
        note = (
            f"\n\n## Airbyte lineage\nmanifest ingested via {mode}; {n} rows in training_rows; "
            f"sync {sync_id}; {DATASET} {LICENSE}.\n"
        )
        if seed.exists():
            seed.write_text(seed.read_text(encoding="utf-8") + note, encoding="utf-8")
    except Exception:
        pass
    lin = db.lineage(M3_RUN)
    return {"ok": True, "mode": mode, "n": n, "sync_id": sync_id, **lin}


def lineage_header() -> str:
    lin = store().lineage(M3_RUN)
    n = lin.get("n") or 0
    if not n:
        return "No training_rows yet — ingest not run."
    connector = lin.get("connector_version") or "unknown"
    sync = lin.get("sync_id") or "?"
    return (
        f"{n:,} training rows · {connector} · sync {sync} · "
        f"{lin.get('dataset') or DATASET} · {lin.get('license') or LICENSE}"
    )
