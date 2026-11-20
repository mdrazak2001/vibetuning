#!/usr/bin/env python3
"""PyAirbyte (OSS) syncs. Run with /home/user/.venv-airbyte/bin/python.

  airbyte_sync.py manifest           Expresso manifest (from HF ylacombe/expresso) -> DuckDB cache -> rows jsonl
  airbyte_sync.py gpu <metrics.jsonl> <job_id>   GPU job snapshots -> DuckDB cache -> rows jsonl

Prints one JSON summary line on stdout (last line).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "airbyte"


def sync(path: Path, stream: str, out_rows: Path) -> dict:
    import airbyte as ab

    t0 = time.time()
    config = {"dataset_name": stream, "format": "jsonl", "url": str(path), "provider": {"storage": "local"}}
    exe = ROOT / ".venv-source-file" / "bin" / "source-file"
    if exe.exists():
        source = ab.get_source("source-file", config=config, local_executable=str(exe))
    else:
        source = ab.get_source("source-file", config=config, install_if_missing=True)
    source.check()
    source.select_all_streams()
    cache = ab.new_local_cache(f"tuno_{stream}")
    result = source.read(cache=cache)
    rows = []
    for name, dataset in result.streams.items():
        rows.extend(dataset.to_pandas().to_dict(orient="records"))
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    with out_rows.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if not str(k).startswith("_airbyte")}, default=str) + "\n")
    return {"ok": True, "connector": "pyairbyte:source-file", "cache": "duckdb", "stream": stream,
            "records": len(rows), "sec": round(time.time() - t0, 1), "rows_file": str(out_rows),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "manifest"
    try:
        if mode == "manifest":
            data = ROOT / "data" / "expresso_whisper"
            combined = OUT / "manifest_all.jsonl"
            combined.parent.mkdir(parents=True, exist_ok=True)
            with combined.open("w", encoding="utf-8") as fh:
                for split in ("train", "val"):
                    src = data / f"manifest_{split}.jsonl"
                    for line in src.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            rec = json.loads(line)
                            rec["_split"] = split
                            rec["split"] = split
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary = sync(combined, "expresso_manifest", OUT / "manifest_rows.jsonl")
        else:
            metrics, job = Path(sys.argv[2]), sys.argv[3]
            summary = sync(metrics, "gpu_snapshots", OUT / f"gpu_{job}.jsonl")
            summary["job_id"] = job
        (OUT / f"last_{mode}.json").write_text(json.dumps(summary, indent=2))
    except Exception as exc:
        summary = {"ok": False, "error": f"{exc.__class__.__name__}: {str(exc)[:300]}"}
    print(json.dumps(summary))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
