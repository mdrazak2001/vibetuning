#!/usr/bin/env python3
"""Launch M1 on an on-demand L4, enforce a 30-minute pause timeout, download artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "m1"
ARTIFACT_DIR = ROOT / "artifacts" / "m1"
TIMEOUT_SEC = 30 * 60
INSTANCE_NAME = "vibetuning-m1"
REMOTE_OUTPUT = "/home/outputs/m1"


def load_env() -> None:
    for path in (ROOT / ".env", Path.home() / ".vibetuning" / "secrets.env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    if os.environ.get("JARVIS_API") and not os.environ.get("JL_API_KEY"):
        os.environ["JL_API_KEY"] = os.environ["JARVIS_API"]


def jl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["jl", *args]
    print(f"[run_m1] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check)


def jl_json(*args: str) -> dict | list:
    proc = jl(*args, check=False)
    blob = (proc.stdout or "").strip()
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        print(proc.stdout, file=sys.stderr, flush=True)
        raise RuntimeError(f"jl {' '.join(args)} failed ({proc.returncode})")
    if not blob:
        raise RuntimeError(f"jl {' '.join(args)} produced empty stdout")
    return json.loads(blob)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def status_snapshot() -> dict:
    data = jl_json("status", "--json")
    bal = data.get("balance") or {}
    return {
        "at": now_iso(),
        "balance": bal.get("balance"),
        "grants": bal.get("grants"),
        "currency": data.get("currency"),
        "running_instances": (data.get("resources") or {}).get("running_instances"),
        "paused_instances": (data.get("resources") or {}).get("paused_instances"),
        "raw": data,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pause_machine(machine_id: int | None) -> None:
    if not machine_id:
        return
    print(f"[run_m1] pausing {machine_id}", flush=True)
    proc = jl("pause", str(machine_id), "--yes", "--json", check=False)
    print(proc.stdout or proc.stderr, flush=True)


def download_outputs(machine_id: int) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACT_DIR
    proc = jl("download", str(machine_id), REMOTE_OUTPUT, str(dest), "-r", check=False)
    print(proc.stdout, flush=True)
    print(proc.stderr, flush=True)
    if proc.returncode != 0:
        print("[run_m1] recursive download failed; trying adapter + metrics", flush=True)
        jl("download", str(machine_id), f"{REMOTE_OUTPUT}/metrics.json", str(dest / "metrics.json"), check=False)
        jl("download", str(machine_id), f"{REMOTE_OUTPUT}/adapter", str(dest / "adapter"), "-r", check=False)


def save_logs(run_id: str | None) -> None:
    if not run_id:
        return
    proc = jl("run", "logs", run_id, "--tail", "400", check=False)
    (LOG_DIR / "run.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")


def main() -> int:
    load_env()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    before = status_snapshot()
    write_json(LOG_DIR / "balance_before.json", before)
    print(f"[run_m1] grants_before={before['grants']} balance_before={before['balance']}", flush=True)

    start_cmd = [
        "run",
        "recipes/sft_whisper/train.py",
        "--gpu",
        "L4",
        "--region",
        "IN2",
        "--storage",
        "40",
        "--name",
        INSTANCE_NAME,
        "--setup",
        "apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1",
        "--json",
        "--yes",
        "--keep",
        "--no-follow",
    ]
    t0 = time.time()
    start = jl_json(*start_cmd)
    write_json(LOG_DIR / "jl_run_start.json", start if isinstance(start, dict) else {"raw": start})
    if not isinstance(start, dict):
        raise RuntimeError(f"unexpected jl run start payload: {start!r}")
    run_id = str(start.get("run_id") or start.get("id") or "")
    machine_id = start.get("machine_id") or start.get("id")
    if machine_id is not None:
        machine_id = int(machine_id)
    print(f"[run_m1] run_id={run_id} machine_id={machine_id}", flush=True)

    timed_out = False
    exit_code = None
    last_state = None
    try:
        # Early failure check.
        time.sleep(15)
        early = jl("run", "logs", run_id, "--tail", "80", check=False)
        print(early.stdout or "", flush=True)
        print(early.stderr or "", flush=True)

        while True:
            elapsed = time.time() - t0
            if elapsed >= TIMEOUT_SEC:
                timed_out = True
                print("[run_m1] HARD TIMEOUT 30min — stopping run and pausing", flush=True)
                jl("run", "stop", run_id, check=False)
                break

            st = jl_json("run", "status", run_id, "--json")
            last_state = st
            write_json(LOG_DIR / "last_status.json", st if isinstance(st, dict) else {"raw": st})
            state = (st or {}).get("status") or (st or {}).get("state")
            exit_code = (st or {}).get("run_exit_code")
            inst = (st or {}).get("instance_status") or (st or {}).get("machine_status")
            print(
                f"[run_m1] t={elapsed:.0f}s state={state} exit={exit_code} instance={inst}",
                flush=True,
            )
            if exit_code is not None or str(state).lower() in {"succeeded", "failed", "stopped", "error"}:
                break
            logs = jl("run", "logs", run_id, "--tail", "40", check=False)
            print(logs.stderr or "", flush=True)
            time.sleep(45)
    finally:
        try:
            save_logs(run_id)
        except Exception as exc:
            print(f"[run_m1] save_logs failed: {exc!r}", flush=True)
        try:
            if machine_id:
                download_outputs(int(machine_id))
        except Exception as timed_exc:
            print(f"[run_m1] download failed: {timed_exc!r}", flush=True)
        try:
            pause_machine(int(machine_id) if machine_id else None)
        except Exception as pause_exc:
            print(f"[run_m1] pause failed: {pause_exc!r}", flush=True)
            # Last-resort: pause any running instance named vibetuning-m1
            listing = jl_json("list", "--json")
            for row in listing if isinstance(listing, list) else []:
                if row.get("name") == INSTANCE_NAME and str(row.get("status", "")).lower() == "running":
                    pause_machine(int(row["machine_id"]))

    after = status_snapshot()
    write_json(LOG_DIR / "balance_after.json", after)
    listing = jl_json("list", "--json")
    write_json(LOG_DIR / "jl_list_after.json", listing if isinstance(listing, list) else {"raw": listing})
    running = [
        row
        for row in (listing if isinstance(listing, list) else [])
        if str(row.get("status", "")).lower() == "running"
    ]
    grants_before = float(before.get("grants") or 0)
    grants_after = float(after.get("grants") or 0)
    billed_inr = round(grants_before - grants_after, 4)
    fx = 93.29
    cost = {
        "predicted_inr": round(41.31 * 12 / 60, 4),
        "predicted_usd": round((41.31 * 12 / 60) / fx, 4),
        "timeout_ceiling_inr": round(41.31 * 0.5, 4),
        "timeout_ceiling_usd": round((41.31 * 0.5) / fx, 4),
        "live_l4_inr_per_hr": 41.31,
        "fx_usd_inr": fx,
        "grants_before": grants_before,
        "grants_after": grants_after,
        "cash_before": before.get("balance"),
        "cash_after": after.get("balance"),
        "actual_inr": billed_inr,
        "actual_usd": round(billed_inr / fx, 4),
        "elapsed_sec": round(time.time() - t0, 1),
        "timed_out": timed_out,
        "run_id": run_id,
        "machine_id": machine_id,
        "run_exit_code": exit_code,
        "last_state": last_state,
        "running_after": running,
    }
    write_json(LOG_DIR / "cost.json", cost)
    print(json.dumps(cost, indent=2), flush=True)
    print("[run_m1] jl list (after pause):", flush=True)
    print(json.dumps(listing, indent=2), flush=True)
    if running:
        print("[run_m1] ERROR: instances still running", flush=True)
        return 2
    return 0 if not timed_out else 3


if __name__ == "__main__":
    raise SystemExit(main())
