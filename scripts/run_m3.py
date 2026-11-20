#!/usr/bin/env python3
"""M3 full-run launcher — ON vibetuner only. 3-epoch whisper-small LoRA.

Hard caps: 45 min wall-clock, ₹150. Pause in finally and on timeout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("M3_ROOT", "/home/user/vibetuning"))
if not (ROOT / "recipes").exists():
    ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "logs" / "m3"
DATA_SRC = ROOT / "data" / "expresso_whisper"
TIMEOUT_SEC = 45 * 60
CAP_INR = 150.0
INSTANCE_NAME = "vibetuning-m3-probe"
REMOTE_OUTPUT = "/home/outputs/m3"
GPU = "L4"
REGION = "IN2"
STORAGE = "40"
PREDICTED_INR = 9.9
PREDICTED_RANGE = [7.4, 12.4]


def log(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jl_env() -> None:
    for path in (Path.home() / ".jl.env", ROOT / ".env"):
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


def jl_bin() -> str:
    cand = Path.home() / ".venv-jl" / "bin" / "jl"
    if cand.exists():
        return str(cand)
    found = shutil.which("jl")
    if not found:
        raise RuntimeError("jl CLI not found")
    return found


def jl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [jl_bin(), *args]
    log(f"[m3] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=check)


def jl_json(*args: str):
    proc = jl(*args, check=False)
    blob = (proc.stdout or "").strip()
    if proc.returncode != 0:
        log(proc.stderr or "")
        log(proc.stdout or "")
        raise RuntimeError(f"jl {' '.join(args)} failed ({proc.returncode})")
    if not blob:
        raise RuntimeError(f"jl {' '.join(args)} empty stdout")
    return json.loads(blob)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def live_l4_rate() -> float:
    rows = jl_json("gpus", "--json")
    for row in rows:
        if row.get("gpu_type") == "L4" and row.get("workload_type") in {"container", None} and row.get("region") == REGION:
            return float(row["price_per_hour"])
    for row in rows:
        if row.get("gpu_type") == "L4":
            return float(row["price_per_hour"])
    return 41.31


def strip_pii(status: dict) -> dict:
    user = dict(status.get("user") or {})
    for k in ("phone_number", "address1", "address2", "tax_id"):
        user.pop(k, None)
    bal = status.get("balance") or {}
    return {
        "at": now_iso(),
        "grants": bal.get("grants"),
        "balance": bal.get("balance"),
        "currency": status.get("currency"),
        "running_instances": (status.get("resources") or {}).get("running_instances"),
        "paused_instances": (status.get("resources") or {}).get("paused_instances"),
        "user_id": user.get("user_id"),
    }


def pack_tarball() -> Path:
    tgz = Path.home() / "m3_full_data.tgz"
    if tgz.exists():
        tgz.unlink()
    cmd = [
        "tar",
        "-czf",
        str(tgz),
        "-C",
        str(DATA_SRC),
        "audio",
        "manifest_train.jsonl",
        "manifest_val.jsonl",
    ]
    log(f"[m3] packing {' '.join(cmd)}")
    subprocess.check_call(cmd)
    log(f"[m3] tarball {tgz} bytes={tgz.stat().st_size}")
    return tgz


def pause_machine(machine_id: int | None) -> None:
    if not machine_id:
        return
    log(f"[m3] pausing {machine_id}")
    proc = jl("pause", str(machine_id), "--yes", "--json", check=False)
    log((proc.stdout or "") + (proc.stderr or ""))


def pause_named_running() -> None:
    listing = jl_json("list", "--json")
    for row in listing if isinstance(listing, list) else []:
        if row.get("name") == INSTANCE_NAME and str(row.get("status", "")).lower() == "running":
            pause_machine(int(row["machine_id"]))


def main() -> int:
    load_jl_env()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rate = live_l4_rate()
    log(f"[m3] full run L4 ₹/hr={rate} cap=₹{CAP_INR} timeout={TIMEOUT_SEC}s predicted=₹{PREDICTED_INR} range={PREDICTED_RANGE}")

    before = strip_pii(jl_json("status", "--json"))
    write_json(LOG_DIR / "full_balance_before.json", before)
    log(f"[m3] grants_before={before['grants']}")

    tgz = pack_tarball()
    train_py = ROOT / "recipes" / "sft_whisper" / "train_m3.py"
    run_id = None
    machine_id = None
    timed_out = False
    exit_code = None
    last_state = None
    start_payload = None

    try:
        listing = jl_json("list", "--json")
        write_json(LOG_DIR / "full_jl_list_before.json", listing)
        named = [r for r in listing if r.get("name") == INSTANCE_NAME]
        running = [r for r in listing if str(r.get("status", "")).lower() == "running"]
        others = [r for r in running if r.get("name") != INSTANCE_NAME]
        if others:
            raise RuntimeError(f"refusing to create GPU; already running: {others}")

        setup = (
            "mkdir -p /home/data && tar -xzf /home/m3_full_data.tgz -C /home/data && "
            "apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 || true"
        )
        start_cmd = [
            "run",
            str(train_py),
            "--setup",
            setup,
            "--json",
            "--yes",
            "--no-follow",
        ]
        if named:
            on_id = int(named[0]["machine_id"])
            st = str(named[0].get("status", "")).lower()
            if st == "paused":
                log(f"[m3] resuming paused {INSTANCE_NAME} id={on_id}")
                resumed = jl_json("resume", str(on_id), "--yes", "--json")
                write_json(LOG_DIR / "full_jl_resume.json", resumed)
                on_id = int(resumed.get("machine_id") or on_id)
            log(f"[m3] using {INSTANCE_NAME} id={on_id}")
            up = jl("upload", str(on_id), str(tgz), "/home/m3_full_data.tgz", check=False)
            log((up.stdout or "") + (up.stderr or ""))
            start_cmd.extend(["--on", str(on_id)])
            machine_id = on_id
        else:
            start_cmd.extend(
                [
                    "--gpu",
                    GPU,
                    "--region",
                    REGION,
                    "--storage",
                    STORAGE,
                    "--name",
                    INSTANCE_NAME,
                    "--keep",
                ]
            )
        start_payload = jl_json(*start_cmd)
        write_json(LOG_DIR / "full_jl_run_start.json", start_payload)
        run_id = str(start_payload.get("run_id") or start_payload.get("id") or "")
        machine_id = start_payload.get("machine_id") or start_payload.get("id") or machine_id
        if machine_id is not None:
            machine_id = int(machine_id)
        log(f"[m3] run_id={run_id} machine_id={machine_id}")

        time.sleep(20)
        early = jl("run", "logs", run_id, "--tail", "80", check=False)
        log(early.stdout or "")
        log(early.stderr or "")

        while True:
            wall = time.time() - t0
            billed_est = rate * wall / 3600.0
            if wall >= TIMEOUT_SEC:
                timed_out = True
                log("[m3] HARD TIMEOUT 45min — stop + pause")
                jl("run", "stop", run_id, check=False)
                break
            if billed_est >= CAP_INR:
                timed_out = True
                log(f"[m3] CAP ₹{CAP_INR} estimated ({billed_est:.2f}) — stop + pause")
                jl("run", "stop", run_id, check=False)
                break
            st = jl_json("run", "status", run_id, "--json")
            last_state = st
            write_json(LOG_DIR / "full_last_status.json", st)
            state = (st or {}).get("status") or (st or {}).get("state")
            exit_code = (st or {}).get("run_exit_code")
            inst = (st or {}).get("instance_status") or (st or {}).get("machine_status")
            log(f"[m3] t={wall:.0f}s est₹={billed_est:.2f} state={state} exit={exit_code} instance={inst}")
            if exit_code is not None or str(state).lower() in {"succeeded", "failed", "stopped", "error"}:
                break
            logs = jl("run", "logs", run_id, "--tail", "40", check=False)
            log(logs.stderr or "")
            time.sleep(40)
    finally:
        try:
            if run_id:
                proc = jl("run", "logs", run_id, "--tail", "500", check=False)
                (LOG_DIR / "full_run.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        except Exception as exc:
            log(f"[m3] save_logs failed: {exc!r}")
        try:
            if machine_id:
                dest = LOG_DIR / "full_out"
                dest.mkdir(parents=True, exist_ok=True)
                dl = jl("download", str(machine_id), REMOTE_OUTPUT, str(dest), "-r", check=False)
                log(dl.stdout or "")
                log(dl.stderr or "")
        except Exception as exc:
            log(f"[m3] download failed: {exc!r}")
        try:
            pause_machine(int(machine_id) if machine_id else None)
        except Exception as exc:
            log(f"[m3] pause failed: {exc!r}")
            try:
                pause_named_running()
            except Exception as exc2:
                log(f"[m3] pause named failed: {exc2!r}")

    after = strip_pii(jl_json("status", "--json"))
    write_json(LOG_DIR / "full_balance_after.json", after)
    listing = jl_json("list", "--json")
    write_json(LOG_DIR / "full_jl_list_after.json", listing)
    running = [r for r in listing if str(r.get("status", "")).lower() == "running"]
    grants_before = float(before.get("grants") or 0)
    grants_after = float(after.get("grants") or 0)
    actual = round(grants_before - grants_after, 4)
    wall = round(time.time() - t0, 1)
    cost = {
        "job": "m3-full-whisper-small-lora",
        "gpu": GPU,
        "spot": False,
        "region": REGION,
        "machine_id": machine_id,
        "run_id": run_id,
        "live_l4_inr_per_hr": rate,
        "predicted_inr": PREDICTED_INR,
        "predicted_range_inr": PREDICTED_RANGE,
        "cap_inr": CAP_INR,
        "timeout_sec": TIMEOUT_SEC,
        "timeout_ceiling_inr": round(rate * TIMEOUT_SEC / 3600, 4),
        "grants_before": grants_before,
        "grants_after": grants_after,
        "actual_inr": actual,
        "elapsed_sec": wall,
        "timed_out": timed_out,
        "run_exit_code": exit_code,
        "running_after": running,
        "m1_calibration_multiplier": 1.84,
    }
    write_json(LOG_DIR / "full_cost.json", cost)
    log(json.dumps(cost, indent=2))
    if running:
        log("[m3] ERROR: instances still running")
        return 2
    ok = (not timed_out) and (exit_code in {0, None}) and str((last_state or {}).get("status") or "").lower() in {
        "succeeded",
        "success",
        "",
    }
    # Treat succeeded-with-null-exit as OK (jl quirk from the probe).
    state = str((last_state or {}).get("status") or (last_state or {}).get("state") or "").lower()
    if running:
        return 2
    if timed_out:
        return 3
    if state == "failed" or (exit_code not in {0, None}):
        return 1
    return 0


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    raise SystemExit(main())
