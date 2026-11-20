#!/usr/bin/env python3
"""Live fine-tune relay — ON vibetuner only. Re-runs the M3 whisper LoRA SFT recipe
on the paused L4 and streams loss / epoch WER to logs/live/ + audit store + Amplitude.

Hard caps: 20 min wall-clock, ₹40. Refuses to create a GPU: warm-resume only.
Pauses in finally.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_m3 as m  # noqa: E402

LIVE_DIR = m.ROOT / "logs" / "live"
STATE = LIVE_DIR / "state.json"
POINTS = LIVE_DIR / "metrics.jsonl"
TIMEOUT_SEC = int(os.getenv("LIVE_TIMEOUT_SEC", str(20 * 60)))
CAP_INR = float(os.getenv("LIVE_CAP_INR", "40"))
POLL_SEC = 15

DICT_RE = re.compile(r"\{'(?:loss|eval_loss)'.*\}")
WER_RE = re.compile(r"\[m3\] WER (\S+)=([0-9.]+)")
PHASES = [
    ("pip install done", "deps ready"),
    ("model loaded", "model loaded"),
    ("datasets ready", "datasets ready"),
    ("WER before", "baseline eval"),
    ("training epochs=", "training"),
    ("training done", "post-train eval"),
    ("WER after", "post-train eval"),
    ("[m3] complete", "complete"),
    ("FATAL", "failed"),
]


def emit(run_id: str, event: str, payload: dict, wer: float | None = None, cost: float | None = None) -> None:
    try:
        from agent.events import emit as _emit

        _emit(run_id, event, payload, wer=wer, cost_inr=cost)
    except Exception as exc:
        m.log(f"[live] emit failed {exc!r}")


def write_state(**kw) -> dict:
    cur = json.loads(STATE.read_text()) if STATE.exists() else {}
    cur.update(kw, updated=m.now_iso())
    m.write_json(STATE, cur)
    return cur


def parse(lines: list[str]) -> tuple[list[dict], str | None]:
    points, phase = [], None
    for raw in lines:
        line = raw.strip()
        for needle, label in PHASES:
            if needle in line:
                phase = label
        hit = DICT_RE.search(line)
        if hit:
            try:
                d = ast.literal_eval(hit.group(0))
            except Exception:
                continue
            if "loss" in d:
                points.append({"kind": "train", "epoch": d.get("epoch"), "loss": d["loss"], "lr": d.get("learning_rate")})
            elif "eval_wer" in d or "eval_loss" in d:
                points.append({"kind": "eval", "epoch": d.get("epoch"), "eval_loss": d.get("eval_loss"), "wer": d.get("eval_wer")})
        else:
            w = WER_RE.search(line)
            if w:
                points.append({"kind": "wer", "split": w.group(1), "wer": float(w.group(2))})
    return points, phase


def main() -> int:
    m.load_jl_env()
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    POINTS.write_text("")
    t0 = time.time()
    live_id = "live_" + time.strftime("%H%M%S", time.gmtime())
    rate = m.live_l4_rate()
    write_state(id=live_id, state="starting", phase="resuming L4", started=m.now_iso(), rate=rate,
                cap_inr=CAP_INR, timeout_sec=TIMEOUT_SEC, est_inr=0.0, run_id=None, machine_id=None)
    emit(live_id, "live_train_start", {"gpu": "L4", "rate_inr_hr": rate, "recipe": "whisper-small LoRA SFT"})

    seen: set[str] = set()
    machine_id = run_id = None
    final_state = "failed"
    try:
        listing = m.jl_json("list", "--json")
        running = [r for r in listing if str(r.get("status", "")).lower() == "running"]
        if running:
            raise RuntimeError(f"refusing: already running {[r.get('name') for r in running]}")
        named = [r for r in listing if r.get("name") == m.INSTANCE_NAME]
        if not named:
            raise RuntimeError("no paused instance to warm-resume; refusing to create a GPU")
        machine_id = int(named[0]["machine_id"])
        if str(named[0].get("status", "")).lower() == "paused":
            resumed = m.jl_json("resume", str(machine_id), "--yes", "--json")
            machine_id = int(resumed.get("machine_id") or machine_id)
        write_state(state="running", phase="uploading data", machine_id=machine_id)
        tgz = Path.home() / "m3_full_data.tgz"
        if not tgz.exists():
            tgz = m.pack_tarball()
        m.jl("upload", str(machine_id), str(tgz), "/home/m3_full_data.tgz", check=False)

        setup = (
            "mkdir -p /home/data && tar -xzf /home/m3_full_data.tgz -C /home/data && "
            "apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 || true"
        )
        start = m.jl_json("run", str(m.ROOT / "recipes" / "sft_whisper" / "train_m3.py"), "--setup", setup,
                          "--json", "--yes", "--no-follow", "--on", str(machine_id))
        run_id = str(start.get("run_id") or start.get("id") or "")
        write_state(phase="setup", run_id=run_id)

        while True:
            wall = time.time() - t0
            est = round(rate * wall / 3600.0, 2)
            if wall >= TIMEOUT_SEC or est >= CAP_INR:
                m.log("[live] HARD STOP (timeout/cap)")
                m.jl("run", "stop", run_id, check=False)
                final_state = "stopped"
                break
            logs = m.jl("run", "logs", run_id, "--tail", "400", check=False)
            points, phase = parse((logs.stdout or "").splitlines())
            new = []
            for p in points:
                key = json.dumps(p, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    new.append({**p, "t": round(wall, 1)})
            if new:
                with POINTS.open("a", encoding="utf-8") as fh:
                    for p in new:
                        fh.write(json.dumps(p) + "\n")
                for p in new:
                    ev = {"train": "live_train_step", "eval": "live_epoch_eval", "wer": "live_wer"}[p["kind"]]
                    emit(live_id, ev, {k: v for k, v in p.items() if v is not None}, wer=p.get("wer"))
            st = m.jl_json("run", "status", run_id, "--json")
            status = str((st or {}).get("status") or (st or {}).get("state") or "").lower()
            exit_code = (st or {}).get("run_exit_code")
            write_state(phase=phase or "setup", est_inr=est, elapsed=round(wall), jl_status=status)
            m.log(f"[live] t={wall:.0f}s est₹={est} phase={phase} status={status} points={len(seen)}")
            if exit_code is not None or status in {"succeeded", "failed", "stopped", "error"}:
                final_state = "complete" if status in {"succeeded", "success"} and exit_code in {0, None} else "failed"
                break
            time.sleep(POLL_SEC)
    except Exception as exc:
        m.log(f"[live] error {exc!r}")
        write_state(error=str(exc)[:200])
    finally:
        if run_id:
            proc = m.jl("run", "logs", run_id, "--tail", "600", check=False)
            (LIVE_DIR / "run.log").write_text(proc.stdout or "", encoding="utf-8")
        if machine_id:
            m.pause_machine(machine_id)
        try:
            after = m.jl_json("list", "--json")
            still = [r.get("name") for r in after if str(r.get("status", "")).lower() == "running"]
            if still:
                m.pause_named_running()
        except Exception as exc:
            m.log(f"[live] list after failed {exc!r}")
        wall = time.time() - t0
        est = round(rate * wall / 3600.0, 2)
        write_state(state=final_state, phase="paused", est_inr=est, elapsed=round(wall), finished=m.now_iso())
        emit(live_id, "live_train_end", {"state": final_state, "elapsed_sec": round(wall)}, cost=est)
    return 0 if final_state == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
