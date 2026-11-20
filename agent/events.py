"""One emit() writes SingleStore/SQLite and mirrors to Amplitude HTTP v2."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid

from agent.audit import store
from agent.flags import status

AMPLITUDE_URL = os.getenv("AMPLITUDE_URL", "https://api2.amplitude.com/2/httpapi")


def _amplitude(event: str, payload: dict, run_id: str, ts: str | None) -> None:
    flags = status()
    key = os.getenv("AMPLITUDE_API_KEY")
    if not flags.amplitude or not key:
        flags.notes.setdefault("amplitude", "flag off or no AMPLITUDE_API_KEY")
        return
    body = {
        "api_key": key,
        "events": [
            {
                "user_id": os.getenv("AMPLITUDE_USER", "vibetuning-operator"),
                "event_type": event,
                "time": int(time.time() * 1000),
                "insert_id": str(uuid.uuid4()),
                "event_properties": {
                    "run_id": run_id,
                    "ts": ts,
                    **{k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))},
                },
            }
        ],
    }
    req = urllib.request.Request(
        AMPLITUDE_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            flags.notes["amplitude"] = f"http {resp.status}"
    except urllib.error.HTTPError as exc:
        flags.notes["amplitude"] = f"http {exc.code}"
    except Exception as exc:
        flags.notes["amplitude"] = f"failed {exc.__class__.__name__}"


def emit(
    run_id: str,
    event: str,
    payload: dict | None = None,
    cost_inr: float | None = None,
    wer: float | None = None,
    ts: str | None = None,
) -> None:
    payload = payload or {}
    try:
        store().emit(run_id, event, payload, cost_inr=cost_inr, wer=wer, ts=ts)
    except Exception as exc:
        status().notes["singlestore"] = f"emit failed {exc.__class__.__name__}"
    _amplitude(event, {**payload, "cost_inr": cost_inr, "wer": wer}, run_id, ts)
