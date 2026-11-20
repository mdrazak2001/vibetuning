"""Live L4 quote with cold vs warm overhead, citing cognee lessons."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from agent.events import emit
from agent.flags import status
from agent.memory import lessons_for_plan
from agent.replay import M3_RUN

L4_FALLBACK_INR = 41.31
FX = 93.29


def _jl_gpus() -> dict[str, Any]:
    env = os.environ.copy()
    if not env.get("JL_API_KEY") and env.get("JARVIS_API"):
        env["JL_API_KEY"] = env["JARVIS_API"]
    try:
        proc = subprocess.run(
            ["jl", "gpus", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        if proc.returncode != 0:
            return {}
        data = json.loads(proc.stdout or "[]")
        rows = data if isinstance(data, list) else data.get("gpus") or data.get("data") or []
        for row in rows:
            name = str(row.get("gpu") or row.get("gpu_type") or row.get("name") or "")
            if name.upper() == "L4" or "L4" in name.upper():
                return row
        return rows[0] if rows else {}
    except Exception:
        return {}


def live_l4_inr() -> tuple[float, str]:
    row = _jl_gpus()
    for key in ("price", "price_per_hour", "inr_per_hr", "on_demand_price", "cost"):
        if row.get(key) is not None:
            try:
                return float(row[key]), "jl gpus live"
            except (TypeError, ValueError):
                continue
    return L4_FALLBACK_INR, "fallback cached ₹41.31 (jl gpus missed)"


def build_card(task: str, research: dict | None = None) -> dict:
    flags = status()
    rate, rate_src = live_l4_inr()
    warm = flags.warm_resume_enabled
    # Probe-calibrated pieces for a 3-epoch whisper-small LoRA.
    train_sec = 144
    eval_sec = 63 + 82
    cold_overhead = 180
    warm_overhead = 90
    overhead = warm_overhead if warm else cold_overhead
    raw_sec = overhead + train_sec + eval_sec
    raw_inr = round(raw_sec / 3600 * rate, 2)
    multiplier = 1.0 if warm else 1.84
    predicted = round(raw_inr * multiplier, 1)
    cap = flags.budget_cap_inr
    lessons = lessons_for_plan()
    card = {
        "task": task,
        "base_model": "openai/whisper-small",
        "method": "LoRA SFT r=8 α=16 q_proj+v_proj fp16",
        "dataset": "ylacombe/expresso read/whisper · CC BY-NC 4.0 · 1025/114",
        "gpu": "on-demand L4 IN2",
        "live_l4_inr_per_hr": rate,
        "rate_source": rate_src,
        "warm_resume": warm,
        "warm_resume_source": flags.statsig_source,
        "budget_cap_inr": cap,
        "raw_inr": raw_inr,
        "multiplier": multiplier,
        "predicted_inr": predicted,
        "predicted_range_inr": [round(predicted * 0.75, 1), round(predicted * 1.25, 1)],
        "m3_actual_inr": 5.24,
        "m3_predicted_inr": 9.9,
        "wer_before": 0.0783,
        "wer_after": 0.0717,
        "no_train_in_demo": True,
        "lessons": lessons,
        "memory": (research or {}).get("memory"),
        "datasets": (research or {}).get("datasets"),
        "web": (research or {}).get("web"),
    }
    emit(M3_RUN, "plan_shown", {"predicted_inr": predicted, "warm": warm, "cap": cap}, cost_inr=predicted)
    return card


def card_markdown(card: dict) -> str:
    lessons = card.get("lessons") or []
    lesson_md = "\n".join(f"- {l.splitlines()[0][:280]}" for l in lessons[:4])
    ds_lines = []
    for d in card.get("datasets") or []:
        mark = "✓ use" if d["id"].startswith("ylacombe") else "✗ skip"
        ds_lines.append(f"- **{d['id']}** ({d['license']}, {d['labels']}) — {mark}. {d['why']}")
    web_lines = []
    for h in card.get("web") or []:
        web_lines.append(f"- [{h.get('title') or h.get('url')}]({h.get('url')}) — {h.get('snippet','')[:160]}")
    mem = card.get("memory") or {}
    return f"""# Plan / cost card

**Task.** {card['task']}

| | |
|---|---|
| Model | `{card['base_model']}` {card['method']} |
| Data | {card['dataset']} |
| GPU | {card['gpu']} @ **₹{card['live_l4_inr_per_hr']}/hr** ({card['rate_source']}) |
| Resume | **{'warm' if card['warm_resume'] else 'cold'}** via `{card['warm_resume_source']}` |
| Raw GPU-sec quote | ₹{card['raw_inr']} |
| Multiplier | ×{card['multiplier']} {'(warm: do not apply 1.84×)' if card['warm_resume'] else '(cold: M1 1.84×)'} |
| **Predicted** | **₹{card['predicted_inr']}** range {card['predicted_range_inr']} |
| Cap | ₹{card['budget_cap_inr']} (Statsig/env) |
| Already spent M3 | predicted ₹{card['m3_predicted_inr']} → actual **₹{card['m3_actual_inr']}** |
| WER (Expresso val) | {card['wer_before']} → **{card['wer_after']}** |

Demo **APPROVE** replays that M3 trail. It does **not** create a GPU or train.

## Memory (cognee)

Source: `{mem.get('source') or 'lexical'}`.

{lesson_md}

## Datasets

{chr(10).join(ds_lines) or '_no rank_'}

## Web (context.dev)

{chr(10).join(web_lines) or '_no live hits (key missing or error)_'}
"""
