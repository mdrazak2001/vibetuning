"""Backfill M1/M3 into the audit DB and replay M3 on demo APPROVE (no training)."""

from __future__ import annotations

from agent.audit import store
from agent.events import emit

M3_RUN = "r_fc806936"
M1_RUN = "r_3a26277a"


def backfill() -> dict:
    db = store()
    db.upsert_run(
        run_id=M1_RUN,
        phase="m1",
        base_model="openai/whisper-tiny",
        status="done",
        predicted_inr=8.26,
        actual_inr=15.20,
        wer_before=0.238261,
        wer_after=0.240000,
        created_at="2026-11-20T01:48:16Z",
    )
    db.upsert_run(
        run_id=M3_RUN,
        phase="m3",
        base_model="openai/whisper-small",
        status="done",
        predicted_inr=9.9,
        actual_inr=5.24,
        wer_before=0.0783,
        wer_after=0.0717,
        created_at="2026-11-20T04:09:27Z",
    )
    if db.events(M3_RUN):
        return {"ok": True, "skipped": True, "backend": db.backend}

    m1 = [
        ("search", "2026-11-20T00:40:00Z", {"query": "whispered speech ASR dataset huggingface", "tool": "context.dev"}, None, None),
        ("dataset_built", "2026-11-20T03:10:00Z", {"note": "M1 used dummy split; Expresso forged in M2"}, 0, None),
        ("plan_shown", "2026-11-20T01:40:00Z", {"predicted_inr": 8.26, "minutes": 12}, 8.26, None),
        ("approved", "2026-11-20T01:47:00Z", {"go": "M1 plumbing"}, None, None),
        ("train_start", "2026-11-20T01:48:16Z", {"machine_id": 580359, "gpu": "L4"}, None, None),
        ("train_end", "2026-11-20T02:06:09Z", {"wer_before": 0.238261, "wer_after": 0.240000}, 15.20, 0.240000),
        ("paused", "2026-11-20T02:06:20Z", {"machine_id": 580359, "running": 0}, None, None),
        ("cost_delta", "2026-11-20T02:06:30Z", {"predicted_inr": 8.26, "actual_inr": 15.20, "multiplier": 1.84, "lesson": "cold start"}, 15.20, None),
    ]
    for event, ts, payload, cost, wer in m1:
        emit(M1_RUN, event, payload, cost_inr=cost, wer=wer, ts=ts)

    m3 = [
        ("search", "2026-11-20T03:00:00Z", {"query": "ylacombe/expresso read whisper", "hits": ["ylacombe/expresso CC BY-NC 4.0"], "rejected": ["nyuuzyou/asmr unspecified+distillation"]}, 0, None),
        ("dataset_built", "2026-11-20T03:40:00Z", {"dataset": "ylacombe/expresso", "license": "CC BY-NC 4.0", "clips": 1139, "hours": 0.993, "train": 1025, "val": 114, "labels": "human transcripts"}, 0, None),
        ("plan_shown", "2026-11-20T04:00:00Z", {"predicted_inr": 9.9, "range": [7.4, 12.4], "epochs": 3, "warm_resume": True, "lesson": "apply 1.84x only on cold create"}, 9.9, None),
        ("approved", "2026-11-20T04:09:00Z", {"go": "M3 full 3-epoch"}, None, None),
        ("train_start", "2026-11-20T04:09:27Z", {"machine_id": 580496, "resumed_from": 580479, "run_id": M3_RUN, "gpu": "L4"}, None, None),
        ("train_end", "2026-11-20T04:16:50Z", {"global_step": 387, "best_step": 258, "wer_before": 0.0783, "wer_after": 0.0717}, 5.24, 0.0717),
        ("paused", "2026-11-20T04:16:55Z", {"machine_id": 580496, "running": 0, "proof": "jl pause 580496"}, None, None),
        ("cost_delta", "2026-11-20T04:17:00Z", {"predicted_inr": 9.9, "actual_inr": 5.24, "lesson": "warm resume beat the 1.84x quote"}, 5.24, 0.0717),
    ]
    for event, ts, payload, cost, wer in m3:
        emit(M3_RUN, event, payload, cost_inr=cost, wer=wer, ts=ts)
    return {"ok": True, "skipped": False, "backend": db.backend}


def replay_m3(task_sentence: str) -> list[dict]:
    """Demo APPROVE: re-emit the real M3 trail tagged as a replay, do not train."""
    stamp_run = f"replay-{M3_RUN}"
    store().upsert_run(
        run_id=stamp_run,
        phase="m3-replay",
        base_model="openai/whisper-small",
        status="replayed",
        predicted_inr=9.9,
        actual_inr=5.24,
        wer_before=0.0783,
        wer_after=0.0717,
    )
    steps = [
        ("search", {"task": task_sentence, "replay_of": M3_RUN}),
        ("dataset_built", {"dataset": "ylacombe/expresso", "license": "CC BY-NC 4.0", "clips": 1139}),
        ("plan_shown", {"predicted_inr": 9.9, "actual_already_known": 5.24, "no_train": True}),
        ("approved", {"demo": True, "no_gpu": True}),
        ("train_start", {"note": "replay — weights already at artifacts/m3/adapter"}),
        ("train_end", {"wer_before": 0.0783, "wer_after": 0.0717}),
        ("paused", {"running": 0}),
        ("cost_delta", {"predicted_inr": 9.9, "actual_inr": 5.24}),
    ]
    for event, payload in steps:
        emit(
            stamp_run,
            event,
            payload,
            cost_inr=5.24 if event in {"train_end", "cost_delta"} else None,
            wer=0.0717 if event == "train_end" else None,
        )
    return store().events(M3_RUN)
