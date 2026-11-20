#!/usr/bin/env python3
"""Mission-control API on :7861. Thin wrapper over agent/*; the Gradio app on :7860 is untouched."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
JL_BIN = Path.home() / ".venv-jl" / "bin"
if JL_BIN.exists():
    os.environ["PATH"] = f"{JL_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import memory as mem
from agent.audit import store
from agent.events import emit
from agent.flags import status
from agent.ingest import ingest_manifests, lineage_header
from agent.plan import build_card, live_l4_inr
from agent.replay import M3_RUN, backfill, replay_m3
from agent.research import KNOWN_DATASETS, _context_search, _superlinked_rank

INDEX = ROOT / "web" / "index.html"
LOGOS = ROOT / "assets" / "logos"
app = FastAPI(title="vibetuning mission control")
LOGOS.mkdir(parents=True, exist_ok=True)
app.mount("/assets/logos", StaticFiles(directory=LOGOS), name="logos")

# ---------------------------------------------------------------- streams

_STREAMS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _new_stream(kind: str) -> dict:
    sid = f"{kind}-{uuid.uuid4().hex[:10]}"
    s = {"id": sid, "events": [], "done": False, "cond": threading.Condition()}
    with _LOCK:
        _STREAMS[sid] = s
    return s


def _push(s: dict, event: str, data: dict) -> None:
    with s["cond"]:
        s["events"].append((event, data))
        s["cond"].notify_all()


def _finish(s: dict) -> None:
    _push(s, "end", {"stream_id": s["id"]})
    with s["cond"]:
        s["done"] = True
        s["cond"].notify_all()


def _sse(s: dict, after: int = -1):
    i = after + 1
    while True:
        with s["cond"]:
            while i >= len(s["events"]) and not s["done"]:
                s["cond"].wait(timeout=15)
                if i >= len(s["events"]) and not s["done"]:
                    break
            batch = s["events"][i:]
            done = s["done"]
        if not batch and not done:
            yield ": keepalive\n\n"
            continue
        for event, data in batch:
            yield f"id: {i}\nevent: {event}\ndata: {json.dumps(data, default=str)}\n\n"
            i += 1
        if done and i >= len(s["events"]):
            return


def _stream_response(s: dict, after: int = -1) -> StreamingResponse:
    return StreamingResponse(
        _sse(s, after),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Stream-Id": s["id"]},
    )


# ---------------------------------------------------------------- jl (HTTP API only, no SSH)


_JL_CACHE: dict = {"t": 0.0, "data": None}


def _jl_json(*args: str):
    env = os.environ.copy()
    if not env.get("JL_API_KEY") and env.get("JARVIS_API"):
        env["JL_API_KEY"] = env["JARVIS_API"]
    try:
        proc = subprocess.run(["jl", *args, "--json"], capture_output=True, text=True, timeout=25, env=env)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout or "null")
    except Exception:
        return None


def jl_snapshot(max_age: float = 30.0) -> dict:
    now = time.time()
    if _JL_CACHE["data"] and now - _JL_CACHE["t"] < max_age:
        return _JL_CACHE["data"]
    st = _jl_json("status") or {}
    listing = _jl_json("list") or []
    running = [m for m in listing if str(m.get("status", "")).lower() == "running"] if isinstance(listing, list) else []
    paused = [m for m in listing if str(m.get("status", "")).lower() == "paused"] if isinstance(listing, list) else []
    grants = None
    for key in ("grants", "grant_balance"):
        if isinstance(st, dict) and st.get(key) is not None:
            grants = st.get(key)
    if grants is None and isinstance(st, dict):
        acct = st.get("account") or st.get("balance") or {}
        if isinstance(acct, dict):
            grants = acct.get("grants")
    data = {
        "ok": bool(st or listing),
        "grants_inr": grants,
        "running": len(running),
        "running_gpu_types": [m.get("gpu_type") for m in running],
        "paused": len(paused),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _JL_CACHE.update(t=now, data=data)
    return data


# ---------------------------------------------------------------- run pipeline

_PARSE_RULES = [
    (r"whisper|asmr|quiet|soft", "whispered speech"),
    (r"asr|transcrib|speech|voice", "speech recognition"),
    (r"english", "English"),
    (r"normal speech|forget", "guard against forgetting"),
]


def _parse(goal: str) -> dict:
    g = goal.lower()
    tags = [label for pat, label in _PARSE_RULES if re.search(pat, g)]
    speech = any(t in tags for t in ("whispered speech", "speech recognition"))
    return {
        "method": "rule-based parse (no LLM)",
        "tags": tags,
        "modality": "audio → text" if speech else "unknown (defaulting to the M3 ASR recipe)",
        "base_model": "openai/whisper-small",
        "recipe": "recipes/sft_whisper · LoRA SFT",
    }


def _lexical_rank(goal: str, candidates: list[dict]) -> list[dict]:
    words = {w for w in re.findall(r"[a-z]+", goal.lower()) if len(w) > 3}
    scored = []
    for c in candidates:
        blob = f"{c['id']} {c['why']} {c['labels']}".lower()
        score = sum(1 for w in words if w in blob)
        if c["labels"].startswith("human"):
            score += 2
        if "do not" in c["why"] or "missing" in c["why"]:
            score -= 1
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    top = max(1, scored[0][0]) if scored else 1
    return [{**c, "score": round(max(0, s) / top, 2)} for s, c in scored]


_VERDICT = {
    "ylacombe/expresso": ("ok", "CC BY-NC 4.0 · human transcripts", "USE"),
    "nyuuzyou/asmr": ("warn", "license unspecified · pseudo-labels · 36 GB", "SKIP"),
    "facebook/expresso": ("off", "missing / auth-walled on HF", "SKIP"),
}


_TEXT_RE = re.compile(r"""['"]text['"]:\s*(['"])((?:\\.|(?!\1).)*)\1""", re.S)


def _clean_hits(hits: list[str]) -> list[str]:
    out = []
    for h in hits or []:
        texts = [m.group(2).replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"') for m in _TEXT_RE.finditer(h)]
        out.extend(t for t in texts if t.strip()) if texts else out.append(h)
    return out


def _walk_texts(obj, out: list[str]) -> None:
    if isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, str) and t.strip():
            out.append(t)
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                _walk_texts(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_texts(v, out)
    else:
        t = getattr(obj, "text", None)
        if isinstance(t, str) and t.strip():
            out.append(t)
        elif hasattr(obj, "model_dump"):
            _walk_texts(obj.model_dump(), out)


def _memory_query(q: str, use_cognee: bool = True) -> dict:
    if use_cognee and status().cognee and mem._COGNEE_OK:
        try:
            import asyncio

            import cognee
            from cognee import SearchType

            res = asyncio.run(cognee.search(query_text=q, query_type=SearchType.CHUNKS_LEXICAL))
            texts: list[str] = []
            _walk_texts(res, texts)
            if texts:
                paras = [p.strip() for t in texts for p in re.split(r"\n\s*\n", t) if p.strip()]
                words = {w for w in re.findall(r"[a-z0-9.]+", q.lower()) if len(w) > 2}
                paras.sort(key=lambda p: -sum(1 for w in words if w in p.lower()))
                return {"source": "cognee", "hits": paras[:12]}
        except Exception as exc:
            status().notes["cognee"] = f"search failed {exc.__class__.__name__} · lexical fallback"
    return {"source": "lexical", "hits": mem._lexical(q)}


def _lesson_line(hits: list[str], needle: str = "1.84") -> str:
    for h in hits:
        if needle in h:
            for line in re.split(r"\n+", h):
                if needle in line:
                    return line.strip(" -#*")[:400]
    return (hits[0] if hits else "")[:400]


def _step(s, sid, title, fn, **meta):
    _push(s, "step_start", {"id": sid, "title": title, **meta})
    t0 = time.perf_counter()
    try:
        out = fn()
        ms = int((time.perf_counter() - t0) * 1000)
        return out, ms, None
    except Exception as exc:
        ms = int((time.perf_counter() - t0) * 1000)
        err = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        _push(s, "error", {"id": sid, "error": err})
        _push(s, "step_done", {"id": sid, "status": "error", "ms": ms, "label": err})
        return None, ms, err


def _run_worker(s: dict, goal: str, choices: dict) -> None:
    flags = status()
    run_id = M3_RUN
    _push(s, "run", {"stream_id": s["id"], "goal": goal, "choices": choices, "audit_run_id": run_id})

    parsed, ms, err = _step(s, "parse", "Parse goal", lambda: _parse(goal))
    if not err:
        _push(s, "step_done", {"id": "parse", "status": "ok", "ms": ms, "label": parsed["method"], "detail": parsed})

    mem_choice = choices.get("memory", "cognee")
    q = "what did we learn about GPU cost estimation?"

    def _memory():
        return _memory_query(q, use_cognee=mem_choice == "cognee")

    packed_mem, ms, err = _step(s, "memory", "Recall lessons", _memory, tool="cognee")
    if not err:
        hits = _clean_hits(packed_mem.get("hits") or [])
        lesson = _lesson_line(hits)
        src = packed_mem.get("source", "lexical")
        label = "cognee" if src.startswith("cognee") else ("lexical fallback (cognee not ready)" if mem_choice == "cognee" else "lexical (chosen)")
        _push(s, "memory_hit", {"source": src, "label": label, "lesson": lesson, "hits": [h[:400] for h in hits[:4]]})
        _push(s, "step_done", {"id": "memory", "status": "ok", "ms": ms, "label": label})

    query = f"{goal} huggingface dataset license whispered speech ASR"
    web, ms, err = _step(s, "search", "Search the web", lambda: _context_search(query, 5), tool="context.dev")
    if not err:
        emit(run_id, "search", {"task": goal, "tool": "context.dev", "n": len(web)})
        has_key = bool(os.getenv("CONTEXT_DEV_API_KEY"))
        rows = [
            {**h, "rank": i + 1, "relevance": round(1 - i * 0.14, 2)}
            for i, h in enumerate(web)
        ]
        _push(s, "search_results", {
            "results": rows,
            "relevance_label": "rank order from context.dev (no score returned)",
            "label": f"{len(rows)} live results" if rows else ("no CONTEXT_DEV_API_KEY" if not has_key else "0 results / error"),
        })
        _push(s, "step_done", {"id": "search", "status": "ok" if rows else "off", "ms": ms,
                               "label": f"{len(rows)} results" if rows else "flag off / no key"})

    rank_choice = choices.get("ranking", "sie")

    def _rank():
        if rank_choice == "sie" and flags.superlinked and os.getenv("SIE_API_KEY"):
            ranked = _superlinked_rank(goal, KNOWN_DATASETS)
            note = flags.notes.get("superlinked", "")
            if note.startswith("ranked"):
                return {"mode": f"SIE Serverless · {note.split('· ', 1)[-1]}", "live": True, "datasets": ranked}
            return {"mode": f"SIE {note or 'failed'} → lexical", "datasets": _lexical_rank(goal, KNOWN_DATASETS)}
        why = "flag off → lexical" if rank_choice == "sie" else "lexical (chosen)"
        return {"mode": why, "datasets": _lexical_rank(goal, KNOWN_DATASETS)}

    ranked, ms, err = _step(s, "rank", "Rank candidate datasets", _rank, tool="superlinked")
    datasets = KNOWN_DATASETS
    if not err:
        datasets = ranked["datasets"]
        live = ranked.get("live", False)
        _push(s, "step_done", {"id": "rank", "status": "ok" if live else "off", "ms": ms, "label": ranked["mode"],
                               "detail": [{"id": d["id"], "score": d.get("score")} for d in datasets]})

    def _licenses():
        out = []
        for d in datasets:
            tone, reason, verdict = _VERDICT.get(d["id"], ("off", d["license"], "SKIP"))
            out.append({"id": d["id"], "license": d["license"], "labels": d["labels"], "url": d["url"],
                        "tone": tone, "reason": reason, "verdict": verdict})
        return out

    verdicts, ms, err = _step(s, "license", "Check licenses", _licenses)
    if not err:
        _push(s, "license_verdicts", {"verdicts": verdicts, "label": "verdicts from the M2 dataset card audit"})
        _push(s, "step_done", {"id": "license", "status": "ok", "ms": ms, "label": "1 usable · 2 rejected"})

    ing, ms, err = _step(s, "ingest", "Ingest manifest → training_rows", ingest_manifests, tool="airbyte")
    if not err:
        mode = ing.get("mode")
        label = {
            "airbyte": "PyAirbyte source-file",
            "already": f"cached lineage ({ing.get('connector_version') or 'prior sync'})",
            "pandas": "pandas-jsonl (AIRBYTE_ENABLED off)",
        }.get(mode, mode)
        _push(s, "ingest_progress", {"done": ing.get("n", 0), "total": ing.get("n", 0), "mode": mode, "label": label,
                                     "sync_id": ing.get("sync_id"), "backend": store().backend,
                                     "lineage": lineage_header()})
        _push(s, "step_done", {"id": "ingest", "status": "ok" if mode == "airbyte" else "off" if mode == "pandas" else "ok",
                               "ms": ms, "label": label})

    rate, ms, err = _step(s, "jarvis", "Price L4 on JarvisLabs", live_l4_inr, tool="jarvislabs")
    if not err:
        inr, src = rate
        _push(s, "step_done", {"id": "jarvis", "status": "ok" if "live" in src else "off", "ms": ms,
                               "label": src, "detail": {"inr_per_hr": inr}})

    research_packed = {"task": goal, "web": web or [], "memory": packed_mem, "datasets": datasets}
    card, ms, err = _step(s, "quote", "Build quote", lambda: build_card(goal, research_packed))
    if not err:
        emit(run_id, "research_done", {"memory_source": (packed_mem or {}).get("source"), "datasets": [d["id"] for d in datasets]})
        _push(s, "quote", {k: card[k] for k in (
            "predicted_inr", "predicted_range_inr", "raw_inr", "multiplier", "warm_resume", "warm_resume_source",
            "budget_cap_inr", "live_l4_inr_per_hr", "rate_source", "m3_predicted_inr", "m3_actual_inr",
            "wer_before", "wer_after", "base_model", "method", "dataset", "gpu")})
        _push(s, "step_done", {"id": "quote", "status": "ok", "ms": ms, "label": f"₹{card['predicted_inr']} predicted"})
        _push(s, "choices_required", _choice_options(choices))
        mem_hits = (packed_mem or {}).get("hits") or _clean_hits(card.get("lessons") or [])
        lessons = [_lesson_line(mem_hits)]
        _push(s, "card", {"task": goal, "lessons": lessons, "memory_source": (packed_mem or {}).get("source"),
                          "no_train_in_demo": True})
    _finish(s)


def _choice_options(current: dict) -> dict:
    f = status()
    return {
        "current": current,
        "groups": [
            {"key": "provider", "title": "Control-plane sandbox", "options": [
                {"id": "boat", "label": "Boat", "sub": "vibetuner · $0.018/h", "available": True,
                 "note": "runs every jl SSH op (laptop :22 times out)"},
                {"id": "jarvis_cpu", "label": "Jarvis CPU", "sub": "stub · not wired", "available": False,
                 "note": "label only"},
                {"id": "local", "label": "Local", "sub": "Windows SSH blocked", "available": False,
                 "note": "AGENTS.md rule"},
            ]},
            {"key": "ranking", "title": "Dataset ranking", "options": [
                {"id": "sie", "label": "SIE Serverless", "sub": "Superlinked", "available": bool(f.superlinked and os.getenv("SIE_API_KEY")),
                 "note": "flag off" if not (f.superlinked and os.getenv("SIE_API_KEY")) else "live"},
                {"id": "lexical", "label": "Lexical", "sub": "keyword overlap", "available": True, "note": "local"},
            ]},
            {"key": "memory", "title": "Memory", "options": [
                {"id": "cognee", "label": "Cognee", "sub": "ready" if mem._COGNEE_OK else "indexing / not ready",
                 "available": bool(f.cognee), "note": "falls back to lexical if not ready"},
                {"id": "lexical", "label": "Lexical", "sub": "seed + progress docs", "available": True, "note": "local"},
            ]},
        ],
    }


def _approve_worker(s: dict, goal: str) -> None:
    _push(s, "receipt_start", {"label": "REPLAY of M3 r_fc806936 — no GPU created, no training"})
    try:
        rows = replay_m3(goal)
    except Exception as exc:
        _push(s, "error", {"error": f"{exc.__class__.__name__}: {exc}"})
        _finish(s)
        return
    for r in rows:
        _push(s, "receipt", {"replay": True, "ts": r.get("ts"), "event": r.get("event"),
                             "cost_inr": r.get("cost_inr"), "wer": r.get("wer"), "payload": r.get("payload")})
        time.sleep(0.18)
    jl = jl_snapshot(max_age=0)
    _push(s, "receipt_done", {
        "replay": True, "predicted_inr": 9.9, "actual_inr": 5.24, "wer_before": 0.0783, "wer_after": 0.0717,
        "running_gpus": jl.get("running") if jl.get("ok") else None,
        "running_source": "jl list (live)" if jl.get("ok") else "jl unavailable",
        "audit_backend": store().backend,
    })
    _finish(s)


# ---------------------------------------------------------------- routes


@app.on_event("startup")
def _startup() -> None:
    try:
        backfill()
    except Exception as exc:
        status().notes["singlestore"] = f"backfill failed {exc.__class__.__name__}"
    threading.Thread(target=_probe_cognee, daemon=True).start()


def _probe_cognee() -> None:
    # Full cognify loads a ~750 MB GLiNER model and OOMs the 4 GB sandbox; the chunk index
    # from an earlier add→cognify persists, and CHUNKS_LEXICAL search only needs that.
    f = status()
    if not f.cognee:
        return
    try:
        import asyncio

        import cognee
        from cognee import SearchType

        res = asyncio.run(cognee.search(query_text="GPU cost warm resume", query_type=SearchType.CHUNKS_LEXICAL))
        if mem._fmt_cognee(res):
            mem._COGNEE_OK = True
            f.notes["cognee"] = "chunk index (graph extraction skipped: 4 GB RAM)"
        else:
            f.notes["cognee"] = "index empty · lexical fallback"
    except Exception as exc:
        f.notes["cognee"] = f"probe failed {exc.__class__.__name__} · lexical fallback"


@app.get("/")
def index():
    return FileResponse(INDEX, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.post("/api/run")
async def api_run(request: Request):
    body = await request.json()
    goal = (body.get("goal") or "").strip()[:500] or "Transcribe whispered English better."
    choices = {"provider": "boat", "ranking": "sie", "memory": "cognee", **(body.get("choices") or {})}
    s = _new_stream("run")
    s["goal"] = goal
    threading.Thread(target=_run_worker, args=(s, goal, choices), daemon=True).start()
    return _stream_response(s)


@app.post("/api/approve")
async def api_approve(request: Request):
    body = await request.json()
    src = _STREAMS.get(body.get("run_id") or "")
    goal = (src or {}).get("goal") or body.get("goal") or "Transcribe whispered English better."
    s = _new_stream("approve")
    threading.Thread(target=_approve_worker, args=(s, goal), daemon=True).start()
    return _stream_response(s)


@app.get("/api/streams/{sid}")
def api_resume(sid: str, after: int = -1):
    s = _STREAMS.get(sid)
    if not s:
        return JSONResponse({"error": "unknown stream"}, status_code=404)
    return _stream_response(s, after)


def _runs_rows() -> list[dict]:
    db = store()
    with db.connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT run_id, phase, base_model, status, predicted_inr, actual_inr, wer_before, wer_after FROM runs")
        rows = cur.fetchall()
    cols = ["run_id", "phase", "base_model", "status", "predicted_inr", "actual_inr", "wer_before", "wer_after"]
    return [dict(zip(cols, tuple(r))) for r in rows]


@app.get("/api/spend")
def api_spend():
    f = status()
    db = store()
    events = db.events()
    m3 = [e for e in events if e["run_id"] == M3_RUN]
    order = ["search", "dataset_built", "plan_shown", "approved", "train_start", "train_end", "paused", "cost_delta"]
    funnel = [{"event": ev, "count": sum(1 for e in m3 if e["event"] == ev)} for ev in order]
    tools = {
        "context.dev searches": sum(1 for e in events if e["event"] == "search"),
        "plan quotes": sum(1 for e in events if e["event"] == "plan_shown"),
        "manifest ingests": sum(1 for e in events if e["event"] in ("manifest_ingested", "airbyte_sync_completed")),
        "replays": sum(1 for e in events if e["run_id"].startswith("replay-") and e["event"] == "approved"),
    }
    runs = [r for r in _runs_rows() if not str(r["run_id"]).startswith("replay-")]
    return {
        "source": f"audit db ({db.backend})",
        "amplitude": {"enabled": bool(f.amplitude and os.getenv("AMPLITUDE_API_KEY")),
                      "label": "mirrored to Amplitude" if f.amplitude and os.getenv("AMPLITUDE_API_KEY") else "local mirror (AMPLITUDE_API_KEY off)",
                      "url": "https://app.amplitude.com"},
        "runs": runs,
        "gpu_spend_inr": round(sum(r["actual_inr"] or 0 for r in runs), 2),
        "funnel": funnel,
        "tools": [{"tool": k, "count": v} for k, v in tools.items()],
    }


@app.get("/api/status")
def api_status():
    f = status()
    jl = jl_snapshot()
    if "l4" not in _JL_CACHE or time.time() - _JL_CACHE.get("l4_t", 0) > 120:
        try:
            _JL_CACHE["l4"] = live_l4_inr()
        except Exception:
            _JL_CACHE["l4"] = (None, "not fetched")
        _JL_CACHE["l4_t"] = time.time()
    rate, src = _JL_CACHE["l4"]
    has = lambda k: bool(os.getenv(k))
    tiles = [
        {"id": "cognee", "name": "Cognee", "on": f.cognee, "live": bool(mem._COGNEE_OK),
         "note": f.notes.get("cognee") or ("chunk index" if mem._COGNEE_OK else "lexical fallback")},
        {"id": "context", "name": "context.dev", "on": has("CONTEXT_DEV_API_KEY"), "live": has("CONTEXT_DEV_API_KEY"),
         "note": "web search" if has("CONTEXT_DEV_API_KEY") else "no key"},
        {"id": "singlestore", "name": "SingleStore", "on": f.singlestore, "live": store().backend == "singlestore",
         "note": f"audit on {store().backend}"},
        {"id": "airbyte", "name": "Airbyte", "on": f.airbyte, "live": f.airbyte,
         "note": "AIRBYTE_ENABLED" if f.airbyte else "flag off · pandas-jsonl"},
        {"id": "amplitude", "name": "Amplitude", "on": f.amplitude, "live": bool(f.amplitude and has("AMPLITUDE_API_KEY")),
         "note": "HTTP v2" if f.amplitude and has("AMPLITUDE_API_KEY") else "flag off · local mirror"},
        {"id": "statsig", "name": "Statsig", "on": f.statsig, "live": f.statsig,
         "note": f.statsig_source if f.statsig else "flag off · env defaults"},
        {"id": "superlinked", "name": "Superlinked SIE", "on": f.superlinked, "live": bool(f.superlinked and has("SIE_API_KEY")),
         "note": "rerank" if f.superlinked and has("SIE_API_KEY") else "flag off"},
        {"id": "jarvis", "name": "JarvisLabs", "on": jl.get("ok", False), "live": jl.get("ok", False),
         "note": f"{jl.get('running', '?')} running · {jl.get('paused', '?')} paused" if jl.get("ok") else "jl unavailable"},
        {"id": "boat", "name": "Boat", "on": True, "live": True, "note": "vibetuner bx_6nvrrech"},
    ]
    return {
        "tiles": tiles,
        "budget_cap_inr": f.budget_cap_inr,
        "warm_resume_enabled": f.warm_resume_enabled,
        "statsig_source": f.statsig_source,
        "whisper_gpu_fallback": f.whisper_gpu,
        "jarvis": {**jl, "l4_inr_per_hr": rate, "rate_source": src},
        "audit_backend": store().backend,
    }


@app.post("/api/memory")
def api_memory(body: dict):
    q = (body.get("q") or "what did we learn about GPU cost estimation?")[:300]
    packed = _memory_query(q)
    return {"source": packed.get("source"), "hits": [h[:600] for h in packed["hits"][:5]]}


@app.get("/api/audit")
def api_audit():
    rows = store().events(M3_RUN)
    return {"lineage": lineage_header(), "backend": store().backend, "events": rows,
            "receipt": {"predicted_inr": 9.9, "actual_inr": 5.24, "wer_before": 0.0783, "wer_after": 0.0717,
                        "machine_id": 580496, "run_id": M3_RUN}}


LIVE_DIR = ROOT / "logs" / "live"
_LIVE_PROC: dict = {"p": None}
_LIVE_LOCK = threading.Lock()


def _live_active() -> bool:
    p = _LIVE_PROC["p"]
    if p is not None and p.poll() is None:
        return True
    try:
        st = json.loads((LIVE_DIR / "state.json").read_text())
        fresh = time.time() - (LIVE_DIR / "state.json").stat().st_mtime < 300
        return fresh and st.get("state") in {"starting", "running"}
    except Exception:
        return False


@app.get("/api/live")
def api_live():
    state = json.loads((LIVE_DIR / "state.json").read_text()) if (LIVE_DIR / "state.json").exists() else {}
    points = []
    if (LIVE_DIR / "metrics.jsonl").exists():
        points = [json.loads(l) for l in (LIVE_DIR / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return {"state": state, "points": points, "active": _live_active(),
            "dispatch_enabled": os.getenv("LIVE_DISPATCH") == "1"}


def _job_summary(blob: dict) -> dict:
    st, pts = blob.get("state", {}), blob.get("points", [])
    evals = [p["wer"] for p in pts if p.get("kind") == "eval" and p.get("wer") is not None]
    base = next((p["wer"] for p in pts if p.get("kind") == "wer" and str(p.get("split", "")).startswith("base/")), None)
    return {"id": st.get("id"), "state": st.get("state"), "phase": st.get("phase"), "started": st.get("started"),
            "finished": st.get("finished"), "elapsed": st.get("elapsed"), "est_inr": st.get("est_inr"),
            "best_wer": min(evals) if evals else None, "base_wer": base, "epochs": len(evals),
            "n_points": len(pts), "store": store().backend, "machine_id": st.get("machine_id")}


@app.get("/api/jobs")
def api_jobs():
    jobs = []
    for f in sorted((LIVE_DIR / "jobs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            jobs.append(_job_summary(json.loads(f.read_text())))
        except Exception:
            continue
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    f = LIVE_DIR / "jobs" / f"{Path(job_id).name}.json"
    if not f.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return json.loads(f.read_text())


@app.post("/api/live/dispatch")
def api_live_dispatch():
    if os.getenv("LIVE_DISPATCH") != "1":
        return JSONResponse({"error": "dispatch disabled (LIVE_DISPATCH!=1)"}, status_code=403)
    with _LIVE_LOCK:
        if _live_active():
            return JSONResponse({"error": "a live run is already in flight — one GPU at a time"}, status_code=409)
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        out = open(LIVE_DIR / "relay.log", "w")
        _LIVE_PROC["p"] = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "live_run.py")],
                                           cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
                                           start_new_session=True)
    return {"ok": True, "pid": _LIVE_PROC["p"].pid}


def _decode_16k(path: str) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[:200])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


@app.post("/api/transcribe")
def api_transcribe(file: UploadFile = File(...)):
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        wav = _decode_16k(path)
    except Exception as exc:
        return JSONResponse({"error": f"decode failed: {exc}"}, status_code=400)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if wav.size < 1600:
        return JSONResponse({"error": "clip too short"}, status_code=400)
    from agent.whisper_cmp import transcribe_pair

    out = transcribe_pair((16000, wav))
    failed = out.get("device") == "none"
    return {
        "base": out.get("base"), "tuned": out.get("tuned"), "sec": out.get("sec"), "device": out.get("device"),
        "note": out.get("note", ""), "error": out.get("base") if failed else None,
        "clip_sec": round(wav.size / 16000, 2), "gpu_fallback_flag": status().whisper_gpu,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MISSION_PORT", "7861")))
