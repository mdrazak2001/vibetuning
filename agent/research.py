"""Task sentence → context.dev search + cognee lessons + optional Superlinked rank."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from agent.events import emit
from agent.flags import status
from agent.memory import lessons_for_plan, query_sync
from agent.replay import M3_RUN

KNOWN_DATASETS = [
    {
        "id": "ylacombe/expresso",
        "license": "CC BY-NC 4.0",
        "labels": "human transcripts",
        "why": "public ungated read/whisper split; 0.993 h; used in M2/M3",
        "url": "https://huggingface.co/datasets/ylacombe/expresso",
    },
    {
        "id": "nyuuzyou/asmr",
        "license": "unspecified (other)",
        "labels": "distil-whisper pseudo-labels",
        "why": "36GB tar; do not download; distillation not human-verified",
        "url": "https://huggingface.co/datasets/nyuuzyou/asmr",
    },
    {
        "id": "facebook/expresso",
        "license": "unknown",
        "labels": "n/a",
        "why": "missing or auth-walled on HF",
        "url": "https://huggingface.co/datasets/facebook/expresso",
    },
]


def _context_search(query: str, n: int = 5) -> list[dict]:
    key = os.getenv("CONTEXT_DEV_API_KEY")
    if not key:
        status().notes["context.dev"] = "no CONTEXT_DEV_API_KEY"
        return []
    body = json.dumps({"query": query, "numResults": n}).encode()
    req = urllib.request.Request(
        "https://api.context.dev/v1/web/search",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        status().notes["context.dev"] = f"search failed {exc.__class__.__name__}"
        return []
    hits = data.get("results") or data.get("data") or data.get("organic") or []
    out = []
    for h in hits[:n]:
        if isinstance(h, dict):
            out.append(
                {
                    "title": h.get("title") or h.get("name") or "",
                    "url": h.get("url") or h.get("link") or "",
                    "snippet": (h.get("snippet") or h.get("text") or h.get("description") or "")[:280],
                }
            )
    status().notes["context.dev"] = f"{len(out)} hits"
    return out


def _superlinked_rank(task: str, candidates: list[dict]) -> list[dict]:
    flags = status()
    key = os.getenv("SIE_API_KEY")
    if not flags.superlinked or not key:
        flags.notes.setdefault("superlinked", "flag off")
        return candidates
    try:
        from sie_sdk import SIEClient

        model = os.getenv("SIE_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        client = SIEClient(api_key=key, base_url=os.getenv("SIE_BASE_URL", "https://api.superlinked.com"), timeout_s=30)
        items = [{"id": c["id"], "text": f"{c['id']} · {c['labels']} · {c['why']}"} for c in candidates]
        out = client.score(model, {"text": task}, items, provision_timeout_s=30)
        by_id = {s["item_id"]: float(s["score"]) for s in (out or {}).get("scores", [])}
        if not by_id:
            flags.notes["superlinked"] = "no scores returned"
            return candidates
        scored = [{**c, "score": round(by_id.get(c["id"], by_id.get(f"item-{i}", 0.0)), 4)} for i, c in enumerate(candidates)]
        scored.sort(key=lambda c: -c["score"])
        flags.notes["superlinked"] = f"ranked · {model}"
        return scored
    except Exception as exc:
        flags.notes["superlinked"] = f"failed {exc.__class__.__name__}"
        return candidates


def research(task: str, run_id: str = M3_RUN) -> dict:
    hits = _context_search(f"{task} huggingface dataset license whispered speech ASR")
    emit(run_id, "search", {"task": task, "tool": "context.dev", "n": len(hits)})
    memory = query_sync("what did we learn about GPU cost estimation?")
    license_mem = query_sync("dataset licensing Expresso CC BY-NC asmr")
    ranked = _superlinked_rank(task, KNOWN_DATASETS)
    emit(
        run_id,
        "research_done",
        {
            "memory_source": memory.get("source"),
            "datasets": [c["id"] for c in ranked],
        },
    )
    return {
        "task": task,
        "web": hits,
        "memory": memory,
        "license_memory": license_mem,
        "datasets": ranked,
    }
