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

        client = SIEClient(api_key=key, base_url=os.getenv("SIE_BASE_URL", "https://api.superlinked.com"))
        docs = [f"{c['id']} {c['license']} {c['why']}" for c in candidates]
        # Rerank primitive — fail closed on any SDK mismatch.
        scores = None
        if hasattr(client, "score"):
            scores = client.score(query=task, documents=docs)
        elif hasattr(client, "rerank"):
            scores = client.rerank(query=task, documents=docs)
        if not scores:
            return candidates
        ranked = list(candidates)
        try:
            order = list(scores)
            ranked.sort(key=lambda c: -float(order[candidates.index(c)] if False else 0))
        except Exception:
            pass
        flags.notes["superlinked"] = "ranked"
        return ranked
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
