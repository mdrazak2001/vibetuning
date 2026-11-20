"""Cognee memory with a lexical fallback so the demo beat never dies."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from agent import ROOT
from agent.flags import status

SEED = ROOT / "data" / "memory_seed.md"
INGEST_FILES = [
    ROOT / "PROGRESS.md",
    ROOT / "AGENTS.md",
    ROOT / "data" / "dataset_card.md",
    ROOT / "run_spec.json",
    ROOT / "logs" / "m1" / "cost.json",
    ROOT / "logs" / "m3" / "cost.json",
    ROOT / "logs" / "m3" / "full_cost.json",
    ROOT / "logs" / "m3" / "plan.md",
    SEED,
]

_READY = False
_COGNEE_OK = False
_LOCK = asyncio.Lock() if False else None  # placeholder; we use a module flag


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


_COST_LESSON = (
    "M1 cold start billed **1.84×** the quote (₹8.26 predicted → ₹15.20 actual). "
    "M3 warm-resume of paused L4 billed **₹5.24 vs predicted ₹9.9** — do not apply "
    "the 1.84× cold multiplier on a warm resume. Live L4 is ₹41.31/hr."
)
_LICENSE_LESSON = (
    "Use **ylacombe/expresso** read/whisper, **CC BY-NC 4.0**, human transcripts. "
    "Do not download 36GB nyuuzyou/asmr (license unspecified; distil-whisper pseudo-labels)."
)


def _lexical(query: str, limit: int = 6) -> list[str]:
    ql = query.lower()
    out: list[str] = []
    if any(w in ql for w in ("cost", "gpu", "1.84", "warm", "overhead", "estimat", "quote", "₹", "inr")):
        out.append(_COST_LESSON)
        seed = _read(SEED)
        m = re.search(r"## GPU cost estimation[\s\S]+?(?=\n## |\Z)", seed)
        if m:
            out.append(m.group(0).strip()[:1200])
    if any(w in ql for w in ("license", "licensing", "expresso", "asmr", "cc-by", "dataset")):
        out.append(_LICENSE_LESSON)
    q = {t for t in re.findall(r"[a-z0-9.×x]+", ql) if len(t) > 3}
    scored: list[tuple[int, str]] = []
    for path in (SEED, ROOT / "PROGRESS.md", ROOT / "data" / "dataset_card.md"):
        blob = _read(path)
        for chunk in re.split(r"\n(?=## )", blob):
            if chunk.startswith("{") or "spot-check" in chunk.lower():
                continue
            low = chunk.lower()
            hit = sum(1 for t in q if t in low)
            if hit >= 2:
                scored.append((hit + (5 if path == SEED else 0), chunk.strip()[:900]))
    scored.sort(key=lambda x: -x[0])
    for _, c in scored:
        if c not in out:
            out.append(c)
        if len(out) >= limit:
            break
    if not out:
        out = [_COST_LESSON]
    return out[:limit]


def _fmt_cognee(results) -> list[str]:
    out = []
    if results is None:
        return out
    if isinstance(results, (str, int, float)):
        return [str(results)]
    for item in results:
        if item is None:
            continue
        if isinstance(item, str):
            out.append(item)
            continue
        text = getattr(item, "text", None) or getattr(item, "answer", None)
        if text:
            out.append(str(text))
            continue
        if isinstance(item, dict):
            out.append(str(item.get("text") or item.get("answer") or item)[:900])
            continue
        out.append(str(item)[:900])
    return [x for x in out if x.strip()]


async def _ingest_cognee() -> None:
    global _COGNEE_OK, _READY
    import cognee

    os.environ.setdefault("TELEMETRY_DISABLED", "true")
    dataset = os.getenv("COGNEE_DATASET", "vibetuning")
    texts = []
    for path in INGEST_FILES:
        if path.exists():
            body = path.read_text(encoding="utf-8")
            texts.append(f"# {path.name}\n\n{body[:12000]}")
    await cognee.add("\n\n---\n\n".join(texts), dataset_name=dataset)
    kwargs = {"datasets": [dataset]}
    try:
        await cognee.cognify(extractor="gliner_demo", **kwargs)
    except TypeError:
        await cognee.cognify(**kwargs)
    _COGNEE_OK = True
    _READY = True


def ingest_sync() -> dict:
    """Add → cognify. Falls back to lexical index if Cognee fights us."""
    global _READY, _COGNEE_OK
    flags = status()
    if not flags.cognee:
        _READY = True
        flags.notes["cognee"] = "COGNEE_ENABLED=off; lexical fallback"
        return {"ok": False, "mode": "lexical", "reason": "flag off"}
    try:
        asyncio.run(_ingest_cognee())
        flags.notes["cognee"] = "local SDK add→cognify"
        return {"ok": True, "mode": "cognee"}
    except RuntimeError:
        # Already in an event loop (Gradio).
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(_ingest_cognee())
            flags.notes["cognee"] = "cognify scheduled"
            return {"ok": True, "mode": "cognee-async"}
        except Exception as exc:
            _READY = True
            _COGNEE_OK = False
            flags.notes["cognee"] = f"failed: {exc.__class__.__name__}"
            return {"ok": False, "mode": "lexical", "reason": str(exc)[:200]}
    except Exception as exc:
        _READY = True
        _COGNEE_OK = False
        flags.notes["cognee"] = f"failed: {exc.__class__.__name__}: {str(exc)[:120]}"
        return {"ok": False, "mode": "lexical", "reason": str(exc)[:200]}


async def query(question: str, limit: int = 6) -> dict:
    flags = status()
    lexical = _lexical(question, limit=limit)
    if not flags.cognee or not _COGNEE_OK:
        return {"source": "lexical", "hits": lexical}
    try:
        import cognee
        from cognee import SearchType

        results = await cognee.search(
            query_text=question,
            query_type=SearchType.CHUNKS_LEXICAL,
        )
        hits = _fmt_cognee(results)
        if not hits:
            hits = lexical
            return {"source": "cognee+lexical", "hits": hits}
        return {"source": "cognee", "hits": hits[:limit]}
    except Exception as exc:
        flags.notes["cognee"] = f"search failed: {exc.__class__.__name__}"
        return {"source": "lexical", "hits": lexical, "error": str(exc)[:160]}


def query_sync(question: str) -> dict:
    try:
        return asyncio.run(query(question))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(query(question))


def lessons_for_plan() -> list[str]:
    packed = query_sync("GPU cost estimation cold vs warm resume 1.84x dataset licensing")
    return packed.get("hits") or []
