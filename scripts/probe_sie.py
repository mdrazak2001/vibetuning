"""Find an SIE reranker and try one score call. Prints shapes only, never keys."""

import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.flags import status  # noqa: E402

status()
from sie_sdk import Item, SIEClient  # noqa: E402

print("Item keys:", getattr(Item, "__annotations__", {}))
c = SIEClient(api_key=os.environ["SIE_API_KEY"], timeout_s=40)
print("base_url:", c.base_url)
try:
    models = c.list_models()
    names = [getattr(m, "id", None) or getattr(m, "name", None) or str(m) for m in models]
    print("n models:", len(names))
    print("rerank-ish:", [n for n in names if any(k in str(n).lower() for k in ("rerank", "score", "cross"))][:15])
except Exception as exc:
    print("list_models failed:", exc.__class__.__name__, str(exc)[:300])
    names = []
model = os.getenv("SIE_RERANK_MODEL") or next((n for n in names if "rerank" in str(n).lower()), "BAAI/bge-reranker-v2-m3")
print("trying model:", model)
try:
    out = c.score(model, {"text": "whispered English ASR dataset"},
                  [{"text": "ylacombe/expresso CC BY-NC human transcripts"}, {"text": "nyuuzyou/asmr unspecified pseudo-labels"}],
                  provision_timeout_s=30)
    print("call ok:", type(out).__name__, repr(out)[:500])
except Exception as exc:
    print("call failed:", exc.__class__.__name__, str(exc)[:400])
