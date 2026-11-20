#!/usr/bin/env python3
"""vibetuning demo: shop data+GPU, approve a replay, hear the adapter, read the audit trail."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_k):
        return False

load_dotenv(ROOT / ".env")
load_dotenv(Path.home() / ".vibetuning" / "secrets.env")

import gradio as gr

from agent.flags import status
from agent.ingest import ingest_manifests, lineage_header
from agent.memory import ingest_sync, query_sync
from agent.plan import build_card, card_markdown
from agent.replay import M3_RUN, backfill, replay_m3
from agent.research import research
from agent.whisper_cmp import transcribe_pair
from agent.audit import store

DEFAULT_TASK = (
    "Fine-tune a tiny ASR model so it transcribes whispered / ASMR English better, "
    "without wrecking normal speech."
)


def _boot() -> str:
    flags = status()
    bf = backfill()
    ing = ingest_manifests()
    threading.Thread(target=ingest_sync, daemon=True).start()
    return (
        f"audit={bf.get('backend')} ingest={ing.get('mode')} n={ing.get('n')} · "
        f"statsig={flags.statsig_source} cap=₹{flags.budget_cap_inr} "
        f"warm={flags.warm_resume_enabled} · cognee ingest in background"
    )


def _flag_panel() -> str:
    s = status()
    rows = [
        ("Cognee", s.cognee, s.notes.get("cognee", "")),
        ("SingleStore", s.singlestore and store().backend == "singlestore", f"{store().backend} {s.notes.get('singlestore','')}"),
        ("Airbyte", s.airbyte, s.notes.get("airbyte", "")),
        ("Amplitude", s.amplitude, s.notes.get("amplitude", "")),
        ("Statsig", s.statsig, s.notes.get("statsig", s.statsig_source)),
        ("Superlinked", s.superlinked, s.notes.get("superlinked", "")),
        ("context.dev", True, s.notes.get("context.dev", "")),
    ]
    lines = ["| Integration | Live | Notes |", "|---|---|---|"]
    for name, live, note in rows:
        lines.append(f"| {name} | {'on' if live else 'off'} | {note} |")
    lines.append(f"| budget_cap_inr | ₹{s.budget_cap_inr} | {s.statsig_source} |")
    lines.append(f"| warm_resume_enabled | {s.warm_resume_enabled} | {s.statsig_source} |")
    return "\n".join(lines)


def do_research(task: str):
    packed = research(task or DEFAULT_TASK)
    card = build_card(task or DEFAULT_TASK, packed)
    return card_markdown(card), json.dumps(card, indent=2)[:8000], _flag_panel()


def do_memory(q: str):
    packed = query_sync(q or "what did we learn about GPU cost estimation?")
    hits = packed.get("hits") or []
    body = f"**source:** `{packed.get('source')}`\n\n" + "\n\n".join(f"> {h}" for h in hits[:6])
    return body


def do_approve(task: str, card_json: str):
    replay_m3(task or DEFAULT_TASK)
    return (
        "Approved. **No GPU was created.** Replayed M3: predicted ₹9.9 → actual **₹5.24**, "
        "WER **7.83% → 7.17%**, `jl pause 580496`, 0 running.\n\nOpen the Audit trail tab.",
        audit_view(),
    )


def audit_view() -> str:
    header = lineage_header()
    rows = store().events(M3_RUN)
    if not rows:
        rows = store().events()
    lines = [
        f"### Lineage\n{header}\n",
        "### M3 timeline — every decision has a row\n",
        "| ts | event | ₹ | WER | payload |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        payload = r.get("payload") or {}
        if isinstance(payload, dict):
            brief = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:4])
        else:
            brief = str(payload)[:120]
        cost = r.get("cost_inr")
        wer = r.get("wer")
        lines.append(
            f"| {r.get('ts','')} | `{r.get('event')}` | {'' if cost is None else cost} | "
            f"{'' if wer is None else wer} | {brief[:160]} |"
        )
    lines.append("\nReceipt: **₹9.9 predicted vs ₹5.24 actual**. WER **0.0783 → 0.0717**. Pause proof in `paused` row.")
    return "\n".join(lines)


def do_whisper(audio):
    out = transcribe_pair(audio)
    note = out.get("note") or ""
    meta = f"{out.get('device')} · {out.get('sec')}s{note}"
    return out.get("base") or "", out.get("tuned") or "", meta


BOOT = _boot()

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .gradio-container { font-family: 'IBM Plex Sans', sans-serif; background: #f4efe6 !important; color: #1c1915; }
.gradio-container { max-width: 1120px !important; }
h1, h2, h3 { font-family: Fraunces, serif; letter-spacing: -0.02em; }
button.primary { background: #1c1915 !important; color: #f4efe6 !important; }
#receipt { background: #fffdf8; border: 1px solid #d9cfc0; padding: 12px 16px; }
"""

with gr.Blocks(title="vibetuning — personal training agent", css=CSS, theme=gr.themes.Soft(primary_hue="stone", font=["IBM Plex Sans", "sans-serif"])) as demo:
    gr.Markdown(
        "# vibetuning\n"
        "A personal agent that shops for **datasets + GPU compute**, buys a LoRA, and leaves "
        "an audit trail from prompt to weights.\n\n"
        f"<span id='receipt'>{BOOT}</span>"
    )
    with gr.Tabs():
        with gr.Tab("1 · Vibe-tune"):
            task = gr.Textbox(label="One-sentence model goal", value=DEFAULT_TASK, lines=2)
            with gr.Row():
                research_btn = gr.Button("Research + quote", variant="primary")
                mem_btn = gr.Button("Ask memory")
            mem_q = gr.Textbox(
                label="Memory question",
                value="what did we learn about GPU cost estimation?",
            )
            mem_out = gr.Markdown()
            card_md = gr.Markdown()
            card_json = gr.Textbox(visible=False)
            flags_md = gr.Markdown(_flag_panel())
            approve = gr.Button("APPROVE — replay M3 (no training)", variant="primary")
            approve_out = gr.Markdown()
            research_btn.click(do_research, inputs=task, outputs=[card_md, card_json, flags_md])
            mem_btn.click(do_memory, inputs=mem_q, outputs=mem_out)
            mem_q.submit(do_memory, inputs=mem_q, outputs=mem_out)
            approve.click(do_approve, inputs=[task, card_json], outputs=[approve_out, flags_md])

        with gr.Tab("2 · Whisper"):
            gr.Markdown(
                "Base `openai/whisper-small` vs the M3 LoRA at `artifacts/m3/adapter/`. "
                "No retraining. If CPU >15 s/clip, GPU fallback stays behind `WHISPER_GPU_FALLBACK` (₹20 cap)."
            )
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Whispered clip")
            go = gr.Button("Transcribe both", variant="primary")
            with gr.Row():
                base_tb = gr.Textbox(label="Base whisper-small", lines=4)
                tuned_tb = gr.Textbox(label="M3 LoRA", lines=4)
            meta = gr.Markdown()
            go.click(do_whisper, inputs=audio, outputs=[base_tb, tuned_tb, meta])

        with gr.Tab("3 · Audit trail"):
            gr.Markdown("SingleStore when the wire is live; SQLite if not. Same schema either way.")
            refresh = gr.Button("Refresh timeline")
            trail = gr.Markdown(audit_view())
            refresh.click(lambda: audit_view(), outputs=trail)

    gr.Markdown(
        "context.dev discovers data anywhere; Airbyte ingests it from 700+ sources; "
        "every training row lands in SingleStore with lineage — audit trail from prompt to weights."
    )

if __name__ == "__main__":
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=bool(os.getenv("GRADIO_SHARE")))
