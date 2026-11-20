# tuno

tuno (repo: vibetuning)

A personal agent that shops for **datasets + GPU compute**, buys a LoRA, and leaves a row for every decision.

> context.dev discovers data anywhere; Airbyte ingests it from 700+ sources; every training row lands in SingleStore with lineage — audit trail from prompt to weights.

```mermaid
flowchart LR
  T[One-sentence goal] --> R[context.dev search]
  R --> M[Cognee memory]
  M --> C[Plan / cost card]
  C -->|APPROVE replay| A[Audit trail]
  subgraph shop [Shop]
    R
    SIE[Superlinked SIE rank]
    R --> SIE
  end
  subgraph buy [Buy]
    C
    JL[jl gpus live ₹]
    C --> JL
  end
  subgraph trail [Every decision has a row]
    AB[Airbyte source-file]
    SS[(SingleStore)]
    AMP[Amplitude]
    AB --> SS
    A --> SS
    A --> AMP
  end
  W[Whisper mic] --> B[whisper-small]
  W --> L[M3 LoRA adapter]
```

## Why each tool

| Tool | Role | Why this one |
| --- | --- | --- |
| **context.dev** | Discover datasets on the live web | The agent has to shop, not assume Expresso. |
| **Cognee** | Agent memory of *our* runs | Quotes the M1 **1.84×** cold overhead and the M3 **warm-resume** lesson before it prices a GPU. |
| **Superlinked SIE** | Rank discovered datasets to the task sentence | Stretch; flag-off if no `SIE_API_KEY`. |
| **Airbyte** | Ingest the curated manifest | Lineage for every training row (not search). |
| **SingleStore** | `runs` + `events` + `training_rows` | The audit DB. MySQL wire. SQLite fallback so the demo never blanks. |
| **Amplitude** | HTTP v2 mirror of the same events | Funnel of the M3 run. |
| **Statsig** | One runtime flag | `warm_resume_enabled` and/or `budget_cap_inr` actually read before the quote. |
| **JarvisLabs `jl`** | Live L4 ₹/hr + the real M0–M3 training | Already ran. Demo **does not retrain**. |
| **Gradio** | Three-tab demo on vibetuner | Sentence → card → approve → audit → live ASR. |

## The audit-trail story

Every decision has a row in `events(run_id, ts, event, payload, cost_inr, wer)`:

`search` → `dataset_built` (license on the row) → `plan_shown` (₹ quote) → `approved` → `train_start` → `train_end` (WER) → `paused` (0 running) → `cost_delta` (predicted vs actual).

M3 receipt: **₹9.9 predicted vs ₹5.24 actual**. WER **7.83% → 7.17%**. Pause proof is a row, not a screenshot.

## Built today vs already there

**Pre-existing core (M0–M3, do not retrain)**

- Boat sandbox `vibetuner`, JarvisLabs L4 jobs, Expresso forge (`ylacombe/expresso`, CC BY-NC 4.0)
- whisper-small LoRA at `artifacts/m3/adapter/`
- Probe + full-run cost cards, `run_spec.json`, WER protocol

**Built today (hackathon demo)**

- Cognee memory over progress/cost/license docs
- SingleStore (or SQLite) audit + M1/M3 backfill
- Airbyte `source-file` → `training_rows` lineage (pandas fallback)
- Amplitude HTTP v2 + Statsig runtime flag
- Superlinked SIE rank (stretch, flagged)
- Gradio: Vibe-tune / Whisper / Audit trail

## Demo

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements-demo.txt
copy .env.example .env   # paste hackathon keys; flags default safe
python app.py      # Gradio, :7860
python server.py   # mission control, :7861
```

1. **Vibe-tune** — task sentence → research (context.dev + cognee) → plan/cost with live `jl gpus` ₹ and cold/warm overhead → **APPROVE** replays M3 (no GPU).
2. **Whisper** — mic/file, base vs adapter. GPU serving fallback is `WHISPER_GPU_FALLBACK=1` with a **₹20** cap, paused after.
3. **Audit trail** — lineage header + M3 timeline.

Ask memory: *what did we learn about GPU cost estimation?* It should return M1 **1.84×** and M3 warm-resume **₹5.24 vs ₹9.9**.

### Mission control (single page, :7861)

| Light | Dark |
|---|---|
| ![hero light](docs/screenshots/mission-light-hero.png) | ![hero dark](docs/screenshots/mission-dark-hero.png) |
| ![receipt light](docs/screenshots/mission-light-receipt.png) | ![receipt dark](docs/screenshots/mission-dark-receipt.png) |

The flow moves through phases: hero → research → stack → quote → receipt, with one primary action at a time. Finished phases collapse into one-line summaries. The theme toggle persists in `localStorage`, and vendor logos live in `assets/logos/`.

`python server.py` serves `web/index.html`. It's a FastAPI wrapper over the same `agent/*` calls, and the Gradio app stays on :7860 as fallback.
- `POST /api/run` streams each step over SSE: parse → cognee → context.dev → SIE → license → ingest → jl price → quote. Every step is a real call or is labelled *flag off* / *fallback*.
- `POST /api/approve` streams the M3 audit rows, labelled **REPLAY**.
- Streams resume via `GET /api/streams/{id}?after=N`.
- `/api/spend`, `/api/status`, `/api/audit` and `/api/memory` feed the drawers. `/api/transcribe` feeds the whisper A/B (CPU torch needed).
- No secrets leave the server.

### Architecture

```mermaid
flowchart LR
  U[Goal in plain English] --> MC[tuno mission control<br/>FastAPI + SSE on Boat]
  MC --> CG[Cognee<br/>cross-context memory]
  MC --> CD[context.dev<br/>dataset search]
  MC --> SL[Superlinked SIE<br/>rerank]
  SL --> LG[License gate]
  LG --> IN[Ingest<br/>Airbyte path / pandas fallback]
  IN --> JQ[JarvisLabs live price<br/>+ Cognee 1.84x lesson]
  JQ --> Q[Quote → Approve]
  Q --> D[Dispatch job]
  D --> R[Relay on Boat<br/>scripts/live_run.py]
  R -->|warm resume · jl run| GPU[JarvisLabs L4<br/>LoRA SFT]
  GPU -->|jl run logs / 15s| R
  R --> ST[(Audit store<br/>SingleStore or SQLite)]
  R --> AM[Amplitude]
  R --> JB[Jobs view · loss + WER]
  R -->|finally| P[GPU paused]
```

### Live training (Jobs)

![jobs](docs/screenshots/mission-jobs.png)

`POST /api/live/dispatch` (needs `LIVE_DISPATCH=1`) launches `scripts/live_run.py` on the Boat control plane. It refuses to dispatch while another run is in flight.
- It warm-resumes the paused L4 and never creates a GPU. Hard caps are 20 min and ₹40, and the GPU is paused in `finally`.
- It re-runs the M3 recipe and reads `jl run logs` every 15 s for loss, epoch WER and phase.
- Points go to `logs/live/metrics.jsonl`, the audit store, and Amplitude (`live_train_step`, `live_epoch_eval`, `live_wer`). `GET /api/live` feeds the charts.

The first live run took 6:12 and cost about ₹4.26. Expresso-val WER went from a base of 0.0783 to 0.0816, then 0.0695, then 0.0750 across epochs 1–3. The best checkpoint (epoch 2, 0.0695) is kept.

## Flags

Every integration degrades. `COGNEE_ENABLED`, `SINGLESTORE_ENABLED`, `AIRBYTE_ENABLED`, `AMPLITUDE_ENABLED`, `STATSIG_ENABLED`, `SUPERLINKED_ENABLED`, `WHISPER_GPU_FALLBACK`. Missing keys → off. The demo still runs.

## License note

Training audio is **CC BY-NC 4.0** (Expresso). Demo and agent code in this repo: see commit history. Do not use the adapter commercially.
