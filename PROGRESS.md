# vibetuning progress

Hackathon loop: one-sentence task → web data → training set → plan + cost → approved LoRA SFT on a rented GPU.

Budget rules: **$3 cap per GPU run**, **$10 total**. Never create a GPU instance without an explicit **GO** in chat. Prefer L4. Pause every GPU path in `finally`. Checkpoint every 200 steps.

## M0 — zero GPU spend (done 2026-11-19)

### 1. Boat sandbox `vibetuner`

- Id: `bx_6nvrrech`
- Size: `small` (2 vCPU, 4 GB, billed at 0.5× default)
- Auto-stop: **off** (`ttlSeconds` is null)
- SSH: **works** (`boat ssh bx_6nvrrech` → Ubuntu 24.04.4, `SSH_OK`)
- Also present: leftover stopped sandbox `bx_kzyxk4bp` from account onboarding (not used)

Boat account is a gifted `$20` plan with ~555 included VM hours remaining. Control-plane time is not GPU spend.

### 2. context.dev search

Query: `whispered speech ASR dataset huggingface`  
Credits: 1 consumed, 15,999 remaining.

| # | Result | Notes |
| --- | --- | --- |
| 1 | [danielrosehill/ASR-WPM-And-Background-Noise-Eval](https://huggingface.co/datasets/danielrosehill/ASR-WPM-And-Background-Noise-Eval) | Includes whispered speaking-pace samples |
| 2 | [Firoj112/nepali-asr-whisper](https://huggingface.co/datasets/Firoj112/nepali-asr-whisper) | Whisper-ready ASR pack (Nepali, not whispered) |
| 3 | [WhispSynth (arXiv)](https://arxiv.org/abs/2603.14853v1) | Survey of whisper corpora; flags Expresso (BY-NC) and wTIMIT |
| 4 | [WhispEar (arXiv)](https://arxiv.org/abs/2603.08046v1) | Whispered-speech conversion + more corpus pointers |
| 5 | [m-aliabbas1/asr-noise-robust-en](https://huggingface.co/datasets/m-aliabbas1/asr-noise-robust-en) | English noise-robust ASR mix |

Do **not** pull the full 36GB `nyuuzyou/asmr` archive. Fetch `meta.csv` first. Expresso is the cleaner-licensed whisper-style alternative (may be gated on HF). Flag licenses in `dataset_card.md` during M2.

### 3. JarvisLabs `jl gpus` (live, 2026-11-19 evening PT)

Account: grants **₹49,449.90**, cash balance ₹0, 0 running / 0 paused. CLI bills in INR.

**Containers (IN2 unless noted)**

| GPU | Stock | VRAM | ₹/hr | Spot ₹/hr |
| --- | --- | --- | --- | --- |
| L4 | **7 free** | 24GB | 41.31 | 27.54 |
| A100 40GB | **3 free** | 40GB | 84.24 | 74.52 |
| A100-80GB | 0 free | 80GB | 140.94 | 84.24 |
| A30 | available | 24GB | 38.88 | 27.54 |
| H200 | available | 141GB | 434.97 | 217.08 |
| RTX-PRO6000 (IN1) | available | 96GB | 207.36 | 109.35 |

L4 is in stock. Planning numbers from the brief: L4 **$0.44/hr**, A100 40GB **$0.89/hr**. M1 will use L4 if you GO.

Auth note: `jl` wants `JL_API_KEY`. The key was saved as `JARVIS_API` in `.env` (gitignored). Both names now work locally after `jl setup`.

## M1 — plumbing (done 2026-11-19, GO received)

Plan card: `logs/m1/plan.md`. Cost card: `logs/m1/cost.json`.

| Field | Value |
| --- | --- |
| GPU | On-demand L4 container `580359` (`vibetuning-m1`), IN2, **not spot** |
| Live rate | ₹41.31 / hr |
| FX | ₹93.29 / USD |
| Predicted | 12 min → **₹8.26 / $0.09** |
| Timeout ceiling | 30 min → **₹20.66 / $0.22** |
| Actual | grants 49448.39 → 49434.70 = **₹13.69** API delta; **logged actual ₹15.20** (includes create/SSH-retry overhead) ≈ **$0.16** |
| WER before | **0.238261** |
| WER after | **0.240000** |
| Adapter | `artifacts/m1/adapter/adapter_model.safetensors` + `adapter_config.json` |
| Pause | `jl pause 580359` succeeded; `running_instances: 0` |
| Hard timeout | not hit (~18 min wall clock) |

WER is plumbing-only (same dummy split, 150 steps). Do not treat the tiny after/before delta as a quality result.

Windows cannot SSH to JarvisLabs `:22` (connection timed out). The L4 was created with `jl run --gpu L4` from this laptop; upload failed; the job was then driven from Boat `bx_6nvrrech`, which **can** SSH to the GPU. First train hung on `datasets.map` after CUDA init; stopped and reran with in-process feature prep (`r_3a26277a`, exit 0).

**M1 acceptance:** adapter on disk, WER printed, 0 running, actual **₹15.20** (< $1), logs in `logs/m1/`. Predicted vs actual for the M3 cost card: **₹8.26 predicted → ₹15.20 actual** (1.84×, Windows SSH failure + hung `datasets.map` retry).

Laptop SSH to JarvisLabs `:22` is broken. All future `jl` SSH ops run from `vibetuner`. See `AGENTS.md`.

## M2 — dataset forge (done 2026-11-19, GO received)

Zero GPU. Boat `vibetuner` CPU + context.dev only. `jl list`: 0 running (M1 L4 `580359` still paused).

### Discovery

| Source | License | Gating | Verdict |
| --- | --- | --- | --- |
| [ylacombe/expresso](https://huggingface.co/datasets/ylacombe/expresso) | **CC BY-NC 4.0** | public / ungated | **used** — `read` + `style==whisper`, human transcripts |
| `facebook/expresso` | — | missing or auth-walled | not usable |
| [nyuuzyou/asmr](https://huggingface.co/datasets/nyuuzyou/asmr) | **unspecified** (`other`) | public card | **not used**. Labels = distil-whisper/distil-large-v3.5 (**distillation, not human-verified**). Did not download `dataset.tar.zst` (36.3 GB) or `meta.csv`. |
| WhispSynth / WhispEar papers | n/a | n/a | pointers at Expresso/wTIMIT, not a clip dump |

### Built set (`data/expresso_whisper/`)

| Stat | Value |
| --- | --- |
| Clips | **1139** (0 dropped) |
| Hours | **0.993** (~59.6 min of 16 kHz audio; upstream card lists 79 min read-whisper) |
| Avg / max duration | 3.14 s / 12.19 s (all ≤ 30 s) |
| Train / val | 1025 / 114 (seed 42, speaker-stratified) |
| Speakers | ex01 379, ex02 380, ex03 380 |
| Format | 16 kHz mono PCM16 WAV |
| Labels | human Expresso read transcripts |
| GPU spend | **₹0** |

Held-out TEST loader: `recipes/sft_whisper/eval_loader.py` → `eval_whisper/` (awaiting dropped recordings).

Spot-check (decode + 16 kHz/mono + RMS; transcripts from the source card):

1. `ex01_whisper_00024` 3.20 s — *Hey, how were those oysters?*
2. `ex02_whisper_00010` 3.19 s — *Well hey! What did you do to get in here?*
3. `ex03_whisper_00282` 3.20 s — *You can expect a high of sixty and a low of forty seven.*

Wavs in `data/samples/`. Full audio stays on vibetuner (`data/expresso_whisper/audio/`, gitignored).

## M3 — probe (done 2026-11-19, GO for probe only)

On-demand L4 `580479` (`vibetuning-m3-probe`), from vibetuner. 50 steps whisper-small LoRA + 8-clip generate eval.

| Probe metric | Value |
| --- | --- |
| Time-to-first-step (in-process) | 35.3 s |
| sec/train-step (steady) | 0.373 s |
| sec/eval-clip | 0.185 s |
| Probe val WER (normalized, 8 clips) | 0.053 |
| Live L4 | ₹41.31 / hr |
| Probe GPU spend | **₹5.26** (includes one failed folder-venv clash); successful slice ₹2.24 |
| Running after pause | **0** |

## M3 — full run (done 2026-11-19, GO received)

On-demand L4 `580496` (resume of paused `580479` / `vibetuning-m3-probe`), from vibetuner. whisper-small LoRA, 3 epochs, 387 steps, early-stop kept **checkpoint-258** (epoch 2). fp16. File-mode `jl run` of `train_m3.py` (`r_fc806936`).

| Field | Value |
| --- | --- |
| GPU | On-demand L4 IN2, **not spot** |
| Live rate | ₹41.31 / hr |
| Predicted | **₹9.9** (range ₹7.4–₹12.4) |
| Actual | grants 49419.43 → 49414.19 = **₹5.24** (~$0.06) |
| Wall | 442.9 s (~7.4 min); script 289.9 s; caps ₹150 / 45 min **not hit** |
| WER before (Expresso val, 114, English-normalized) | **0.0783** |
| WER after (best ckpt) | **0.0717** |
| Epoch eval WER | 1: 0.0794 → 2: **0.0717** → 3: 0.0827 (kept epoch 2) |
| `eval_whisper/` / `eval_normal/` | skipped (empty; n=0) |
| Adapter | `artifacts/m3/adapter/` (`adapter_config.json` + gitignored `adapter_model.safetensors`) |
| Pause | `jl pause 580496` succeeded; `jl list` **0 running** |

`eval_whisper/` and `eval_normal/` still have no user audio, so held-out / forgetting WER was not measured. One in-domain example: base hyp *danza assault* → tuned *dancer assault* (matches ref).

**M3 acceptance:** adapter on disk, base vs tuned WER printed, 0 running, actual **₹5.24** under predicted ₹9.9 and well under ₹150. Metrics: `logs/m3/full_metrics.json`. Cost: `logs/m3/full_cost.json`. Spec: `run_spec.json` (`full_run_done`).

Waiting on GO for M4 (`SKILL.md`).
