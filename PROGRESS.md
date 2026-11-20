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

## M1 — plumbing (waiting for GO)

Hardcoded whisper-tiny + LoRA ~150 steps on `hf-internal-testing/librispeech_asr_dummy` via `jl run --gpu L4`. Adapter download, auto-pause, local WER before/after. Acceptance: spend < $1, adapter on disk, WER printed.

**Not started.** Reply **GO** to run M1.
