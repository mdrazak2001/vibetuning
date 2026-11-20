# M3 plan / cost card — STOPPED for full-run GO

Probe is done. **Do not start the full run until a new GO in chat.**

## What we will train (full run — not executed)

| Field | Proposal |
| --- | --- |
| Base model | `openai/whisper-small` |
| Method | LoRA SFT (PEFT), `r=8`, `alpha=16`, targets `q_proj`+`v_proj`, fp16 |
| Dataset | Expresso read whisper, **0.993 h**, **CC BY-NC 4.0**, 1025 train / 114 val. Audio on vibetuner. |
| Epochs / steps | **3 epochs** (in the 2–4 band) ≈ **385 steps** at batch 8. Early stop on **val WER** (patience 1 epoch), keep best checkpoint. |
| GPU | On-demand **L4** IN2, not spot. Reuse paused `580479` (`vibetuning-m3-probe`) or create if resume fails. |
| Caps | **₹150** and **45 min** wall-clock. Pause on every exit path (success, fail, timeout). |
| Launch | **From vibetuner only** (Windows SSH to `:22` times out). |

## Eval protocol (full run)

jiwer WER after **English text normalization on both refs and hyps** (`transformers` `BasicTextNormalizer`). Report **base vs tuned** on:

| Split | Path | Notes |
| --- | --- | --- |
| (a) Expresso whisper val | `data/expresso_whisper/manifest_val.jsonl` (114) | in-domain |
| (b) Recorded whisper | `eval_whisper/` | drop m4a/mp3/wav; loader converts to 16 kHz mono |
| (c) Normal voice | `eval_normal/` | catastrophic-forgetting check; same loader |

Empty (b)/(c) are skipped with a log line, not a crash. Probe WER on 8 val clips after 50 steps: **0.053** (sanity only, not a quality claim).

## Probe measurements (r_2ebaedf6, L4, whisper-small)

| Quantity | Value |
| --- | --- |
| Time-to-first-step (in-process: pip+model+200 clips+step 1) | **35.3 s** |
| Probe wall (resume+upload+apt+script+pause) | **197.7 s** |
| Fixed wall beyond script | **135 s** |
| sec/train-step (steady, skip warmup) | **0.373 s** |
| sec/eval-clip (8× generate, ~3 s studio clips) | **0.185 s** (first clip 0.35 s) |
| Live L4 | **₹41.31 / hr** |
| This successful slice | ₹2.24 |
| All M3 GPU including failed folder-venv attempt | **₹5.26** (grants 49426.19 → 49420.93) |
| `jl list` | **0 running** (580479 Paused) |

Generation eval is in the cost model below, including per-epoch val generate.

## Cost model

```
estimated_sec = fixed_overhead_sec
              + steps × sec_per_train_step
              + n_epoch_evals × n_val × sec_per_eval_clip
              + 2 × (n_val + n_eval_whisper + n_eval_normal) × sec_per_eval_clip_heldout

estimated_inr = estimated_sec / 3600 × live_₹/hr
```

Calibrated knobs:

- `sec_per_train_step` = 0.373 (probe)
- `sec_per_eval_clip` = 0.185 on Expresso val; **0.50** for phone m4a/mp3 (2.7×, clips may be longer)
- `fixed_overhead_sec` = 180 (probe 135 s wall-beyond-script + create/pause slack)
- M1 lesson: predicted ₹8.26 vs actual **₹15.20** = **1.84×**. Apply that as the upper calibration, not a silent fudge inside the point estimate.

Assume 20 recorded-whisper + 20 normal clips if you have not dropped them yet (zeros if empty).

### 3-epoch point estimate (recommended)

| Piece | Seconds |
| --- | --- |
| Fixed overhead | 180 |
| Train 385 × 0.373 | 144 |
| Epoch val generate 3 × 114 × 0.185 | 63 |
| Final base+tuned: 2×(114×0.185 + 40×0.50) | 82 |
| **Raw total** | **469 s ≈ 7.8 min** |
| Raw ₹ | 41.31 × 7.8/60 = **₹5.4** |
| M1-calibrated (×1.84) | **₹9.9** |
| **Range ±25% around calibrated** | **₹7.4 – ₹12.4** |
| Wall-clock range | **~8–15 min** |

2 epochs: ~₹6.5–₹11. 4 epochs: ~₹8.5–₹14. Still under ₹150 / 45 min.

Do **not** treat ₹5.4 as the number you will be billed. M1 billed 1.84× the naive GPU-seconds estimate. Quote **₹7.4–₹12.4** (likely ~₹10).

## Acceptance for the full run (after GO)

- Adapter under `artifacts/m3/adapter/`
- `jl list` = 0 running
- Balance delta logged predicted-vs-actual
- This card + `run_spec.json` + probe metrics already committed; full-run metrics follow the GO
- Hard pause on success/fail/timeout; caps ₹150 / 45 min

**Waiting for GO to execute the full run.**
