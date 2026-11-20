# M1 plan card

Approved: user GO in chat (2026-11-19). Proceed without a second wait.

| Field | Value |
| --- | --- |
| Job | whisper-tiny + LoRA SFT, ~150 steps, plumbing only |
| Dataset | `hf-internal-testing/librispeech_asr_dummy` (`clean`/`validation`, 73 clips) |
| GPU | **On-demand L4** container, region **IN2**, 1 GPU — **not spot** |
| Live stock | 7 free L4 devices (`jl gpus --json`) |
| Live rate | **₹41.31 / hr** (container L4 IN2) |
| FX used for $1 gate | **₹93.29 / USD** (USDINR 2026-11-20) → $1 = ₹93.29 |
| Expected wall clock | ~12 min (pip + ffmpeg + 150 steps + WER before/after) |
| Expected cost | ₹41.31 × 12/60 = **₹8.26 ≈ $0.09** |
| Hard timeout | **30 min**, then pause no matter what |
| Timeout ceiling | ₹41.31 × 30/60 = **₹20.66 ≈ $0.22** |
| Budget cap | estimated cost ≤ $1, actual spend < $1 |
| Storage | 40 GB (minimize pause-disk) |
| Instance name | `vibetuning-m1` |
| Pause | `finally` + timeout path; only this machine_id |
| Do not touch | pre-existing paused VMs `579489` (L4), `576152` (CPU) |

## Gate

- On-demand L4: yes (no `--spot`)
- Estimated cost $0.09 ≤ $1: **PASS**
- Timeout ceiling $0.22 ≤ $1: **PASS**
- 30 min hard pause: **PASS**

**Decision: PROCEED.**
