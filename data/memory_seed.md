# vibetuning agent memory (hackathon seed)

These are lessons the personal agent must recall before quoting a GPU.

## GPU cost estimation — cold vs warm

- **M1 cold start** (new L4 `580359`, Windows SSH timeout + hung `datasets.map` retry): predicted **₹8.26**, actual **₹15.20**. Calibration multiplier **1.84×**. Never quote raw GPU-seconds as the bill.
- **M3 probe** (cold-ish, folder-venv clash then file-mode): all-in **₹5.26**; successful slice ₹2.24.
- **M3 full run warm-resume** (paused `580479` → new machine_id `580496`, pip cache hit 0.8s): predicted **₹9.9** (range ₹7.4–₹12.4 after 1.84×), actual **₹5.24**. Warm resume beat the calibrated quote because overhead was already paid.
- Cost model: `estimated_sec = fixed_overhead + steps×sec/step + eval_clips×sec/eval`. Then `₹ = sec/3600 × live L4 ₹/hr`. Apply **1.84× only for cold creates**. For warm-resume, quote closer to raw GPU-seconds plus ~2–3 min resume/upload.
- Live L4 IN2: **₹41.31 / hr**. Caps: ₹150 / 45 min full run; demo-serving fallback **₹20**.
- Launch GPU SSH from Boat `vibetuner` (`bx_6nvrrech`). Laptop `jl run` times out on `:22` and can leave a billing instance.

## Dataset licensing

- Preferred: **ylacombe/expresso** `read` + `style==whisper`, **CC BY-NC 4.0**, ungated, **human transcripts**. 1139 clips, 0.993 h, train 1025 / val 114.
- Do **not** download 36GB `nyuuzyou/asmr` (`license=other`, distil-whisper **pseudo-labels**).
- Held-out TEST is `eval_whisper/` (user recordings), never Expresso val.

## WER (English BasicTextNormalizer on refs and hyps)

- M1 plumbing (whisper-tiny dummy): 0.238 → 0.240 (not a quality claim).
- M3 full (whisper-small LoRA r=8, 3 epochs, best = epoch 2 / step 258): Expresso val **0.0783 → 0.0717**.
- Epoch 3 WER 0.0827 — early stop kept the best checkpoint.

## Receipt

- Predicted ₹9.9 vs actual ₹5.24. Pause proof: `jl pause 580496`, `jl list` 0 running.


## Airbyte lineage
manifest ingested via pandas; 1139 rows in training_rows; sync 48e3ba00-311c-4854-bcc2-86012cec808f; ylacombe/expresso CC BY-NC 4.0.
