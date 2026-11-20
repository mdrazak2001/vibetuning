# vibetuning agent notes

## Cloud sandboxes (Boat)

Control plane is the persistent Boat sandbox **`vibetuner`** (`bx_6nvrrech`). Auto-stop is off. CPU only — no GPUs on Boat.

Docs: https://docs.boat.dev/llms.txt. Prices/limits: `boat limits`.

## JarvisLabs GPU — do not SSH from the laptop

**Windows SSH to JarvisLabs port 22 times out** (`ssh: connect to host <gpu-ip> port 22: Connection timed out`). `jl run` / `jl exec` / `jl upload` / `jl download` from this laptop fail after the instance is created, which can leave a GPU billing.

**Rule: all `jl` and GPU SSH operations run from `vibetuner`, never the laptop.**

- Create/list/status/pause via the JarvisLabs HTTP API (`jl list`, `jl status`, `jl gpus`, `jl pause`) is OK from the laptop.
- Anything that needs SSH into a GPU (run, exec, upload, download) must be invoked inside Boat: install `jl` in `/home/user/.venv-jl` on `bx_6nvrrech` and `source ~/.jl.env`.
- After any `jl run --gpu` that might have created an instance, immediately `jl list` and pause stray Running machines.

## Budget

Never create a GPU instance without an explicit **GO** in chat. $3 cap per run, $10 total. Prefer on-demand L4. Pause in `finally` and on a hard wall-clock timeout. Checkpoint every 200 steps.

## Dataset

Preferred whisper source: **`ylacombe/expresso`** (public, ungated, **CC BY-NC 4.0**), `read` split, `style==whisper`, human transcripts. `facebook/expresso` is missing/inaccessible on HF.

Do not download the full 36GB `nyuuzyou/asmr` archive. License is **unspecified** (`other`). Labels are **distil-whisper pseudo-labels** (distillation, not human-verified). Fetch `meta.csv` first only if Expresso is blocked; subset; stream only needed clips.

Held-out TEST is `eval_whisper/` (user recordings + `sentences.txt`), never the Expresso train/val split.
