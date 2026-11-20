---
base_model: openai/whisper-small
library_name: peft
license: cc-by-nc-4.0
---

# whisper-small LoRA (Expresso read-whisper)

PEFT LoRA adapter (`r=8`, `alpha=16`, `q_proj`+`v_proj`) on `openai/whisper-small`, trained 3 epochs on `ylacombe/expresso` `read`/`style==whisper` (CC BY-NC 4.0). Best checkpoint is epoch 2 (step 258). Weights live in `adapter_model.safetensors` (gitignored).

| Split | WER before | WER after |
| --- | --- | --- |
| Expresso val (114, English-normalized) | 0.0783 | **0.0717** |
| `eval_whisper/` / `eval_normal/` | skipped (empty) | skipped (empty) |
