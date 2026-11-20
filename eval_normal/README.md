# Catastrophic-forgetting check — normal-voice clips

Same layout as `eval_whisper/`. Drop **normal (not whispered)** recordings here so WER on this set can catch the tuned model forgetting regular speech.

```
eval_normal/
  sentences.txt
  clip_001.m4a
  ...
```

`sentences.txt`: `filename<TAB>transcript` (m4a / mp3 / wav all fine; the loader converts to 16 kHz mono).

Loaded via `load_eval_whisper("eval_normal")` / `EVAL_NORMAL_DIR`.
