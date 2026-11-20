# Held-out TEST — recorded whisper clips

Drop your whispered recordings here. Train/val come from Expresso; this folder is TEST only.

## Layout

```
eval_whisper/
  sentences.txt
  clip_001.wav
  clip_002.wav
  ...
```

`sentences.txt` (UTF-8), one clip per line. `#` comments and blank lines are ignored.

```
# filename<TAB>transcript
clip_001.wav	the sentence you actually whispered
clip_002.wav	another quiet line
```

Also accepted: `filename|transcript`, `filename transcript`, or transcript-only lines aligned to audio sorted by filename.

Audio: wav / flac / mp3 / ogg / m4a. Prefer 16 kHz mono, but the trainer resamples.

Loaded by `recipes/sft_whisper/eval_loader.py` (`EVAL_WHISPER_DIR` overrides the path).
