# Held-out TEST — recorded whisper clips

Drop your whispered recordings here. Train/val come from Expresso; this folder is TEST only.

## Layout

```
eval_whisper/
  sentences.txt
  clip_001.m4a
  clip_002.wav
  ...
```

Any of **m4a / mp3 / wav / flac / ogg / aac** is fine. `recipes/sft_whisper/eval_loader.py` converts to 16 kHz mono (ffmpeg first, then soundfile, then librosa).

`sentences.txt` (UTF-8), one clip per line. `#` comments and blank lines are ignored.

```
clip_001.m4a	the sentence you actually whispered
clip_002.wav	another quiet line
```

Also accepted: `filename|transcript`, `filename transcript`, or transcript-only lines aligned to audio sorted by filename.

Override path with `EVAL_WHISPER_DIR`. Normal-voice (forgetting) clips go in `eval_normal/` with the same layout.
