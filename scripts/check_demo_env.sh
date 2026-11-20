#!/usr/bin/env bash
PY=/home/user/vibetuning/.venv-demo/bin/python
for m in fastapi uvicorn multipart torch transformers peft librosa soundfile cognee; do
  if "$PY" -c "import $m" 2>/dev/null; then echo "$m ok"; else echo "$m MISSING"; fi
done
which ffmpeg || echo "ffmpeg MISSING"
ls /home/user/.venv-jl/bin/jl || echo "jl MISSING"
free -m | head -2
