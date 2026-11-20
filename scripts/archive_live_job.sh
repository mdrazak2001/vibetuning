#!/usr/bin/env bash
# Archive the current logs/live state + points into logs/live/jobs/<id>.json.
cd /home/user/vibetuning
.venv-demo/bin/python - <<'EOF'
import json
from pathlib import Path
d = Path("logs/live")
st = json.loads((d / "state.json").read_text())
pts = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
(d / "jobs").mkdir(exist_ok=True)
(d / "jobs" / f"{st['id']}.json").write_text(json.dumps({"state": st, "points": pts}))
print("archived", st["id"], len(pts))
EOF
