#!/usr/bin/env bash
# Airbyte-sync the latest live job's GPU log and attach the result to its job record.
cd /home/user/vibetuning
DO_NOT_TRACK=1 .venv-demo/bin/python - <<'EOF'
import json
from pathlib import Path
from agent.ingest import airbyte_sync
from agent.events import emit
d = Path("logs/live")
st = json.loads((d / "state.json").read_text())
ab = airbyte_sync("gpu", str(d / "metrics.jsonl"), st["id"], timeout=300)
print(ab)
st["airbyte"] = ab
(d / "state.json").write_text(json.dumps(st, indent=2))
pts = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
(d / "jobs").mkdir(exist_ok=True)
(d / "jobs" / f"{st['id']}.json").write_text(json.dumps({"state": st, "points": pts}))
if ab.get("ok"):
    emit(st["id"], "airbyte_sync_completed", {"stream": ab["stream"], "records": ab["records"], "connector": ab["connector"]})
EOF
