#!/usr/bin/env bash
# Run mission control on :7862 with every integration flag off, smoke it, then stop it.
cd /home/user/vibetuning
COGNEE_ENABLED=0 SINGLESTORE_ENABLED=0 AIRBYTE_ENABLED=0 AMPLITUDE_ENABLED=0 SUPERLINKED_ENABLED=0 \
CONTEXT_DEV_API_KEY= JL_API_KEY= JARVIS_API= PATH=/usr/bin:/bin MISSION_PORT=7862 \
  .venv-demo/bin/python -u server.py > /tmp/flags_off.log 2>&1 &
PID=$!
for _ in $(seq 1 30); do curl -s -m 2 localhost:7862/api/spend >/dev/null && break; sleep 1; done
bash scripts/smoke_mission.sh http://localhost:7862 \
  | grep -oE '"id": "[a-z]+", "status": "[a-z]+", "ms": [0-9]+, "label": "[^"]*"|event: (error|end)|receipt|== .*' \
  | uniq -c | cut -c1-160
kill $PID
