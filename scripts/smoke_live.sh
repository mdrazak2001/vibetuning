#!/usr/bin/env bash
# Smoke the live :7861 server and print step labels + integration notes.
bash /home/user/vibetuning/scripts/smoke_mission.sh \
  | grep -oE '"id": "[a-z]+", "status": "[a-z]+", "ms": [0-9]+, "label": "[^"]*"|event: (error|end)|receipt|== .*' \
  | uniq -c
curl -s -m 30 localhost:7861/api/status | grep -oE '"(amplitude|superlinked)"[^}]*'
curl -s -m 30 localhost:7861/api/spend | grep -oE '"amplitude":\{[^}]*\}'
