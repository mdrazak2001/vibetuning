#!/usr/bin/env bash
# Dispatch one live run via mission control, then show state.
for i in $(seq 1 20); do curl -sf localhost:7861/api/live >/dev/null && break; sleep 3; done
curl -s -X POST localhost:7861/api/live/dispatch; echo
sleep 45
curl -s localhost:7861/api/live | head -c 600; echo
tail -n 15 /home/user/vibetuning/logs/live/relay.log
