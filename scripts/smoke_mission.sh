#!/usr/bin/env bash
# Smoke-test the mission-control API on :7861.
set -u
B=${1:-http://localhost:7861}
echo '{"goal":"Transcribe whispered English better"}' > /tmp/mc_body.json
echo "== /api/run"
curl -sN -m 150 -X POST "$B/api/run" -H 'Content-Type: application/json' -d @/tmp/mc_body.json | grep -E '^event|^data' | cut -c1-200
echo "== /api/approve"
curl -sN -m 60 -X POST "$B/api/approve" -H 'Content-Type: application/json' -d @/tmp/mc_body.json | grep -E '^event' | sort | uniq -c
echo "== /api/spend"
curl -s -m 20 "$B/api/spend" | cut -c1-600; echo
free -m | head -2
