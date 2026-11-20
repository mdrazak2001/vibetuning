#!/usr/bin/env bash
# Print env key names (never values) from the demo env files.
for f in /home/user/vibetuning/.env /home/user/.vibetuning/secrets.env; do
  echo "== $f"
  [ -f "$f" ] && grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$f" | sed 's/=$/=<set>/'
done
