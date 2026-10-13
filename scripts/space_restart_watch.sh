#!/bin/bash
# Watch the HF training Space for restarts (which wipe /tmp). Polls every INTERVAL seconds (default 600, one request each,
# well inside the Space rate limit). Exits with a RESTART line the moment the boot time changes, so a session running
# this in the background is woken up and can restore /tmp from /data (persist_daemon.py --restore) and relaunch.
# Usage: scripts/space_restart_watch.sh [INTERVAL]
INTERVAL=${1:-600}
HERE=$(cd "$(dirname "$0")" && pwd); STATE=~/.config/open-text-detector/space-boot.json
check() {
  (cd ~/.config/open-text-detector && uv run -q --with 'huggingface_hub>=1.0' --with httpx --with websockets python remote.py "$HERE/space_boot.py" 2>&1) \
    | sed -E 's/token=[A-Za-z0-9_-]+/token=REDACTED/g' | grep '^BOOT ' | sed 's/^BOOT //'
}
while true; do
  now=$(check)
  if [ -n "$now" ]; then
    boot=$(echo "$now" | python3 -c 'import json,sys; print(json.load(sys.stdin)["boot_epoch_min"])')
    last=$(python3 -c "import json; print(json.load(open('$STATE'))['boot_epoch_min'])" 2>/dev/null)
    echo "$now" > "$STATE"
    if [ -n "$last" ] && [ "$(( boot - last ))" -gt 120 ]; then
      echo "RESTART $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M %Z'): Space rebooted (boot time moved $(( (boot - last) / 60 )) min). $now"
      exit 0
    fi
    echo "$(TZ=America/Los_Angeles date +%H:%M) ok $now"
  else
    echo "$(TZ=America/Los_Angeles date +%H:%M) check failed (Space unreachable or rate-limited); retrying"
  fi
  sleep "$INTERVAL"
done
