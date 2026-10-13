#!/bin/bash
# Backoff watcher: sleeps until the next check offset (minutes after launch), prints the check, exits so the session wakes up.
# Usage: watch_abl.sh LAUNCH_EPOCH MINUTES [TAGS...]
L=$1; M=$2; shift 2; now=$(date +%s); wait=$(( L + M*60 - now )); [ $wait -gt 0 ] && sleep $wait
echo "check at +${M} min ($(TZ=America/Los_Angeles date +%H:%M) PDT)"; "$(dirname "$0")"/check_abl.sh "$@"
