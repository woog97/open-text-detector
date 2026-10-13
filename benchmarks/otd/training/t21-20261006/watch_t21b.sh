#!/bin/bash
# Backoff checks (1,2,4,8,16,32 min, then hourly) from START; exits on PROBLEM, on all scoring done, or after 6 h.
H=$(cd "$(dirname "$0")" && pwd); start=$(date -j -f "%Y-%m-%d %H:%M" "$1" +%s)
for m in 1 2 4 8 16 32 92 152 212 272 332; do
  now=$(date +%s); tgt=$((start + m*60)); [ $tgt -gt $now ] && sleep $((tgt-now))
  out=$(cd ~/.config/open-text-detector && uv run -q --with 'huggingface_hub>=1.0' --with httpx --with websockets python remote.py $H/check_t21.py 2>&1 | sed -E 's/hf_[A-Za-z0-9]+/hf_***/g; s/token=[A-Za-z0-9_-]+/token=***/g')
  echo "=== $(TZ=America/Los_Angeles date +%H:%M) PDT check at +${m}m"; echo "$out" | tail -30
  if echo "$out" | grep -q "ROUND2 done 36 of 36"; then echo ALLDONE; exit 0; fi
  if echo "$out" | grep -q -E "PROBLEM|RuntimeError|round2 workers alive 0$"; then echo PROBLEM; exit 0; fi
done
