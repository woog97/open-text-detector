#!/bin/bash
# Idempotent (re)launch of the full-size mixB run on the H200 after a reboot or crash (Oct 8).
# Usage: bash relaunch_h200.sh [--setup-only]
#   stdin line 1: W&B key (empty is refused unless WANDB_MODE=offline); line 2: HF token, only used to restore the MoE weights to
#   /dev/shm from the training bucket when a reboot wiped them (~41 s). Neither is written to disk.
# Config: final-20261007/launch-mixB-full.json {"model", "tag", "gpus", "port"}. Prints one JSON status line.
set -u; R=/workspace/woog/pangram/final-20261007; cd $R
read -r WKEY || true; read -r HFTOK || true
CFG=launch-mixB-full.json
get() { python3 -c "import json;print(json.load(open('$CFG'))['$1'])"; }
MODEL=$(get model); TAG=$(get tag); GPUS=$(get gpus); PORT=$(get port)
SHM=/dev/shm/woog-pangram/assets/qwen36-35b-a3b
if [ ! -f $SHM/.restored ] && [ ! -f $SHM/config.json ]; then
  [ -n "$HFTOK" ] || { echo '{"run": "MoE weights missing in /dev/shm and no HF token given"}'; exit 1; }
  mkdir -p $SHM
  HF_TOKEN=$HFTOK HF_HUB_OFFLINE=0 venv/bin/python - <<'PY' || { echo '{"run": "weight restore failed"}'; exit 1; }
import os, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN']); pre = 'workspace/backbone-launch-20261003/assets/qwen36-35b-a3b/'; dst = '/dev/shm/woog-pangram/assets/qwen36-35b-a3b/'
files = [f for f in api.list_bucket_tree('open-text-detector/training-storage', prefix=pre, recursive=True) if hasattr(f, 'size')]
api.download_bucket_files('open-text-detector/training-storage', [(f, dst + f.path[len(pre):]) for f in files], raise_on_missing_files=True)
open(dst + '.restored', 'w').write(time.strftime('%F %T'))
PY
fi
[ "$(readlink assets/$MODEL)" = "$SHM" ] || ln -sfn $SHM assets/$MODEL
[ -f runs/$MODEL/prepared-v2/manifest.json ] || { echo '{"run": "mix data missing"}'; exit 1; }
if [ "${1:-}" = "--setup-only" ]; then echo '{"run": "setup ok"}'; exit 0; fi
STATE=$(python3 -c "import json;print(json.load(open('sweeps/$TAG/status.json'))['state'])" 2>/dev/null || echo none)
if [ -f sweeps/$TAG/train.done ] && grep -q '^rc=0' sweeps/$TAG/train.done && [ "$STATE" = trained ]; then echo '{"run": "finished"}'; exit 0; fi
if [ "$STATE" = diverged ]; then echo '{"run": "diverged: not restarting"}'; exit 0; fi
WP=$(cat sweeps/$TAG.wrapper.pid 2>/dev/null); if [ -n "$WP" ] && kill -0 $WP 2>/dev/null; then echo '{"run": "alive"}'; exit 0; fi
[ -n "$WKEY" ] || [ "${WANDB_MODE:-}" = offline ] || { echo '{"run": "refusing: empty W&B key (set WANDB_MODE=offline to run without W&B)"}'; exit 1; }
rm -f sweeps/$TAG/train.done; mkdir -p sweeps/$TAG
# Wait (detached) until every assigned GPU is free, then start; the key only travels through the pipe.
printf '%s\n' "$WKEY" | nohup bash mixgrid/start_when_free.sh $MODEL $TAG $GPUS $PORT > sweeps/$TAG.wrapper.out 2>&1 &
echo $! > sweeps/$TAG.wrapper.pid
echo "{\"run\": \"started (waits for GPUs $GPUS) pid $!\"}"
