#!/bin/bash
set -euo pipefail
cd /workspace/pangram
mkdir -p research/benchmarks/meld-cuda
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq gcc g++ libgomp1) > research/benchmarks/meld-cuda/system-setup.log 2>&1 &
apt_pid=$!
python -m pip install --no-cache-dir 'torch==2.12.0' 'transformers==5.17.0' safetensors numpy huggingface-hub zstandard pyarrow > research/benchmarks/meld-cuda/python-setup.log 2>&1
wait "$apt_pid"
python - <<'PY'
import torch, transformers, triton, json
assert torch.cuda.is_available()
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,
 'triton':triton.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}))
PY
touch research/benchmarks/meld-cuda/setup-complete
