#!/bin/bash
# Download each staged MoE detail tar (1.5 GB) from the bucket, extract its sentence scores, delete it. Resumable.
# Usage: run_iclr.sh WORK CATALOGUE.json OUT_DIR
set -e
WORK=$1; CAT=$2; OUT=$3; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/parts"
for i in 0 1 2 3 4 5 6 7; do
  T=$OUT/objects-00$i.tar
  [ -f "$OUT/parts/objects-00$i.parquet" ] && continue
  echo "$(date '+%H:%M:%S') download $i"
  [ -f "$T" ] || uv run -q --with 'huggingface_hub>=1.0' python -c "from huggingface_hub import HfApi;HfApi().download_bucket_files('open-text-detector/training-storage',files=[('workspace/atlas-rescore-moe-20261007/staging/objects-00$i.tar','$T')])"
  echo "$(date '+%H:%M:%S') extract $i ($(stat -f%z "$T") bytes)"
  uv run -q --with numpy --with pyarrow python "$HERE/extract.py" iclr "$WORK" "$CAT" "$OUT" "$T"
  rm -f "$T"
done
echo ALL_DONE
