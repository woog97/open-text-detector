#!/bin/bash
# H200 setup of /workspace/woog/pangram/final-20261007 (idempotent). Code is copied in by launch.py (scp) before this runs.
# vendor/ = transformers 5.19.0 only (the venv has 5.18.0, which lacks embedding_gemma2), layered over the backbones venv.
# Assets: MoE from the RAM disk (symlink); 4B and EmbeddingGemma 2 downloaded anonymously from HF (both ungated) to /workspace.
set -e
R=/workspace/woog/pangram/final-20261007; V=/workspace/woog/pangram/backbones-20261003/venv; cd $R
mkdir -p assets runs sweeps
ln -sfn $V venv
if [ ! -f vendor/.complete ]; then
  $V/bin/pip install -q --target vendor --no-deps transformers==5.19.0 && touch vendor/.complete
fi
PYTHONPATH=$R/vendor:/workspace/woog/pangram/backbones-20261003/moe-vendor $V/bin/python -c "import transformers, torch, peft, accelerate, fla; import transformers.models.embedding_gemma2, transformers.models.qwen3_5_moe; print('vendor ok', transformers.__version__, torch.__version__, peft.__version__)"
ln -sfn /dev/shm/woog-pangram/assets/qwen36-35b-a3b assets/qwen36-35b-a3b
fetch() {  # repo revision name
  [ -f assets/$3/.complete ] && return 0
  HF_HUB_OFFLINE=0 $V/bin/python -c "from huggingface_hub import snapshot_download as s; s('$1', revision='$2', local_dir='assets/$3')" && touch assets/$3/.complete
}
fetch Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a qwen35-4b
fetch google/embeddinggemma-2 914f7f89142e33e77833254d9c9b90c3cef7303b embeddinggemma-2
du -sh assets/*/ 2>/dev/null; echo setup done
