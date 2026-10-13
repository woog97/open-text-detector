"""Resolve/download preparation assets only; never runs a model or optimizer."""
from runtime import require_space, SPACE_CACHE
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
ROOT=Path(__file__).resolve().parent
MODELS={'encoder':'answerdotai/ModernBERT-large','causal':'Qwen/Qwen3-0.6B','qwen35':'Qwen/Qwen3.5-4B'}
def main():
 require_space()
 api=HfApi();lockpath=ROOT/'models.lock.json';old=json.loads(lockpath.read_text()) if lockpath.exists() else {}
 lock={}
 for name,repo in MODELS.items():
  revision=old.get(name,{}).get('revision') or api.model_info(repo).sha
  path=snapshot_download(repo,revision=revision,allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'])
  lock[name]={'repo':repo,'revision':revision};print(name,repo,revision)
 lockpath.write_text(json.dumps(lock,indent=2)+'\n')
if __name__=='__main__':main()
