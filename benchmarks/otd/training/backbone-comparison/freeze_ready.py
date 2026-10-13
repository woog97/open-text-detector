"""Finalize preparation only after every backbone passes both no-update preflights."""
from runtime import require_space,SPACE_CACHE
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 require_space();results=json.loads((ROOT/'preflight-results.json').read_text());optim=json.loads((ROOT/'optimizer-preflight-results.json').read_text());manifest=json.loads((ROOT/'prepared/manifest.json').read_text());audit=json.loads((ROOT/'prepared/audit.json').read_text())
 assert set(results)==set(optim)==set(manifest['models'])==set(audit['models'])=={'encoder','causal','qwen35'}
 assets={}
 for key,r in results.items():
  assert r['model']==manifest['models'][key] and r['optimizer_steps']==0 and not r['trained_checkpoint_saved']
  assert all(c['finite_gradients'] and c['forward_dtype']=='torch.bfloat16' for c in r['checks'])
  if key!='encoder':assert r['repeat2_visibility']['second_copy_sees_later_source_content']
  o=optim[key];assert o['model_optimizer_steps']==0 and o['model_parameter_versions_unchanged'] and o['finite_gradients']
  assert o['actual_peak_with_allocated_optimizer_states_bytes']<.9*o['gpu_total_bytes']
  info=manifest['models'][key];folder=Path(SPACE_CACHE)/('models--'+info['repo'].replace('/','--'))/'snapshots'/info['revision'];weights=list(folder.glob('*.safetensors'));assert weights
  assets[key]={'snapshot':str(folder),'weight_bytes':sum(p.stat().st_size for p in weights)}
 record={'status':'prepared_not_trained','models':manifest['models'],'data_manifest_sha256':digest(ROOT/'prepared/manifest.json'),'preflight_sha256':digest(ROOT/'preflight-results.json'),'optimizer_preflight_sha256':digest(ROOT/'optimizer-preflight-results.json'),'source_code_sha256':{p.name:digest(p) for p in ROOT.glob('*.py')},'configs_sha256':{p.name:digest(p) for p in (ROOT/'configs').glob('*.json')},'model_assets':assets,'model_optimizer_steps':0,'paper_partitions_disjoint':audit['paper_partitions_disjoint'],'shared_windows_validated_per_model':audit['unique_windows']}
 (ROOT/'READY.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
