from pathlib import Path
import json,sys,inspect
sys.path.insert(0,'/data/workspace/current-data-v1/vendor')
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_MAPPING
R=Path('/data/workspace/backbone-launch-20261003')
for p in (R/'assets').iterdir():
 try:
  c=AutoConfig.from_pretrained(p,local_files_only=True);tc=c.get_text_config();cls=MODEL_MAPPING[type(tc)]
  print(p.name,'types',c.model_type,tc.model_type,'class',cls.__name__,'hidden',tc.hidden_size,'source',inspect.getfile(cls),flush=True)
  print('params', {k:getattr(tc,k,None) for k in ['num_hidden_layers','num_experts','num_local_experts','num_experts_per_tok','vocab_size','layer_types']},flush=True)
 except Exception as e:print(p.name,repr(e),flush=True)
