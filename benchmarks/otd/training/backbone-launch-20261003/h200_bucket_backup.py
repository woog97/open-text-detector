from pathlib import Path
import time,json,fcntl
from huggingface_hub import HfApi
R=Path('/workspace/woog/pangram/backbones-20261003');B='open-text-detector/training-storage';PREFIX='workspace/backbone-fast10-20261003/h200'
lock=(R/'h200-bucket-backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
api=HfApi();receipt=R/'h200-bucket-backup.json';saved=json.loads(receipt.read_text()) if receipt.exists() else {}
while True:
 finished=[]
 for name in ['gemma4-12b','qwen36-35b-a3b']:
  root=R/'runs'/name;out=root/'run';s=out/'status.json'
  finished.append(s.exists() and json.loads(s.read_text()).get('state')=='trained_calibration_pending')
  for p in list(out.glob('*.safetensors'))+list(out.glob('*.json'))+list(root.glob('*.json')):
   dest=PREFIX+'/'+name+'/'+str(p.relative_to(root));stamp=[p.stat().st_mtime_ns,p.stat().st_size]
   if saved.get(dest)==stamp:continue
   try:
    api.batch_bucket_files(B,add=[(p,dest)])
    assert api.get_bucket_file_metadata(B,dest).size==p.stat().st_size
    saved[dest]=stamp;receipt.write_text(json.dumps(saved));print(json.dumps({'copied':dest,'bytes':stamp[1],'time':time.time()}),flush=True)
   except Exception as e:print(json.dumps({'retry':dest,'error':type(e).__name__}),flush=True)
 if all(finished):break
 time.sleep(45)
