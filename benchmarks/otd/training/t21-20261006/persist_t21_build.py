import json, shutil, time
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); D = Path('/data/workspace/pangram-t2-20261006')
shutil.copytree(R / 'runs/qwen35-4b-t21/prepared-v2', D / 'prepared/qwen35-4b-t21/prepared-v2', dirs_exist_ok=True)
for n in ['build-t21.json', 'models.json', 'build_t21.py', 'nt21.py']: shutil.copy(R / n, D / n)
(D / 'inputs-t21').mkdir(exist_ok=True)
for n in ['t21-heldout-never-train.json', 't21-heldout-score-rows.jsonl.gz']: shutil.copy(R / 'inputs' / n, D / 'inputs-t21' / n)
open(R / 'build-t21.log', 'a').write(json.dumps({'t': time.strftime('%H:%M:%S'), 'event': 'persisted_build'}) + '\n')
