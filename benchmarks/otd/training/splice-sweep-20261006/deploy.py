import base64, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
T = Path(__file__).resolve().parents[1]; SW = T / 'overnight-sweep-20261004'; BL = T / 'backbone-launch-20261003'
HERE = Path(__file__).resolve().parent
files = {n: (SW / n).read_text() for n in ['train_sweep.py', 'modeling_sweep.py', 'sweep_eval.py', 'queue_runner.py']}
files.update({n: (BL / n).read_text() for n in ['data.py', 'adapters_short.py', 'modeling.py', 'models.json']})
files['setup_splice.py'] = (HERE / 'setup_splice.py').read_text()
files['runtime.py'] = ("from pathlib import Path\nimport sys,os\nROOT=Path(__file__).resolve().parent\nsys.path.insert(0,'/tmp/pangram-wandb-vendor')\n"
                       "sys.path.insert(0,str(ROOT/'vendor'))\nSPACE_CACHE=str(ROOT/'assets')\ndef require_space():\n assert str(ROOT)=='/tmp/pangram-splice-20261006'\n"
                       "os.environ.setdefault('HF_HUB_OFFLINE','1')\nos.environ.setdefault('TOKENIZERS_PARALLELISM','false')\n")
blobs = {'splices-v1.jsonl.gz': base64.b64encode((T.parents[2] / 'research/splice-edits-20261005/splices-v1.jsonl.gz').read_bytes()).decode(),
         'sweeps/sweep-eval-rows.jsonl.gz': base64.b64encode((SW / 'sweep-eval-rows.jsonl.gz').read_bytes()).decode()}
code = r'''
import base64, json, subprocess, sys
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006'); assert not R.exists(), 'exists'; R.mkdir(); (R / 'sweeps').mkdir()
for n, s in FILES.items(): (R / n).write_text(s)
for n, b in BLOBS.items(): (R / n).write_bytes(base64.b64decode(b))
p = subprocess.Popen([sys.executable, '-u', str(R / 'setup_splice.py')], cwd=R, stdout=open(R / 'setup.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'setup_pid': p.pid, 'files': sorted(x.name for x in R.iterdir())}))
'''
run('FILES=' + repr(files) + '\nBLOBS=' + repr(blobs) + '\n' + code, timeout=300)
