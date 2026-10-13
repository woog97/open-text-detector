import base64, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent; P = H.parents[3]
files = {'setup_wave2.py': (H / 'setup_wave2.py').read_text(), 'cross_daemon.py': (H / 'cross_daemon.py').read_text(),
         'cross_model_eval.py': (P / 'benchmarks/otd/training/overnight-sweep-20261004/cross_model_eval.py').read_text()}
blob = base64.b64encode((P / 'research/llm-sentence-edits-20261005/llm-edits-v1.jsonl.gz').read_bytes()).decode()
code = r'''
import base64, json, subprocess, sys
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006')
for n, s in FILES.items(): (R / n).write_text(s)
(R / 'llm-edits-v1.jsonl.gz').write_bytes(base64.b64decode(BLOB))
p = subprocess.Popen([sys.executable, '-u', str(R / 'setup_wave2.py')], cwd=R, stdout=open(R / 'setup-wave2.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'pid': p.pid, 'sp_runs': sorted(x.name for x in (R / 'sweeps').glob('q4b-*'))}))
'''
run('FILES=' + repr(files) + '\nBLOB=' + repr(blob) + '\n' + code, timeout=300)
