"""Prepare the baseline recipe on the approved Space; never start training."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'));from remote import run
files={name:(HERE/name).read_text() for name in ['policy.py','build_remote.py','worker.py']}
files['track.py']=(HERE.parent/'wandb-tracking/track.py').read_text()
files['reservation.json']=(PROJECT/'research/evaluation/full-paper-holdout-20261003/reservation.json').read_text()
run('FILES='+repr(files)+'''\nfrom pathlib import Path
import sys,subprocess,json,time
r=Path('/data/workspace/baseline-mix-v1');assert not r.exists(),'Preserve existing recipe; inspect rather than retry'
r.mkdir()
for name,content in FILES.items():(r/name).write_text(content)
with (r/'build.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',str(r/'build_remote.py')],cwd=r,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({'root':str(r),'pid':p.pid,'training_started':False}))
''')
