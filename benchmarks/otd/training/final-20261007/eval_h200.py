"""Mac side: set up and start final-run scoring on the H200.
  eval_h200.py setup            copy eval code, pull eval-inputs from the bucket (HF token over ssh stdin only)
  eval_h200.py start TAG:GPU..  start eval_watch.sh detached with those jobs (e.g. final-eg2:2 final-qwen35-4b:1 final-moe:0)
  eval_h200.py collect          upload each run's sweeps/TAG/eval/ to bucket workspace/final-20261007/eval/TAG/ (no .npz>50MB filter)
"""
import os, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; HR = '/workspace/woog/pangram/final-20261007'; BUCKET = 'open-text-detector/training-storage'
sys.path.insert(0, str(H))
from launch import h200, h200_script  # noqa: E402


def setup():
    for f in ['eval_final.py', 'eval_watch.sh']:
        subprocess.run(['scp', '-q', str(H / f), f'otd-h200:{HR}/'], check=True)
    from huggingface_hub import get_token
    script = f'''read -r HF_TOKEN
cd {HR}; mkdir -p eval-inputs
HF_TOKEN=$HF_TOKEN HF_HUB_OFFLINE=0 venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).download_bucket_files('{BUCKET}', [('workspace/final-20261007/eval/inputs.tgz', 'eval-inputs/inputs.tgz')], raise_on_missing_files=True)
PY
unset HF_TOKEN
cd eval-inputs && tar xzf inputs.tgz && rm inputs.tgz && cp sweep-eval-rows.jsonl.gz ../sweeps/ && ls
'''
    print(h200_script(script, [get_token()], timeout=1800))


def start(jobs):
    print(h200(f'cd {HR} && (nohup bash eval_watch.sh {" ".join(jobs)} > /dev/null 2>&1 &) && sleep 1 && tail -2 eval-watch.log'))


def collect():
    from huggingface_hub import get_token
    script = f'''read -r HF_TOKEN; cd {HR}; HF_TOKEN=$HF_TOKEN HF_HUB_OFFLINE=0 venv/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
for d in sorted(list(Path('sweeps').glob('final-*/eval')) + list(Path('sweeps').glob('ref-*/eval'))):
    files = [p for p in d.rglob('*') if p.is_file()]
    for i in range(0, len(files), 200):
        api.batch_bucket_files('{BUCKET}', add=[(str(p), 'workspace/final-20261007/eval/' + d.parent.name + '/' + str(p.relative_to(d))) for p in files[i:i + 200]])
    print(d.parent.name, len(files), 'eval files uploaded')
PY
'''
    print(h200_script(script, [get_token()], timeout=3600))


if __name__ == '__main__':
    {'setup': lambda: setup(), 'start': lambda: start(sys.argv[2:]), 'collect': lambda: collect()}[sys.argv[1]]()
