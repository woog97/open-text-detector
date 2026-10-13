"""Space-side setup of /tmp/final-20261007 (idempotent; rerun after a Space restart). Run detached; log: /data/workspace/final-20261007/setup.log.
Code comes from /data/workspace/final-20261007/code (pushed by push.py). Vendor from the T2 workspace (transformers 5.19.0, which has
embedding_gemma2). Assets: 4B and MoE copied from the bucket's backbone-launch assets; EmbeddingGemma 2 downloaded on the Space from HF
(pinned revision) into the bucket assets first, then copied."""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
R = Path('/tmp/final-20261007'); D = Path('/data/workspace/final-20261007'); A = Path('/data/workspace/backbone-launch-20261003/assets')
T2 = Path('/data/workspace/pangram-t2-20261006'); log = open(D / 'setup.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()

def copy_dir(src, dst):
    if dst.exists() and (dst / '.complete').exists(): return 'present'
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.cache'), dirs_exist_ok=True)
    bad = [p.name for p in src.iterdir() if p.is_file() and (dst / p.name).stat().st_size != p.stat().st_size]
    assert not bad, bad
    (dst / '.complete').write_text(time.strftime('%F %T')); return 'copied'

try:
    say(event='start'); R.mkdir(exist_ok=True); (R / 'assets').mkdir(exist_ok=True); (R / 'runs').mkdir(exist_ok=True); (R / 'sweeps').mkdir(exist_ok=True)
    for p in (D / 'code').iterdir(): shutil.copy(p, R / p.name)
    if not (R / 'vendor' / '.complete').exists():
        # Local copy from the live T2 workspace when it exists; copying thousands of small files off the bucket mount is very slow.
        V = Path('/tmp/pangram-t2-20261006/vendor'); V = V if V.exists() else T2 / 'vendor'
        shutil.rmtree(R / 'vendor', ignore_errors=True); shutil.copytree(V, R / 'vendor', symlinks=True); (R / 'vendor' / '.complete').write_text(str(V))
    if not Path('/tmp/pangram-wandb-vendor/wandb').exists(): shutil.copytree(T2 / 'wandb-vendor', '/tmp/pangram-wandb-vendor', dirs_exist_ok=True)
    r = subprocess.run([sys.executable, '-c', 'import sys;sys.path[:0]=["/tmp/pangram-wandb-vendor","%s"];import transformers,peft,bitsandbytes,fla,wandb,accelerate;'
                        'import transformers.models.embedding_gemma2 as m;print(transformers.__version__,peft.__version__,accelerate.__version__,wandb.__version__)' % (R / 'vendor')],
                       capture_output=True, text=True, cwd='/tmp')
    say(event='vendor', ok=r.returncode == 0, msg=(r.stdout + r.stderr)[-400:]); assert r.returncode == 0
    eg = A / 'embeddinggemma-2'
    if not (eg / '.complete').exists():
        from huggingface_hub import snapshot_download
        snapshot_download('google/embeddinggemma-2', revision='914f7f89142e33e77833254d9c9b90c3cef7303b', local_dir=str(eg))
        (eg / '.complete').write_text(time.strftime('%F %T'))
    say(event='eg2_bucket', files=sorted(p.name for p in eg.iterdir()))
    src4 = Path('/tmp/pangram-t2-20261006/assets/qwen35-4b')
    for name, src in [('embeddinggemma-2', eg), ('qwen35-4b', src4 if src4.exists() else A / 'qwen35-4b'), ('qwen36-35b-a3b', A / 'qwen36-35b-a3b')]:
        t = time.time(); say(event='asset', name=name, result=copy_dir(src, R / 'assets' / name), seconds=round(time.time() - t))
    r = subprocess.run([sys.executable, '-c', '''import sys;sys.path[:0]=["%s"]
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained("%s",local_files_only=True)
e=t("A test sentence.",add_special_tokens=False,return_offsets_mapping=True)
print(t.cls_token_id,t.sep_token_id,t.bos_token_id,t.eos_token_id,t.pad_token_id,len(e["input_ids"]),e["offset_mapping"][:3])''' % (R / 'vendor', R / 'assets/embeddinggemma-2')],
                       capture_output=True, text=True, cwd='/tmp')
    say(event='eg2_tokenizer', msg=(r.stdout + r.stderr)[-500:])
    say(event='done')
except Exception as e:
    say(event='failed', error=repr(e), tb=traceback.format_exc()[-1500:]); raise
