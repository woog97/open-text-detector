"""Loss ablation (Oct 10): archive each run (tags from argv, else the Oct 10 arms) to the bucket once its evaluation is complete, then free the local adapter.
Runs detached on the H200 from the final-20261007 work dir; HF token on stdin line 1 (kept in memory only).
For each tag, once sweeps/TAG/eval/summary.json exists: upload the run folder (adapter, run/status/history files, train.log,
eval/; not wandb/ or *.claim) to workspace/final-20261007/h200/TAG/, check every file's size in the bucket, re-download the
adapter and compare sha256, and only then delete the local adapter and write sweeps/TAG/ARCHIVED.json.
A run whose training failed is left untouched. Exits when every tag is archived or failed. Log: sweeps/abl-archive.log."""
import hashlib, json, os, sys, time
from pathlib import Path
from huggingface_hub import HfApi
B = 'open-text-detector/training-storage'; PRE = 'workspace/final-20261007/h200/'
TAGS = sys.argv[1:] or ['abl-nostage1', 'abl-noseg-nomix', 'abl-tok-sent', 'abl-uniform-tok', 'abl-tok-only', 'abl-sent1', 'abl-control']
api = HfApi(token=sys.stdin.readline().strip()); S = Path('sweeps'); log = open(S / 'abl-archive.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def archive(tag):
    d = S / tag; ad = d / 'stage2-epoch0-adapters.safetensors'; local_sha = sha(ad)
    files = [p for p in d.rglob('*') if p.is_file() and 'wandb' not in p.relative_to(d).parts and not p.name.endswith('.claim')]
    api.batch_bucket_files(B, add=[(str(p), PRE + str(p.relative_to(S))) for p in files])
    got = {f.path: f.size for f in api.list_bucket_tree(B, prefix=PRE + tag + '/', recursive=True) if hasattr(f, 'size')}
    bad = [str(p) for p in files if got.get(PRE + str(p.relative_to(S))) != p.stat().st_size]
    if bad: say(tag=tag, event='size-mismatch', files=bad[:10]); return False
    remote = next(f for f in api.list_bucket_tree(B, prefix=PRE + tag + '/stage2-epoch0-adapters.safetensors') if hasattr(f, 'size'))
    tmp = f'/dev/shm/woog-pangram/verify-{tag}.safetensors'
    try:
        api.download_bucket_files(B, [(remote, tmp)], raise_on_missing_files=True); remote_sha = sha(tmp)
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    if remote_sha != local_sha: say(tag=tag, event='sha-mismatch', local=local_sha, remote=remote_sha); return False
    (d / 'ARCHIVED.json').write_text(json.dumps({'bucket': B, 'prefix': PRE + tag + '/', 'files': len(files), 'adapter_sha256': local_sha,
                                                 'adapter_bytes': ad.stat().st_size, 'deleted_local_adapter': True, 'time_utc': time.strftime('%F %T')}, indent=1))
    ad.unlink(); say(tag=tag, event='archived', files=len(files), adapter_sha256=local_sha); return True
say(event='start', pid=os.getpid()); settled = set(); fails = {}
while len(settled) < len(TAGS):
    for tag in TAGS:
        if tag in settled: continue
        d = S / tag
        if (d / 'ARCHIVED.json').exists(): settled.add(tag); continue
        done = d / 'train.done'
        if done.exists() and not done.read_text().startswith('rc=0'):
            say(tag=tag, event='training failed; not archived', done=done.read_text().strip()); settled.add(tag); continue
        if not (d / 'eval' / 'summary.json').exists(): continue
        try:
            if archive(tag): settled.add(tag)
            else: fails[tag] = fails.get(tag, 0) + 1
        except Exception as e:
            fails[tag] = fails.get(tag, 0) + 1; say(tag=tag, event='error', error=repr(e)[:300])
        if fails.get(tag, 0) >= 3: say(tag=tag, event='giving up after 3 failures; local adapter kept'); settled.add(tag)
    if len(settled) < len(TAGS): time.sleep(120)
say(event='exit', settled=sorted(settled))
