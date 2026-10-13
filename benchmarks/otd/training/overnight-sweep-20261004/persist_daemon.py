"""Mirror a Space sweep's outputs from /tmp to persistent /data every few minutes, so a Space restart loses at most one interval.

Usage (on the Space): persist_daemon.py SWEEP_DIR [--every 300] [--once]
  SWEEP_DIR  working sweep root in /tmp, e.g. /tmp/pangram-splice-20261006 (its sweeps/ folder is mirrored)
Copies new or changed files to /data/workspace/<basename of SWEEP_DIR>/sweeps/..., skipping resume state, rolling
snapshots and temp files (those are big, change constantly and are only needed while a run is alive).
Restore after a restart with:  persist_daemon.py SWEEP_DIR --restore
"""
import argparse, json, os, shutil, time
from pathlib import Path

SKIP_SUFFIX = ('.tmp', '.lock', '.pt')
SKIP_PARTS = {'snapshots', 'wandb', '__pycache__'}


def mirror(src, dst):
    copied = 0
    for p in src.rglob('*'):
        if not p.is_file() or p.name.endswith(SKIP_SUFFIX) or SKIP_PARTS & set(p.relative_to(src).parts):
            continue
        q = dst / p.relative_to(src)
        try:
            st = p.stat()
            if q.exists() and q.stat().st_size == st.st_size and q.stat().st_mtime >= st.st_mtime:
                continue
            q.parent.mkdir(parents=True, exist_ok=True)
            tmp = q.with_name(q.name + '.partial'); shutil.copy2(p, tmp); tmp.replace(q); copied += 1
        except FileNotFoundError:
            continue  # file vanished mid-copy (pruned checkpoint); next pass catches up
    # Propagate retention pruning: when a run's pruned.json lists removed checkpoints, remove the mirrored copies too
    # (the retention rule authorizes deleting superseded checkpoints of scored runs; final and selected ones are never listed).
    for pj in src.rglob('pruned.json'):
        try:
            gone = json.loads(pj.read_text()).get('removed', [])
        except ValueError:
            continue
        for name in gone:
            q = dst / pj.parent.relative_to(src) / name
            if q.exists():
                q.unlink()
    return copied


def main():
    a = argparse.ArgumentParser(); a.add_argument('sweep_dir'); a.add_argument('--every', type=int, default=300)
    a.add_argument('--once', action='store_true'); a.add_argument('--restore', action='store_true'); args = a.parse_args()
    src = Path(args.sweep_dir) / 'sweeps'; dst = Path('/data/workspace') / Path(args.sweep_dir).name / 'sweeps'
    if args.restore:
        print(json.dumps({'restored_files': mirror(dst, src), 'from': str(dst)})); return
    log = dst.parent / 'persist.log'; dst.mkdir(parents=True, exist_ok=True)
    index = Path('/data/workspace/README.md')  # bucket index kept by the user's sessions: list every top-level folder
    if index.exists() and f'`{dst.parent.name}/`' not in index.read_text():
        with open(index, 'a') as f:
            f.write(f"\n| `{dst.parent.name}/` | Persisted outputs of Space sweep `{args.sweep_dir}` (mirrored by persist_daemon.py; working copy in /tmp) |\n")
    while True:
        t0 = time.time(); n = mirror(src, dst)
        with open(log, 'a') as f:
            f.write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), 'copied': n, 'seconds': round(time.time() - t0, 1)}) + '\n')
        if args.once:
            break
        time.sleep(args.every)


if __name__ == '__main__':
    main()
