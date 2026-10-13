"""Space-side early start of the Claude part of wide-pool round 1 (Oct 7), from job 1's clean v2 text (ICLR/NeurIPS 2021-2022)
instead of waiting for the full corpus job:
  1. build_pool.py on /tmp/pangram-pre2023text/clean/data -> pool-early/
  2. sample 2,600 sections (one per paper), then 650 + 1,400 hosts from other papers (one per paper)
  3. items for opus,sonnet (no held-out papers): early/items-opus-sonnet-{d,e}.jsonl; sections in early/sections-sample.jsonl
  4. early/exclude-papers.json: every paper used here; wide1_run.py excludes them and skips its own Claude slices
Log: early.log in this folder. Rerunnable: finished steps are skipped.
"""
import json, subprocess, sys, time
from pathlib import Path

W = Path(__file__).resolve().parent; POOL = W / 'pool-early'; R = W / 'early'
log = open(W / 'early.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()
def py(*args, **kw):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=W, capture_output=True, text=True, **kw)
    if r.returncode: raise RuntimeError(f'{args[0]} failed: {r.stderr[-1500:]}')
    return r.stdout

SLICES = [('d', 650), ('e', 1400)]


def main():
    R.mkdir(exist_ok=True)
    if not (POOL / 'stats.json').exists():
        say(event='pool_start')
        py('build_pool.py', '--clean-parts', '/tmp/pangram-pre2023text/clean/data/*.parquet', '--workers', 48,
           '--calibration', W / 'repo/research/human-calibration-20261006/selection.json', '--out', POOL, timeout=6 * 3600)
        say(event='pool_done', total=json.loads((POOL / 'stats.json').read_text())['total'])
    if not (R / 'sections-sample.jsonl').exists():
        say(event='sections_sample', out=py('sample.py', 'sections', '--pool', POOL, '--n', 2600, '--max-per-paper', 1, '--out', R / 'sections-sample.jsonl')[-600:])
    sec_papers = sorted({json.loads(l)['paper_id'] for l in open(R / 'sections-sample.jsonl')})
    if not (R / 'hosts-sample.jsonl').exists():
        (R / 'exclude-sections.json').write_text(json.dumps(sec_papers))
        say(event='hosts_sample', out=py('sample.py', 'hosts', '--pool', POOL, '--n', sum(n for _, n in SLICES), '--max-per-paper', 1,
                                         '--exclude', R / 'exclude-sections.json', '--out', R / 'hosts-sample.jsonl')[-600:])
    hosts = open(R / 'hosts-sample.jsonl').read().splitlines(); k = 0
    for job, n in SLICES:
        part = R / f'hosts-opus-sonnet-{job}.jsonl'; items = R / f'items-opus-sonnet-{job}.jsonl'
        if not items.exists():
            part.write_text('\n'.join(hosts[k:k + n]) + '\n')
            py('make_items.py', '--job', job, '--hosts', part, '--writers', 'opus,sonnet', '--out', items, '--heldout-frac', 0., '--round', 'wide1')
        k += n
    used = sorted(set(sec_papers) | {json.loads(h)['record_id'] for h in hosts})
    (R / 'exclude-papers.json').write_text(json.dumps(used))
    say(event='done', section_papers=len(sec_papers), host_papers=len(hosts), excluded=len(used))


if __name__ == '__main__':
    try: main()
    except Exception as e:
        import traceback; say(event='failed', error=repr(e), tb=traceback.format_exc()[-2000:]); raise
