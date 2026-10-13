"""Space-side orchestration of the wide-pool round 1 (Oct 7). Waits for the corpus job, then:
  1. build_pool.py on corpus-clean-text-v2 -> pool-v1/
  2. sample 2,600 sections (one per paper), then hosts from other papers (one per paper)
     (if early_claude.py ran: no section sample; its papers are excluded and the Claude slices are skipped)
  3. split hosts (contiguous slices of the stratified round-robin order, so each slice keeps the venue/year spread):
       eval-only writers (split heldout): glm, mimo (space-bunny skipped: no endpoints): 300 small edits + 100 paragraph edits each
       deepseek: 4,000 small edits + 1,200 paragraph edits, 15% of papers held out
       claude (opus, sonnet; run locally): 650 small edits + 1,400 paragraph edits
       luna: 470 paragraph edits
  4. write wide1-heldout-never-train.json (eval + DeepSeek held-out papers and their host paragraphs)
  5. run api_writer.py for luna, deepseek and the eval writers (in parallel)
Env: OPENROUTER_API_KEY. Log: wide1.log in this folder. Rerunnable: finished steps are skipped.
"""
import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

W = Path(__file__).resolve().parent; POOL = W / 'pool-v1'; R = W / 'wide1'
log = open(W / 'wide1.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()
def py(*args, **kw):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=W, capture_output=True, text=True, **kw)
    if r.returncode: raise RuntimeError(f'{args[0]} failed: {r.stderr[-1500:]}')
    return r.stdout

SLICES = [('glm', 'd', 300, 1.0), ('glm', 'e', 100, 1.0), ('mimo', 'd', 300, 1.0), ('mimo', 'e', 100, 1.0),
          ('deepseek', 'd', 4000, .15), ('deepseek', 'e', 1200, .15),
          ('opus,sonnet', 'd', 650, 0.), ('opus,sonnet', 'e', 1400, 0.), ('luna', 'e', 470, 0.)]


def main():
    corpus = Path('/tmp/pangram-corpus/pipeline.log')
    early_pool = os.environ.get('WIDE1_POOL')  # e.g. pool-early: skip the corpus wait and sample from that pool
    if early_pool:
        global POOL; POOL = W / early_pool; say(event='pool_override', pool=str(POOL))
    while not early_pool and not (corpus.exists() and '"event": "done"' in corpus.read_text()):
        if corpus.exists() and '"event": "failed"' in corpus.read_text(): say(event='corpus_failed'); return
        time.sleep(300)
    R.mkdir(exist_ok=True)
    if not (POOL / 'stats.json').exists():
        say(event='pool_start')
        py('build_pool.py', '--calibration', W / 'repo/research/human-calibration-20261006/selection.json', '--out', POOL, timeout=6 * 3600)
        say(event='pool_done', total=json.loads((POOL / 'stats.json').read_text())['total'])
    early = W / 'early/exclude-papers.json'
    slices = [s for s in SLICES if ',' not in s[0]] if early.exists() else SLICES
    if early.exists(): (R / 'exclude.json').exists() or (R / 'exclude.json').write_text(early.read_text())
    elif not (R / 'sections-sample.jsonl').exists():
        say(event='sections_sample', out=py('sample.py', 'sections', '--pool', POOL, '--n', 2600, '--max-per-paper', 1, '--out', R / 'sections-sample.jsonl')[-600:])
    if not (R / 'hosts-sample.jsonl').exists():
        if not early.exists():
            papers = sorted({json.loads(l)['paper_id'] for l in open(R / 'sections-sample.jsonl')}); (R / 'exclude.json').write_text(json.dumps(papers))
        n = sum(s[2] for s in slices)
        say(event='hosts_sample', out=py('sample.py', 'hosts', '--pool', POOL, '--n', n, '--max-per-paper', int(os.environ.get('WIDE1_MAX_PER_PAPER', 1)), '--exclude', R / 'exclude.json',
                                         '--out', R / 'hosts-sample.jsonl')[-600:])
    hosts = open(R / 'hosts-sample.jsonl').read().splitlines(); k = 0; made = []
    for writers, job, n, held in slices:
        name = f"{writers.replace(',', '-')}-{job}"; part = R / f'hosts-{name}.jsonl'; items = R / f'items-{name}.jsonl'
        if not items.exists():
            part.write_text('\n'.join(hosts[k:k + n]) + '\n')
            py('make_items.py', '--job', job, '--hosts', part, '--writers', writers, '--out', items, '--heldout-frac', held, '--round', 'wide1')
        k += n; made.append((writers, job, items))
    held_items = [json.loads(l) for _, _, f in made for l in open(f)]
    held_items = [h for h in held_items if h['split'] == 'heldout']
    norm = lambda t: hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()
    paras = {x for h in held_items for x in (h['prev'], h['target'], h['next']) if x and len(re.sub(r'\W+', '', x)) >= 100}
    (R / 'wide1-heldout-never-train.json').write_text(json.dumps({'paper_ids': sorted({h['record_id'] for h in held_items}),
        'paragraph_sha256': sorted(norm(x) for x in paras), 'paragraph_shingle_sha256_12': [],
        'note': 'wide1 evaluation papers: GLM/MiMo/space-bunny (eval-only writers) and 15% of DeepSeek papers'}))
    say(event='heldout_list', papers=len({h['record_id'] for h in held_items}), paragraphs=len(paras))
    procs = []
    for writers, job, items in made:
        if ',' in writers: continue  # Claude items run locally
        out = R / f"out-{writers}-{job}"
        if (out / 'stats.json').exists() and json.loads((out / 'stats.json').read_text()).get('finished'): continue
        procs.append((writers, job, subprocess.Popen([sys.executable, 'api_writer.py', '--job', job, '--writer', writers, '--items', items, '--out', out,
                                                      '--concurrency', '48', '--budget-usd', '15'], cwd=W, stdout=open(R / f'api-{writers}-{job}.log', 'a'),
                                                     stderr=subprocess.STDOUT)))
    say(event='api_started', runs=[f'{w}-{j}' for w, j, _ in procs])
    for w, j, p in procs:
        p.wait(); out = R / f'out-{w}-{j}'
        st = json.loads((out / 'stats.json').read_text()) if (out / 'stats.json').exists() else {}
        say(event='api_done', run=f'{w}-{j}', rc=p.returncode, **{k: st.get(k) for k in ('done', 'of', 'spent_usd', 'stats', 'providers')})
    say(event='done')


if __name__ == '__main__':
    try: main()
    except Exception as e:
        import traceback; say(event='failed', error=repr(e), tb=traceback.format_exc()[-2000:]); raise
