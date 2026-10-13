"""Space-side: draw hosts or sections from the wide pool with maximum paper spread.

Uniform over papers: papers are shuffled (seeded) within venue/year strata and taken round-robin, one item per paper
per round; strata are interleaved in proportion to their eligible papers, so every eligible paper gives one item before
any gives a second. Already-used hosts/papers (--exclude: JSON list of host_ids or paper ids) are skipped.
Writes --out (JSONL items in the existing formats) and prints only summary counts.

  hosts:    python sample.py hosts    --pool POOL --n 3000 --out hosts-sample.jsonl [--max-per-paper 2]
  sections: python sample.py sections --pool POOL --n 2600 --out sections-sample.jsonl [--max-per-paper 2]
            (each line: the paper's segment record plus 'target_section' = [index, role])
"""
import argparse, gzip, json, random
from collections import Counter, defaultdict
from pathlib import Path


def round_robin(groups, n, cap, seed):
    """groups: {stratum: {paper: [items]}} -> list of items, one per paper per round, strata interleaved by size."""
    rng = random.Random(seed); queues = {}
    for g, papers in groups.items():
        order = sorted(papers); rng.shuffle(order)
        for p in order: rng.shuffle(papers[p])
        queues[g] = [(p, papers[p][:cap]) for p in order]
    out = []
    for rnd in range(cap):
        live = {g: [(p, it[rnd]) for p, it in q if len(it) > rnd] for g, q in queues.items()}
        total = sum(len(v) for v in live.values())
        # interleave strata proportionally (largest-remainder by position) so a partial round is still balanced
        merged = sorted(((i + .5) / len(v), g, x) for g, v in live.items() for i, x in enumerate(v))
        for _, g, (p, item) in merged:
            out.append(item)
            if len(out) >= n: return out
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('kind', choices=['hosts', 'sections']); ap.add_argument('--pool', required=True)
    ap.add_argument('--n', type=int, required=True); ap.add_argument('--out', required=True); ap.add_argument('--max-per-paper', type=int, default=2)
    ap.add_argument('--exclude', help='JSON list of host_ids / paper ids already used'); ap.add_argument('--seed', type=int, default=20261007)
    a = ap.parse_args(); pool = Path(a.pool); ex = set(json.load(open(a.exclude))) if a.exclude else set()
    groups = defaultdict(lambda: defaultdict(list))
    if a.kind == 'hosts':
        import pyarrow.parquet as pq
        for h in pq.read_table(pool / 'hosts.parquet').to_pylist():
            if h['host_id'] in ex or h['record_id'] in ex: continue
            groups[f"{h['venue']}/{h['year']}"][h['record_id']].append(h)
    else:
        for l in gzip.open(pool / 'segments.jsonl.gz', 'rt'):
            s = json.loads(l)
            if s['paper_id'] in ex: continue
            for i, r in s['eligible']:
                groups[f"{s['conference']}/{s['year']}"][s['paper_id']].append({**{k: v for k, v in s.items() if k != 'eligible'}, 'target_section': [i, r]})
    items = round_robin(groups, a.n, a.max_per_paper, a.seed)
    with open(a.out, 'w') as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + '\n')
    pid = (lambda x: x['record_id']) if a.kind == 'hosts' else (lambda x: x['paper_id'])
    per = Counter(map(pid, items))
    print(json.dumps({'items': len(items), 'papers': len(per), 'max_per_paper': max(per.values()) if per else 0,
                      'pool_papers': sum(len(v) for v in groups.values()),
                      'by_stratum': dict(Counter(f"{x.get('venue') or x.get('conference')}/{x['year']}" for x in items).most_common()),
                      **({'roles': dict(Counter(x['target_section'][1] for x in items))} if a.kind == 'sections' else {})}, indent=1))


if __name__ == '__main__':
    main()
