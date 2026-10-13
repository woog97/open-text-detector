"""Add each paper's average sentence score to the Oct 10 flag index, for the Submissions page's average-score sort and
the flagged-share tie-break (Alice, Oct 13 2026).

Reads the per-sentence scores extracted from the published Oct 10 detail objects (extract_oct10.py: iclr-papers.parquet,
iclr-sentences.parquet; section 0 = main text, 1 = bibliography, 2 = appendices) and the flag index now published
(indexes/flags/current.json). Writes a copy of that index with `mean` {body, nobib} per paper: the mean score over the
sentences each counting mode counts (body: main text; nobib: main text and appendices). Checks first that every paper's
sentence count per mode equals the index's counted sentences, and its flagged count at the index's thresholds equals
the index's flagged count, so the mean is taken over exactly the sentences the site counts.
Usage: mean_scores_oct10.py EXTRACT_DIR FLAGS.json OUT.json
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq

MODES = {'body': (0,), 'nobib': (0, 2)}


def main(extract, flags_path, out):
    flags = json.load(open(flags_path))
    ids = pq.read_table(f'{extract}/iclr-papers.parquet', columns=['paper', 'id']).to_pydict()
    id_of = dict(zip(ids['paper'], ids['id']))
    s = pq.read_table(f'{extract}/iclr-sentences.parquet', columns=['section', 'score', 'paper'])
    sec, score, paper = (s[c].to_numpy() for c in ('section', 'score', 'paper'))
    order = np.argsort(paper, kind='stable'); sec, score, paper = sec[order], score[order], paper[order]
    starts = np.flatnonzero(np.r_[True, paper[1:] != paper[:-1]]); ends = np.r_[starts[1:], len(paper)]
    done, bad = 0, []
    for a, b in zip(starts, ends):
        pid = id_of[int(paper[a])]; entry = flags['papers'].get(pid)
        if entry is None: continue
        mean = {}
        for mode, keep in MODES.items():
            v = score[a:b][np.isin(sec[a:b], keep)]
            for m in (mode, mode + '0.1', mode + '0.01'):
                if m not in entry['modes']: continue
                flagged, total = entry['modes'][m]
                if total != len(v) or flagged != int((v >= flags['thresholds'][m]).sum()): bad.append((pid, m, entry['modes'][m], len(v)))
            mean[mode] = float(f'{v.mean():.6g}') if len(v) else None
        entry['mean'] = mean; done += 1
    missing = [k for k, e in flags['papers'].items() if 'mean' not in e]
    print(f'means for {done:,} papers; {len(missing):,} index papers without sentences; {len(bad):,} count mismatches')
    if bad: print(bad[:10]); sys.exit(1)
    flags['mean_rule'] = 'mean sentence score per paper over the sentences each mode counts (body: main text; nobib: main text and appendices), 6 significant digits (research/atlas-oct10-20261012/mean_scores_oct10.py)'
    json.dump(flags, open(out, 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main(*sys.argv[1:])
