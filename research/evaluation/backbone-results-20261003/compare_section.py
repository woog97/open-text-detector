"""'Old vs new' tab: Oct 3 backbones vs overnight sweep, all scored on the sweep's fixed test half."""
import json, statistics as st, html
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parents[2] / 'benchmarks' / 'otd' / 'training' / 'overnight-sweep-20261004' / 'results'
E = html.escape
COLS = [('edits_all', 'recall_at_1pct', 'All edits'), ('edits_small', 'recall_at_1pct', 'Small edits<span class="sub">1–2 sentences</span>'),
        ('edits_one', 'recall_at_1pct', 'One sentence'), ('edits_para', 'recall_at_1pct', 'Paragraph edits'),
        ('v3_sentences', 'recall_at_1pct', 'paper_v3 sentences'), ('standalone_rewrites', 'recall_at_1pct', 'Standalone rewrites'),
        ('public_docs', 'auroc', 'Public AUROC')]


def final(run):
    ev = {k: e for k, e in run['eval'].items() if k.startswith('stage2')}
    return ev[sorted(ev)[-1]]['test'] if len(ev) == 3 else None


def agg(tests):
    out = {}
    for sec, met, _ in COLS:
        v = [t[sec][met] for t in tests if t[sec][met] is not None]
        out[sec] = (st.mean(v), st.stdev(v) if len(v) > 1 else None)
    return out


def table(caption, rows, best_cols=True, delta=None):
    """rows: list of (label, sublabel, group, {sec: (mean, sd)})."""
    best = {sec: max(r[3][sec][0] for r in rows) for sec, _, _ in COLS} if best_cols else {}
    head = ''.join(f'<th scope="col">{c}</th>' for _, _, c in COLS)
    body = ''; last = None
    for label, sub, group, vals in rows:
        if group != last:
            body += f'<tr class="grp"><th scope="rowgroup" colspan="{len(COLS)+1}">{E(group)}</th></tr>'; last = group
        cells = ''.join(f'<td class="{"best" if best and abs(vals[sec][0]-best[sec])<1e-9 else ""}">{vals[sec][0]:.2f}'
                        f'<span class="sub">{"± %.2f" % vals[sec][1] if vals[sec][1] is not None else "&nbsp;"}</span></td>' for sec, _, _ in COLS)
        body += f'<tr><th scope="row"><span class="mname">{E(label)}</span><span class="mkind">{E(sub)}</span></th>{cells}</tr>'
    if delta:
        cells = ''.join(f'<td class="{"good" if d > 0.005 else "worse" if d < -0.005 else ""}">{d:+.2f}</td>' for d in delta)
        body += f'<tr class="delta"><th scope="row"><span class="mname">Best now − best earlier</span><span class="mkind">per column</span></th>{cells}</tr>'
    return f'<div class="tblwrap"><table><caption>{E(caption)}</caption><thead><tr><th scope="col">Run</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


HCOLS = [('test_human_fpr', 'Human FPR<span class="sub">sentences, realized</span>', True), ('recall_all', 'All edits', False),
         ('recall_small', 'Small edits', False), ('recall_one', 'One sentence', False), ('recall_para', 'Paragraph', False),
         ('v3_recall', 'paper_v3<span class="sub">recall · FPR</span>', 'v3_test_fpr'), ('standalone_recall', 'Standalone<span class="sub">recall · FPR</span>', 'standalone_test_fpr'),
         ('public_recall', 'Public docs<span class="sub">recall · FPR</span>', 'public_test_fpr')]


OLD_TIME = {'old:qwen36-35b-a3b': (126, 'H200'), 'old:gemma4-12b': (75, 'H200'), 'old:qwen35-9b': (65, 'A100'),
            'old:qwen35-4b': (65, 'A100'), 'old:modernbert': (93, 'A100')}  # W&B train/elapsed_seconds, Oct 3


def train_time(keys, runs):
    """'(126 min, H200)' or '(72–88 min, A100)' for a set of runs; training only, excludes evaluation."""
    vals = []
    for k in keys:
        if k in OLD_TIME:
            vals.append(OLD_TIME[k])
        else:
            r = runs.get(k.split(':', 1)[1], {}); s = r.get('status') or {}
            if s.get('elapsed_seconds'):
                vals.append((round(s['elapsed_seconds'] / 60), r.get('host', '?')))
    if not vals:
        return ''
    out = []
    for host in sorted({h for _, h in vals}):
        m = sorted(v for v, h in vals if h == host)
        span = f'{m[0]}' if m[0] == m[-1] else f'{m[0]}–{m[-1]}'
        hrs = f' ≈ {m[-1]/60:.1f} h' if m[-1] >= 120 else ''
        out.append(f'{span} min{hrs} on {host}')
    return ' (' + '; '.join(out) + ')'


def honest_table(hon, groups, runs=None):
    """groups: list of (group, [(label, sub, [keys])]). Values averaged over keys (seeds)."""
    head = ''.join(f'<th scope="col">{c}</th>' for _, c, _ in HCOLS); body = ''
    for group, items in groups:
        body += f'<tr class="grp"><th scope="rowgroup" colspan="{len(HCOLS)+1}">{E(group)}</th></tr>'
        for label, sub, keys in items:
            vs = [hon[k] for k in keys if k in hon]
            if not vs:
                continue
            cells = ''
            for key, _, extra in HCOLS:
                m = st.mean(v[key] for v in vs)
                if extra is True:
                    cls = 'bad' if m >= .02 else 'warn' if m >= .0125 else ''
                    cells += f'<td class="{cls}">{m*100:.1f}%</td>'
                elif extra:
                    f = st.mean(v[extra] for v in vs)
                    cells += f'<td>{m:.2f}<span class="sub">at {f*100:.1f}% FPR</span></td>'
                else:
                    sd = st.stdev(v[key] for v in vs) if len(vs) > 1 else None
                    cells += f'<td>{m:.2f}<span class="sub">{"± %.2f" % sd if sd is not None else "&nbsp;"}</span></td>'
            body += f'<tr><th scope="row"><span class="mname">{E(label)}</span><span class="mkind">{E(sub + train_time(keys, runs or {}))}</span></th>{cells}</tr>'
    return f'<div class="tblwrap"><table><caption>Held-out 1% operating point</caption><thead><tr><th scope="col">Run</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def section():
    old = json.loads((HERE / 'comparison-old-on-sweep-set.json').read_text())
    runs = json.loads((SWEEP / 'runs.json').read_text())
    o = lambda k: agg([old[k]['test']])
    n = lambda *tags: agg([final(runs[t]) for t in tags if final(runs.get(t, {'eval': {}}))])
    arm = lambda prefix: agg([final(r) for t, r in runs.items() if t.startswith(prefix + '-s') and final(r)])
    narm = lambda prefix: sum(1 for t, r in runs.items() if t.startswith(prefix + '-s') and final(r))

    early = [('Qwen3.6 35B-A3B MoE', 'Oct 3 · 10% length · best earlier detector', 'Best earlier', o('qwen36-35b-a3b')),
             ('ModernBERT-large', 'Oct 3 · full length · strongest earlier on edits', 'Best earlier', o('modernbert'))]
    now = [('Qwen3.6 35B-A3B MoE, baseline', 'overnight · 20% length · 1 seed', 'Best now', n('moe-A-s1')),
           ('Qwen3.5 9B, baseline', 'overnight · full length · 1 seed', 'Best now', n('curve-9b-A')),
           ('Qwen3.5 4B, baseline', 'overnight · full length · 2 seeds', 'Best now', n('curve-4b-A', 'curve-4b-A2'))]
    delta = [max(r[3][sec][0] for r in now) - max(r[3][sec][0] for r in early) for sec, _, _ in COLS]
    headline = table('Best earlier vs best now', early + now, delta=delta)

    q4 = [('Oct 3 run', '10% length · micro-batch 1', 'Qwen3.6 35B-A3B MoE', o('qwen36-35b-a3b')),
          ('Overnight recipe, 20% length', '2e-4 cosine · micro-batch 8 · final epoch · 1 seed', 'Qwen3.6 35B-A3B MoE', n('moe-A-s1')),
          ('Oct 3 run', '10% length · micro-batch 1 · loss-picked epoch-0 checkpoint', 'Qwen3.5 4B', o('qwen35-4b')),
          ('Overnight, 20% length', f'2e-4 cosine · final epoch · {narm("q4b-A")} seeds', 'Qwen3.5 4B', arm('q4b-A')),
          ('Overnight, full length', '2e-4 cosine · final epoch · 2 seeds', 'Qwen3.5 4B', n('curve-4b-A', 'curve-4b-A2')),
          ('Oct 3 run', '10% length · loss-picked checkpoint', 'Qwen3.5 9B', o('qwen35-9b')),
          ('Overnight, 20% length', f'2e-4 cosine · final epoch · {narm("q9b-A")} seeds', 'Qwen3.5 9B', arm('q9b-A')),
          ('Overnight, full length', '2e-4 cosine · final epoch · 1 seed', 'Qwen3.5 9B', n('curve-9b-A'))]
    same = table('Same backbone, then and now', q4, best_cols=False)

    others = [(m['label'].split(' (')[0], m['label'][m['label'].find('(') + 1:-1] + ' · Oct 3', 'Other Oct 3 backbones', o(k))
              for k, m in old.items() if k not in ('qwen36-35b-a3b', 'modernbert', 'qwen35-4b', 'qwen35-9b')]
    others += [('Qwen3.5 4B, 5e-4 cosine (J)', 'overnight · full length · 1 seed', 'Other overnight full-length runs', n('curve-4b-J')),
               ('Qwen3.5 4B, 5e-4 constant (B)', 'overnight · full length · collapsed', 'Other overnight full-length runs', n('curve-4b-B')),
               ('Qwen3.5 9B, 5e-4 constant (B)', 'overnight · full length · collapsed', 'Other overnight full-length runs', n('curve-9b-B2'))]
    rest = table('Other runs on the same set', others, best_cols=False)

    hon = json.loads((HERE / 'honest-1pct.json').read_text())
    seeds = lambda pre, n: [f'new:{pre}-s{i}' for i in range(1, n + 1)]
    honest = honest_table(hon, runs=runs, groups=[
        ('Best earlier', [('Qwen3.6 35B-A3B MoE', 'Oct 3 · 10% length', ['old:qwen36-35b-a3b']), ('ModernBERT-large', 'Oct 3 · full length', ['old:modernbert'])]),
        ('Best now', [('Qwen3.6 35B-A3B MoE, baseline', 'overnight · 20% length · 1 seed', ['new:moe-A-s1']),
                      ('Qwen3.5 9B, baseline', 'overnight · full length · 1 seed', ['new:curve-9b-A']),
                      ('Qwen3.5 4B, baseline', 'overnight · full length · 2 seeds', ['new:curve-4b-A', 'new:curve-4b-A2'])]),
        ('Same backbone, then and now', [('Qwen3.6 35B-A3B MoE · Oct 3', '10% length', ['old:qwen36-35b-a3b']),
                                         ('Qwen3.6 35B-A3B MoE · 20%', 'overnight · 1 seed', ['new:moe-A-s1']),
                                         ('Qwen3.5 4B · Oct 3', '10% length', ['old:qwen35-4b']), ('Qwen3.5 4B · 20%', 'overnight · 3 seeds', seeds('q4b-A', 3)),
                                         ('Qwen3.5 9B · Oct 3', '10% length', ['old:qwen35-9b']), ('Qwen3.5 9B · 20%', 'overnight · 3 seeds', seeds('q9b-A', 3))]),
        ('Other overnight arms at 20% (stable schedule)', [('4B · I: 1e-4', '2 seeds', seeds('q4b-I', 2)), ('4B · H: 3e-4', '2 seeds', seeds('q4b-H', 2)),
                                         ('4B · J: 5e-4 cosine', '2 seeds', seeds('q4b-J', 2)), ('4B · K: sentence weight 1.0', '3 seeds', seeds('q4b-K', 3)),
                                         ('4B · L: more short spans', '3 seeds', seeds('q4b-L', 3)), ('4B · J at full length', '1 seed', ['new:curve-4b-J']),
                                         ('Gemma 4 12B · Oct 3', '10% length', ['old:gemma4-12b'])])])
    matched = honest_table(hon, runs=runs, groups=[
        ('10% of the training schedule · Oct 3 (all matched)', [
            ('Qwen3.6 35B-A3B MoE', 'Oct 3', ['old:qwen36-35b-a3b']), ('Gemma 4 12B', 'Oct 3', ['old:gemma4-12b']),
            ('Qwen3.5 9B', 'Oct 3', ['old:qwen35-9b']), ('Qwen3.5 4B', 'Oct 3 · loss picked the earliest stage-2 checkpoint', ['old:qwen35-4b'])]),
        ('20% of the schedule · overnight (closest to 10%; twice the data)', [
            ('Qwen3.6 35B-A3B MoE, baseline', '1 seed', ['new:moe-A-s1']),
            ('Qwen3.5 9B, baseline', '3 seeds', seeds('q9b-A', 3)), ('Qwen3.5 4B, baseline', '3 seeds', seeds('q4b-A', 3)),
            ('Qwen3.5 4B, best other stable arm (H: 3e-4)', '2 seeds', seeds('q4b-H', 2))]),
        ('100% of the schedule · same length on both dates', [
            ('ModernBERT-large', 'Oct 3', ['old:modernbert']), ('Qwen3.5 9B, baseline', 'overnight · 1 seed', ['new:curve-9b-A']),
            ('Qwen3.5 4B, baseline', 'overnight · 2 seeds', ['new:curve-4b-A', 'new:curve-4b-A2']),
            ('Qwen3.5 4B, 5e-4 cosine (J)', 'overnight · 1 seed', ['new:curve-4b-J'])])])
    matched_intro = ('<h3>Matched training length</h3><p class="lede">Same held-out 1% cutoffs as above, grouped by how much of the training schedule each run saw. Times in parentheses are training only (excluding evaluation), with the GPU type; overnight A100 runs shared their GPU with background evaluation. '
                     'The 10% and 100% groups are exactly matched; there are no overnight 10% runs, so 20% is the closest cross-date match. '
                     'Oct 3 decoder runs also used micro-batch 1 and a checkpoint picked by validation loss, while overnight runs used micro-batch 8 and the final epoch, '
                     'so 10% vs 20% mixes length with those recipe differences.</p>')
    honest_intro = ('<h3>Held-out 1% operating point</h3><p class="lede">The tables below pick each model\'s cutoff on the test set itself, which flatters every model. '
                    'Here each cutoff is set to flag 1% of human text on the <strong>dev half</strong> (paper-disjoint from test) and applied unchanged to the test half. '
                    'Realized test FPR lands at 0.6–2.1%, so this transfers well, unlike the calibration windows (2–23%). Compare recall together with the realized FPR in the first column: '
                    'amber is ≥1.25%, red ≥2%. Separate cutoffs are set for edit sentences, paper_v3 sentences, standalone documents and public documents.</p>')
    lede = ('Every number on this tab comes from the same fixed evaluation set: the test half of the overnight sweep set (paper-grouped, 1,514 rows). '
            'Oct 3 runs were rescored from their saved token scores, with no new inference. Edits and paper_v3 are sentence recall at 1% human-sentence FPR, with the cutoff chosen on this set; '
            'public is document AUROC. Ettin is missing because its saved scores are incomplete.')
    finds = ('<ul class="finds">'
             f'<li><strong>All-edit recall improved only slightly over the best earlier model</strong> ({delta[0]:+.2f} over ModernBERT), but paper_v3 and standalone rewrites improved a lot ({delta[4]:+.2f} and {delta[5]:+.2f}).</li>'
             '<li><strong>Small edits did not move.</strong> At the held-out 1% operating point, the Oct 3 MoE still leads on small edits (0.35 at 1.6% FPR, against 0.30 at 1.4% for the overnight 9B). This is the main open problem and looks data-limited.</li>'
             '<li><strong>The same Qwen3.5 4B went from 0.20 to 0.79 all-edit recall.</strong> The Oct 3 run was short (10%), ran at micro-batch 1, and its loss-picked checkpoint was the earliest stage-2 epoch. Overnight runs train longer and use the final epoch.</li>'
             '<li><strong>The MoE with the overnight recipe (20%, 1 seed) is the best edit detector so far:</strong> 0.84 all edits and 0.98 paragraph edits at 1.5% held-out FPR, against 0.80 and 0.92 for the Oct 3 MoE. Small edits (0.30 vs 0.35) and standalone rewrites (0.55 vs 0.61) did not improve, within the noise for one seed. It trained in 62 minutes on one H200.</li>'
             '<li><strong>Trade-off for the 9B:</strong> best on small edits (0.26), but public AUROC fell to 0.93 at full length. The 4B keeps 0.97.</li>'
             '<li><strong>Calibration:</strong> cutoffs fit on the calibration windows flag 2–23% of test human sentences, not 1%. Cutoffs fit on the dev half of this set land at 0.6–2.1%; the held-out table below uses those.</li></ul>')
    return lede, finds + honest_intro + honest + matched_intro + matched + '<h3>Recall at 1% FPR, cutoff chosen on the test set</h3>' + headline + '<h3>Same backbone, then and now</h3>' + same + '<h3>Other runs</h3>' + rest
