"""Overnight-sweep section for the results page (data from the sweep's results/ folder)."""
import json, statistics as st, html
from pathlib import Path

SWEEP = Path(__file__).resolve().parents[3] / 'benchmarks' / 'otd' / 'training' / 'overnight-sweep-20261004' / 'results'
E = html.escape
ARMS = [  # (key, label, recipe)
    ('q4b-A', '4B · A', 'Baseline: 2e-4, cosine, 6% warmup'),
    ('q4b-I', '4B · I', '1e-4, cosine'),
    ('q4b-H', '4B · H', '3e-4, cosine'),
    ('q4b-J', '4B · J', '5e-4, cosine (LoRA Without Regret LR)'),
    ('q4b-K', '4B · K', 'A + sentence-loss weight 1.0'),
    ('q4b-L', '4B · L', 'A + more short-span windows'),
    ('q9b-A', '9B · A', 'Baseline'),
    ('q4b-B', '4B · B', '5e-4, constant, no warmup (LoRA Without Regret)'),
    ('q4b-C', '4B · C', 'B + heads at 5e-4'),
    ('q4b-D', '4B · D', 'B + sentence-loss weight 1.0'),
    ('q4b-E', '4B · E', '1e-3, constant'),
    ('q4b-F', '4B · F', 'C + D'),
    ('q4b-G', '4B · G', 'B with standard AdamW'),
    ('q9b-B', '9B · B', '5e-4, constant'),
    ('q9b-C', '9B · C', 'B + heads at 5e-4'),
    ('q9b-D', '9B · D', 'B + sentence-loss weight 1.0'),
]
COLS = [('edits_all', 'recall_at_1pct', 'All edits'), ('edits_small', 'recall_at_1pct', 'Small edits<span class="sub">1–2 sentences</span>'),
        ('edits_para', 'recall_at_1pct', 'Paragraph edits'), ('v3_sentences', 'recall_at_1pct', 'paper_v3 sentences'),
        ('standalone_rewrites', 'recall_at_1pct', 'Standalone rewrites'), ('public_docs', 'auroc', 'Public AUROC')]
CURVES = [('curve-4b-A', '4B · baseline (A)'), ('curve-4b-A2', '4B · baseline, seed 2'), ('curve-4b-J', '4B · 5e-4 cosine (J)'),
          ('curve-4b-B', '4B · 5e-4 constant (B)'), ('curve-9b-A', '9B · baseline (A)'), ('curve-9b-B2', '9B · 5e-4 constant (B)')]


def final_test(run):
    ev = {k: e for k, e in run['eval'].items() if k.startswith('stage2')}
    return ev[sorted(ev)[-1]]['test'] if len(ev) == 3 else None


def arm_table(runs):
    groups = {}
    for tag, r in runs.items():
        if tag.startswith(('q4b', 'q9b')):
            t = final_test(r)
            if t:
                groups.setdefault('-'.join(tag.split('-')[:2]), []).append(t)
    base = groups.get('q4b-A')
    head = ''.join(f'<th scope="col">{c}</th>' for _, _, c in COLS)
    body = ''
    for key, label, recipe in ARMS:
        rs = groups.get(key)
        if not rs:
            continue
        stable = key in ('q4b-A', 'q4b-I', 'q4b-H', 'q4b-J', 'q4b-K', 'q4b-L', 'q9b-A')
        cells = ''
        for sec, met, _ in COLS:
            v = [x[sec][met] for x in rs if x[sec][met] is not None]
            m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else None
            cls = 'bad' if m < 0.3 and sec != 'edits_small' and sec != 'public_docs' else ''
            cells += f'<td class="{cls}">{m:.2f}<span class="sub">{"± %.2f" % sd if sd is not None else "1 seed"}</span></td>'
        body += (f'<tr class="{"" if stable else "unstable"}"><th scope="row"><span class="mname">{E(label)}</span>'
                 f'<span class="mkind">{E(recipe)} · {len(rs)} seed{"s" if len(rs) > 1 else ""}</span></th>{cells}</tr>')
    return f'<div class="tblwrap"><table><caption>Sweep arms</caption><thead><tr><th scope="col">Arm</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def curve_points(run, key):
    hist = {(h['stage'], h['epoch']): h.get('optimizer_steps') for h in run.get('history.json') or []}
    pts = []
    for ck, e in run['eval'].items():
        if ck.startswith('step'):
            step = int(ck[4:])
        elif ck.startswith('stage2-epoch'):
            step = hist.get((2, int(ck[-1])))
            if step is None:
                continue
        else:
            continue
        v = e['test'][key]['recall_at_1pct']
        if v is not None:
            pts.append((step, v))
    return sorted(pts)


def curve_chart(runs, key, title):
    W, H, m = 360, 240, {'l': 38, 'r': 10, 't': 10, 'b': 34}
    iw, ih = W - m['l'] - m['r'], H - m['t'] - m['b']; xmax = 2625
    xs = lambda s: m['l'] + iw * s / xmax; ys = lambda v: m['t'] + ih * (1 - v)
    g = ''.join(f'<line class="gl" x1="{xs(s):.1f}" x2="{xs(s):.1f}" y1="{m["t"]}" y2="{m["t"]+ih}"/><text class="ax" x="{xs(s):.1f}" y="{m["t"]+ih+12}" text-anchor="middle">{s}</text>' for s in (0, 500, 1000, 1500, 2000, 2500))
    g += ''.join(f'<line class="gl" x1="{m["l"]}" x2="{m["l"]+iw}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/><text class="ax" x="{m["l"]-4}" y="{ys(v)+3:.1f}" text-anchor="end">{v:.1f}</text>' for v in (0, .2, .4, .6, .8, 1))
    g += f'<text class="axt" x="{m["l"]+iw/2}" y="{H-3}" text-anchor="middle">Optimizer steps (full schedule = 2,625)</text>'
    for i, (tag, label) in enumerate(CURVES, 1):
        r = runs.get(tag)
        if not r:
            continue
        p = curve_points(r, key)
        if not p:
            continue
        g += f'<polyline class="curve" stroke="var(--s{i})" points="{" ".join(f"{xs(s):.1f},{ys(v):.1f}" for s, v in p)}"/>'
        g += ''.join(f'<circle cx="{xs(s):.1f}" cy="{ys(v):.1f}" r="3" fill="var(--s{i})"><title>{E(label)} · step {s}: {v:.3f}</title></circle>' for s, v in p)
    return (f'<figure class="roc big sweepfig"><figcaption><span class="ftitle">{E(title)}</span><span class="fsub">test half, recall at 1% human-sentence FPR</span></figcaption>'
            f'<div class="plot"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{E(title)}">{g}</svg></div></figure>')


def calibration_table():
    p = SWEEP / 'calibrated-operating-points.json'
    if not p.exists():
        return '<p class="note">Calibrated operating points are still being computed.</p>'
    rep = json.loads(p.read_text()); rows = ''
    for k, v in rep.items():
        m = v['0.010']
        rows += (f'<tr><th scope="row"><span class="mname">{E(k.split("/")[0])}</span><span class="mkind">cutoff {m["cutoff"]:.3f}</span></th>'
                 f'<td>{m["test_human_fpr"]*100:.2f}%</td><td>{m["recall_all"]:.2f}</td><td>{m["recall_small"]:.2f}</td><td>{m["recall_one"]:.2f}</td>'
                 f'<td>{m["recall_para"]:.2f}</td><td>{m["v3_recall"]:.2f}</td><td>{m["v3_human_fpr"]*100:.2f}%</td></tr>')
    return ('<h3>Calibrated operating points</h3><p class="lede">Each cutoff was chosen on the separate calibration windows (195 papers, none in the evaluation set) to flag 1% of human calibration sentences, then applied unchanged to the test half. '
            '<strong>The cutoffs do not transfer.</strong> On ordinary untouched human paper passages they flag about 0.2–1.6% of sentences, close to the target. '
            'But they flag 4.5–7% of human sentences right next to AI edits, 2–5% of the matched untouched passages, and 16–29% of human paper_v3 sentences. '
            'The calibration windows contain almost none of this hard human text, so the cutoff comes out far too low. A deployable cutoff needs a calibration set that includes it.</p>'
            '<div class="tblwrap"><table><caption>Calibrated operating points</caption><thead><tr><th scope="col">Run</th><th scope="col">Realized human FPR</th>'
            '<th scope="col">All edits</th><th scope="col">Small edits</th><th scope="col">One sentence</th><th scope="col">Paragraph</th>'
            f'<th scope="col">paper_v3 recall</th><th scope="col">paper_v3 human FPR</th></tr></thead><tbody>{rows}</tbody></table></div>')


def section():
    runs = json.loads((SWEEP / 'runs.json').read_text())
    lede = ('About 40 Qwen3.5 LoRA runs overnight on 4×A100 and 3×H200. Each arm changes one thing from the baseline recipe; most are 20% of the training '
            'schedule with 2–3 seeds, plus full-length learning curves. Every checkpoint is scored on a fixed 3,511-row set split into dev and test halves by paper; '
            'the table shows the test half at the final epoch. These numbers run lower than the sentence ROC section above because this set includes more human '
            'sentences right next to edits, but every run is scored on the same set.')
    finds = ('<ul class="finds">'
             '<li><strong>Recipe changes plateau.</strong> Every stable recipe lands near 0.77 all-edit recall and 0.21–0.25 small-edit recall at 1% FPR. '
             'Learning rate (1e-4 to 5e-4), sentence-loss weight, short-span oversampling and 9B vs 4B all fall within seed noise.</li>'
             '<li><strong>The constant schedule, not the LoRA Without Regret learning rate, caused the collapses.</strong> At 5e-4 with cosine decay and warmup (J), results match the baseline; '
             'constant 5e-4 (B) trails or collapses, and 1e-3 flags every human sentence.</li>'
             '<li><strong>Training length helps paper_v3 sentences and standalone rewrites, but not small edits.</strong> Two full-length baseline seeds reach paper_v3 0.889/0.852 and standalone rewrites 0.70/0.80, against 0.77 and 0.51 at 20%; small edits stay at 0.21–0.24. '
             'Full-length 9B is best on small edits (0.26) but drops to 0.926 public AUROC. Full-length 5e-4 cosine (J) generalizes worse (paper_v3 0.66, public AUROC 0.83). '
             'Small edits look data-limited: only 2 of 24,000 training windows contain an AI span under 200 characters.</li>'
             '<li><strong>Speed:</strong> a 20% 4B run takes about 39 minutes on an H200 and 82 on an A100.</li></ul>')
    charts = (f'<div class="rocmain">{curve_chart(runs, "edits_all", "All edits vs training steps")}'
              f'{curve_chart(runs, "edits_small", "Small edits vs training steps")}</div>')
    legend = ''.join(f'<span class="lg2"><i style="background:var(--s{i})"></i>{E(l)}</span>' for i, (_, l) in enumerate(CURVES, 1))
    note = '<p class="note">Faded rows used the constant no-warmup schedule. Red cells mean recall under 0.3 on a metric where the baseline is far higher. Hover a point for its value. All six full-length runs are complete.</p>'
    return ('sweep', 'Overnight training sweep', lede,
            finds + arm_table(runs) + note + f'<h3>Learning curves</h3><div class="legend2 static">{legend}</div>' + charts + calibration_table())
