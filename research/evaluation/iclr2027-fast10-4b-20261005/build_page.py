"""Build the 1%-FPR calibration page (aggregates only) from results.json.

Usage: build_page.py ICLR_HIST.json HUMAN_HIST.json OUT.html
ICLR_HIST.json: {"flag": [20 bins of per-paper flagged-sentence share]}; HUMAN_HIST.json: {"human_flag_hist": [...]}
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text())
iclr = json.loads(Path(sys.argv[1]).read_text())
human = json.loads(Path(sys.argv[2]).read_text())
s = R['sentence']; d = R['document']
pick = lambda c: {k: c[k] for k in ('fpr', 'tpr', 'auc', 'n_ai', 'n_human')}
DATA = {
    'sentence': [
        {'key': 'one', 'name': 'One-sentence edits', **pick(s['roc_vs_human_papers']['papers_one']), 'tpr1': s['tpr_at_threshold']['papers_one']},
        {'key': 'two', 'name': 'Two-sentence edits', **pick(s['roc_vs_human_papers']['papers_two']), 'tpr1': s['tpr_at_threshold']['papers_two']},
        {'key': 'para', 'name': 'Paragraph edits', **pick(s['roc_vs_human_papers']['papers_para']), 'tpr1': s['tpr_at_threshold']['papers_para']},
    ],
    'document': [
        {'key': 'doc', 'name': 'Document head', **pick(d['document_prob']), 'tpr1': d['document_prob']['tpr_at_threshold']},
        {'key': 'tok', 'name': 'Mean token score', **pick(d['token_prob_mean']), 'tpr1': d['token_prob_mean']['tpr_at_threshold']},
    ],
    'hist': {'human': human['human_flag_hist'], 'iclr': iclr['flag']},
}
N = {
    'sent_t': s['threshold_all_human'], 'heldout_rate': s['heldout_fpr']['rate'], 'heldout_ci': s['heldout_fpr']['ci95'],
    'heldout_n': s['heldout_fpr']['n'], 'n_hs': s['n_human_sentences'], 'n_hp': s['n_human_papers'],
    'doc_t': d['document_prob']['threshold_1pct'], 'doc_ci': d['document_prob']['human_fpr_ci95'],
    'human_p50': d['sentence_flag_fraction']['human_p50'], 'human_p99': d['sentence_flag_fraction']['human_p99'],
}
pct = lambda v, n=1: f'{100 * v:.{n}f}%'
iclr_n = sum(DATA['hist']['iclr'])

html = r'''<title>Qwen3.5-4B 1% FPR Calibration</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>
/* Layout: one reading column of findings; charts sit full column width with a data table under each. */
:root {
  --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #6f6e69;
  --grid: #e1e0d9; --axis: #c3c2b7; --rule: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --warn-bg: #fdf3e6; --warn-ink: #8a4b00; --warn-rule: #eda100;
  --display: "Source Serif 4", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #9a9890;
  --grid: #2c2c2a; --axis: #383835; --rule: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --warn-bg: #2a1f0f; --warn-ink: #f2c46b; --warn-rule: #c98500; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #9a9890;
  --grid: #2c2c2a; --axis: #383835; --rule: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --warn-bg: #2a1f0f; --warn-ink: #f2c46b; --warn-rule: #c98500; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }
main { max-width: 860px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; display: grid; gap: 40px; }
h1 { font: 600 2rem/1.15 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 1.35rem/1.25 var(--display); margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 68ch; color: var(--ink-2); }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
header { display: grid; gap: 12px; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.fact { background: var(--surface); padding: 14px 16px; display: grid; gap: 4px; min-width: 0; }
.fact b { font: 500 1.5rem/1.1 var(--mono); font-variant-numeric: tabular-nums; }
.fact span { font-size: 13px; color: var(--ink-2); }
.warn { background: var(--warn-bg); border-left: 3px solid var(--warn-rule); padding: 16px 18px; border-radius: 0 6px 6px 0; display: grid; gap: 8px; }
.warn strong { color: var(--warn-ink); }
.warn p { color: var(--ink); }
section { display: grid; gap: 14px; min-width: 0; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 16px; display: grid; gap: 10px; min-width: 0; position: relative; }
figcaption { font-size: 13px; color: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--ink-2); }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { fill: var(--muted); font: 11px var(--mono); }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 6px 9px; font: 12px/1.45 var(--mono); color: var(--ink); box-shadow: 0 2px 8px rgba(0,0,0,.12); white-space: nowrap; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 500; color: var(--ink-2); }
ul { margin: 0; padding-left: 20px; color: var(--ink-2); max-width: 68ch; display: grid; gap: 6px; }
code { font: 13px var(--mono); }
</style>
<main>
<header>
  <div class="eyebrow">Paper detector · fast10 Qwen3.5-4B · stride 510, merged LoRA, BF16</div>
  <h1>Qwen3.5-4B flags at 1% false-positive rate</h1>
  <p>Thresholds are set so 1% of sentences in human papers (written 2022 or earlier, never seen in training) get flagged. The curves show how many AI edits the detector catches at each false-positive rate.</p>
</header>

<div class="facts">
  <div class="fact"><b>__SENT_T__</b><span>Sentence threshold at 1% FPR</span></div>
  <div class="fact"><b>__HELD__</b><span>Held-out human FPR (95% CI __HELD_CI__)</span></div>
  <div class="fact"><b>__TPR_ONE__</b><span>One-sentence edits caught</span></div>
  <div class="fact"><b>__TPR_PARA__</b><span>Paragraph edits caught</span></div>
</div>

<div class="warn">
  <strong>Do not publish ICLR 2027 flags from this model yet</strong>
  <p>At this threshold the median ICLR 2027 paper has __ICLR_MED__ of its sentences flagged, against __HUMAN_MED__ for human papers from 2022 or earlier. __ICLR_OVER__ of the __ICLR_N__ ICLR 2027 papers exceed the 99th percentile of human papers. A shift that large more likely reflects topic or era drift, or an extraction difference, than AI writing. Scoring 2023–2025 papers (ICLR 2023 submissions were written before ChatGPT) would separate these explanations.</p>
</div>

<section>
  <h2>Sentence level: AI edits in human papers</h2>
  <p>Edited sentences from the evaluation suite (shown in their paper context) against every sentence of __N_HP__ human papers (__N_HS__ sentences). Log scale on false-positive rate; the dashed line marks 1%.</p>
  <figure id="f-sent"><div class="legend" id="l-sent"></div><svg id="c-sent" viewBox="0 0 800 420" role="img" aria-label="Sentence-level ROC curves"></svg>
    <figcaption>Threshold is set on half of the human papers; the other half confirms the 1% rate (__HELD__ on __HELD_N__ sentences).</figcaption></figure>
  <div class="tablewrap"><table id="t-sent"></table></div>
</section>

<section>
  <h2>Document level: AI manuscripts vs human papers</h2>
  <p>120 AI-involved manuscripts against the same __N_HP__ human papers. Both document scores separate them completely, but with only __N_HP__ human papers the 1% threshold rests on two papers, so its false-positive rate is only known to lie between __DOC_CI__.</p>
  <figure id="f-doc"><div class="legend" id="l-doc"></div><svg id="c-doc" viewBox="0 0 800 420" role="img" aria-label="Document-level ROC curves"></svg></figure>
  <div class="tablewrap"><table id="t-doc"></table></div>
</section>

<section>
  <h2>Share of sentences flagged per paper</h2>
  <p>Each paper's share of sentences above the 1% threshold, as a percentage of papers. Human papers sit near zero, as calibrated; ICLR 2027 papers spread across the whole range.</p>
  <figure id="f-hist"><div class="legend" id="l-hist"></div><svg id="c-hist" viewBox="0 0 800 340" role="img" aria-label="Histogram of flagged sentence share"></svg></figure>
  <div class="tablewrap"><table id="t-hist"></table></div>
</section>

<section>
  <h2>Method</h2>
  <ul>
    <li>Model: fast10 Qwen3.5-4B LoRA classifier (10% training mix, checkpoint <code>stage2-epoch0</code>), run as deployed on ICLR 2027: 510-token windows without overlap, LoRA merged into BF16 weights, compiled.</li>
    <li>Sentence score: mean AI probability of the tokens in the sentence, the same rule as the six-backbone comparison.</li>
    <li>Human set: 363 papers from 2020–2022 in the clean-text archive; 148 that share text with the model's training windows were removed, leaving __N_HP__.</li>
    <li>AI sets: frozen evaluation suite v1 (paper edit reconstructions in context; full AI manuscripts).</li>
  </ul>
</section>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = ['--s1','--s2','--s3'];
const NS = 'http://www.w3.org/2000/svg';
const el = (t, a, p) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); p && p.appendChild(e); return e; };
const fmtP = v => (v*100).toFixed(v < 0.01 ? 2 : 1) + '%';

function legend(id, items) {
  document.getElementById(id).innerHTML = items.map((it, i) => `<span><i style="background:var(${COL[i]})"></i>${it}</span>`).join('');
}

function rocChart(id, series, opt) {
  const svg = document.getElementById(id), W = 800, H = 420, m = {l: 56, r: 20, t: 14, b: 44};
  const xmin = 1e-4, lx = v => m.l + (Math.log10(Math.max(v, xmin)) + 4) / 4 * (W - m.l - m.r);
  const y = v => m.t + (1 - v) * (H - m.t - m.b);
  const g = el('g', {}, svg);
  [1e-4, 1e-3, 1e-2, 1e-1, 1].forEach(v => { el('line', {x1: lx(v), x2: lx(v), y1: m.t, y2: H - m.b, stroke: 'var(--grid)', 'stroke-width': 1}, g);
    el('text', {x: lx(v), y: H - m.b + 18, 'text-anchor': 'middle'}, g).textContent = v >= 0.01 ? (v*100) + '%' : (v*100).toFixed(v === 1e-4 ? 2 : 1) + '%'; });
  [0, .25, .5, .75, 1].forEach(v => { el('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'var(--grid)', 'stroke-width': 1}, g);
    el('text', {x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end'}, g).textContent = (v*100) + '%'; });
  el('line', {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: 'var(--axis)'}, g);
  el('line', {x1: lx(0.01), x2: lx(0.01), y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-dasharray': '4 4'}, g);
  el('text', {x: lx(0.01) + 6, y: m.t + 12}, g).textContent = '1% FPR';
  el('text', {x: (W + m.l) / 2, y: H - 6, 'text-anchor': 'middle'}, g).textContent = 'False-positive rate on human text (log scale)';
  const yl = el('text', {x: 14, y: (H - m.b + m.t) / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${(H - m.b + m.t) / 2})`}, g); yl.textContent = 'AI text caught (recall)';
  series.forEach((s, i) => {
    const pts = s.fpr.map((f, j) => [f, s.tpr[j]]).filter(p => p[0] >= xmin);
    el('path', {d: pts.map((p, j) => (j ? 'L' : 'M') + lx(p[0]).toFixed(1) + ' ' + y(p[1]).toFixed(1)).join(' '), fill: 'none', stroke: `var(${COL[i]})`, 'stroke-width': 2, 'stroke-linejoin': 'round'}, g);
    el('circle', {cx: lx(0.01), cy: y(s.tpr1), r: 4.5, fill: `var(${COL[i]})`, stroke: 'var(--surface)', 'stroke-width': 2}, g);
  });
  // direct labels at the 1% markers, spaced to avoid collisions
  const lab = series.map((s, i) => ({i, yy: y(s.tpr1), t: `${s.name.split(' ')[0]} ${fmtP(s.tpr1)}`})).sort((a, b) => a.yy - b.yy);
  lab.forEach((L, k) => { if (k && L.yy - lab[k-1].yy < 14) L.yy = lab[k-1].yy + 14; el('text', {x: lx(0.01) - 8, y: L.yy + 4, 'text-anchor': 'end', fill: 'var(--ink-2)'}, g).textContent = L.t; });
  hover(svg, id, series, lx, y, m, W, H, xmin);
}

function hover(svg, id, series, lx, y, m, W, H, xmin) {
  const fig = svg.parentElement, tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; fig.appendChild(tip);
  const cross = el('line', {y1: m.t, y2: H - m.b, stroke: 'var(--axis)', 'stroke-width': 1, visibility: 'hidden'}, svg);
  const dots = series.map((s, i) => el('circle', {r: 4, fill: `var(${COL[i]})`, stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden'}, svg));
  const hit = el('rect', {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'}, svg);
  hit.addEventListener('pointermove', e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
    const f = Math.pow(10, (px - m.l) / (W - m.l - m.r) * 4 - 4);
    cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
    const rows = series.map((s, i) => { let j = 0; while (j + 1 < s.fpr.length && s.fpr[j + 1] <= f) j++;
      dots[i].setAttribute('cx', lx(Math.max(s.fpr[j], xmin))); dots[i].setAttribute('cy', y(s.tpr[j])); dots[i].setAttribute('visibility', 'visible');
      return `<span style="color:var(${COL[i]})">●</span> ${s.name}: ${fmtP(s.tpr[j])}`; });
    tip.innerHTML = `FPR ≤ ${fmtP(f)}<br>` + rows.join('<br>'); tip.hidden = false;
    const fx = (e.clientX - fig.getBoundingClientRect().left);
    tip.style.left = Math.min(fx + 14, fig.clientWidth - tip.offsetWidth - 8) + 'px'; tip.style.top = '40px';
  });
  hit.addEventListener('pointerleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); });
}

function rocTable(id, series) {
  const at = (s, f) => { let j = 0; while (j + 1 < s.fpr.length && s.fpr[j + 1] <= f) j++; return s.tpr[j]; };
  document.getElementById(id).innerHTML = '<tr><th>Set</th><th>AI items</th><th>Human items</th><th>AUROC</th><th>Recall @ 0.1%</th><th>Recall @ 1%</th><th>Recall @ 5%</th></tr>' +
    series.map(s => `<tr><td>${s.name}</td><td>${s.n_ai.toLocaleString()}</td><td>${s.n_human.toLocaleString()}</td><td>${s.auc.toFixed(4)}</td><td>${fmtP(at(s, 0.001))}</td><td>${fmtP(s.tpr1)}</td><td>${fmtP(at(s, 0.05))}</td></tr>`).join('');
}

function histChart() {
  const svg = document.getElementById('c-hist'), W = 800, H = 340, m = {l: 56, r: 20, t: 14, b: 44};
  const h = D.hist, nh = h.human.reduce((a, b) => a + b), ni = h.iclr.reduce((a, b) => a + b);
  const sets = [{name: `Human, 2022 or earlier (${nh} papers)`, v: h.human.map(c => c / nh)}, {name: `ICLR 2027 (${ni.toLocaleString()} papers)`, v: h.iclr.map(c => c / ni)}];
  const ymax = 1, bw = (W - m.l - m.r) / 20, y = v => m.t + (1 - v / ymax) * (H - m.t - m.b);
  const g = el('g', {}, svg);
  [0, .25, .5, .75, 1].forEach(v => { el('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'var(--grid)'}, g); el('text', {x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end'}, g).textContent = (v*100) + '%'; });
  [0, .25, .5, .75, 1].forEach(v => el('text', {x: m.l + v * 20 * bw, y: H - m.b + 18, 'text-anchor': 'middle'}, g).textContent = (v*100) + '%');
  el('text', {x: (W + m.l) / 2, y: H - 6, 'text-anchor': 'middle'}, g).textContent = 'Share of the paper’s sentences flagged at 1% FPR';
  const tipFig = svg.parentElement, tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; tipFig.appendChild(tip);
  sets.forEach((s, si) => s.v.forEach((v, b) => {
    const x0 = m.l + b * bw + 2 + si * (bw - 4) / 2, w = (bw - 4) / 2 - 1, top = y(v), bh = Math.max(y(0) - top, v > 0 ? 1.5 : 0);
    const r = el('rect', {x: x0, y: y(0) - bh, width: w, height: bh, rx: Math.min(2, w / 2), fill: `var(${COL[si]})`}, g);
    const hitR = el('rect', {x: m.l + b * bw, y: m.t, width: bw, height: H - m.t - m.b, fill: 'transparent'}, g);
    if (si === 1) hitR.addEventListener('pointermove', e => { tip.innerHTML = `${b*5}–${b*5+5}% flagged<br>` + sets.map((S, k) => `<span style="color:var(${COL[k]})">●</span> ${S.name.split(' (')[0]}: ${fmtP(S.v[b])}`).join('<br>');
      tip.hidden = false; const fx = e.clientX - tipFig.getBoundingClientRect().left; tip.style.left = Math.min(fx + 14, tipFig.clientWidth - tip.offsetWidth - 8) + 'px'; tip.style.top = '40px'; });
    hitR.addEventListener('pointerleave', () => tip.hidden = true);
  }));
  el('line', {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: 'var(--axis)'}, g);
  legend('l-hist', sets.map(s => s.name));
  document.getElementById('t-hist').innerHTML = '<tr><th>Flagged share</th>' + sets.map(s => `<th>${s.name.split(' (')[0]}</th>`).join('') + '</tr>' +
    h.iclr.map((_, b) => `<tr><td>${b*5}–${b*5+5}%</td>${sets.map(s => `<td>${fmtP(s.v[b])}</td>`).join('')}</tr>`).join('');
}

legend('l-sent', D.sentence.map(s => `${s.name} (AUROC ${s.auc.toFixed(3)})`)); rocChart('c-sent', D.sentence); rocTable('t-sent', D.sentence);
legend('l-doc', D.document.map(s => `${s.name} (AUROC ${s.auc.toFixed(3)})`)); rocChart('c-doc', D.document); rocTable('t-doc', D.document);
histChart();
</script>
'''
hist = DATA['hist']; ni = sum(hist['iclr'])
# share of ICLR papers above the human 99th percentile (6.2%), from the summary printed by the flag-rate job
ICLR_OVER = float(sys.argv[4]) if len(sys.argv) > 4 else None
cum = 0; med = None
for b, c in enumerate(hist['iclr']):
    cum += c
    if med is None and cum >= ni / 2: med = b
rep = {
    '__DATA__': json.dumps(DATA), '__SENT_T__': f"{N['sent_t']:.4f}", '__HELD__': pct(N['heldout_rate'], 2),
    '__HELD_CI__': f"{pct(N['heldout_ci'][0], 2)}–{pct(N['heldout_ci'][1], 2)}", '__HELD_N__': f"{N['heldout_n']:,}",
    '__TPR_ONE__': pct(s['tpr_at_threshold']['papers_one']), '__TPR_PARA__': pct(s['tpr_at_threshold']['papers_para']),
    '__N_HP__': str(N['n_hp']), '__N_HS__': f"{N['n_hs']:,}", '__DOC_CI__': f"{pct(N['doc_ci'][0])} and {pct(N['doc_ci'][1])}",
    '__ICLR_MED__': sys.argv[5] if len(sys.argv) > 5 else f'{med*5}–{med*5+5}%', '__HUMAN_MED__': pct(N['human_p50']),
    '__ICLR_OVER__': pct(ICLR_OVER) if ICLR_OVER is not None else 'most', '__ICLR_N__': f'{ni:,}',
}
for k, v in rep.items(): html = html.replace(k, v)
Path(sys.argv[3]).write_text(html)
print('wrote', sys.argv[3], len(html))
