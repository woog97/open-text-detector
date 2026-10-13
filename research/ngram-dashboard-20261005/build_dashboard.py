"""Render ngrams.json into a self-contained interactive dashboard (dashboard.html)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'ngrams-all.json').read_text())

PAGE = r'''<title>N-gram Tells</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
/* Layout: controls row, then a two-column workspace (table | scatter + detail) that stacks on narrow screens. */
:root {
  --bg: #f6f7f9; --surface: #fff; --ink: #18202b; --muted: #5d6877; --rule: #dde2e9; --accent: #2c5b8f; --accent-soft: #e6eef7;
  --ai: #2a78d6; --hu: #eb6834; --ai-soft: #e3eefb; --hu-soft: #fdeee6; --grid: #e9edf2;
  --display: "Fraunces", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12161c; --surface: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --rule: #2c3440; --accent: #8db8e8; --accent-soft: #1f2d3e;
  --ai: #3987e5; --hu: #d95926; --ai-soft: #17283d; --hu-soft: #3a2116; --grid: #252c36; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #12161c; --surface: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --rule: #2c3440; --accent: #8db8e8; --accent-soft: #1f2d3e;
  --ai: #3987e5; --hu: #d95926; --ai-soft: #17283d; --hu-soft: #3a2116; --grid: #252c36; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 var(--body); }
.wrap { max-width: 1280px; margin: 0 auto; padding-inline: 16px; padding-block: 20px 40px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }
h1 { font: 600 clamp(26px, 5vw, 36px)/1.1 var(--display); margin: 0; }
.intro { color: var(--muted); max-width: 80ch; margin: 0; }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--rule); overflow-x: auto; }
.tabs button { font: 600 14px var(--body); color: var(--muted); background: none; border: 0; border-bottom: 2px solid transparent; padding: 9px 14px; cursor: pointer; margin-bottom: -1px; flex: none; }
.tabs button[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font: 500 12px var(--mono); padding: 3px 9px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); }
.controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: end; }
.controls label { display: grid; gap: 3px; font: 500 12px var(--body); color: var(--muted); }
.controls input, .controls select { font: 14px var(--body); color: var(--ink); background: var(--surface); border: 1px solid var(--rule); border-radius: 7px; padding: 6px 9px; }
.controls input[type=search] { width: 220px; max-width: 70vw; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: 500 13px var(--body); padding: 6px 11px; border: 0; background: none; color: var(--muted); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent); }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.work { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 900px) { .work { grid-template-columns: minmax(0, 1fr); } }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; min-width: 0; }
.tblwrap { overflow: auto; max-height: 72vh; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
thead th { position: sticky; top: 0; background: var(--surface); font: 500 12px var(--body); color: var(--muted); cursor: pointer; user-select: none; z-index: 1; }
thead th:first-child, td:first-child { text-align: left; }
td:first-child { font: 500 14px var(--mono); max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
td { font: 13px var(--mono); }
tbody tr { cursor: pointer; } tbody tr:hover td { background: var(--accent-soft); }
tbody tr.sel td { background: var(--accent-soft); }
.bar { display: inline-block; height: 6px; border-radius: 3px; vertical-align: middle; margin-left: 6px; }
.side { display: grid; gap: 16px; }
.plot { padding: 10px; position: relative; }
.plot h3, .detail h3 { margin: 0 0 6px; font: 600 14px var(--body); }
.plot .sub { font: 12px var(--body); color: var(--muted); }
canvas { width: 100%; height: auto; display: block; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--rule); border-radius: 7px; padding: 5px 8px; font: 12px var(--mono); box-shadow: 0 4px 12px rgb(0 0 0 / .12); }
.detail { padding: 12px 14px; }
.detail .g { font: 600 18px var(--mono); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin: 8px 0; }
.stat { background: var(--bg); border-radius: 7px; padding: 6px 8px; }
.stat b { display: block; font: 600 15px var(--mono); }
.stat span { font: 11px var(--body); color: var(--muted); }
.ex { display: grid; gap: 6px; margin-top: 6px; }
.ex div { font: 13px/1.45 var(--body); padding: 6px 9px; border-radius: 6px; }
.ex .a { background: var(--ai-soft); } .ex .h { background: var(--hu-soft); }
.ex mark { background: none; color: inherit; font-weight: 700; text-decoration: underline; text-underline-offset: 3px; }
.legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
.note { font-size: 12px; color: var(--muted); max-width: 90ch; margin: 0; }
details.explain { border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }
details.explain > summary { cursor: pointer; padding: 10px 14px; font: 600 14px var(--body); color: var(--accent); }
.exb { padding: 0 16px 12px; display: grid; gap: 8px; max-width: 95ch; } .exb p { margin: 0; }
.ex-t table td, .ex-t table th { white-space: normal; text-align: left; font: 13px var(--body); vertical-align: top; }
.ex-t { max-height: none; border: 1px solid var(--rule); border-radius: 8px; }

.bd { display: grid; gap: 3px; }
.bdr { display: grid; grid-template-columns: minmax(0, 190px) minmax(0, 1fr) 56px; gap: 8px; align-items: center; font-size: 12px; }
.bdl { color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bdb { background: var(--bg); border-radius: 3px; height: 9px; overflow: hidden; } .bdb i { display: block; height: 100%; border-radius: 3px; }
.bdv { font: 12px var(--mono); text-align: right; }
</style>
<div class="wrap">
  <h1>N-gram Tells</h1>
  <p class="intro">Which word sequences are far more common in AI text than in human text, across all training-source data? <strong>All AI vs all human</strong> pools everything; the <strong>matched</strong> tabs hold topic constant (mirrors vs the passages they were written from; paper rewrites vs their originals), so they isolate style. Ranked by log-odds z-score (Monroe et al. 2008), which discounts rare n-grams; the ratio is the smoothed rate in AI text divided by the rate in human text. <strong>Many top n-grams are formatting, not style</strong> (curly quotes, Markdown pipes and headers, LaTeX); use <em>Words only</em> to hide them. Click any n-gram for its rate in every source category.</p>
  <details class="explain"><summary>What are "soft n-gram labels", and how is this page different?</summary><div class="exb">
    <p><strong>Two different uses of n-grams.</strong> This page asks <em>which phrases does AI use more than people?</em> It is a vocabulary comparison: "rather than", "whether", "by contrast," are tells. <strong>Soft n-gram labels</strong> answer a different question: <em>in a document an AI edited, which words are still the human's?</em> They are a way of labeling training data, not a list of AI phrases.</p>
    <p><strong>Why they're needed.</strong> When an AI edits a human document, it rewrites some sentences heavily, lightly touches others, and copies some almost word for word. Labeling the whole edited document "AI" would teach a detector that untouched human sentences are AI. So each sentence (or clause) of the edited text is matched back to the original, and n-gram overlap measures how much survived:</p>
    <div class="tblwrap ex-t"><table><thead><tr><th>Original (human)</th><th>After AI editing</th><th>Shared word 2-grams</th><th>Label</th></tr></thead><tbody>
      <tr><td>The results show a clear improvement over the baseline.</td><td>The results show a clear improvement over the baseline.</td><td>100%</td><td>human (0.0)</td></tr>
      <tr><td>We ran the test three times to be sure.</td><td>We repeated the test three times to ensure reliability.</td><td>~40%</td><td>partly AI (soft, e.g. 0.6)</td></tr>
      <tr><td>Our method is fast.</td><td>By contrast, the proposed approach markedly reduces computational overhead.</td><td>0%</td><td>AI (1.0)</td></tr>
    </tbody></table></div>
    <p>"Soft" means the label can be in between, graded by how much of the original survived, instead of a hard human-or-AI call. Pangram's reported pipeline combines word n-gram overlap, character n-gram overlap and embedding similarity for this matching. This description comes from our reconstruction of their technical report (<code>research/pangram4-reconstruction.md</code>), not a line-by-line reading of their code.</p>
    <p><strong>How it relates to our work.</strong> Our paper-edit training data labels a whole rewritten paragraph as AI, even where the rewrite copied parts nearly unchanged. Soft labels would fix that. The sentence-splice builder uses the same matching idea (word-overlap alignment between human and AI sentences) to decide which human sentence each AI sentence replaces.</p>
  </div></details>
  <div class="tabs" role="tablist" id="tabs"></div>
  <div class="chips" id="chips"></div>
  <div class="controls">
    <label>Length<span class="seg" id="nseg"></span></label>
    <label>Leaning<span class="seg" id="dseg"></span></label>
    <label for="q">Search<input type="search" id="q" placeholder="e.g. crucial, —, delve"></label>
    <label for="mindoc">Min share of documents (%)<input type="number" id="mindoc" min="0" max="50" step="0.1" value="0.2"></label>
    <label>Tokens<span class="seg" id="wseg"></span></label>
  </div>
  <div class="work">
    <div class="card"><div class="tblwrap"><table><thead><tr id="hdr"></tr></thead><tbody id="rows"></tbody></table></div></div>
    <div class="side">
      <div class="card plot"><h3>Frequency vs skew</h3><div class="sub">Each dot is an n-gram. Right = more common overall; up = more AI-leaning. Hover for values, click to inspect.</div>
        <div class="legend"><span><i style="background:var(--ai)"></i>AI-leaning</span><span><i style="background:var(--hu)"></i>Human-leaning</span></div>
        <canvas id="sc" width="640" height="420" role="img" aria-label="Scatter of n-gram frequency against AI vs human skew"></canvas><div class="tip" id="tip" hidden></div></div>
      <div class="card detail" id="detail"></div>
    </div>
  </div>
  <p class="note">Counts include every occurrence; "docs" is the share of documents containing the n-gram at least once. Only n-grams seen at least 15 times across both sides are scored, and each tab keeps the 500 most AI-leaning and 500 most human-leaning per length (before the Words-only filter). Data: all training-source pools (human source mix, paper originals and rewrites, Luna mirrors, generated manuscripts, GRADTEX); no evaluation text. Example contexts come from the same texts. Punctuation marks are separate tokens, so “ — ” and “ ; ” appear as n-grams.</p>
</div>
<script>
const DATA = __DATA__;
const S = { corpus: 'matched_general', n: 'all', dir: 'ai', q: '', mindoc: 0.2, sort: 'z', desc: true, sel: null, words: 'words' };
const ORDER = ['overall', 'matched_general', 'matched_papers', 'manuscripts', 'gradtex'].filter(k => DATA.comparisons[k]);
const SHORT = { overall: 'All AI vs all human', matched_general: 'Matched: general writing', matched_papers: 'Matched: papers', manuscripts: 'Manuscripts vs human science', gradtex: 'GRADTEX' };
const WORDY = /^[a-z0-9'’]+( [a-z0-9'’]+)*$/;
const COLS = [['g', 'N-gram'], ['ai_pm', 'AI /M'], ['h_pm', 'Human /M'], ['ratio', 'Ratio'], ['ai_doc', 'AI docs %'], ['h_doc', 'Human docs %'], ['z', 'z']];
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function allRows() {
  const c = DATA.comparisons[S.corpus].by_n; const ns = S.n === 'all' ? Object.keys(c) : [S.n];
  return ns.flatMap(n => c[n].rows.map(r => ({ ...r, n: +n })));
}
function filtered() {
  const q = S.q.trim().toLowerCase();
  return allRows().filter(r => (S.dir === 'ai' ? r.z > 0 : r.z < 0) && (!q || r.g.includes(q)) && Math.max(r.ai_doc, r.h_doc) >= S.mindoc && (S.words === 'all' || WORDY.test(r.g)))
    .sort((a, b) => { const k = S.sort, x = k === 'z' && S.dir === 'hu' ? -a[k] : a[k], y = k === 'z' && S.dir === 'hu' ? -b[k] : b[k]; return (S.desc ? -1 : 1) * (x > y ? 1 : x < y ? -1 : 0); });
}
function seg(el, items, key) {
  el.innerHTML = items.map(([v, t]) => `<button type="button" data-v="${v}" aria-pressed="${S[key] === v}">${t}</button>`).join('');
  el.querySelectorAll('button').forEach(b => b.onclick = () => { S[key] = b.dataset.v; render(); });
}
function render() {
  $('tabs').innerHTML = ORDER.map(c => `<button role="tab" aria-selected="${S.corpus === c}" data-c="${c}" title="${esc(DATA.comparisons[c].title)}">${SHORT[c]}</button>`).join('');
  $('tabs').querySelectorAll('button').forEach(b => b.onclick = () => { S.corpus = b.dataset.c; S.sel = null; render(); });
  const d = DATA.comparisons[S.corpus], t1 = d.by_n['1'];
  $('chips').innerHTML = `<span class="chip">${d.docs_ai.toLocaleString()} AI docs</span><span class="chip">${d.docs_h.toLocaleString()} human docs</span><span class="chip">${(t1.tokens_ai / 1e6).toFixed(1)}M AI tokens</span><span class="chip">${(t1.tokens_h / 1e6).toFixed(1)}M human tokens</span>`;
  seg($('nseg'), [['all', 'All'], ['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], 'n');
  seg($('dseg'), [['ai', 'AI-leaning'], ['hu', 'Human-leaning']], 'dir');
  seg($('wseg'), [['words', 'Words only'], ['all', 'Include punctuation & formatting']], 'words');
  $('hdr').innerHTML = COLS.map(([k, t]) => `<th scope="col" data-k="${k}">${t}${S.sort === k ? (S.desc ? ' ▾' : ' ▴') : ''}</th>`).join('');
  $('hdr').querySelectorAll('th').forEach(th => th.onclick = () => { const k = th.dataset.k; if (S.sort === k) S.desc = !S.desc; else { S.sort = k; S.desc = k !== 'g'; } render(); });
  const rows = filtered(); const show = rows.slice(0, 400); const maxLog = Math.max(...show.map(r => Math.abs(Math.log2(r.ratio))), 1);
  $('rows').innerHTML = show.map((r, i) => { const w = 60 * Math.abs(Math.log2(r.ratio)) / maxLog;
    return `<tr data-i="${i}" class="${S.sel && S.sel.g === r.g && S.sel.n === r.n ? 'sel' : ''}"><td title="${esc(r.g)}">${esc(r.g)}</td><td>${r.ai_pm}</td><td>${r.h_pm}</td><td>${r.ratio.toFixed(2)}<span class="bar" style="width:${w}px;background:var(${r.z > 0 ? '--ai' : '--hu'})"></span></td><td>${r.ai_doc}</td><td>${r.h_doc}</td><td>${r.z.toFixed(1)}</td></tr>`; }).join('')
    || '<tr><td colspan="7">No n-grams match these filters.</td></tr>';
  $('rows').querySelectorAll('tr[data-i]').forEach(tr => tr.onclick = () => { S.sel = show[+tr.dataset.i]; render(); });
  if (!S.sel && show.length) S.sel = show[0];
  detail(); scatter(rows);
}
function mark(ex) { return ex.map(([a, m, b]) => `${esc(a)}<mark>${esc(m)}</mark>${esc(b)}`); }
function detail() {
  const r = S.sel; if (!r) { $('detail').innerHTML = '<h3>Select an n-gram</h3>'; return; }
  const exa = r.ex_ai ? mark(r.ex_ai) : [], exh = r.ex_h ? mark(r.ex_h) : [];
  $('detail').innerHTML = `<h3>Selected n-gram</h3><div class="g">${esc(r.g)}</div>
    <div class="stats"><div class="stat"><b>${r.ratio.toFixed(2)}×</b><span>AI rate ÷ human rate</span></div><div class="stat"><b>${r.z.toFixed(1)}</b><span>log-odds z</span></div>
    <div class="stat"><b>${r.ai.toLocaleString()} / ${r.h.toLocaleString()}</b><span>count AI / human</span></div><div class="stat"><b>${r.ai_doc}% / ${r.h_doc}%</b><span>docs AI / human</span></div></div>
    <div class="ex">${exa.map(x => `<div class="a"><b>AI</b> · ${x}</div>`).join('')}${exh.map(x => `<div class="h"><b>Human</b> · ${x}</div>`).join('')}
    ${!exa.length && !exh.length ? '<p class="note">Example contexts are stored for the 100 strongest n-grams in each direction per length.</p>' : ''}</div>${breakdown(r)}`;
}
function breakdown(r) {
  const v = DATA.breakdown.rates[String(r.id)]; if (!v) return '';
  const G = DATA.breakdown.groups, mx = Math.max(...v, 1e-9);
  const rows = G.map((g, i) => [g, v[i]]).sort((a, b) => (a[0] < b[0] ? -1 : 1));
  return `<h3 style="margin-top:12px">Rate by source category <span class="note">(per million n-grams)</span></h3><div class="bd">` + rows.map(([g, x]) => {
    const kind = g.startsWith('Human') ? '--hu' : g.startsWith('GRADTEX') ? '--muted' : '--ai';
    return `<div class="bdr"><span class="bdl">${esc(g)}</span><span class="bdb"><i style="width:${(100 * x / mx).toFixed(1)}%;background:var(${kind})"></i></span><span class="bdv">${x >= 100 ? x.toFixed(0) : x.toFixed(1)}</span></div>`; }).join('') + '</div>';
}
let pts = [];
function scatter(rows) {
  const cv = $('sc'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height, m = { l: 46, r: 10, t: 10, b: 34 };
  const css = getComputedStyle(document.documentElement); const col = v => css.getPropertyValue(v).trim();
  ctx.clearRect(0, 0, W, H);
  const all = allRows().filter(r => Math.max(r.ai_doc, r.h_doc) >= S.mindoc && (S.words === 'all' || WORDY.test(r.g)));
  const fx = r => Math.log10((r.ai_pm + r.h_pm) / 2 + 1e-3), fy = r => Math.max(-10, Math.min(10, Math.log2(Math.max(r.ratio, 1e-3))));
  const xs = all.map(fx), ys = all.map(fy); const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.min(10, Math.max(4, ...ys.map(Math.abs)));
  const X = v => m.l + (W - m.l - m.r) * (v - x0) / (x1 - x0 || 1), Y = v => m.t + (H - m.t - m.b) * (1 - (v + y1) / (2 * y1));
  ctx.strokeStyle = col('--grid'); ctx.fillStyle = col('--muted'); ctx.font = '11px IBM Plex Mono, monospace'; ctx.lineWidth = 1;
  for (let v = Math.ceil(x0); v <= x1; v++) { ctx.beginPath(); ctx.moveTo(X(v), m.t); ctx.lineTo(X(v), H - m.b); ctx.stroke(); ctx.fillText(v >= 0 ? (10 ** v >= 1000 ? (10 ** v / 1000) + 'k' : 10 ** v) : 10 ** v, X(v) - 8, H - m.b + 14); }
  for (const v of [-y1, -y1 / 2, 0, y1 / 2, y1].map(Math.round)) { ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(W - m.r, Y(v)); ctx.stroke(); ctx.fillText((v >= 0 ? '' : '') + (2 ** v >= 1 ? (2 ** v) + '×' : '1/' + (2 ** -v)), 4, Y(v) + 4); }
  ctx.fillText('occurrences per million (log)', W / 2 - 80, H - 4);
  const hot = new Set(rows.slice(0, 400).map(r => r.n + '|' + r.g));
  pts = all.map((r, i) => ({ r, x: X(xs[i]), y: Y(ys[i]) }));
  for (const p of pts) { const on = hot.has(p.r.n + '|' + p.r.g); ctx.globalAlpha = on ? .85 : .18; ctx.fillStyle = col(p.r.z > 0 ? '--ai' : '--hu');
    ctx.beginPath(); ctx.arc(p.x, p.y, on ? 3 : 2, 0, 7); ctx.fill(); }
  ctx.globalAlpha = 1;
  if (S.sel) { const p = pts.find(p => p.r.g === S.sel.g && p.r.n === S.sel.n); if (p) { ctx.strokeStyle = col('--ink'); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.stroke(); } }
}
function nearest(ev) {
  const cv = $('sc'), b = cv.getBoundingClientRect(), x = (ev.clientX - b.left) * cv.width / b.width, y = (ev.clientY - b.top) * cv.height / b.height;
  let best = null, bd = 64; for (const p of pts) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = p; } } return [best, b];
}
$('sc').addEventListener('pointermove', ev => { const [p, b] = nearest(ev); const tip = $('tip');
  if (!p) { tip.hidden = true; return; } tip.hidden = false; tip.textContent = `${p.r.g}  ·  ${p.r.ratio.toFixed(2)}×  ·  z ${p.r.z.toFixed(1)}`;
  tip.style.left = Math.min(ev.clientX - b.left + 12, b.width - 200) + 'px'; tip.style.top = (ev.clientY - b.top + 40) + 'px'; });
$('sc').addEventListener('pointerleave', () => $('tip').hidden = true);
$('sc').addEventListener('click', ev => { const [p] = nearest(ev); if (p) { S.sel = p.r; S.dir = p.r.z > 0 ? 'ai' : 'hu'; render(); } });
$('q').addEventListener('input', e => { S.q = e.target.value; render(); });
$('mindoc').addEventListener('input', e => { S.mindoc = +e.target.value || 0; render(); });
render();
</script>
'''
(HERE / 'dashboard.html').write_text(PAGE.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/').replace('<!--', '<\\!--')))
print('wrote dashboard.html', (HERE / 'dashboard.html').stat().st_size // 1024, 'KB')
