"""Render metrics.json into a static, phone-friendly results page (results.html)."""
import json, html, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / 'metrics.json').read_text())
MODELS = D['models']
DONE = [m for m in MODELS if m['complete']]
PENDING = [m for m in MODELS if not m['complete']]
E = html.escape


def pct(x, d=1):
    return '–' if x is None else f'{x*100:.{d}f}%'


def num(x, d=3):
    return '–' if x is None else f'{x:.{d}f}'


def cell(text, sub=None, cls='', title=None):
    t = f' title="{E(title)}"' if title else ''
    s = f'<span class="sub">{E(sub)}</span>' if sub else ''
    return f'<td class="{cls}"{t}>{E(text)}{s}</td>'


def table(caption, cols, rows, note=None, best=None):
    """rows: list of (model, [(value_for_ranking, html_cell), ...]). best: per-col 'max'/'min'/None."""
    if best:
        for j, how in enumerate(best):
            if not how:
                continue
            vals = [r[1][j][0] for r in rows if r[1][j][0] is not None]
            if len(vals) < 2:
                continue
            target = max(vals) if how == 'max' else min(vals)
            for r in rows:
                if r[1][j][0] == target:
                    r[1][j] = (r[1][j][0], r[1][j][1].replace('<td class="', '<td class="best ', 1))
    head = ''.join(f'<th scope="col">{c}</th>' for c in cols)
    body = ''
    for m, cells in rows:
        body += f'<tr><th scope="row"><span class="mname">{E(m["name"])}</span><span class="mkind">{E(m["kind"])}</span></th>' + ''.join(c for _, c in cells) + '</tr>'
    for m in PENDING:
        body += f'<tr class="pending"><th scope="row"><span class="mname">{E(m["name"])}</span><span class="mkind">{E(m["kind"])}</span></th><td colspan="{len(cols)-1}">Evaluation still running</td></tr>'
    n = f'<p class="note">{note}</p>' if note else ''
    return f'<div class="tblwrap"><table><caption>{caption}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{n}'


def rate_cell(r, warn=None, bad=None, invert=False):
    if not r or r['rate'] is None:
        return (None, cell('–'))
    v = r['rate']; cls = ''
    if warn is not None:
        if (v >= bad) if not invert else (v <= bad):
            cls = 'bad'
        elif (v >= warn) if not invert else (v <= warn):
            cls = 'warn'
    lo, hi = r['ci']
    return (v, cell(pct(v), f"{r['k']:,}/{r['n']:,}", cls, f"95% CI {pct(lo)}–{pct(hi)}"))


rows_summary, rows_fpr, rows_tpr, rows_auc, rows_sparse, rows_span, rows_asst, rows_ms = [], [], [], [], [], [], [], []
HUMAN_GROUPS = ['Research-paper paragraphs', 'Paper paragraphs + context', 'English-learner essays', 'Other human text (public sets)', 'All human documents']
AUC_SETS = ['Comparison (pooled)', 'Papers: human vs rewritten/generated', 'detectrl', 'epoch', 'meld_eval', 'sem_detect', 'local_binary', 'local_length']
AI_SETS = ['arena50', 'detectrl', 'epoch', 'gede', 'local_binary', 'local_length', 'meld_eval', 'perkins', 'saha', 'sem_detect', 'vub']
GENS = sorted({g for m in DONE for g in m['metrics']['manuscripts_by_generator']})

for m in DONE:
    s = m['metrics']
    o1 = s['oracle_tpr']['1% FPR']['tpr']
    wf = s['spans']['Workflow reconstructions']
    rows_summary.append((m, [
        rate_cell(s['human_fpr']['All human documents'], .01, .02),
        rate_cell(s['ai_tpr']['Public AI text (comparison)']),
        (s['auroc']['Comparison (pooled)'], cell(num(s['auroc']['Comparison (pooled)']))),
        (o1, cell(pct(o1))),
        (s['auroc']['Papers: human vs rewritten/generated'], cell(num(s['auroc']['Papers: human vs rewritten/generated']))),
        (wf['f1'], cell(num(wf['f1'], 2), f"P {pct(wf['precision'],0)} · R {pct(wf['recall'],0)}")),
    ]))
    rows_fpr.append((m, [rate_cell(s['human_fpr'][g], .01, .02) for g in HUMAN_GROUPS]))
    rows_tpr.append((m, [rate_cell(s['ai_tpr']['Full generated manuscripts']), rate_cell(s['ai_tpr']['Fully rewritten paper paragraphs'])]
                     + [rate_cell(s['ai_tpr_by_dataset'].get(d)) for d in AI_SETS]))
    rows_auc.append((m, [(s['auroc'].get(k), cell(num(s['auroc'].get(k)))) for k in AUC_SETS]))
    se = s['sparse_edits']
    rows_sparse.append((m, [rate_cell(se[c]['detected']) for c in ['sentence', 'two_sentence', 'paragraph_concise', 'paragraph_v3']]
                        + [rate_cell(s['sparse_edits_untouched'], .02, .05)]))
    rows_span.append((m, [(v['f1'], cell(num(v['f1'], 2), f"P {pct(v['precision'],0)} · R {pct(v['recall'],0)}"))
                          for v in s['spans'].values()]))
    a = s['assistance']
    rows_asst.append((m, [(a[c]['doc_flag']['rate'], cell(pct(a[c]['doc_flag']['rate']), f"{pct(a[c]['mean_flagged_tokens'],0)} tokens"))
                          for c in ['proofread', 'light_polish', 'substantial_rewrite']]))
    rows_ms.append((m, [rate_cell(s['manuscripts_by_generator'][g]) for g in GENS]))

status = (f'<span class="chip ok">All {len(MODELS)} complete</span>' if not PENDING else
          f'<span class="chip run">{len(DONE)} of {len(MODELS)} complete</span><span class="chip">{", ".join(E(m["name"]) for m in PENDING)} running</span>')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%b %-d, %Y · %H:%M UTC')

sections = [
    ('summary', 'Summary', 'Human false-positive rate and AI recall use the uncalibrated 0.5 cutoff. AUROC needs no cutoff, so it is the fairest comparison across models today.',
     table('Headline metrics', ['Model', 'Human FPR<span class="sub">all human docs</span>', 'AI recall<span class="sub">public AI text</span>', 'AUROC<span class="sub">public sets</span>', 'Recall @ 1% FPR<span class="sub">oracle cutoff</span>', 'AUROC<span class="sub">papers</span>', 'Edit-span F1<span class="sub">paper rewrites</span>'],
           rows_summary, best=['min', 'max', 'max', 'max', 'max', 'max'],
           note='“Recall @ 1% FPR” picks the cutoff on the evaluation humans themselves, so it is optimistic. It shows what calibration could reach, not a deployable number.')),
    ('fpr', 'Human false positives', 'Share of human-written documents flagged as AI. This is the costliest error for a detector. Amber marks 1% or more, red 2% or more.',
     table('Human documents flagged as AI', ['Model'] + [E(g) for g in HUMAN_GROUPS], rows_fpr, best=['min'] * 5,
           note='English-learner essays combine ELLIPSE, Liang (TOEFL) and PELIC, the usual check for bias against non-native writers. Tap or hover a cell for the 95% interval.')),
    ('tpr', 'AI detection', 'Share of fully AI-written documents flagged at the 0.5 cutoff.',
     table('AI documents flagged', ['Model', 'Generated manuscripts', 'Rewritten paper paragraphs'] + [E(d) for d in AI_SETS], rows_tpr, best=['max'] * (2 + len(AI_SETS)),
           note='Manuscript titles and abstracts were written by humans, so manuscripts count as AI-involved documents. ModernBERT saw some reserved manuscript families in training, so its manuscript score is not an unseen test.')),
    ('auc', 'Ranking quality (AUROC)', 'How well each model ranks AI above human text, regardless of cutoff. 0.5 is chance and 1.0 is perfect. Only datasets with at least 20 of each class are shown.',
     table('AUROC by dataset', ['Model'] + [E(k) for k in AUC_SETS], rows_auc, best=['max'] * len(AUC_SETS))),
    ('sparse', 'Small edits inside human papers', 'Share of edited paper passages where at least one AI-edited sentence was flagged. The last column is the false-alarm rate on matching untouched passages.',
     table('Edited passages caught', ['Model', 'One sentence', 'Two sentences', 'Paragraph (concise)', 'Paragraph (full)', 'Untouched: any flag'], rows_sparse, best=['max', 'max', 'max', 'max', 'min'])),
    ('spans', 'Locating AI text', 'Token-level F1, with precision (P) and recall (R), for marking which tokens are AI-written inside mixed documents.',
     table('Token-level span detection', ['Model'] + [E(k) for k in DONE[0]['metrics']['spans']], rows_span, best=['max'] * 4,
           note='Paper edits (paper_v3_target) label only the edited spans, so their precision is always 100%. Read recall there.')),
    ('assist', 'Light AI assistance', 'Documents flagged after AI proofreading or polishing, with the mean share of tokens flagged. Whether these should be flagged is a policy choice, so this is a diagnostic, not accuracy.',
     table('Assistance flag rates', ['Model', 'Proofread', 'Light polish', 'Substantial rewrite'], rows_asst)),
    ('ms', 'Manuscripts by generator', 'Recall on the 120 reserved full manuscripts, 40 per generator.',
     table('Manuscript recall', ['Model'] + [E(g) for g in GENS], rows_ms)),
]

ROC_MAIN = [('Comparison (pooled)', 'Public sets, pooled'), ('Papers: human vs rewritten/generated', 'Research papers')]
ROC_SMALL = ['detectrl', 'epoch', 'sem_detect', 'meld_eval', 'local_binary', 'local_length']
SENT = json.loads((ROOT / 'sentence_roc.json').read_text()) if (ROOT / 'sentence_roc.json').exists() else {}
SENT_MAIN = [('papers_all', 'Edited sentences in papers, all sizes'), ('papers_small', 'One- and two-sentence edits')]
SENT_SMALL = [('papers_one', 'One sentence'), ('papers_two', 'Two sentences'), ('papers_para', 'Paragraph'),
              ('paper_v3_target', 'paper_v3_target'), ('paper_pilots_exploratory', 'paper_pilots'), ('opai', 'opai')]
roc_data = {'grid': D['fpr_grid'], 'models': [{'name': m['name'], 'slot': i + 1,
            'roc': {**{k: m['metrics']['roc'].get(k) for k, _ in ROC_MAIN + [(d, d) for d in ROC_SMALL]},
                    **{'s:' + k: SENT.get(m['key'], {}).get(k) for k, _ in SENT_MAIN + SENT_SMALL}}}
            for i, m in enumerate(MODELS) if m['complete']]}


def fig(key, title, big, unit='documents'):
    c = next((m['roc'][key] for m in roc_data['models'] if m['roc'][key]), None)
    sub = f"{c['n_ai']:,} AI · {c['n_human']:,} human {unit}" if c else ''
    return (f'<figure class="roc {"big" if big else "small"}" data-key="{E(key)}" data-unit="{unit}"><figcaption><span class="ftitle">{E(title)}</span>'
            f'<span class="fsub">{sub}</span></figcaption><div class="plot" role="img" aria-label="ROC curves for {E(title)}"></div></figure>')


legend = ''.join(f'<button type="button" class="lg" data-slot="{m["slot"]}" aria-pressed="false"><i style="background:var(--s{m["slot"]})"></i>{E(m["name"])}</button>'
                 for m in roc_data['models'])
rocbar = (f'<div class="rocbar"><div class="seg" role="group" aria-label="False-positive axis">'
          f'<button type="button" class="scale" data-log="0" aria-pressed="true">Full curve</button><button type="button" class="scale" data-log="1" aria-pressed="false">Low-FPR zoom</button></div>'
          f'<div class="legend2">{legend}</div></div>')
roc_html = (rocbar + f'<div class="rocmain">{"".join(fig(k, t, True) for k, t in ROC_MAIN)}</div>'
            f'<h3>Per public dataset</h3><div class="rocgrid">{"".join(fig(d, d, False) for d in ROC_SMALL)}</div>'
            '<p class="note">Drag or hover across a chart to read every model\'s AI recall at that false-positive rate. Tap a model name to highlight it. '
            'The dashed vertical line marks 1% human FPR; the diagonal is chance. Papers compare human paper paragraphs with fully rewritten paragraphs and generated manuscripts. '
            'meld_eval and local_binary have only 34 and 32 human documents, so their curves are coarse. Exact AUROC values are in the table under Ranking quality.</p>')
sections.insert(1, ('roc', 'Document ROC curves', 'Each document gets one score (the mean of its token probabilities) and counts as fully human or fully AI. Each curve shows AI recall against the human false-positive rate across every cutoff. Higher and further left is better; the low-FPR end matters most for a detector.', roc_html))

sent_rows = []
for m in MODELS:
    c = SENT.get(m['key'])
    if not c:
        continue
    cells = []
    for k, _ in [('papers_one', 0), ('papers_two', 0), ('papers_para', 0), ('papers_all', 0)]:
        v = c[k]
        cells.append((v['tpr_at_1pct'], cell(pct(v['tpr_at_1pct']), f"AUROC {v['auc']:.3f}")))
    op = c['papers_all']['op_05']
    cells.append((op['fpr'], cell(pct(op['fpr'], 2), f"recall {pct(op['tpr'], 0)}")))
    sent_rows.append((m, cells))
missing = [m['name'] for m in MODELS if m['key'] not in SENT]
sent_table = table('Sentence recall at 1% human-sentence FPR', ['Model', 'One-sentence edits', 'Two-sentence edits', 'Paragraph edits', 'All edits',
                   'At the 0.5 cutoff<span class="sub">human sentences flagged</span>'], sent_rows, best=['max', 'max', 'max', 'max', None])
sent_html = (rocbar
             + f'<div class="rocmain">{"".join(fig("s:" + k, t, True, "sentences") for k, t in SENT_MAIN)}</div>'
             + f'<h3>By edit size and dataset</h3><div class="rocgrid">{"".join(fig("s:" + k, t, False, "sentences") for k, t in SENT_SMALL)}</div>'
             + '<p class="note">Positives are AI-edited sentences inside paper passages, shown with their surrounding paper text (each edit counted once). '
               'Negatives are human sentences from the same paper workflow: the untouched sentences around each edit, the matching untouched passages and 4,901 other human paper passages. '
               'A sentence score is the mean of its token probabilities, the same rule the evaluation uses. Open circles on the large charts mark where the 0.5 cutoff falls for each model. '
               'paper_v3_target, paper_pilots and opai compare AI and human sentences within those public datasets.</p>'
             + sent_table
             + '<p class="note">Recall at 1% FPR picks the cutoff on these evaluation sentences, so it is optimistic. It compares models fairly but is not a deployable number. '
             + (f'{", ".join(missing)} is missing because its saved token scores are incomplete in the storage backup. ' if missing else '')
             + 'Each recall cell also shows that set\'s sentence AUROC.</p>')
sections.insert(2, ('sentroc', 'Sentence ROC curves', 'Each sentence gets its own score and label, so passages with small AI edits count directly. This answers the main question for papers: how many AI-edited sentences a model catches while flagging a fixed share of human sentences.', sent_html))

from sweep_section import section as sweep_section
from compare_section import section as compare_section
_, _, sw_lede, sw_html = sweep_section()
cmp_lede, cmp_html = compare_section()

nav = ''.join(f'<a href="#{sid}">{E(t)}</a>' for sid, t, _, _ in sections)
old_body = ''.join(f'<section id="{sid}"><h2>{E(t)}</h2><p class="lede">{E(d)}</p>{tbl}</section>' for sid, t, d, tbl in sections)
OLD_INTRO = ('Six backbones trained on the same current-data mixture and scored on the full frozen evaluation suite in BF16. '
             'Flags use an uncalibrated 0.5 cutoff. ModernBERT ran the full training schedule; the other five ran at 10% length, so differences partly reflect training budget.')
import takeaways
TABS = [('compare', 'Old vs new', takeaways.COMPARE + f'<p class="lede">{E(cmp_lede)}</p>{cmp_html}'),
        ('sweep', 'Overnight sweep · Oct 4–5', takeaways.SWEEP + f'<p class="lede">{E(sw_lede)}</p>{sw_html}'),
        ('oct3', 'Backbone comparison · Oct 3', takeaways.OCT3 + f'<p class="lede">{E(OLD_INTRO)}</p>'
                 '<div class="legend"><span class="lb">Best in column</span><span class="lw">Human FPR ≥ 1%</span><span class="lr">Human FPR ≥ 2%</span></div>'
                 f'<nav aria-label="Sections">{nav}</nav>{old_body}')]
tabbar = ('<div class="tabs" role="tablist" aria-label="Result sets">' + ''.join(
    f'<button type="button" role="tab" id="tab-{tid}" aria-controls="panel-{tid}" aria-selected="{str(i == 0).lower()}" data-tab="{tid}">{E(t)}</button>'
    for i, (tid, t, _) in enumerate(TABS)) + '</div>')
body = tabbar + ''.join(f'<div class="panel" role="tabpanel" id="panel-{tid}" aria-labelledby="tab-{tid}"{"" if i == 0 else " hidden"}>{h}</div>'
                        for i, (tid, _, h) in enumerate(TABS))

page = f'''<title>Backbone Eval Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
/* Layout: single reading column; each question is a section with one horizontally scrollable table and a pinned model column. */
:root {{
  --bg: #f6f7f9; --surface: #ffffff; --ink: #18202b; --muted: #5d6877; --rule: #dde2e9;
  --accent: #2c5b8f; --accent-soft: #e6eef7;
  --good: #1f7a4d; --good-bg: #e3f3ea; --warn: #8a5a00; --warn-bg: #fbf0d9; --bad: #a3322a; --bad-bg: #f8e2df;
  --display: "Fraunces", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --grid: #e9edf2;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --grid: #252c36; }} }}
:root[data-theme="dark"] {{ --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --grid: #252c36; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12161c; --surface: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --rule: #2c3440;
  --accent: #8db8e8; --accent-soft: #1f2d3e;
  --good: #7fd0a5; --good-bg: #183026; --warn: #f0c46a; --warn-bg: #352a12; --bad: #f0958c; --bad-bg: #3a1d1a; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12161c; --surface: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --rule: #2c3440;
  --accent: #8db8e8; --accent-soft: #1f2d3e;
  --good: #7fd0a5; --good-bg: #183026; --warn: #f0c46a; --warn-bg: #352a12; --bad: #f0958c; --bad-bg: #3a1d1a; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.5 var(--body); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-inline: 16px; padding-block: 20px 48px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 28px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 500 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 600 clamp(28px, 6vw, 40px)/1.1 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 22px/1.2 var(--display); margin: 0; text-wrap: balance; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ font: 500 12px var(--mono); padding: 3px 9px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); }}
.chip.ok {{ background: var(--good-bg); color: var(--good); }}
.chip.run {{ background: var(--warn-bg); color: var(--warn); }}
.intro {{ max-width: 68ch; color: var(--muted); margin: 0; }}
.intro strong {{ color: var(--ink); font-weight: 600; }}
nav {{ position: sticky; top: env(safe-area-inset-top, 0px); z-index: 3; background: var(--bg); display: flex; gap: 6px; overflow-x: auto; padding-block: 8px; border-bottom: 1px solid var(--rule); scrollbar-width: none; }}
nav a {{ flex: none; font: 500 13px var(--body); color: var(--accent); text-decoration: none; padding: 5px 10px; border-radius: 6px; background: var(--surface); border: 1px solid var(--rule); }}
nav a:focus-visible, nav a:hover {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
section {{ display: grid; gap: 10px; scroll-margin-top: 56px; min-width: 0; }}
.lede {{ margin: 0; max-width: 68ch; color: var(--muted); }}
.tblwrap {{ overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }}
table {{ border-collapse: separate; border-spacing: 0; width: 100%; font-variant-numeric: tabular-nums; }}
caption {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }}
th, td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; vertical-align: top; }}
tbody tr:last-child > * {{ border-bottom: 0; }}
thead th {{ font: 500 12px/1.3 var(--body); color: var(--muted); background: var(--surface); white-space: normal; min-width: 92px; vertical-align: bottom; }}
thead th:first-child, tbody th {{ text-align: left; position: sticky; left: 0; z-index: 1; background: var(--surface); border-right: 1px solid var(--rule); min-width: 132px; }}
td {{ font: 500 14px var(--mono); }}
.sub {{ display: block; font: 400 11px var(--mono); color: var(--muted); margin-top: 2px; }}
thead .sub {{ font-family: var(--body); }}
.mname {{ display: block; font: 600 14px var(--body); }}
.mkind {{ display: block; font: 400 11px var(--body); color: var(--muted); }}
td.best {{ background: var(--accent-soft); color: var(--accent); }}
td.best .sub {{ color: var(--accent); }}
td.warn {{ background: var(--warn-bg); color: var(--warn); }}
td.bad {{ background: var(--bad-bg); color: var(--bad); }}
tr.pending td {{ text-align: left; font: italic 400 13px var(--body); color: var(--muted); }}
.note {{ margin: 0; font-size: 13px; color: var(--muted); max-width: 72ch; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: var(--muted); }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}
.legend .lb::before {{ background: var(--accent-soft); outline: 1px solid var(--accent); }}
.legend .lw::before {{ background: var(--warn-bg); outline: 1px solid var(--warn); }}
.legend .lr::before {{ background: var(--bad-bg); outline: 1px solid var(--bad); }}
h3 {{ font: 600 16px/1.3 var(--body); margin: 8px 0 0; }}
.rocbar {{ display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }}
.seg {{ display: inline-flex; border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; background: var(--surface); }}
.seg button {{ font: 500 13px var(--body); padding: 6px 12px; border: 0; background: none; color: var(--muted); cursor: pointer; }}
.seg button[aria-pressed="true"] {{ background: var(--accent-soft); color: var(--accent); }}
.legend2 {{ display: flex; flex-wrap: wrap; gap: 4px 6px; }}
.lg {{ font: 500 12px var(--body); color: var(--ink); background: var(--surface); border: 1px solid var(--rule); border-radius: 999px; padding: 3px 9px 3px 7px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }}
.lg i {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
.lg[aria-pressed="true"] {{ border-color: var(--ink); }}
button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.rocmain {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 12px; }}
.rocgrid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 160px), 1fr)); gap: 10px; }}
@media (max-width: 420px) {{ .rocgrid {{ grid-template-columns: 1fr 1fr; }} }}
figure.roc {{ margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 10px 10px 6px; min-width: 0; position: relative; }}
figcaption {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 8px; align-items: baseline; }}
.ftitle {{ font: 600 14px var(--body); }}
.small .ftitle {{ font: 500 12px var(--mono); }}
.fsub {{ font: 400 11px var(--mono); color: var(--muted); }}
.small .fsub {{ display: none; }}
.plot {{ position: relative; touch-action: pan-y; }}
.plot svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
.plot .ax {{ font: 400 10px var(--mono); fill: var(--muted); }}
.plot .axt {{ font: 500 10px var(--body); fill: var(--muted); }}
.plot .gl {{ stroke: var(--grid); stroke-width: 1; }}
.plot .chance {{ stroke: var(--rule); stroke-width: 1.5; stroke-dasharray: 4 4; fill: none; }}
.plot .ref {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: .7; }}
.plot .curve {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; transition: opacity .15s; }}
.small .plot .curve {{ stroke-width: 1.5; }}
.plot.dim .curve {{ opacity: .15; }}
.plot.dim .curve.on {{ opacity: 1; stroke-width: 2.5; }}
.plot .xh {{ stroke: var(--ink); stroke-width: 1; opacity: .5; }}
.plot .op {{ fill: var(--surface); stroke-width: 2; }}
.plot .op.faded {{ opacity: .15; }}
.plot .dot {{ stroke: var(--surface); stroke-width: 2; }}
.tip {{ position: absolute; z-index: 4; pointer-events: none; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 6px 8px; box-shadow: 0 4px 14px rgb(0 0 0 / .12); font: 400 11px var(--mono); color: var(--ink); min-width: 150px; }}
.tip b {{ display: block; font: 600 11px var(--body); margin-bottom: 3px; color: var(--muted); }}
.tip div {{ display: flex; align-items: center; gap: 6px; }}
.tip i {{ width: 10px; height: 3px; border-radius: 2px; flex: none; }}
.tip span {{ flex: 1; font-family: var(--body); }}
@media (prefers-reduced-motion: reduce) {{ .plot .curve {{ transition: none; }} }}
tr.unstable th, tr.unstable td {{ opacity: .55; }}
ul.finds {{ margin: 0; padding-left: 18px; display: grid; gap: 6px; max-width: 72ch; }}
ul.finds strong {{ font-weight: 600; }}
.legend2.static {{ display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: 12px; }}
.lg2 {{ display: inline-flex; align-items: center; gap: 6px; }}
.lg2 i {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
.tabs {{ display: flex; gap: 4px; border-bottom: 1px solid var(--rule); overflow-x: auto; scrollbar-width: none; }}
.tabs button {{ flex: none; font: 600 14px var(--body); color: var(--muted); background: none; border: 0; border-bottom: 2px solid transparent; padding: 10px 14px; cursor: pointer; margin-bottom: -1px; }}
.tabs button[aria-selected="true"] {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tabs button:hover {{ color: var(--ink); }}
.panel {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 20px; }}
.panel[hidden] {{ display: none; }}
.panel > section {{ padding-top: 8px; }}
tr.grp th {{ text-align: left; font: 600 11px var(--body); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); background: var(--bg); position: static; }}
tr.delta th, tr.delta td {{ border-top: 2px solid var(--rule); }}
td.good {{ color: var(--good); background: var(--good-bg); }}
td.worse {{ color: var(--bad); background: var(--bad-bg); }}
details.takeaways {{ border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }}
details.takeaways > summary {{ cursor: pointer; padding: 12px 16px; font: 600 15px var(--body); color: var(--accent); list-style-position: inside; }}
details.takeaways[open] > summary {{ border-bottom: 1px solid var(--rule); }}
.tk {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 0; }}
.tkg {{ padding: 12px 16px; border-left: 3px solid var(--rule); margin: 10px 12px; }}
.tkg h4 {{ margin: 0 0 6px; font: 600 12px var(--body); letter-spacing: .06em; text-transform: uppercase; }}
.tkg ul, .tkg ol {{ margin: 0; padding-left: 18px; display: grid; gap: 6px; font-size: 14px; line-height: 1.45; max-width: 75ch; }}
.tkg.clear {{ border-left-color: var(--good); }} .tkg.clear h4 {{ color: var(--good); }}
.tkg.likely {{ border-left-color: var(--accent); }} .tkg.likely h4 {{ color: var(--accent); }}
.tkg.unclear {{ border-left-color: var(--warn); }} .tkg.unclear h4 {{ color: var(--warn); }}
.tkg.next {{ border-left-color: var(--ink); }} .tkg.next h4 {{ color: var(--ink); }}
footer {{ font-size: 12px; color: var(--muted); display: grid; gap: 6px; max-width: 72ch; }}
footer ul {{ margin: 0; padding-left: 18px; display: grid; gap: 4px; }}
</style>
<div class="wrap">
<header>
  <span class="eyebrow">Open Text Detector · AI-text classifiers</span>
  <h1>Backbone Eval Results</h1>
  <div class="chips"><span class="chip">Updated {stamp}</span></div>
  <p class="intro">Two rounds of training. <strong>Oct 3:</strong> six backbones compared on the full evaluation suite. <strong>Oct 4–5:</strong> an overnight Qwen3.5 LoRA sweep of about 40 runs testing learning rate, schedule, loss and training length. Start with <strong>Old vs new</strong> for the head-to-head on a common test set.</p>
</header>
{body}
<footer>
  <strong>Notes</strong>
  <ul>
    <li>The four profiles overlap (workflow, comparison, assistance, manuscripts). Don't add their counts together.</li>
    <li>“Document flagged” means at least half the document's tokens scored ≥ 0.5. AUROC ranks documents by mean token probability.</li>
    <li>Human false positives pool clean human documents from the workflow and comparison profiles. Rows marked assisted or ambiguous are excluded.</li>
    <li>Cutoffs fit on the separate calibration windows don't transfer to hard human text (see Calibrated operating points under the overnight sweep). A deployable cutoff needs a calibration set that includes human sentences next to edits and paired paper originals.</li>
    <li>Source: research/evaluation/backbone-results-20261003 (analyze.py → metrics.json → build_page.py).</li>
  </ul>
</footer>
</div>
'''
SCRIPT = r'''<script>
(() => {
const D = __DATA__;
const G = D.grid, NS = 'http://www.w3.org/2000/svg';
let logX = false, hi = null;
const el = (t, a, p) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
const pctf = v => v === 0 ? '0%' : v < 0.01 ? (v * 100).toFixed(2) + '%' : v < 0.1 ? (v * 100).toFixed(1) + '%' : Math.round(v * 100) + '%';
const LOGMIN = 1e-3;
function draw(fig) {
  const key = fig.dataset.key, big = fig.classList.contains('big'), box = fig.querySelector('.plot');
  box.textContent = '';
  const W = big ? 360 : 200, H = big ? 300 : 190, m = big ? {l: 38, r: 10, t: 8, b: 32} : {l: 28, r: 6, t: 6, b: 24};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = v => m.l + iw * (logX ? (Math.log10(Math.max(v, LOGMIN)) - Math.log10(LOGMIN)) / -Math.log10(LOGMIN) : v);
  const ys = v => m.t + ih * (1 - v);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`}, box);
  const xt = logX ? [0.001, 0.01, 0.1, 1] : [0, 0.2, 0.4, 0.6, 0.8, 1], yt = [0, 0.2, 0.4, 0.6, 0.8, 1];
  for (const v of xt) { el('line', {class: 'gl', x1: xs(v), x2: xs(v), y1: m.t, y2: m.t + ih}, svg); const t = el('text', {class: 'ax', x: xs(v), y: m.t + ih + 12, 'text-anchor': 'middle'}, svg); t.textContent = logX ? pctf(v) : v.toFixed(1); }
  for (const v of yt) { el('line', {class: 'gl', x1: m.l, x2: m.l + iw, y1: ys(v), y2: ys(v)}, svg); const t = el('text', {class: 'ax', x: m.l - 4, y: ys(v) + 3, 'text-anchor': 'end'}, svg); t.textContent = v.toFixed(1); }
  if (big) {
    const a = el('text', {class: 'axt', x: m.l + iw / 2, y: H - 3, 'text-anchor': 'middle'}, svg); const u = fig.dataset.unit === 'sentences' ? 'Human-sentence false-positive rate' : 'Human false-positive rate'; a.textContent = logX ? u + ' (log)' : u;
    const b = el('text', {class: 'axt', x: 10, y: m.t + ih / 2, 'text-anchor': 'middle', transform: `rotate(-90 10 ${m.t + ih / 2})`}, svg); b.textContent = 'AI recall (TPR)';
  }
  const ch = []; for (let i = 0; i <= 60; i++) { const v = logX ? Math.pow(10, Math.log10(LOGMIN) * (1 - i / 60)) : i / 60; ch.push(`${xs(v).toFixed(1)},${ys(v).toFixed(1)}`); }
  el('polyline', {class: 'chance', points: ch.join(' ')}, svg);
  el('line', {class: 'ref', x1: xs(0.01), x2: xs(0.01), y1: m.t, y2: m.t + ih}, svg);
  const series = [];
  for (const md of D.models) {
    const c = md.roc[key]; if (!c) continue;
    const pts = []; G.forEach((g, i) => { if (!logX || g >= LOGMIN) pts.push(`${xs(g).toFixed(1)},${ys(c.tpr[i]).toFixed(1)}`); });
    const p = el('polyline', {class: 'curve' + (hi === md.slot ? ' on' : ''), points: pts.join(' '), stroke: `var(--s${md.slot})`}, svg);
    series.push({md, c, p});
    if (big && c.op_05) { const ox = Math.max(c.op_05.fpr, logX ? LOGMIN : 0); el('circle', {class: 'op' + (hi !== null && hi !== md.slot ? ' faded' : ''), cx: xs(ox), cy: ys(c.op_05.tpr), r: 4.5, stroke: `var(--s${md.slot})`}, svg); }
  }
  box.classList.toggle('dim', hi !== null);
  const xh = el('line', {class: 'xh', y1: m.t, y2: m.t + ih, visibility: 'hidden'}, svg);
  const dots = series.map(s => el('circle', {class: 'dot', r: big ? 4 : 3, fill: `var(--s${s.md.slot})`, visibility: 'hidden'}, svg));
  const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; box.appendChild(tip);
  const inv = px => { const f = Math.min(1, Math.max(0, (px - m.l) / iw)); return logX ? Math.pow(10, Math.log10(LOGMIN) * (1 - f)) : f; };
  function move(ev) {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    if (px < m.l - 4 || px > m.l + iw + 4) return hide();
    const x = inv(px); let i = 0; while (i < G.length - 1 && G[i + 1] <= x) i++;
    const g = G[i];
    xh.setAttribute('x1', xs(g)); xh.setAttribute('x2', xs(g)); xh.setAttribute('visibility', 'visible');
    const rows = series.map((s, k) => { dots[k].setAttribute('cx', xs(g)); dots[k].setAttribute('cy', ys(s.c.tpr[i])); dots[k].setAttribute('visibility', hi === null || hi === s.md.slot ? 'visible' : 'hidden'); return [s.c.tpr[i], s.md]; })
      .sort((a, b) => b[0] - a[0]);
    tip.innerHTML = `<b>At ${pctf(g)} human FPR</b>` + rows.map(([v, md]) => `<div><i style="background:var(--s${md.slot})"></i><span>${md.name}</span>${pctf(v)}</div>`).join('');
    tip.hidden = false;
    const bw = box.clientWidth, sx = (xs(g) / W) * bw, tw = tip.offsetWidth;
    tip.style.left = Math.max(0, Math.min(bw - tw, sx > bw / 2 ? sx - tw - 10 : sx + 10)) + 'px';
    tip.style.top = (big ? 8 : 0) + 'px';
  }
  function hide() { tip.hidden = true; xh.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); }
  svg.addEventListener('pointermove', move); svg.addEventListener('pointerdown', move); svg.addEventListener('pointerleave', hide);
}
const figs = [...document.querySelectorAll('figure.roc[data-key]')];
const redraw = () => figs.forEach(draw);
document.querySelectorAll('.scale').forEach(b => b.addEventListener('click', () => { logX = b.dataset.log === '1'; document.querySelectorAll('.scale').forEach(x => x.setAttribute('aria-pressed', String((x.dataset.log === '1') === logX))); redraw(); }));
document.querySelectorAll('.lg').forEach(b => b.addEventListener('click', () => {
  const s = +b.dataset.slot; hi = hi === s ? null : s;
  document.querySelectorAll('.lg').forEach(x => x.setAttribute('aria-pressed', String(+x.dataset.slot === hi)));
  redraw();
}));
redraw();
const tabs = [...document.querySelectorAll('.tabs button')];
const show = id => { tabs.forEach(b => { const on = b.dataset.tab === id; b.setAttribute('aria-selected', String(on)); document.getElementById('panel-' + b.dataset.tab).hidden = !on; }); };
tabs.forEach(b => b.addEventListener('click', () => { show(b.dataset.tab); try { history.replaceState(null, '', '#' + b.dataset.tab); } catch (e) {} }));
const h = location.hash.slice(1); if (tabs.some(b => b.dataset.tab === h)) show(h);
})();
</script>'''
page += SCRIPT.replace('__DATA__', json.dumps(roc_data, separators=(',', ':')))
(ROOT / 'results.html').write_text(page)
print('wrote results.html', len(page))
