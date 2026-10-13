"""Cutoff explorer: classify every ICLR 2027 Atlas paper at any sentence cutoff, next to the 2,403 human calibration
papers, for either MoE: Oct 10 (final-moe, the live model from Oct 12) or Oct 7 (final-moe-oct7). A sidebar dropdown
switches between them; each model's files sit in its own directory (MODELS).

Data: extract.py output (per-sentence scores and sections). A sentence is flagged when score > cutoff, the Atlas rule.
A cutoff set by human false-positive rate uses cal_analyze.threshold_at on the human sentences the mode counts, so the
1% cutoffs equal the published ones.
Run:  uv run --with streamlit --with pyarrow --with pandas --with altair streamlit run app.py -- [DATA_DIR]
DATA_DIR defaults to $CUTOFF_EXPLORER_DATA or ~/pangram-local/cutoff-explorer. Oct 7 files are in DATA_DIR itself, Oct 10
files in DATA_DIR/oct10 (human files: extract.py human on the Oct 10 out-cal; ICLR files: extract.py iclr/finish on the
Oct 10 staged detail objects).
"""
import os, sys
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('CUTOFF_EXPLORER_DATA', '~/pangram-local/cutoff-explorer')).expanduser()
MODES = {'All sentences': (0, 1, 2), 'Bibliography excluded': (0, 2), 'Appendices excluded': (0, 1), 'Main text only': (0,)}
SECTION = np.array(['body', 'bibliography', 'appendix'])
PUBLISHED = {0.001: 'Atlas 0.1%', 0.01: 'Atlas 1%', 0.05: '5%', 0.1: 'Atlas 10%'}
# label: (data directory, description, default raw-score cutoff = its published 1% cutoff with every sentence counted)
MODELS = {'Oct 10 (final-moe, live)': (DATA / 'oct10', 'Oct 10 MoE (final-moe = abl-noseg-nomix, Qwen3.6 35B-A3B, single checkpoint)', 0.0052963),
          'Oct 7 (final-moe-oct7)': (DATA, 'Oct 7 MoE (final-moe-oct7, Qwen3.6 35B-A3B, last checkpoint)', 0.0058388710)}

st.set_page_config(page_title='Cutoff explorer', layout='wide')


@st.cache_resource(show_spinner='Loading sentence scores…')
def load(model):
    d = MODELS[model][0]
    h = pq.read_table(d / 'human-sentences.parquet').to_pandas(); hp = pq.read_table(d / 'human-papers.parquet').to_pandas()
    s = pq.read_table(d / 'iclr-sentences.parquet').to_pandas(); p = pq.read_table(d / 'iclr-papers.parquet').to_pandas().sort_values('paper')
    iclr = {k: s[k].to_numpy() for k in ('paper', 'section', 'score', 'start', 'end')}
    iclr['first'] = np.searchsorted(iclr['paper'], np.arange(len(p) + 1))  # sentences of paper i: first[i]:first[i+1]
    human = {k: h[k].to_numpy() for k in ('paper', 'section', 'score')}; human['half'] = hp.sort_values('paper')['half'].to_numpy()
    return human, iclr, p.reset_index(drop=True), len(hp)


@st.cache_resource(show_spinner=False)
def mode_arrays(model, mode):
    human, iclr, papers, n_human = load(model); keep = MODES[mode]
    hm = np.isin(human['section'], keep); im = np.isin(iclr['section'], keep)
    hs, hpap = human['score'][hm], human['paper'][hm]; is_, ipap = iclr['score'][im], iclr['paper'][im]
    return {'hs': hs, 'hpap': hpap, 'hs_desc': np.sort(hs)[::-1], 'h_total': np.bincount(hpap, minlength=n_human),
            'hs_a_desc': np.sort(hs[human['half'][hpap] == 0])[::-1], 'hs_b': hs[human['half'][hpap] == 1],
            'is': is_, 'ipap': ipap, 'i_total': np.bincount(ipap, minlength=len(papers))}


def threshold_at(desc, fpr):  # cal_analyze.threshold_at on a pre-sorted (descending) array
    return float(desc[min(int(np.floor(fpr * len(desc))), len(desc) - 1)])


def shares(m, t):
    hf = np.bincount(m['hpap'][m['hs'] > t], minlength=len(m['h_total'])); f = np.bincount(m['ipap'][m['is'] > t], minlength=len(m['i_total']))
    return f, f / np.maximum(m['i_total'], 1), hf / np.maximum(m['h_total'], 1)


@st.cache_data(show_spinner='Sweeping cutoffs…')
def sweep(model, mode, paper_rule):
    m = mode_arrays(model, mode); rows = []
    for fpr in np.logspace(-5, 0, 61)[:-1]:
        t = threshold_at(m['hs_desc'], fpr); _, ish, hsh = shares(m, t); ok = m['i_total'] > 0
        rows.append({'human_fpr': float(np.mean(m['hs'] > t)), 'cutoff': t, 'iclr_pooled': float(np.mean(m['is'] > t)),
                     'iclr_median_share': float(np.median(ish[ok])), 'iclr_papers_over_rule': float(np.mean(ish[ok] >= paper_rule)),
                     'human_papers_over_rule': float(np.mean(hsh[m['h_total'] > 0] >= paper_rule))})
    return pd.DataFrame(rows)


def pct(x, d=2): return f'{100 * x:.{d}f}%'


# ---- controls
with st.sidebar:
    st.header('Model')
    model = st.selectbox('Model', list(MODELS), label_visibility='collapsed')
MDIR, MDESC, MDEFAULT = MODELS[model]
missing = [f for f in ('human-sentences.parquet', 'human-papers.parquet', 'iclr-sentences.parquet', 'iclr-papers.parquet') if not (MDIR / f).exists()]
if missing:
    st.error(f'No {model} data yet in {MDIR} (missing: {", ".join(missing)}). Run extract.py (see its docstring).'); st.stop()
human, iclr, papers, n_human = load(model)
with st.sidebar:
    st.header('Cutoff')
    mode = st.radio('Sentences counted', list(MODES), help='Bibliography and appendix sentences are found by bib.py. Each mode sets its cutoffs on the human sentences it counts.')
    m = mode_arrays(model, mode)
    by = st.radio('Set the cutoff by', ['Human false-positive rate', 'Raw sentence score'], horizontal=True)
    if by == 'Human false-positive rate':
        log_fpr = st.slider('log₁₀ human FPR', -5.0, 0.0, -2.0, 0.01, key='log_fpr', help='−2 = 1%, −3 = 0.1%')
        fpr_in = st.number_input('or type the human FPR (%)', min_value=0.0001, max_value=100.0, value=None, format='%.4f', placeholder=f'{100 * 10 ** log_fpr:.4f}')
        target = (fpr_in / 100) if fpr_in else 10 ** log_fpr
        t = threshold_at(m['hs_desc'], target)
    else:
        t = st.number_input('Score cutoff (flag when score > cutoff)', min_value=0.0, max_value=1.0, value=MDEFAULT, step=0.0001, format='%.10f')
        target = None
    st.header('Paper rule')
    rule = st.slider('Flag a paper when its flagged share is at least (%)', 0.0, 100.0, 50.0, 0.5) / 100
    st.caption(f'Data: {MDIR}')

f, ish, hsh = shares(m, t); ok = m['i_total'] > 0; hok = m['h_total'] > 0
fpr_all = float(np.mean(m['hs'] > t))

# ---- summary
st.title('Cutoff explorer')
st.caption(f'{MDESC}. {ok.sum():,} ICLR 2027 papers ({len(m["is"]):,} sentences) and '
           f'{hok.sum():,} human calibration papers ({len(m["hs"]):,} sentences), counting: {mode.lower()}.')
c = st.columns(6)
c[0].metric('Score cutoff', f'{t:.6g}')
c[1].metric('Human sentence FPR', pct(fpr_all, 3))
if target is not None:
    ta = threshold_at(m['hs_a_desc'], target); c[2].metric('Held-out FPR', pct(float(np.mean(m['hs_b'] > ta)), 3), help='Cutoff set on half A of the human papers (cal-split-v1), flag rate measured on half B.')
else:
    c[2].metric('Held-out FPR', '—', help='Shown when the cutoff is set by human FPR.')
c[3].metric('ICLR sentences flagged', pct(float(np.mean(m['is'] > t)), 1))
c[4].metric('ICLR median paper share', pct(float(np.median(ish[ok])), 1))
c[5].metric(f'ICLR papers ≥ {pct(rule, 1)}', pct(float(np.mean(ish[ok] >= rule)), 1), help=f'{int((ish[ok] >= rule).sum()):,} papers. Human papers at the same rule: {pct(float(np.mean(hsh[hok] >= rule)), 2)} ({int((hsh[hok] >= rule).sum())} of {hok.sum():,}).')
st.caption(f'Human papers flagged by the paper rule: {pct(float(np.mean(hsh[hok] >= rule)), 2)} ({int((hsh[hok] >= rule).sum())} of {hok.sum():,}).')

tab1, tab2, tab3, tab4 = st.tabs(['Per-paper flagged share', 'Sentence scores', 'Cutoff sweep', 'Papers'])

with tab1:
    nb = st.select_slider('Bins', [10, 20, 25, 50, 100], value=20)
    edges = np.linspace(0, 1, nb + 1); ci = np.histogram(ish[ok], edges)[0]; chh = np.histogram(hsh[hok], edges)[0]
    hist = pd.DataFrame({'bin': [f'{100 * a:.0f}–{100 * b:.0f}%' for a, b in zip(edges[:-1], edges[1:])], 'lo': edges[:-1],
                         'ICLR 2027 papers': ci, 'ICLR 2027 %': ci / ci.sum(), 'human papers': chh, 'human %': chh / chh.sum()})
    long = pd.concat([pd.DataFrame({'lo': hist.lo, 'bin': hist.bin, 'share of papers': hist['ICLR 2027 %'], 'papers': hist['ICLR 2027 papers'], 'set': 'ICLR 2027'}),
                      pd.DataFrame({'lo': hist.lo, 'bin': hist.bin, 'share of papers': hist['human %'], 'papers': hist['human papers'], 'set': 'Human 2019–2022'})])
    st.altair_chart(alt.Chart(long).mark_bar(opacity=.85).encode(
        x=alt.X('bin:N', sort=list(hist.bin), title='Flagged share of the paper’s sentences'), xOffset='set:N',
        y=alt.Y('share of papers:Q', axis=alt.Axis(format='%'), title='Share of papers'), color=alt.Color('set:N', title=None),
        tooltip=['set', 'bin', alt.Tooltip('papers:Q', format=','), alt.Tooltip('share of papers:Q', format='.2%')]), width='stretch')
    st.dataframe(hist.drop(columns='lo'), hide_index=True, width='stretch',
                 column_config={'ICLR 2027 %': st.column_config.NumberColumn(format='percent'), 'human %': st.column_config.NumberColumn(format='percent')})
    st.download_button('Download bins (CSV)', hist.drop(columns='lo').to_csv(index=False), 'flagged-share-bins.csv')

with tab2:
    lo = st.slider('Score range shown (log₁₀)', -8.0, 0.0, (-6.0, 0.0), 0.5)
    e = np.logspace(lo[0], lo[1], 81)
    a = np.histogram(np.clip(m['is'], e[0], e[-1]), e)[0]; b = np.histogram(np.clip(m['hs'], e[0], e[-1]), e)[0]
    sd = pd.concat([pd.DataFrame({'score': e[:-1], 'to': e[1:], 'share': a / a.sum(), 'sentences': a, 'set': 'ICLR 2027'}),
                    pd.DataFrame({'score': e[:-1], 'to': e[1:], 'share': b / b.sum(), 'sentences': b, 'set': 'Human 2019–2022'})])
    base = alt.Chart(sd).mark_line(interpolate='step-after').encode(
        x=alt.X('score:Q', scale=alt.Scale(type='log'), title='Sentence score (log scale; end bins hold everything beyond)'),
        y=alt.Y('share:Q', axis=alt.Axis(format='%'), title='Share of sentences'), color=alt.Color('set:N', title=None),
        tooltip=['set', alt.Tooltip('score:Q', format='.3g', title='from'), alt.Tooltip('to:Q', format='.3g'), alt.Tooltip('sentences:Q', format=','), alt.Tooltip('share:Q', format='.2%')])
    rule_line = alt.Chart(pd.DataFrame({'score': [t]})).mark_rule(strokeDash=[5, 4]).encode(x='score:Q', tooltip=[alt.Tooltip('score:Q', format='.6g', title='cutoff')])
    st.altair_chart(base + rule_line, width='stretch')

with tab3:
    sw = sweep(model, mode, rule)
    long = sw.melt(id_vars=['human_fpr', 'cutoff'], value_vars=['iclr_pooled', 'iclr_median_share', 'iclr_papers_over_rule', 'human_papers_over_rule'], var_name='measure')
    long['measure'] = long.measure.map({'iclr_pooled': 'ICLR sentences flagged', 'iclr_median_share': 'ICLR median paper share',
                                        'iclr_papers_over_rule': f'ICLR papers ≥ {pct(rule, 1)}', 'human_papers_over_rule': f'Human papers ≥ {pct(rule, 1)}'})
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X('human_fpr:Q', scale=alt.Scale(type='log'), axis=alt.Axis(format='%'), title='Human sentence FPR (log scale)'),
        y=alt.Y('value:Q', axis=alt.Axis(format='%'), title=None), color=alt.Color('measure:N', title=None),
        tooltip=['measure', alt.Tooltip('human_fpr:Q', format='.4%'), alt.Tooltip('cutoff:Q', format='.6g'), alt.Tooltip('value:Q', format='.2%')])
    now = alt.Chart(pd.DataFrame({'human_fpr': [max(fpr_all, 1e-5)]})).mark_rule(strokeDash=[5, 4]).encode(x='human_fpr:Q')
    st.altair_chart(lines + now, width='stretch')
    st.dataframe(sw, hide_index=True, width='stretch')

with tab4:
    tbl = papers.assign(sentences_counted=m['i_total'], flagged=f, share=ish)[ok]
    q = st.text_input('Search titles or OpenReview ids')
    if q: tbl = tbl[tbl.title.str.contains(q, case=False, regex=False) | tbl.openreview.str.contains(q, case=False, regex=False)]
    st.download_button('Download all papers at this cutoff (CSV)', tbl.drop(columns=['paper']).to_csv(index=False), 'papers-at-cutoff.csv')
    ev = st.dataframe(tbl[['paper', 'openreview', 'title', 'sentences_counted', 'flagged', 'share']].sort_values('share', ascending=False),
                      hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row',
                      column_config={'share': st.column_config.ProgressColumn(format='percent', min_value=0, max_value=1)})
    rows = ev.selection.rows
    if rows:
        pid = int(tbl.sort_values('share', ascending=False).iloc[rows[0]]['paper'])
        text = pq.read_table(MDIR / 'iclr-text.parquet', filters=[('paper', '=', pid)], columns=['text']).column('text')[0].as_py()
        a, b = iclr['first'][pid], iclr['first'][pid + 1]
        sd = pd.DataFrame({'section': SECTION[iclr['section'][a:b]], 'score': iclr['score'][a:b],
                           'sentence': [text[x:y] for x, y in zip(iclr['start'][a:b], iclr['end'][a:b])]})
        sd['counted'] = np.isin(iclr['section'][a:b], MODES[mode]); sd['flagged'] = sd.counted & (sd.score > t)
        t2 = st.number_input('Compare with a second cutoff (raw score)', min_value=0.0, max_value=1.0, value=None, format='%.10f', placeholder='optional')
        if t2 is not None:
            sd['flagged at 2nd'] = sd.counted & (sd.score > t2); st.caption(f'{int((sd.flagged != sd["flagged at 2nd"]).sum())} sentences change status.')
        st.markdown(f'**{papers.title[pid]}** · [OpenReview](https://openreview.net/forum?id={papers.openreview[pid]})')
        st.dataframe(sd, hide_index=True, width='stretch', column_config={'score': st.column_config.NumberColumn(format='%.6f')})
