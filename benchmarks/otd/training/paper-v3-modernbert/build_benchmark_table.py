"""Present completed BF16 scores beside explicitly separate Pangram 4 claims.
No inference or deployment-threshold changes. Empirical ROC is descriptive only.
"""
from pathlib import Path
import sys,json,gzip,collections,hashlib
ROOT=Path(__file__).parent;OUT=ROOT/'comparison-v1';sys.path.insert(0,str(ROOT.parents[1]))
from metrics import ranking
MODELS=['ours','meld-v5','meld-v8'];SOURCE='https://arxiv.org/html/2607.27183v1'
summary=json.loads((OUT/'summary.json').read_text());results={m:json.loads((OUT/m/'results.json').read_text()) for m in MODELS};pred={}
for m in MODELS:
 blob=(OUT/m/'predictions.jsonl.gz').read_bytes();assert hashlib.sha256(blob).hexdigest()==summary['verification'][m]['predictions_sha256'];pred[m]=[json.loads(l) for l in gzip.decompress(blob).splitlines()]
rows=[]
def add(section,benchmark,metric,direction,values=None,reported=None,source=None,note='',digits=2):
 rows.append(dict(section=section,benchmark=benchmark,metric=metric,direction=direction,values=values or {m:None for m in MODELS},pangram4_reported_percent=reported,source=(SOURCE+'#'+source) if source else None,note=note,digits=digits))
def docs(m,ds):return results[m]['datasets'][ds]['overall']['document_native_label_metrics']
def local_metric(key,metric):return {m:summary['models'][m][key][metric]*100 if summary['models'][m][key][metric] is not None else None for m in MODELS}
def native(m,ds,cohort=None):
 rr=[r for r in pred[m] if r['dataset']==ds and r['native_label'] in ('human','ai') and (cohort is None or r.get('cohort')==cohort)]
 g=collections.defaultdict(list)
 for r in rr:g[r['text_sha256']].append(r)
 return [v[0] for v in g.values() if len({r['native_label'] for r in v})==1]
def fpr(m,ds,cohort):
 rr=[r for r in native(m,ds,cohort) if r['native_label']=='human'];return 100*sum(r['document_flag'] for r in rr)/len(rr) if rr else None
def roc(m,ds):
 return ranking([(int(r['native_label']=='ai'),r['mean_ai_probability']) for r in native(m,ds)])
for key,label,fields in [('v3_target_tokens','v3 reconstruction — tokens',['precision','recall','human_fpr']),('v3_target_sentences','v3 reconstruction — sentences',['precision','recall','human_fpr']),('novel_clean_human_isolated','Untouched human paragraphs — isolated',['human_fpr']),('novel_clean_human_context','Untouched human paragraphs — with neighbors',['human_fpr']),('arena_tokens','Arena, 50 generators — tokens',['recall']),('arena_sentences','Arena, 50 generators — sentences',['recall']),('opai_tokens','OpAI — tokens',['precision','recall','human_fpr']),('opai_sentences','OpAI — sentences',['precision','recall','human_fpr'])]:
 for metric in fields:add('Our common span suite',label,{'human_fpr':'Human FPR','precision':'Precision','recall':'Recall'}[metric],'min' if metric=='human_fpr' else 'max',local_metric(key,metric),note='Pangram did not report this metric on these examples.')
add('Public benchmark reference','PELIC','Document FPR','min',{m:docs(m,'pelic')['human_fpr']*100 for m in MODELS},.0067,'S5.T9','400 local texts vs 15,005 in report; different thresholds.',4)
add('Public benchmark reference','Liang TOEFL originals','Document FPR','min',{m:fpr(m,'liang','TOEFL_real_91') for m in MODELS},0,'S5.T9','91 local original rows (90 distinct texts) vs 89 in report; excludes polished copies.')
add('Public benchmark reference','VUB','Document recall','max',{m:docs(m,'vub')['recall']*100 for m in MODELS},100,'S5.T11','40 local papers vs 39 in report; local validation-calibrated binary rule differs from report score/verdict rule.')
for metric,field,pg in [('Document FNR','recall',2.86),('Document FPR','human_fpr',0)]:
 add('Public benchmark reference','Epoch style imitation',metric,'min',{m:(1-docs(m,'epoch')[field])*100 if field=='recall' else docs(m,'epoch')[field]*100 for m in MODELS},pg,'A3.T23','Same class counts (594 AI, 495 human); local binary rule vs report strict AI/Mixed verdict protocol.')
rocs={ds:{m:roc(m,ds) for m in MODELS} for ds in ['gede','meld_eval','sem_detect']}
for ds,label,pgauc,pgtpr,limit,source in [('gede','GEDE',100,100,'0.01','A3.T21'),('meld_eval','MELD-eval',99.99,99.99,'0.01','A3.T26'),('sem_detect','Sem-Detect',97.8,95.5,'0.001','A3.T27')]:
 n=rocs[ds]['ours']['human_n'];note=f'Local sampled cohort, {n} distinct human controls; different score/protocol from Pangram. GEDE uses publisher binary labels including improved-human positives.' if ds=='gede' else f'Local sampled cohort, {n} distinct human controls; different score/protocol from Pangram.'
 add('Public benchmark reference',label,'AUROC','max',{m:100*rocs[ds][m]['auroc'] for m in MODELS},pgauc,source,note)
 add('Public benchmark reference',label,f'TPR @ {float(limit)*100:g}% FPR','max',{m:100*rocs[ds][m]['tpr_at_fpr'][limit] for m in MODELS},pgtpr,source,note+' Empirical ROC only. With so few controls the requested FPR permits zero false positives; this is not evidence of a reliably calibrated low-FPR operating point.')
add('Public benchmark reference','DetectRL — local pooled sample','Pooled binary F1','max',{m:docs(m,'detectrl')['f1']*100 for m in MODELS},note='Not the report’s native average across task/test subsets; do not compare to 95.30 below.')
# All available document-label suites retain their local results, even when the report lacks a matching metric.
for ds,name in [('perkins','Perkins — local binary proxy'),('gede','GEDE — local binary labels'),('detectrl','DetectRL — local pooled sample'),('sem_detect','Sem-Detect — local binary subset'),('saha','Saha — local sample'),('local_binary','Local binary proxy'),('local_length','Local length proxy')]:
 for metric,field,direction in [('Document recall','recall','max'),('Document FPR','human_fpr','min')]:
  add('Other local diagnostics',name,metric,direction,{m:None if docs(m,ds)[field] is None else 100*docs(m,ds)[field] for m in MODELS},note='No report score with the same metric/cohort/verdict rule; Pangram cell intentionally missing.')
for ds,name in [('local_mixed','Local interleaving proxy'),('paper_pilots_exploratory','Development-exposed paper pilots')]:
 for field in ['precision','recall','human_fpr']:
  vals={m:100*results[m]['datasets'][ds]['overall']['span_or_historical_human_token_metrics'][field] for m in MODELS}
  add('Other local diagnostics',name,'Token '+field.replace('human_fpr','human FPR'),'min' if field=='human_fpr' else 'max',vals,note='Exploratory/proxy cohort, not Pangram’s held-out interleaving or editing benchmark.')
missing=[
 ('Overall private holdout','AUROC','max',99.16,'abstract'),('Overall private human holdout','FPR','min',.0041,'S5.SS3'),('Overall private AI holdout','FNR','min',.3396,'S5.T3'),
 ('ELLIPSE','FPR','min',0,'S5.T9'),('ICNALE','FPR','min',0,'S5.T9'),
 ('UChicago standard, full length','TPR @ 1% FPR','max',100,'A3.T18'),('UChicago standard, <50 words','TPR @ 1% FPR','max',99.7,'A3.T18'),('UChicago humanizer, full length','TPR @ 1% FPR','max',98.93,'A3.T18'),('UChicago humanizer, <50 words','TPR @ 1% FPR','max',73.32,'A3.T18'),('UChicago human controls','FPR','min',0,'A3.T19'),
 ('GEDE fully generated (report split)','TPR @ 1% FPR','max',100,'A3.T21'),('GEDE improved human (report split)','TPR @ 1% FPR','max',100,'A3.T21'),('GEDE humanized (report split)','TPR @ 1% FPR','max',100,'A3.T21'),
 ('Perkins baseline AI','Native three-method mean accuracy','max',100,'A3.T22'),('Perkins manipulated AI','Native three-method mean accuracy','max',94.1,'A3.T22'),
 ('DetectRL native average','Mean F1 across native subsets','max',95.30,'A3.T24'),
 ('Private light AI polish','AI-verdict FPR','min',.01,'S5.T4'),('Private editing prompts','Mixed-verdict recall','max',55.01,'S5.T5'),('Private WildChat edits','Mixed-verdict recall','max',65.17,'S5.T6'),
 ('Saha easy AI-BP','AI-or-Mixed TPR','max',98.2,'A3.T28'),('Saha easy AI-EP','AI-or-Mixed TPR','max',100,'A3.T28'),('Saha easy AI-HI','AI-or-Mixed TPR','max',96.3,'A3.T28'),('Saha easy H-AI','AI-only FPR','min',1.4,'A3.T28'),
 ('Saha hard AI-BP','AI-or-Mixed TPR','max',100,'A3.T28'),('Saha hard AI-EP','AI-or-Mixed TPR','max',100,'A3.T28'),('Saha hard AI-HI','AI-or-Mixed TPR','max',98.9,'A3.T28'),('Saha hard H-AI','AI-only FPR','min',2.5,'A3.T28'),('Saha human','AI-or-Mixed FPR','min',0,'A3.T28'),
 ('Commercial humanizers','AI recall','max',97.69,'S5.T14'),('Commercial humanizers','AI-or-Mixed recall','max',98.83,'S5.T14'),('Humanizer auxiliary head','Accuracy','max',96.82,'S5.T13'),('BLADER','FNR','min',.430,'S5.T15')]
for name,metric,direction,value,source in missing:add('Reported categories without matching local results',name,metric,direction,reported=value,source=source,note='Matching cohort, native metric, or required Mixed/auxiliary head unavailable. A proxy elsewhere is not substituted.',digits=4 if value in [.0041,.3396] else 2)
# Native descriptive fractions have no monotonic best direction.
for v,value in enumerate([0,.01,3.18,13.59,5.48,32.57,56.17,67.94,69.50]):
 add('Reported categories without matching local results',f'OpAI v{v}','Mean AI+Assisted fraction','none',reported=value,source='A3.T29',note='A predicted fraction is not an accuracy/recall score; neither largest nor smallest is best. Our binary reference-token fraction is a different quantity.')
# Table 8 is copied as factual scalar data, with explicit missing local coverage.
from bs4 import BeautifulSoup
html=BeautifulSoup((ROOT.parents[1]/'sources/report.html').read_text(),'html.parser')
table=html.find(id='S5.T8')
for tr in table.select('tbody tr'):
 cells=[td.get_text(' ',strip=True) for td in tr.find_all(['td','th'])]
 for i in range(0,len(cells)-2,3):
  language,fprv,fnrv=cells[i:i+3]
  if '%' not in fprv:continue
  for metric,value in [('FPR',fprv),('FNR',fnrv)]:
   add('Multilingual coverage',language,metric,'min',reported=float(value.rstrip('%')) if '%' in value else None,source='S5.T8',note='No corresponding multilingual local corpus evaluated.',digits=4)
# Keep the unavailable private interleaving cohort separate from our synthetic proxy.
resolution=None
for tr in html.find(id='S5.T7').select('tbody tr'):
 cells=[td.get_text(' ',strip=True) for td in tr.find_all(['td','th'])]
 if len(cells)!=6:continue
 if cells[0]:resolution=cells[0]
 if cells[1]!='Pangram 4':continue
 for metric,value,direction in zip(['Token accuracy','Token precision','Token recall','Document AI-fraction MAE'],cells[2:],['max','max','max','min']):
  add('Private interleaving coverage',f'Interleaving N={resolution}',metric,direction,reported=float(value.rstrip('%')),source='S5.T7',note='Different private corpus; our local interleaving proxy is not substituted.')
# Validate table anchors to avoid invented links.
for r in rows:
 if r['source']:
  anchor=r['source'].split('#')[1]
  if not html.find(id=anchor):r['source']=SOURCE
assert all(set(r['values'])==set(MODELS) for r in rows)
for section in ['Our common span suite','Public benchmark reference','Other local diagnostics','Reported categories without matching local results','Multilingual coverage']:
 assert any(r['section']==section for r in rows),section

def render(r):
 vals=r['values'];available=[round(v,r['digits']) for v in vals.values() if v is not None]
 best=(min(available) if r['direction']=='min' else max(available)) if available and r['direction']!='none' else None
 def fmt(v):return '-' if v is None else f"{v:.{r['digits']}f}%"
 cells=[]
 for m in MODELS:
  v=vals[m];t=fmt(v)
  if v is not None and best is not None and round(v,r['digits'])==best:t='**'+t+'**'
  cells.append(t)
 pg=fmt(r['pangram4_reported_percent'])
 if r['source'] and pg!='-':pg=f"[{pg}]({r['source']})"
 arrow=' ↓' if r['direction']=='min' else ' ↑' if r['direction']=='max' else ''
 return '| '+' | '.join([r['benchmark'],r['metric']+arrow,*cells,pg])+' |'
intro=['# Benchmark results — BF16','', '**Bold marks the best local score in each row; ties at displayed precision are all bold.** ↑ higher is better; ↓ lower is better. `-` means unavailable or not the same metric. Pangram 4 is a separately reported reference, excluded from local winner highlighting because its samples, operating points and verdict definitions differ. These are descriptive maxima/minima, not significance tests.','','Our three models used the identical frozen 13,751-example suite and BF16 inference. Token/sentence results share boundaries and validation-only thresholds; document diagnostics use the existing validation-calibrated mean-score rule. No inference was rerun for this table. [Full local protocol and confidence intervals](README.md). Pangram values come from the [Pangram 4 technical report, v1](https://arxiv.org/html/2607.27183v1), chiefly Tables 9, 11, 18–29.','','**Important:** ROC operating points below are reaggregated descriptive test-set ROC statistics, not changes to deployed thresholds. GEDE has only 8 human controls, MELD-eval 34, and Sem-Detect 161; their requested 1%/0.1% FPR points permit zero empirical false positives and have weak low-FPR resolution. Pangram used much larger cohorts. GEDE ROC uses the benchmark’s native binary labels, including improved-human positives, unlike the stricter primary provenance metrics.','','PELIC and TOEFL sample sizes differ from the report; VUB is 40 vs 39 papers. Epoch has matching class counts (594 AI, 495 human), but the document verdict rules differ. Public-benchmark overlap with MELD training is unknown. Do not interpret the reference column as a controlled four-model leaderboard.']
lines=intro.copy()
for section in dict.fromkeys(r['section'] for r in rows):
 lines+=['','## '+section,'','| Benchmark | Metric | Our ModernBERT | MELD v5 | MELD v8 | Pangram 4 (reported) |','|---|---|---:|---:|---:|---:|']
 lines.extend(render(r) for r in rows if r['section']==section)
lines+=['','The native DetectRL mean-F1 protocol, Perkins three-method mean accuracy, three-way Saha decisions and OpAI AI+Assisted fractions have no matching results in this run. Their cells remain `-`; locally available proxy metrics appear separately. Manual/agent red-team evaluations and backbone ablations are not substituted with scalar classifier scores.','', 'All scalar values, direction flags, provenance links and per-row caveats are retained in `benchmark_table.json`.']
(OUT/'BENCHMARK_TABLE.md').write_text('\n'.join(lines)+'\n')
(OUT/'benchmark_table.json').write_text(json.dumps({'source':SOURCE,'rows':rows,'roc_diagnostics':rocs,'bold_policy':'Best local score, including displayed ties; external Pangram reference unranked.'},indent=2))
print('Wrote',len(rows),'benchmark rows. No inference calls.')
for r in rows:
 if r['section']=='Public benchmark reference':print(render(r))
