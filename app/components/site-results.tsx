"use client";
/* Evaluation results of the current model (final-moe = abl-noseg-nomix, the Oct 10 loss-ablation winner), Oct 11 2026.
   Numbers: eval_final.py outputs in final-20261007/sweeps/abl-noseg-nomix/eval on the H200; pooled small-edit recall and
   per-cohort/per-writer counts from loss-ablation-20261010/pooled_heldout.py (standard and --suffix=-lc); 95% intervals
   from a 1,000-draw paper bootstrap (same definitions). Facts: research/site-architecture-20261011/RESULTS_FACTS.md. */
import {useEffect,useRef,useState} from 'react';
import './site-architecture.css';
import './site-data.css';
import './site-results.css';

const pct=(v:number,d=1)=>(100*v).toFixed(d)+'%';

/* Small-edit recall at 1% and 0.1% FPR (p1; scratchpad multi_fpr.py on the H200, same definitions as pooled_heldout.py), pooled over heldout-writers, t21-heldout-v2 and wide1-heldout. */
const HEAD={std:{r:.7462,lo:.732,hi:.759},p1:{r:.4773,lo:.448,hi:.504},n:7243};
const COHORTS:[string,string,number,number,number][]=[
 ['Insert one','A new sentence added',1630,.9276,.8110],
 ['Rewrite two','Two adjacent sentences reworded',3071,.7431,.4256],
 ['Rewrite one','One sentence reworded',1540,.6422,.3455],
 ['Split one','One sentence split into two',1002,.6208,.2954],
];
/* name, seen in training, n small-edit sentences, recall at 1% FPR, recall at 0.1% FPR */
const WRITERS:[string,boolean,number,number,number][]=[
 ['GPT-6 Sol',false,663,.8944,.6335],
 ['Claude Fable 5.1',false,1736,.8203,.5536],
 ['MiMo V2.6 Flash',false,391,.6880,.4373],
 ['GLM 5.3 Flash',false,385,.6416,.3610],
 ['GPT-6.1 Sol',true,152,.8816,.6316],
 ['Claude Haiku 4.5',true,453,.7329,.4790],
 ['Claude Opus 5.5',true,1154,.7270,.4324],
 ['DeepSeek V4.1 Flash',true,791,.7206,.4766],
 ['GPT-6 Luna',true,407,.6585,.3931],
 ['Claude Sonnet 5.5',true,1111,.6562,.3753],
];
/* set, what it holds out, n small-edit sentences, recall at 1%, recall at 0.1%, false positives beside edits at 1% */
const SETS:[string,string,number,number,number,number][]=[
 ['Held-out papers','new papers, writers seen in training',1590,.6503,.3799,.0068],
 ['Held-out papers and unseen writers','new papers; Fable 5.1 and GPT-6 Sol never trained',4086,.8040,.5301,.0133],
 ['Site text format and unseen writers','the site’s extraction; GLM and MiMo never trained',1567,.6930,.4384,.0092],
];
/* larger AI text and stress tests: name, detail, n AI sentences, recall at 1% and at 0.1% FPR on that set's controls, AUROC
   (0.1%: other_01.py in the H200 folder atlas-rescore/evals-01pct-20261012; its 1% values reproduce these). */
const OTHER:[string,string,string,number|null,number|null,number|null][]=[
 ['Paragraph drafts','held-out papers and unseen writers','1,421',.994,.975,1.0],
 ['Paragraph rewrites','held-out papers and unseen writers','1,495',.969,.870,.996],
 ['Paragraph rewrites','site text format','903',.694,.449,.970],
 ['Section drafts','held-out papers and unseen writers','6,574',.965,.848,.994],
 ['Section rewrites','held-out papers and unseen writers','6,898',.941,.717,.997],
 ['Humanized paragraphs','Claude Fable 5.1 rewrote AI paragraphs to sound human','5,375',.963,.782,.998],
 ['Co-writing','three rounds mixing human, AI and AI-edited text','339',.263,.118,.729],
 ['Other kinds of writing','fiction, reports, Wikipedia, older ML papers','8,439',.985,.866,.998],
];

/* Runs compared: how the current model was chosen. Backbones: research/evaluation/backbone-results-20261003/metrics.json
   (Oct 3) and research/evaluation/final-runs-20261007/eval/report.md (Oct 7, selected checkpoints). Later tables copied
   from the loss-ablation page (artifact DKeu7bAWT2MgVPo7V3jnf3, Results tab, Oct 10-11): pooled small-edit recall at 1% FPR
   over the three held-out sets, paired paper bootstrap vs the reference row. */
type Cmp={title:string;lc?:boolean;note?:string;cols:string[];higher:boolean[];rows:(string|number|null)[][]};
const CMPS:Cmp[]=[
 {title:'Six backbones (Oct 3)',note:'The same classifier on six language models, each trained on 10% of the training data at the time (ModernBERT-large at full length), scored on the evaluation suite of the time. AUROC needs no cutoff. Paper-edit spans: F1 of the AI spans found in rebuilt paper-editing workflows. At this stage the training data had almost no short AI spans, so no backbone caught small edits yet.',
  cols:['Backbone','Type','Public text AUROC','Papers AUROC','Paper-edit spans F1','Public AI caught at 1% FPR'],higher:[true,true,true,true,true,true],
  rows:[['ModernBERT-large','encoder',0.732,0.953,0.737,26.2],['Ettin 1B','encoder',0.742,0.964,0.691,22.7],['Qwen3.5 4B','decoder',0.922,0.896,0.257,68.7],
   ['Qwen3.5 9B','decoder',0.902,0.962,0.743,60.0],['Gemma 4 12B','decoder',0.846,0.942,0.700,50.0],['★ Qwen3.6 35B-A3B','mixture of experts',0.900,0.970,0.750,57.5]]},
 {title:'Three finalists at full length (Oct 7)',note:'The Oct 7 recipe on the same 72,000 windows, one run each. Recall at one cutoff per model, set to flag 1% of human sentences in the screening set (on these sets it flags 1–2.7% of their own human sentences). Selected checkpoints.',
  cols:['Backbone','Unseen papers, small edits','Six writers, small edits','Site text format, small edits','Other models’ text','Humanized','Co-writing'],higher:[true,true,true,true,true,true,true],
  rows:[['★ Qwen3.6 35B-A3B',69.9,79.3,75.1,93.1,87.9,26.8],['Qwen3.5 4B',61.1,72.0,66.4,53.4,82.7,22.4],['EmbeddingGemma 2',32.9,30.5,37.9,28.0,73.0,8.3]]},
 {title:'Loss-term ablations (35B MoE, Oct 10)',note:'Δ vs the control (the Oct 7 recipe re-run on the fast trainer). The winner is “No segment/mixed”, now final-moe.',
  cols:['Run','Small-edit recall','Δ vs control','Spill-over FPR','Rewrite one','Split one','Paragraph edits','Fable co-write'],higher:[true,true,true,false,true,true,true,true],
  rows:[['Control',71.9,'—',1.24,59.6,59.3,92.7,26.0],['No stage 1',73.2,'+1.3 [+0.3, +2.5]',1.18,61.8,59.5,92.6,24.8],['★ No segment/mixed (final-moe)',74.6,'+2.7 [+1.7, +3.8]',1.09,64.2,62.1,94.7,26.3],
   ['Token + sentence',73.6,'+1.7 [+0.7, +2.8]',1.15,61.2,60.8,92.8,27.7],['Token only',73.5,'+1.5 [+0.5, +2.8]',1.20,61.8,60.7,92.6,30.1],['Uniform token mean',72.8,'+0.9 [+0.0, +1.8]',1.31,60.3,60.6,93.3,23.9],
   ['Sentence ×1.0',72.4,'+0.5 [-0.5, +1.5]',1.14,60.1,61.1,91.2,25.7],['● final-moe-oct7 (reference)',72.3,'+0.3 [-0.9, +1.5]',1.14,61.0,59.2,92.0,25.1],['mixA-full (reference)',72.6,'+0.6 [-0.5, +1.9]',1.17,61.8,59.4,93.4,25.7],['lc-1020 *',74.9,'+2.9 [+1.8, +4.0]',1.06,63.4,63.3,92.7,22.7]]},
 {title:'Standard held-out sets: 510 vs long-context training',cols:['Run','Small-edit recall','Δ vs final-moe','Spill-over FPR','heldout-writers','t21-heldout-v2','wide1-heldout','Insert one','Rewrite one','Rewrite two','Split one'],
  higher:[true,true,true,false,true,true,true,true,true,true,true],
  rows:[['★ final-moe (510)',74.6,'—',1.09,65.0,80.4,69.3,92.8,64.2,74.3,62.1],['lc-1020',74.9,'+0.2 [-0.9, +1.4]',1.06,66.5,80.0,69.9,93.9,63.4,74.3,63.3]]},
 {title:'Follow-up runs on the winner (Oct 10 evening and night)',cols:['Run','Small-edit recall','Δ vs winner','Spill-over FPR','Δ spill-over','Split one'],higher:[true,true,true,false,false,true],
  rows:[['★ Winner (final-moe)',74.6,'—',1.09,'—',62.1],['Document loss on GRADTEX rows only',74.2,'−0.5 [−1.6, +0.4]',1.34,'+0.25 [+0.15, +0.33]',62.1],['Constant LR 2e-4, no warmup',72.7,'−2.0 [−3.2, −0.6]',0.91,'−0.18 [−0.26, −0.07]',57.4],
   ['Sentence weight 0.5',74.6,'−0.0 [−0.9, +1.0]',1.25,'+0.16 [+0.08, +0.26]',62.1],['Sentence weight 0',73.1,'−1.5 [−2.4, −0.4]',1.22,'+0.13 [+0.03, +0.21]',59.4],
   ['Document weight 0.3',73.6,'−1.0 [−1.9, +0.1]',1.15,'+0.06 [−0.01, +0.15]',60.8],['Uniform token mean (no ½ / ½ split)',75.3,'+0.6 [−0.4, +1.5]',1.16,'+0.07 [−0.02, +0.16]',62.8],['lc-1020 *',74.9,'+0.2 [−0.9, +1.4]',1.06,'−0.03 [−0.11, +0.08]',63.3]]},
 {title:'Other evaluation sets',note:'Recall at 1% FPR on each set’s own controls, except where noted.',
  cols:['Run','Humanized paragraphs','Co-writing','Co-writing AUROC','Light polish (document AUROC)','Other kinds of writing','Screening set small edits, dev','Screening set small edits, test'],
  higher:[true,true,true,true,true,true,true,true],
  rows:[['★ final-moe (510)',97.0,26.3,0.729,0.814,98.5,45.5,60.0],['lc-1020 *',98.7,22.7,0.786,0.810,95.7,37.6,55.3]]},
];
function CmpTable({c}:{c:Cmp}){
 const nums=(j:number)=>c.rows.map(r=>r[j]).filter((v):v is number=>typeof v==='number');
 const cls=(j:number,v:unknown)=>{if(typeof v!=='number'||c.rows.length<2)return '';const n=nums(j);const best=c.higher[j]?Math.max(...n):Math.min(...n);const worst=c.higher[j]?Math.min(...n):Math.max(...n);return v===best?'rs-best':v===worst?'rs-worst':'';};
 return <div className={`ar-demo${c.lc?' rs-lc':''}`}><h3>{c.lc&&<span className="rs-lcmark" aria-label="inside the full paper">◆ </span>}{c.title}</h3>{c.note&&<p className="sp-fine">{c.note}</p>}
  <div className="sp-scroll"><table className="sp-table rs-cmp"><thead><tr>{c.cols.map((h,j)=><th key={h}>{c.lc&&j>0&&<span className="rs-lcmark">◆ </span>}{h}</th>)}</tr></thead>
   <tbody>{c.rows.map(r=><tr key={String(r[0])}>{r.map((v,j)=>j===0?<td key={j}><b>{v}</b></td>:<td key={j} className={cls(j,v)}>{typeof v==='number'?v.toFixed(c.cols[j].startsWith('Spill')?2:c.cols[j].includes('AUROC')||c.cols[j].includes('F1')?3:1):v}</td>)}</tr>)}</tbody></table></div></div>;
}
function RunsCompared(){
 return <>
  <p className="sp-lede">How the current model was chosen: backbones first, then the training recipe. Every table compares runs on the same evaluation sets.</p>
  <div className="ar-draft"><b>How to read these tables</b><span>Percentages unless the column says AUROC or F1. From the Oct 10 tables on, small-edit recall is pooled over the three held-out sets at 1% FPR. Green = best in its column, red = worst (lower is better for spill-over). ★ marks the run that was kept; from Oct 10 on that is final-moe, the model that scores every paper on this site. ● is final-moe-oct7, the Oct 7 model it replaced.</span></div>
  {CMPS.map(c=><CmpTable key={c.title} c={c}/>)}
  <p className="sp-fine"><b>*</b> lc-1020 was trained on 1,020-token windows and is scored with its own setup: 1,020-token windows moved 510 tokens at a time, not the 510-token windows moved 256 at a time used for every other run. Its numbers therefore compare the two models each in its own setup, not under identical windows.</p>
 </>;
}

function Bar({v,cls}:{v:number;cls:string}){return <span className="rs-bar"><i className={cls} style={{width:pct(v)}}/></span>;}

export function ResultsPage(){
 const [tab,setTab]=useState<'main'|'runs'>(()=>new URLSearchParams(location.search).get('tab')==='runs'?'runs':'main');
 // The tab is kept in the address so a copied section link (#id) opens the right tab.
 const firstTab=useRef(true);
 useEffect(()=>{const p=new URLSearchParams(location.search);if(tab==='runs')p.set('tab','runs');else p.delete('tab');history.replaceState(null,'',`/?${p}${firstTab.current?location.hash:''}`);firstTab.current=false;},[tab]);
 return <div className="sp-col sp-col-wide ar">
  <section className="sp-hero">
   <h1>Evaluation</h1>
   <p>How well the current model catches AI text in held-out evaluation sets. What each set contains, and why, is on the Data page under Evaluation sets.</p>
   <p className="sp-fine">Model: Qwen3.6 35B-A3B MoE · Oct 10, the one that scores every paper on this site (see Architecture).</p>
  </section>
  <div role="tablist" className="sp-chips" aria-label="Evaluation views">
   <button type="button" role="tab" aria-selected={tab==='main'} aria-pressed={tab==='main'} onClick={()=>setTab('main')}>Current model</button>
   <button type="button" role="tab" aria-selected={tab==='runs'} aria-pressed={tab==='runs'} onClick={()=>setTab('runs')}>Runs compared</button>
  </div>
  {tab==='runs'?<RunsCompared/>:<>
  <div className="ar-demo">
   <div className="ar-demo-head"><h3>Larger AI text and harder cases</h3></div>
   <table className="sp-table rs-table">
    <thead><tr><th>Case</th><th>AI sentences</th><th>Caught at 1% FPR</th><th>Caught at 0.1% FPR</th><th>AUROC</th></tr></thead>
    <tbody>{OTHER.map(o=><tr key={o[0]+o[1]}><td><b>{o[0]}</b><div className="sp-fine">{o[1]}</div></td><td>{o[2]}</td><td>{o[3]===null?'–':pct(o[3])}</td><td>{o[4]===null?'–':pct(o[4])}</td><td>{o[5]===null?'–':o[5].toFixed(3)}</td></tr>)}</tbody>
   </table>
   <p className="sp-fine">Whole AI paragraphs and sections are caught almost always. Co-writing, where human and AI text are mixed sentence by sentence over three rounds, is the weakest case measured. Light polishing of a whole paragraph (about 9% of words changed) is scored only per document, since few sentences are mostly AI: document AUROC 0.814. AUROC needs no cutoff: it is the chance that a random AI sentence scores above a random human one. The 0.1% cutoff rests on few human sentences in the smaller sets (the top 2 of 1,824 for co-writing, the top 12 of 12,841 for humanized paragraphs), so those numbers are rough.</p>
  </div>

  <div className="rs-tiles">
   <div className="rs-tile">
    <b>{pct(HEAD.std.r)}</b>
    <span>of small AI edits caught while flagging 1% of human sentences</span>
    <small>95% interval {pct(HEAD.std.lo)}–{pct(HEAD.std.hi)} · {HEAD.n.toLocaleString()} AI sentences in held-out papers, three sets pooled</small>
   </div>
   <div className="rs-tile">
    <b>{pct(HEAD.p1.r)}</b>
    <span>of the same edits caught while flagging 0.1% of human sentences, the site’s default</span>
    <small>95% interval {pct(HEAD.p1.lo)}–{pct(HEAD.p1.hi)} · same sentences, stricter cutoff</small>
   </div>
  </div>
  <p className="sp-lede">A small edit is one or two sentences an AI rewrote or added inside a human paragraph. Each passage is scored on its own, as in training. ROC curves across all false-positive rates, measured against whole human papers in the site’s own setup, are on the <a href="/?view=calibration">Calibration page</a>.</p>

  <div className="ar-demo">
   <div className="ar-demo-head"><h3>By kind of edit</h3>
   </div>
   <table className="sp-table rs-table">
    <thead><tr><th>Edit</th><th>AI sentences</th><th>At 1% FPR</th><th>At 0.1% FPR</th><th/></tr></thead>
    <tbody>{COHORTS.map(c=><tr key={c[0]}><td><b>{c[0]}</b><div className="sp-fine">{c[1]}</div></td><td>{c[2].toLocaleString()}</td><td>{pct(c[3])}</td><td>{pct(c[4])}</td><td className="rs-barcell"><Bar v={c[3]} cls="std"/></td></tr>)}</tbody>
   </table>
   <p className="sp-fine">Added sentences are easiest; rewording an existing sentence leaves the most human structure behind and is hardest.</p>
  </div>

  <div className="ar-demo">
   <div className="ar-demo-head"><h3>By writing model</h3></div>
   <table className="sp-table rs-table">
    <thead><tr><th>Writer</th><th>In training</th><th>AI sentences</th><th>At 1% FPR</th><th>At 0.1% FPR</th></tr></thead>
    <tbody>{WRITERS.map(w=><tr key={w[0]}><td><b>{w[0]}</b></td><td>{w[1]?'Yes':'No'}</td><td>{w[2].toLocaleString()}</td>
     <td><span className="rs-num">{pct(w[3])}</span><Bar v={w[3]} cls="std"/></td><td><span className="rs-num">{pct(w[4])}</span><Bar v={w[4]} cls="std"/></td></tr>)}</tbody>
   </table>
   <p className="sp-fine">Writers never seen in training are caught about as often as those that were. Some “Claude Fable 5.1” edits were probably written by Claude Opus 5 (see the Data page’s limitations). Writer counts are small for GPT-6.1 Sol (152).</p>
  </div>

  <div className="ar-demo">
   <div className="ar-demo-head"><h3>By set</h3></div>
   <table className="sp-table rs-table">
    <thead><tr><th>Set</th><th>Holds out</th><th>AI sentences</th><th>At 1% FPR</th><th>At 0.1% FPR</th><th>False positives next to an edit (1% cutoff)</th></tr></thead>
    <tbody>{SETS.map(s=><tr key={s[0]}><td><b>{s[0]}</b></td><td>{s[1]}</td><td>{s[2].toLocaleString()}</td><td>{pct(s[3])}</td><td>{pct(s[4])}</td><td>{pct(s[5],2)}</td></tr>)}</tbody>
   </table>
   <p className="sp-fine">Each set’s cutoff is set on its own human control passages, so that 1% of their sentences are flagged. The last column is the share of human sentences flagged inside the edited passages themselves: human text next to an AI edit is flagged about as often as other human text.</p>
  </div>

  <div className="ar-limits">
   <h3>Not measured</h3>
   <ul>
    <li><b>Whole AI-written papers.</b> No evaluation set scores complete AI-generated papers against human papers the way the site does; held-out AI papers exist but are not yet scored.</li>
    <li><b>Edits inside whole papers.</b> Every set above scores the edited passage on its own. The site scores whole papers, with headings, captions, references and math around each passage, which is not measured here. An experiment with passages inside up to 1,000 tokens of their paper was built for a longer-context model that is not released, so its numbers are left out.</li>
    <li><b>The papers on this site.</b> The ICLR 2027 submissions have no known labels, so the site’s flag rates are not accuracy measurements.</li>
    <li><b>The site’s exact setup.</b> These results use overlapping windows (every 256 tokens) and per-set cutoffs; the site uses non-overlapping windows and cutoffs set on separate human papers (see the Calibration page).</li>
   </ul>
  </div>

  <div className="ar-history">
   <h3>How these numbers were computed</h3>
   <ul>
    <li><b>Cutoff.</b> For each set, the score above which 1% of the human control sentences fall. Recall is the share of AI sentences above it.</li>
    <li><b>Pooling.</b> The headline counts all small-edit sentences of the three held-out paper sets together, each judged at its own set’s cutoff.</li>
    <li><b>0.1% cutoffs.</b> The sets hold 16,591, 76,504 and 16,974 human control sentences, so each 0.1% cutoff rests on the top 16 to 76 of them; the intervals are wider than at 1%.</li>
    <li><b>Intervals.</b> 1,000 bootstrap draws that resample whole papers within each set and recompute the cutoffs.</li>
    <li><b>Scoring.</b> Passages are cut into 510-token windows every 256 tokens; each token’s probability is averaged over the windows that cover it, and a sentence’s score is the mean over its tokens.</li>
   </ul>
  </div>
  </>}
 </div>;
}
