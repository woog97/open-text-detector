"use client";
import {useRef,useState} from "react";
import {useHeadingAnchors} from "./heading-anchors";
import {SiteHeader,SiteFooter,useSavedCount} from "./site-chrome";
import calibrationAll from "@/lib/model-calibration/qwen36-35b-a3b-oct10-all.json";
import calibrationNobib from "@/lib/model-calibration/qwen36-35b-a3b-oct10-nobib.json";
import calibrationNoapp from "@/lib/model-calibration/qwen36-35b-a3b-oct10-noapp.json";
import calibrationBody from "@/lib/model-calibration/qwen36-35b-a3b-oct10-body.json";
import {FLAG_FPRS,FLAG_MODES,DEFAULT_FLAG_MODE,asSiteFlagMode,flagModeOf,fprLabel,thresholdText,type BaseFlagMode,type FlagMode,type FlagFpr} from "@/lib/paper-discovery";
import "./model-calibration.css";

type Curve={name:string;auc:number;n_ai:number;n_human:number;tpr_at_1pct:number;fpr:number[];tpr:number[];other_model?:boolean};
type Threshold={target_fpr:number;threshold:number;threshold_ci95?:number[];heldout_fpr:number;heldout_ci95:number[];heldout_sentences:number;heldout_papers?:number;stable:boolean};
type Candidate={id:string;name:string;detail:string;papers:number;threshold_1pct:number;threshold_ci95:number[];heldout_fpr:number;heldout_ci95:number[];document_threshold:number;document_threshold_ci95:number[]};
type YearRow={label:string;kind:string;year:number;papers:number;flagged_share_median:number;flagged_share_p25:number;flagged_share_p75:number;over_human_p99:number|null;document_score_median:number|null};
/** Calibration page data; older exports (the 215-paper 4B file) lack the optional fields. */
type Calibration={model:{id:string;name:string;detail:string};calibration_set:{papers:number;sentences:number;description:string;method:string};sentence_score:string;
 thresholds:Threshold[];sentence_curves:Curve[];document_curves:Curve[];document_threshold:{threshold:number;threshold_ci95?:number[];human_fpr_ci95:number[];heldout_fpr?:number;note:string};
 years:YearRow[];years_p1?:YearRow[];years_p01?:YearRow[];candidates?:Candidate[];caveats:string[];years_note?:string;curves_note?:string;score_map?:{fpr:number[];score:number[];n_sentences:number;max_score:number}};
/** The live model's calibration (Oct 10 MoE, the only model on the site since Oct 12 2026), per counting mode
 * (bibliography / appendices in or out). The Oct 7 and 4B files stay in lib/model-calibration for the record. */
const BY_MODE={all:calibrationAll,nobib:calibrationNobib,noapp:calibrationNoapp,body:calibrationBody} as unknown as Record<BaseFlagMode,Calibration>;
const SERIES=["var(--cal-s1)","var(--cal-s2)","var(--cal-s3)","var(--cal-s4)","var(--cal-s5)"];
/** ROC curves are ordered by how much text is AI-written (one sentence, two, a paragraph, a section), so they take one
 * hue light to dark (Alice, Oct 13 2026): steps 250/400/550/700 of the blue ramp; the lightest still clears 2:1 on white. */
const ROC_RAMP=["var(--cal-r1)","var(--cal-r2)","var(--cal-r3)","var(--cal-r4)"];
const pct=(v:number,d=1)=>`${(v*100).toFixed(v>0&&v<0.001?3:v<0.01?2:d)}%`;
const opt=(v:number|null,f:(x:number)=>string)=>v==null?"—":f(v);
const at=(c:Curve,f:number)=>{let j=0;while(j+1<c.fpr.length&&c.fpr[j+1]<=f)j++;return c.tpr[j];};

/** ROC curves on a log false-positive axis, with markers at the published thresholds. */
const base=(name:string)=>name.replace(/, new final-moe$/,"");
function RocChart({curves,label,marks,xmin=1e-4,scoreAt}:{curves:Curve[];label:string;marks:number[];xmin?:number;scoreAt?:(f:number)=>string|null}){
 const sets=[...new Set(curves.map(c=>base(c.name)))], colour=(c:Curve)=>ROC_RAMP[Math.min(sets.indexOf(base(c.name)),ROC_RAMP.length-1)];
 const W=760,H=380,m={l:56,r:16,t:12,b:44},dec=-Math.log10(xmin);
 const x=(f:number)=>m.l+(Math.log10(Math.max(f,xmin))+dec)/dec*(W-m.l-m.r), y=(t:number)=>m.t+(1-t)*(H-m.t-m.b);
 const svg=useRef<SVGSVGElement>(null);
 const [hover,setHover]=useState<{px:number;f:number}|null>(null);
 const move=(e:React.PointerEvent)=>{const r=svg.current!.getBoundingClientRect(),px=(e.clientX-r.left)*W/r.width;
  if(px<m.l||px>W-m.r){setHover(null);return;}setHover({px,f:Math.pow(10,(px-m.l)/(W-m.l-m.r)*dec-dec)});};
 return <figure className="cal-figure">
  <div className="cal-legend">{curves.map((c,i)=><span key={c.name}><i className={c.other_model?"dash":undefined} style={{background:colour(c),color:colour(c)}}/>{c.name} <span className="site-num">AUROC {c.auc.toFixed(3)}</span></span>)}</div>
  <div className="cal-plot">
   <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onPointerMove={move} onPointerLeave={()=>setHover(null)}>
    {Array.from({length:dec+1},(_,i)=>10**(i-dec)).map(f=><g key={f}><line x1={x(f)} x2={x(f)} y1={m.t} y2={H-m.b} className="cal-grid"/><text x={x(f)} y={H-m.b+18} textAnchor="middle">{f>=.01?pct(f,0):`${+(f*100).toPrecision(1)}%`}</text></g>)}
    {[0,.25,.5,.75,1].map(t=><g key={t}><line x1={m.l} x2={W-m.r} y1={y(t)} y2={y(t)} className="cal-grid"/><text x={m.l-8} y={y(t)+4} textAnchor="end">{t*100}%</text></g>)}
    <line x1={m.l} x2={W-m.r} y1={y(0)} y2={y(0)} className="cal-axis"/>
    {marks.map(f=><g key={f}><line x1={x(f)} x2={x(f)} y1={m.t} y2={H-m.b} className="cal-mark"/><text x={x(f)+5} y={m.t+12} className="cal-mark-label">{pct(f)} FPR</text></g>)}
    <text x={(W+m.l)/2} y={H-6} textAnchor="middle">False-positive rate on human papers (log scale)</text>
    <text x={14} y={(H-m.b+m.t)/2} textAnchor="middle" transform={`rotate(-90 14 ${(H-m.b+m.t)/2})`}>AI text caught</text>
    {curves.map((c,i)=><path key={c.name} fill="none" stroke={colour(c)} strokeWidth={2} strokeLinejoin="round" strokeDasharray={c.other_model?"6 4":undefined}
      d={c.fpr.map((f,j)=>[f===0?xmin:f,c.tpr[j]] as const).filter(([f])=>f>=xmin).map(([f,t],j)=>`${j?"L":"M"}${x(f).toFixed(1)} ${y(t).toFixed(1)}`).join(" ")}/>)}
    {curves.map((c,i)=><circle key={c.name} cx={x(.01)} cy={y(c.tpr_at_1pct)} r={4.5} fill={colour(c)} className="cal-dot"/>)}
    {hover&&<g><line x1={hover.px} x2={hover.px} y1={m.t} y2={H-m.b} className="cal-axis"/>
     {curves.map((c,i)=><circle key={c.name} cx={x(Math.max(c.fpr[c.fpr.findLastIndex(f=>f<=hover.f)]??xmin,xmin))} cy={y(at(c,hover.f))} r={4} fill={colour(c)} className="cal-dot"/>)}</g>}
    <rect x={m.l} y={m.t} width={W-m.l-m.r} height={H-m.t-m.b} fill="transparent"/>
   </svg>
   {hover&&<div className="cal-tip" style={{left:`${Math.min(hover.px/W*100,70)}%`}}>
    <b className="site-num">FPR ≤ {hover.f<1e-5?`${+(hover.f*100).toPrecision(2)}%`:pct(hover.f,2)}</b>{scoreAt&&scoreAt(hover.f)!=null&&<span>Sentence score cutoff: <span className="site-num">{scoreAt(hover.f)}</span></span>}
    {curves.map((c,i)=><span key={c.name}><i style={{background:colour(c)}}/>{c.name}: <span className="site-num">{pct(at(c,hover.f))}</span></span>)}
   </div>}
  </div>
  <div className="cal-table-wrap"><table className="cal-table">
   <thead><tr><th>Set</th><th>AI items</th><th>Human items</th><th>AUROC</th></tr></thead>
   <tbody>{curves.map(c=><tr key={c.name}><td>{c.name}</td><td>{c.n_ai.toLocaleString()}</td><td>{c.n_human.toLocaleString()}</td><td>{c.auc.toFixed(4)}</td></tr>)}</tbody>
  </table></div>
 </figure>;
}

/** Median share of sentences flagged per paper (bar) with the interquartile range (whisker). */
function YearChart({rows,fpr}:{rows:Calibration["years"];fpr:number}){
 const W=760,rowH=30,m={l:250,r:60,t:8,b:36},H=m.t+m.b+rows.length*rowH;
 const x=(v:number)=>m.l+v*(W-m.l-m.r);
 return <figure className="cal-figure">
  <div className="cal-plot"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Share of sentences flagged per paper, by year">
   {[0,.25,.5,.75,1].map(v=><g key={v}><line x1={x(v)} x2={x(v)} y1={m.t} y2={H-m.b} className="cal-grid"/><text x={x(v)} y={H-m.b+18} textAnchor="middle">{v*100}%</text></g>)}
   <text x={(W+m.l)/2} y={H-4} textAnchor="middle">Share of the paper’s sentences flagged at {fprLabel(fpr)} FPR (median, with middle half of papers)</text>
   {rows.map((r,i)=>{const cy=m.t+i*rowH+rowH/2,cls=r.kind==="human"?"cal-bar human":r.kind==="iclr"?"cal-bar iclr":"cal-bar";
    return <g key={r.label}>
     <text x={m.l-10} y={cy+4} textAnchor="end" className="cal-row-label">{r.label}</text>
     <rect x={x(0)} y={cy-7} width={Math.max(x(r.flagged_share_median)-x(0),2)} height={14} rx={3} className={cls}/>
     <line x1={x(r.flagged_share_p25)} x2={x(r.flagged_share_p75)} y1={cy} y2={cy} className="cal-whisker"/>
     <text x={x(Math.max(r.flagged_share_p75,r.flagged_share_median))+6} y={cy+4} className="cal-value">{pct(r.flagged_share_median)}</text>
    </g>;})}
  </svg></div>
  <div className="cal-table-wrap"><table className="cal-table">
   <thead><tr><th>Group</th><th>Papers</th><th>Median flagged</th><th>Middle half</th><th>Papers above the human 99th percentile</th></tr></thead>
   <tbody>{rows.map(r=><tr key={r.label}><td>{r.label}</td><td>{r.papers.toLocaleString()}</td><td>{pct(r.flagged_share_median)}</td><td>{pct(r.flagged_share_p25)}–{pct(r.flagged_share_p75)}</td><td>{r.kind==="human"?"reference":opt(r.over_human_p99,pct)}</td></tr>)}</tbody>
  </table></div>
  <p className="cal-note">“Papers above the human 99th percentile” is the share of a group’s papers whose flagged share is higher than that of 99% of the human calibration papers; the human groups are that reference.{rows.some(r=>r.papers<50)&&<> Groups of fewer than 50 papers ({rows.filter(r=>r.papers<50).map(r=>`${r.label} (${r.papers})`).join("; ")}) are too small to read much into.</>}</p>
 </figure>;
}

export default function ModelCalibration(){
 const savedCount=useSavedCount();
 const main=useRef<HTMLElement>(null);useHeadingAnchors(main);
 // Same default and URL parameter as Discover, so a link carries the counting mode between the two pages.
 const [mode,setModeState]=useState<FlagMode>(()=>typeof location==='undefined'?DEFAULT_FLAG_MODE:asSiteFlagMode(new URLSearchParams(location.search).get('mode')));
 function setMode(m:FlagMode){setModeState(m);const q=new URLSearchParams(location.search);if(m===DEFAULT_FLAG_MODE)q.delete('mode');else q.set('mode',m);history.replaceState(null,'',`?${q}${location.hash}`);}
 const c=BY_MODE[FLAG_MODES[mode].base],curve=(name:string)=>(c.sentence_curves as Curve[]).find(k=>k.name===name)!,one=curve("One-sentence edits"),two=curve("Two-sentence edits"),para=curve("Paragraph edits"),sec=curve("AI sections"),m=FLAG_MODES[mode];
 const fpr=m.fpr,other:FlagFpr=fpr===0.01?0.001:0.01,th=(f:number)=>c.thresholds.find(t=>t.target_fpr===f);
 const years=fpr===0.0001?c.years_p01:fpr===0.001?c.years_p1:c.years;
 // Below one human sentence (1/n) no cutoff has a measurable rate: any score above the highest human sentence gives zero.
 const scoreAt=c.score_map?(f:number)=>{const g=c.score_map!;if(f<g.fpr[0])return `above ${thresholdText(g.max_score)}, the highest human sentence`;let j=0;while(j+1<g.fpr.length&&g.fpr[j+1]<=f)j++;return thresholdText(g.score[j]);}:undefined;
 return <div className="site cal">
  <SiteHeader page="calibration" savedCount={savedCount}/>
  <main className="cal-main" ref={main}>
   <section className="cal-hero">
    <h1>Calibration</h1>
    <p>How the site decides which sentences to flag: the cutoffs set on human-written papers, how often they flag human text, and how much AI text they catch at each rate.</p>
   </section>
   <header className="cal-head">
    <div className="cal-model cal-modes" role="group" aria-label="Sentences counted">Count
     <label><input type="checkbox" checked={m.app} onChange={e=>setMode(flagModeOf(m.bib,e.target.checked,m.fpr))}/> Appendices</label>
    </div>
    <div className="cal-model cal-modes" role="radiogroup" aria-label="False-positive rate">Rate
     {FLAG_FPRS.map(f=><label key={f}><input type="radio" name="cal-fpr" checked={fpr===f} onChange={()=>setMode(flagModeOf(m.bib,m.app,f))}/> {fprLabel(f)}{f===0.001?" (site default)":""}</label>)}
    </div>
    <h2>{c.model.name}</h2>
    <p>{c.model.detail} Sentence scores are the model’s raw probabilities. A sentence is flagged when its score is above the threshold that flags a set share of sentences in human-written papers: 0.1% by default on this site, or 1% for a more sensitive reading.{<> Counting: {(c as Calibration&{counting?:{label:string}}).counting?.label.toLowerCase()}; thresholds, false-positive checks, sentence curves and paper shares below all use the same sentences.</>}</p>
   </header>

   <section className="cal-facts" aria-label="Key figures">
    <div><b className="site-num">{opt(th(fpr)?.threshold??null,thresholdText)}</b><span>Sentence threshold at {fprLabel(fpr)} FPR{fpr===0.001?", the site’s default":fpr===0.0001?", not reliably calibrated (see below)":""}</span></div>
    <div><b className="site-num">{th(fpr)?pct(th(fpr)!.heldout_fpr,fpr<0.01?3:2):"—"}</b><span>False-positive rate on held-out human papers at that threshold</span></div>
    <div><b className="site-num">{opt(th(other)?.threshold??null,thresholdText)}</b><span>Sentence threshold at {fprLabel(other)} FPR</span></div>
    <div><b className="site-num">{c.calibration_set.papers.toLocaleString()}</b><span>Human papers behind the thresholds</span></div>
   </section>

   <section className="cal-section">
    <h2>Choosing a cutoff</h2>
    <p>A cutoff is a sentence score: sentences at or above it are flagged. Every cutoff trades the share of human sentences wrongly flagged (the false-positive rate) against the share of AI sentences caught. The table gives that trade at the site’s three rates (catch rates here are measured against whole human papers and run high; the Evaluation page has harder comparisons); the chart below shows it at every rate, and hovering over it gives the score to use. The site flags at 0.1% by default and also offers 0.01% and 1%; in the reader, “Custom colour thresholds” lets you set any score and shows the rate it corresponds to.</p>
    <div className="cal-table-wrap"><table className="cal-table cal-ops">
     <thead><tr><th>Target rate</th><th>Sentence score</th><th>Rate on held-out human papers</th><th>One-sentence edits</th><th>Two-sentence edits</th><th>Paragraph edits</th><th>AI sections</th></tr></thead>
     <tbody>{c.thresholds.filter(t=>t.target_fpr<0.05).map(t=><tr key={t.target_fpr} className={t.target_fpr===0.001?"is-default":undefined}>
      <td>{pct(t.target_fpr)}{t.target_fpr===0.001?<small>site default</small>:<small>site option</small>}</td>
      <td>{thresholdText(t.threshold)}{t.threshold_ci95&&<small>{thresholdText(t.threshold_ci95[0])}–{thresholdText(t.threshold_ci95[1])}</small>}</td>
      <td>{pct(t.heldout_fpr,3)}<small>{pct(t.heldout_ci95[0],3)}–{pct(t.heldout_ci95[1],3)}</small></td>
      {[one,two,para,sec].map(k=><td key={k.name}>{pct(at(k,t.target_fpr))}</td>)}</tr>)}</tbody>
    </table></div>
    <p className="cal-note">Small grey numbers are 95% intervals; the held-out rate is measured on half of the papers with a cutoff set on the other half, resampling whole papers. At 0.01% the cutoff rests on the top {Math.round(0.0001*c.calibration_set.sentences).toLocaleString()} or so of {c.calibration_set.sentences.toLocaleString()} human sentences and lands at two to three times its target on held-out papers, so rates that low are not reliably calibrated.</p>
    <h3>Which cutoff to use</h3>
    <ul className="cal-guide">
     <li><b>When a false flag is costly</b> (judging a person’s work), stay at 0.1% or below. About one human sentence in a thousand is still flagged, so a long human paper can show a few; look for clusters, not single sentences.</li>
     <li><b>For screening</b> (deciding what to read more closely), 1% catches clearly more small edits: on the Evaluation page’s harder held-out passages, 74.6% against 47.7% at 0.1%. The cost is that two thirds of human papers then show at least one flagged sentence.</li>
     <li><b>Above 1%</b> most human papers show several flagged sentences, so a paper’s share of flags says little on its own.</li>
     <li><b>Whole AI paragraphs and sections</b> are caught at almost any of these rates; single edited sentences are where the choice matters.</li>
    </ul>
    <p>{c.calibration_set.description} {c.calibration_set.method}</p>
    <RocChart curves={c.sentence_curves as Curve[]} label="Sentence-level ROC curves" marks={[.0001,.001,.01]} xmin={1e-6} scoreAt={scoreAt}/>
    <p>Edited sentences come from the three held-out sets behind the small-edit numbers on the <a href="/?view=results">Evaluation page</a> (heldout-writers, t21-heldout-v2 and wide1-heldout): {one.n_ai.toLocaleString()} one-sentence, {two.n_ai.toLocaleString()} two-sentence and {para.n_ai.toLocaleString()} paragraph-edit AI sentences from ten writers (Claude Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 4.5; GPT-6 Luna, GPT-6 Sol and GPT-6.1 Sol; DeepSeek V4.1 Flash, GLM 5.3 Flash and MiMo v2.6 Flash), each edited passage scored in one pass, against every counted sentence of the human calibration papers.{c.curves_note?` ${c.curves_note}`:""}</p>
    <p>AI sections are whole sections drafted or rewritten in held-out papers by Claude Sonnet 5.5, Opus 5.5, Haiku 4.5 and Fable 5.1 (Fable is never trained on), {sec.n_ai.toLocaleString()} AI sentences in 320 sections, each section scored in one pass; 177 more sections were left out because their AI-written text shares a sequence of 10 or more words with the training data.</p>
    <p>Why these rates are higher than on the <a href="/?view=results">Evaluation page</a>: here the false-positive rate counts every counted sentence of the 2,400 human papers, including equations and routine method text that are easy to call human. The Evaluation page measures it on human sentences from the same kind of passage each edit was made in, which is harder, and there the model catches 74.6% of small edits at 1%. The edits here are also scored as passages on their own, not inside a whole paper as on this site, so these catch rates are likely optimistic for whole papers.</p>
   </section>

   <section className="cal-section">
    <h2>Flagged share by year</h2>
    <p>The same threshold applied to papers from each year. Papers that share text with the model’s training data are excluded. {fpr===0.0001?"At 0.01% the median paper has no flagged sentence in every group from 2019 to 2025; ICLR 2026 reaches about 7%, and the median ICLR 2027 submission has about 86% of its sentences flagged.":fpr===0.001?"At this stricter cutoff the median paper has no flagged sentence in every group from 2019 to 2024. It reaches about half a percent in 2025, a fifth in ICLR 2026, and nearly all sentences in the ICLR 2027 submissions, which is not explained yet (see Limitations).":"At 1% the median human paper from 2022 or earlier has about half a percent of its sentences flagged. The median rises to about 1–1.5% in 2023 and 2024, about 5% in 2025, over 40% in ICLR 2026, and nearly all sentences in the ICLR 2027 submissions."}</p>
    {years?<YearChart rows={years} fpr={fpr}/>:<><p className="cal-note">Not yet computed at {fprLabel(fpr)} for this model; shown at 1%.</p><YearChart rows={c.years} fpr={0.01}/></>}
   </section>

   {(c.candidates?.length??0)>0&&<section className="cal-section">
    <h2>Candidate models</h2>
    <p>Two newer training mixes on the same Qwen3.5-4B base, scored with the same rules on the {c.candidates?.[0]?.papers.toLocaleString()} new human papers. Their thresholds are recorded here for when one replaces the site’s model; the site does not use them.</p>
    <div className="cal-table-wrap"><table className="cal-table">
     <thead><tr><th>Model</th><th>Training mix</th><th>Sentence threshold at 1%</th><th>95% interval</th><th>Held-out false-positive rate</th><th>Document threshold at 1%</th></tr></thead>
     <tbody>{(c.candidates??[]).map(m=><tr key={m.id}><td>{m.name}</td><td>{m.detail}</td><td>{m.threshold_1pct.toFixed(4)}</td><td>{m.threshold_ci95[0].toFixed(4)}–{m.threshold_ci95[1].toFixed(4)}</td>
      <td>{pct(m.heldout_fpr,2)} ({pct(m.heldout_ci95[0],2)}–{pct(m.heldout_ci95[1],2)})</td><td>{m.document_threshold.toFixed(3)}</td></tr>)}</tbody>
    </table></div>
   </section>}

   <section className="cal-section">
    <h2>Limits</h2>
    <ul className="cal-limits">{c.caveats.map(t=><li key={t}>{t}</li>)}</ul>
   </section>
  </main>
  <SiteFooter/>
 </div>;
}
