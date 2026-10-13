"use client";
/* Beta reader: "Results by model", its controls and the flagged-passage count, shown in the paper's left column. */
import {useMemo,type ReactNode} from "react";
import {fprReadout} from "@/lib/score-fpr";
import {hasTextContent, summarizePassages, paperRate, modelLabel, MIN_SENTENCES} from "@/lib/paper-scores";
import {FLAG_FPRS, FLAG_MODES, FLAG_MODEL, flagModeOf, fprLabel, thresholdText, type FlagMode} from "@/lib/paper-discovery";
import {segmentSections} from "@/lib/paper-sections";
import type {ReaderDetail} from "@/lib/pdf-reader-data";

type Report=ReaderDetail["reports"][number];
export type FlaggedItem={id:number;page?:number;score:number;strong:boolean};

function share(r:number){return r<.1&&r>0?`${(r*100).toFixed(1)}%`:`${Math.round(r*100)}%`;}
const KNOWN=[FLAG_MODEL,"qwen35-4b","laya","v8","v5"];
function modelName(r:Report){const k=KNOWN.find(m=>r.id===m||r.id.startsWith(`${m}-`));return k?modelLabel(k):r.model.name;}

/** Flagged share of one report, counted the same way as the submissions list. */
function resultFor(r:Report,mode:FlagMode,threshold:number,custom?:number):{value:string;detail:string}{
 if(r.id.startsWith(`${FLAG_MODEL}-`)){
  const flag=FLAG_MODES[mode],secs=segmentSections(r.text,r.result.segments),chars=Array.from(r.text);let counted=0,flagged=0;
  r.result.segments.forEach((s,i)=>{if(!hasTextContent(chars.slice(s.start,s.end).join("")))return;const sec=secs[i];if(!(sec==="body"||(sec==="bib"&&flag.bib)||(sec==="app"&&flag.app)))return;counted++;if(s.score>=(custom??flag.threshold))flagged++;});
  if(counted<MIN_SENTENCES)return {value:"—",detail:`Too short to score reliably (${counted} sentences)`};
  return {value:share(flagged/counted),detail:custom!==undefined?`${flagged.toLocaleString()} of ${counted.toLocaleString()} sentences at or above your custom cutoff ${thresholdText(custom)} (${fprReadout(flag.base,custom)})`:`${flagged.toLocaleString()} of ${counted.toLocaleString()} sentences flagged at ${fprLabel(flag.fpr)} false positives`};
 }
 const s=summarizePassages(r.result.segments,r.text),rate=paperRate(s,threshold);
 return rate===null?{value:"—",detail:"No scored passages"}:{value:share(rate),detail:`passages above ${threshold.toFixed(2)} · not calibrated`};
}

export default function ReaderSummary({reports,reportId,onPick,loading,flagMode,onFlagMode,threshold,flagged,onSelect,onOpenPanel,calibrated,custom,customYellow}:{
 reports:Report[];reportId:string;onPick:(id:string)=>void;loading:boolean;flagMode:FlagMode;onFlagMode:(m:FlagMode)=>void;threshold:number;
 flagged:FlaggedItem[];onSelect:(id:number)=>void;onOpenPanel:()=>void;calibrated:boolean;custom?:ReactNode;customYellow?:number;
}){
 const flag=FLAG_MODES[flagMode];
 const results=useMemo(()=>reports.map(r=>({r,...resultFor(r,flagMode,threshold,customYellow)})),[reports,flagMode,threshold,customYellow]);
 return <div className="rs">
  <section className="rs-models">
   <h2>Results by model</h2>
   {loading?<p className="rs-note">Loading results…</p>:!reports.length?<p className="rs-note">This paper has not been scored yet. You can still read the PDF.</p>:<>
    <fieldset><legend className="rs-sr">Model shown on the PDF</legend>
     {results.map(({r,value,detail})=><label key={r.id} className={r.id===reportId?"is-on":undefined}>
      <input type="radio" name="rs-model" checked={r.id===reportId} onChange={()=>onPick(r.id)}/>
      <span className="rs-name"><b>{modelName(r)}</b>{r.id.startsWith(`${FLAG_MODEL}-`)&&<small>live</small>}</span>
      <span className="rs-value">{value}</span>
      <span className="rs-detail">{detail}</span>
     </label>)}
    </fieldset>
    {calibrated&&<div className="rs-count" role="group" aria-label="Sentences counted"><span>Count</span>
     <label><input type="checkbox" checked={flag.app} onChange={e=>onFlagMode(flagModeOf(flag.bib,e.target.checked,flag.fpr))}/> Appendices</label>
     <label>FPR <select aria-label="False-positive rate" value={flag.fpr} onChange={e=>onFlagMode(flagModeOf(flag.bib,flag.app,Number(e.target.value) as typeof flag.fpr))}>{FLAG_FPRS.map(f=><option key={f} value={f}>{fprLabel(f)}</option>)}</select></label>
    </div>}
    {calibrated&&custom}
    <p className="rs-note">The selected model’s scores are highlighted on the PDF. A flag means the text resembles AI-generated text; it is not proof of authorship. <a href="/?view=transparency">How much to trust it</a></p>
   </>}
  </section>
  {!loading&&reports.length>0&&<section className="rs-flags">
   <h2>Flagged sentences <span>{flagged.length.toLocaleString()}</span></h2>
   {!flagged.length&&<p className="rs-note">No passages are flagged by this model.</p>}
   <button type="button" className="rs-more" onClick={onOpenPanel}>{flagged.length?"Show them in the passages panel":"Open the passages panel"}</button>
  </section>}
 </div>;
}
