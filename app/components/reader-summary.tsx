"use client";
/* Beta reader: "Results by model", its controls and the flagged-passage count, shown in the paper's left column. */
import {useMemo,useState,type ReactNode} from "react";
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

const MAX_FEEDBACK=4000,MIN_FEEDBACK=100;
/** "Think this result is wrong?": a free-text message about this paper's result, stored privately (atlas-cloud/feedback.ts). */
function Feedback({paperId,context,marked,canMark,onClearMarks,onStartMarking,showMarks,onShowMarks,marking,onStopMarking,open,setOpen}:{paperId:string;context:Record<string,string>;marked:number[];canMark:boolean;showMarks:boolean;onShowMarks:(v:boolean)=>void;onClearMarks:()=>void;onStartMarking:()=>void;marking:boolean;onStopMarking:(keep:boolean)=>void;open:boolean;setOpen:(o:boolean)=>void}){
 const [text,setText]=useState(""),[trap,setTrap]=useState(""),[name,setName]=useState(""),[reach,setReach]=useState("");
 const [state,setState]=useState<"idle"|"sending"|"sent"|"error">("idle"),[error,setError]=useState("");
 async function send(){
  setState("sending");setError("");
  try{
   const r=await fetch("/backend/v1/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({paper:paperId,comment:text,context,passages:marked,contact:{name,reach},website:trap})});
   if(r.ok){setState("sent");onClearMarks();return;}
   const d=(await r.json().catch(()=>({}))) as {detail?:unknown};setError(typeof d.detail==="string"?d.detail:"The message could not be sent.");setState("error");
  }catch{setError("The message could not be sent. Check your connection and try again.");setState("error");}
 }
 // A message needs at least 100 characters, also when passages are marked (Alice, Oct 13 2026); the server checks the same.
 const short=MIN_FEEDBACK-text.trim().length,ready=short<=0&&state!=="sending";
 return <section className="rs-feedback" id="rs-feedback">
  <h2>Think this result is wrong?</h2>
  {/* The marking mode can be left from here as well as from the bar over the PDF. */}
  {marking&&<div className="rs-marking" role="status"><p><b>Marking mode is on.</b> Click or drag across flagged sentences on the PDF. {marked.length} marked.</p>
   <div className="rs-feedback-actions"><button type="submit" onClick={()=>onStopMarking(true)}>Done</button><button type="button" className="rs-more" onClick={()=>onStopMarking(false)}>Cancel</button></div></div>}
  {state==="sent"?<p className="rs-note" role="status">Thank you. Your message was saved.</p>:!open?<>
   <p className="rs-note">If this is your paper, or you know how it was written, you can say where the detector got it wrong{canMark?", and mark the flagged sentences you wrote without AI":""}.</p>
   {marked.length>0&&<p className="rs-note rs-marked">{marked.length} passage{marked.length===1?"":"s"} marked as written without AI · <button type="button" className="rs-more" onClick={onClearMarks}>Clear</button></p>}
   {marked.length>0&&<label className="rs-showmarks"><input type="checkbox" checked={showMarks} onChange={e=>onShowMarks(e.target.checked)}/> Show my marks on the PDF</label>}
   <div className="rs-feedback-start">
    <button type="button" className="rs-more" onClick={()=>setOpen(true)}>Write a message</button>
    {canMark&&!marking&&<button type="button" className="rs-more" onClick={onStartMarking}>Mark passages on the PDF</button>}
   </div>
  </>:<form onSubmit={e=>{e.preventDefault();if(ready)send();}}>
   {marked.length>0&&<p className="rs-note rs-marked">{marked.length} passage{marked.length===1?"":"s"} marked as written without AI will be sent with your message · <button type="button" className="rs-more" onClick={onStartMarking}>Edit</button> · <button type="button" className="rs-more" onClick={onClearMarks}>Clear</button></p>}
   {marked.length>0&&<label className="rs-showmarks"><input type="checkbox" checked={showMarks} onChange={e=>onShowMarks(e.target.checked)}/> Show my marks on the PDF</label>}
   {canMark&&!marking&&!marked.length&&<p className="rs-note"><button type="button" className="rs-more" onClick={onStartMarking}>Mark passages on the PDF</button> to send them with your message.</p>}
   <label className="rs-sr" htmlFor="rs-feedback-text">Your message</label>
   <textarea id="rs-feedback-text" value={text} maxLength={MAX_FEEDBACK} rows={5} autoFocus onChange={e=>setText(e.target.value)}
    placeholder={marked.length?"What should I know about the marked passages? For example, how they were written, or how AI was used elsewhere.":"For example: which sections were written without AI, or how AI was used."}/>
   {/* Left empty by people; bots that fill every field are ignored by the server. */}
   <input className="rs-sr" tabIndex={-1} autoComplete="off" aria-hidden="true" name="website" value={trap} onChange={e=>setTrap(e.target.value)}/>
   <div className="rs-contact">
    <label>Name <span>optional</span><input value={name} maxLength={200} autoComplete="name" onChange={e=>setName(e.target.value)}/></label>
    <label>Email, Discord or X <span>optional</span><input value={reach} maxLength={200} autoComplete="email" onChange={e=>setReach(e.target.value)}/></label>
   </div>
   <p className="rs-note">Your message is saved privately with this paper and the settings shown above. Contact details are optional; leave them if you would like a reply (reply not guaranteed).</p>
   {state==="error"&&<p className="rs-note rs-error" role="alert">{error}</p>}
   <div className="rs-feedback-actions">
    <button type="submit" disabled={!ready}>{state==="sending"?"Sending…":"Send"}</button>
    <button type="button" className="rs-more" onClick={()=>{setOpen(false);setState("idle");setError("");}}>Cancel</button>
    <small>{short>0?`${short} more character${short===1?"":"s"} needed`:`${text.length.toLocaleString()} / ${MAX_FEEDBACK.toLocaleString()}`}</small>
   </div>
  </form>}
 </section>;
}

export default function ReaderSummary({paperId,locked=false,showMarks=true,onShowMarks=()=>{},marked=[],canMark=false,onClearMarks=()=>{},onStartMarking=()=>{},marking=false,onStopMarking=()=>{},feedbackOpen=false,onFeedbackOpen=()=>{},reports,reportId,onPick,loading,flagMode,onFlagMode,threshold,flagged,onSelect,onOpenPanel,calibrated,custom,customYellow}:{
 paperId:string;locked?:boolean;showMarks?:boolean;onShowMarks?:(v:boolean)=>void;marked?:number[];canMark?:boolean;onClearMarks?:()=>void;onStartMarking?:()=>void;marking?:boolean;onStopMarking?:(keep:boolean)=>void;feedbackOpen?:boolean;onFeedbackOpen?:(o:boolean)=>void;reports:Report[];reportId:string;onPick:(id:string)=>void;loading:boolean;flagMode:FlagMode;onFlagMode:(m:FlagMode)=>void;threshold:number;
 flagged:FlaggedItem[];onSelect:(id:number)=>void;onOpenPanel:()=>void;calibrated:boolean;custom?:ReactNode;customYellow?:number;
}){
 const flag=FLAG_MODES[flagMode];
 const results=useMemo(()=>reports.map(r=>({r,...resultFor(r,flagMode,threshold,customYellow)})),[reports,flagMode,threshold,customYellow]);
 return <div className="rs">
  <section className="rs-models">
   <h2>Results by model</h2>
   {loading?<p className="rs-note">Loading results…</p>:!reports.length?<p className="rs-note">This paper has not been scored yet. You can still read the PDF.</p>:<>
    <fieldset disabled={locked} className={locked?"rs-locked":undefined}><legend className="rs-sr">Model shown on the PDF</legend>
     {results.map(({r,value,detail})=><label key={r.id} className={r.id===reportId?"is-on":undefined}>
      <input type="radio" name="rs-model" checked={r.id===reportId} onChange={()=>onPick(r.id)}/>
      <span className="rs-name"><b>{modelName(r)}</b>{r.id.startsWith(`${FLAG_MODEL}-`)&&<small>live</small>}</span>
      <span className="rs-value">{value}</span>
      <span className="rs-detail">{detail}</span>
     </label>)}
    </fieldset>
    {paperId&&<button type="button" className="rs-more rs-wrong" onClick={()=>onFeedbackOpen(true)}>Think this result is wrong? Tell me</button>}
    {/* Locked while feedback is in progress, so the message, marks and recorded settings all match what was on screen. */}
    <fieldset className={`rs-lockable${locked?" rs-locked":""}`} disabled={locked}>
    {calibrated&&<div className="rs-count" role="group" aria-label="Sentences counted"><span>Count</span>
     <label><input type="checkbox" checked={flag.app} onChange={e=>onFlagMode(flagModeOf(flag.bib,e.target.checked,flag.fpr))}/> Appendices</label>
     <label>FPR <select aria-label="False-positive rate" value={flag.fpr} onChange={e=>onFlagMode(flagModeOf(flag.bib,flag.app,Number(e.target.value) as typeof flag.fpr))}>{FLAG_FPRS.map(f=><option key={f} value={f}>{fprLabel(f)}</option>)}</select></label>
    </div>}
    {calibrated&&custom}
    </fieldset>
    {locked&&<p className="rs-note rs-lock-note" role="status">Settings are locked while you give feedback, so your message matches what you saw.</p>}
    <p className="rs-note">The selected model’s scores are highlighted on the PDF. A flag means the text resembles AI-generated text; it is not proof of authorship. <a href="/?view=transparency">How much to trust it</a></p>
   </>}
  </section>
  {!loading&&reports.length>0&&<section className="rs-flags">
   <h2>Flagged sentences <span>{flagged.length.toLocaleString()}</span></h2>
   {!flagged.length&&<p className="rs-note">No passages are flagged by this model.</p>}
   <button type="button" className="rs-more" onClick={onOpenPanel}>{flagged.length?"Show them in the passages panel":"Open the passages panel"}</button>
  </section>}
  {!loading&&reports.length>0&&paperId&&<Feedback key={paperId} paperId={paperId} marked={marked} canMark={canMark} onClearMarks={onClearMarks} onStartMarking={onStartMarking} showMarks={showMarks} onShowMarks={onShowMarks} marking={marking} onStopMarking={onStopMarking} open={feedbackOpen} setOpen={onFeedbackOpen} context={{report:reportId,mode:flagMode,shown:results.find(x=>x.r.id===reportId)?.value??"",...(customYellow!==undefined?{cutoff:String(customYellow)}:{})}}/>}
 </div>;
}
