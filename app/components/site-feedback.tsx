"use client";
/* Site-wide feedback (Alice, Oct 13 2026): a "Feedback" button in the bottom-right corner of every page except the PDF
 * reader, which has its own "Think this result is wrong?" form. Messages go to the same private store as the reader's
 * (atlas-cloud/feedback.ts, kind "site"), with only the page name as context. Replaces the public email address. */
import {useEffect,useState} from "react";
import "./site-feedback.css";

const MIN=100,MAX=4000;
/** The page the message was sent from, without search text or other filters (they can contain names). */
function pageName(){const p=new URLSearchParams(location.search);return p.get("view")||([...p.keys()].length?"submissions":"home");}

export default function SiteFeedback(){
 const [open,setOpen]=useState(false),[text,setText]=useState(""),[trap,setTrap]=useState("");
 const [state,setState]=useState<"idle"|"sending"|"sent"|"error">("idle"),[error,setError]=useState("");
 const short=MIN-text.trim().length,ready=short<=0&&state!=="sending";
 useEffect(()=>{if(!open)return;const key=(e:KeyboardEvent)=>{if(e.key==="Escape")setOpen(false);};addEventListener("keydown",key);return ()=>removeEventListener("keydown",key);},[open]);
 async function send(){
  setState("sending");setError("");
  try{
   const r=await fetch("/backend/v1/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({kind:"site",comment:text,context:{page:pageName()},website:trap})});
   if(r.ok){setState("sent");setText("");return;}
   const d=(await r.json().catch(()=>({}))) as {detail?:unknown};setError(typeof d.detail==="string"?d.detail:"The message could not be sent.");setState("error");
  }catch{setError("The message could not be sent. Check your connection and try again.");setState("error");}
 }
 return <div className="sfb">
  {open&&<div className="sfb-panel" role="dialog" aria-label="Feedback on this site">
   <div className="sfb-head"><h2>Feedback</h2><button type="button" className="sfb-close" aria-label="Close" onClick={()=>setOpen(false)}>×</button></div>
   {state==="sent"?<p className="sfb-note" role="status">Thank you. Your message was saved.</p>:<form onSubmit={e=>{e.preventDefault();if(ready)send();}}>
    <p className="sfb-note">Bugs, confusing pages, ideas, anything. For one paper’s result, open the paper and use “Think this result is wrong?” instead.</p>
    <label className="sfb-sr" htmlFor="sfb-text">Your message</label>
    <textarea id="sfb-text" value={text} maxLength={MAX} rows={6} autoFocus onChange={e=>setText(e.target.value)}/>
    {/* Left empty by people; bots that fill every field are ignored by the server. */}
    <input className="sfb-sr" tabIndex={-1} autoComplete="off" aria-hidden="true" name="website" value={trap} onChange={e=>setTrap(e.target.value)}/>
    <p className="sfb-note">Your message is saved privately with the name of this page. I can't reply, so if you want to be contacted later, leave some way to contact you like a name or email address or discord/x account.</p>
    {state==="error"&&<p className="sfb-note sfb-error" role="alert">{error}</p>}
    <div className="sfb-actions">
     <button type="submit" disabled={!ready}>{state==="sending"?"Sending…":"Send"}</button>
     <small>{short>0?`${short} more character${short===1?"":"s"} needed`:`${text.length.toLocaleString()} / ${MAX.toLocaleString()}`}</small>
    </div>
   </form>}
  </div>}
  <button type="button" className="sfb-toggle" aria-expanded={open} onClick={()=>{setOpen(o=>!o);if(state==="sent")setState("idle");}}>Feedback</button>
 </div>;
}
