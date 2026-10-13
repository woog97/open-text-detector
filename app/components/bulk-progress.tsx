"use client";
import { Activity, ArrowRight, CheckCircle2, CirclePause, Clock3, Loader2, Sparkles } from "lucide-react";
export type BulkRun = {
  id: string; status: string; total: number; model: {name: string}; counts: Record<string, number>;
  current?: {paper_id: string; title: string; status: string; scan_id: string} | null;
  upcoming?: {paper_id: string; title: string; ordinal: number}[];
  errors: {paper_id: string; error: string}[];
};
export default function BulkProgress({run, select, open}: {run: BulkRun; select: (id: string)=>void; open: (id: string)=>void}) {
  const active = ['running','stopping'].includes(run.status);
  const completed = run.counts.completed || 0, skipped = run.counts.skipped || 0;
  const failed = run.counts.failed || 0, cancelled = run.counts.cancelled || 0;
  const handled = completed + skipped + failed;
  const percent = run.total ? Math.floor(handled / run.total * 10000) / 100 : 0;
  const pending = run.counts.pending || 0;
  const upcoming = run.upcoming || [];
  const title = run.status === 'running' ? 'Processing papers' : run.status === 'stopping' ? 'Finishing the current paper' : run.status === 'stopped' ? 'Processing stopped' : 'Run complete';
  return <div className={`bulk-progress bulk-${run.status}`}>
    <div className="bulk-progress-heading"><div className="bulk-heading-icon">{active ? <Activity size={22}/> : run.status==='stopped' ? <CirclePause size={22}/> : <CheckCircle2 size={22}/>}</div><div><h3>{title}</h3><span>{run.model.name}</span></div><span className="bulk-live">{active&&<i/>}{active?'Live · updates every 3s':'Saved run'}</span></div>
    <div className="bulk-progress-label"><strong>{handled.toLocaleString()} <span>/ {run.total.toLocaleString()} processed</span></strong><b>{percent}%</b></div>
    <div className="bulk-progress-track" role="progressbar" aria-label="Papers processed" aria-valuemin={0} aria-valuemax={run.total || 1} aria-valuenow={handled} aria-valuetext={`${handled} processed, ${cancelled} cancelled, ${run.total} selected`}>
      {([['completed',completed],['skipped',skipped],['failed',failed]] as const).map(([state,count])=><span key={state} className={`bulk-segment ${state}`} style={{width:`${run.total?count/run.total*100:0}%`}}/>)}
    </div>
    <div className="bulk-stat-grid"><div className="bulk-stat saved"><CheckCircle2 size={16}/><strong>{completed.toLocaleString()}</strong><span>Saved</span></div><div className="bulk-stat waiting"><Clock3 size={16}/><strong>{pending.toLocaleString()}</strong><span>Waiting</span></div><div className="bulk-stat skipped"><ArrowRight size={16}/><strong>{skipped.toLocaleString()}</strong><span>Skipped</span></div><div className={`bulk-stat ${failed?'failed':''}`}><strong>{failed.toLocaleString()}</strong><span>Failed</span></div>{cancelled>0&&<div className="bulk-stat"><strong>{cancelled.toLocaleString()}</strong><span>Cancelled</span></div>}</div>
    {active&&<div className="bulk-queue-grid">
      <div className="bulk-current"><div className="bulk-eyebrow"><Loader2 size={15} className="bulk-spinner"/>{run.current?.status==='running'?'Processing now':run.current?'Waiting for model':'Preparing next paper'}</div>{run.current?<><button className="bulk-paper-title" onClick={()=>select(run.current!.paper_id)}>{run.current.title}</button><button className="bulk-report-link" onClick={()=>open(run.current!.scan_id)}>View classification <ArrowRight size={15}/></button></>:<p>The queue will advance automatically.</p>}<div className="bulk-activity-wave" aria-hidden="true">{Array.from({length:18},(_,i)=><i key={i} style={{animationDelay:`${i*0.08}s`}}/>)}</div><small>{run.status==='stopping'?'The current result will be saved, then processing will stop.':'Full-paper classification · result saved when complete'}</small></div>
      <div className="bulk-upcoming"><div className="bulk-eyebrow"><Clock3 size={15}/>Up next <span>{pending.toLocaleString()} waiting</span></div>{upcoming.length?<ol>{upcoming.map((paper,i)=><li key={paper.paper_id}><span>{i+1}</span><button onClick={()=>select(paper.paper_id)}>{paper.title}</button>{i===0&&<ArrowRight size={14}/>}</li>)}</ol>:<p>{run.status==='stopping'?'No more papers will be started.':'No more papers waiting in this run.'}</p>}{pending>upcoming.length&&<small>+ {(pending-upcoming.length).toLocaleString()} more in the selected order</small>}</div>
    </div>}
    <p className="bulk-run-note" role="status">{active?'This queue follows the filters chosen at start. You can browse or leave this page while it runs.':run.total===0?'No matching papers need classification with this model.':<><Sparkles size={14}/> {completed.toLocaleString()} results saved and ready to view.</>}</p>
    {run.errors.length>0&&<details><summary>Processing issues ({failed})</summary>{run.errors.map(e=><p key={e.paper_id}>{e.paper_id}: {e.error}</p>)}</details>}
  </div>;
}
