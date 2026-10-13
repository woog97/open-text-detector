import { memo, useMemo, useState } from "react";
import { collectionBreakdown, modelLabel, type ScoredPaper } from "@/lib/paper-scores";
function Pie({ title, slices }: { title: string; slices: { label: string; count: number; color: string }[] }) {
  const total=slices.reduce((sum,s)=>sum+s.count,0); let at=0;
  const gradient=slices.map(s=>{const start=at;at+=total?s.count/total*100:0;return `${s.color} ${start}% ${at}%`;}).join(",");
  return <section className="atlas-chart"><div className="atlas-pie" role="img" aria-label={`${title}: ${slices.map(s=>`${s.label}: ${s.count.toLocaleString()}`).join(', ')}`} style={{background:total?`conic-gradient(${gradient})`:'#e2e5dd'}}><span title={total.toLocaleString()}><b>{new Intl.NumberFormat("en",{notation:"compact",maximumFractionDigits:1}).format(total)}</b><small>total</small></span></div><div><h3>{title}</h3><ul>{slices.map(s=><li key={s.label}><i style={{background:s.color}}/>{s.label}<b>{s.count.toLocaleString()}</b><small>{total?`${(s.count/total*100).toFixed(1)}%`:'—'}</small></li>)}</ul></div></section>;
}
function ScoreDashboard({papers,model,threshold}:{papers:(ScoredPaper & {collection:string})[];model:string;threshold:number}) {
  const [breakdownMode,setBreakdownMode]=useState("passages");
  const groups=useMemo(()=>collectionBreakdown(papers,model,threshold),[papers,model,threshold]);
  const palette=breakdownMode==="passages"?["#afd7b4","#e58d82"]:["#afd7b4","#efd578","#e58d82","#d4d7d0"];
  const labels=breakdownMode==="passages"?["At or below threshold","Above threshold"]:["0–10% of passages","Over 10–50%","Over 50%","No score available"];
  const {above,below,scored,bins}=useMemo(()=>{
    let above=0,below=0,scored=0;const bins=[0,0,0,0];
    for(const group of groups){above+=group.above;below+=group.below;scored+=group.scored;group.bins.forEach((n,i)=>bins[i]+=n);}
    return {above,below,scored,bins};
  },[groups]);
  return <div className="atlas-dashboard"><div className="atlas-dashboard-heading"><strong>In this search · {papers.length.toLocaleString()} papers</strong><span>{scored.toLocaleString()} with scores · {model==='all'?'Default model per paper (final MoE, then MELD v8, v5)':modelLabel(model)}</span></div><div className="atlas-charts"><Pie title={`Passages · threshold ${threshold.toFixed(2)}`} slices={[{label:'At or below threshold',count:below,color:'#afd7b4'},{label:'Above threshold',count:above,color:'#e58d82'}]}/><Pie title="Papers · share above threshold" slices={[{label:'0–10% of passages',count:bins[0],color:'#afd7b4'},{label:'Over 10–50%',count:bins[1],color:'#efd578'},{label:'Over 50%',count:bins[2],color:'#e58d82'},{label:'No score available',count:bins[3],color:'#d4d7d0'}]}/></div><p>Text-passage counts, not word counts or probabilities of authorship. Punctuation-only fragments and papers without scores are excluded from passage totals.</p>
    <details className="atlas-breakdown"><summary><span>Conference &amp; year breakdown<small>{groups.length} collections in this search</small></span><span className="atlas-breakdown-chevron" aria-hidden="true">⌄</span></summary>
      <div className="atlas-breakdown-content">
        <div className="atlas-breakdown-toolbar"><div className="atlas-breakdown-tabs" role="group" aria-label="Breakdown measure"><button aria-pressed={breakdownMode==="passages"} onClick={()=>setBreakdownMode("passages")}>Passages</button><button aria-pressed={breakdownMode==="papers"} onClick={()=>setBreakdownMode("papers")}>Papers</button></div><span>Score threshold {threshold.toFixed(2)} · each bar totals 100%</span></div>
        <div className="atlas-breakdown-legend">{labels.map((label,i)=><span key={label}><i style={{background:palette[i]}}/>{label}</span>)}</div>
        <p className="atlas-breakdown-note">{breakdownMode==="passages"?"All scored passages pooled within each collection. Papers without scores are excluded.":"Papers grouped by the share of their passages above the selected threshold."}</p>
        {groups.length===0?<p>No collections match this search.</p>:groups.map(group=>{
          const values=breakdownMode==="passages"?[group.below,group.above]:group.bins;
          const total=values.reduce((a,b)=>a+b,0);
          const description=values.map((v,i)=>`${labels[i]}: ${v.toLocaleString()} (${total?(v/total*100).toFixed(1):'0'}%)`).join('; ');
          return <div className="atlas-breakdown-row" key={group.collection}>
            <div className="atlas-breakdown-label"><strong>{group.collection.replace('/', ' · ').replaceAll('_',' ')}</strong><small>{group.papers.toLocaleString()} papers · {group.scored.toLocaleString()} scored</small></div>
            <div className="atlas-breakdown-bar" role="img" aria-label={`${group.collection}: ${description}`} title={description}>{total?values.map((v,i)=>v>0&&<span key={labels[i]} style={{width:`${v/total*100}%`,background:palette[i]}}>{v/total>=.09?`${(v/total*100).toFixed(1)}%`:''}</span>):<span className="atlas-breakdown-no-data">No scored passages</span>}</div>
            <span className="atlas-breakdown-total">{total.toLocaleString()}<small>{breakdownMode}</small></span>
          </div>;
        })}
      </div>
    </details></div>;
}

export default memo(ScoreDashboard);
