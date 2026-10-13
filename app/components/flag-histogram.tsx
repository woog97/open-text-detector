import {useState,type CSSProperties,type ReactNode} from 'react';
import {flagCutLabel,parseFlagCut,parseFlagRange} from '../lib/paper-discovery';
const tone=(bin:number)=>bin<2?'low':bin<10?'mid':'high';
/**
 * Distribution of the share of sentences flagged as AI over the papers matching the current query, in 5% bins.
 * Counts ignore the bin filter itself, so the chart keeps its shape while one bin is selected.
 * Clicking a column filters to that bin; clicking it again (or the selected one) clears the filter.
 * A cutoff (`cut`) removes the papers outside it from the counts: those bars go empty and the others rescale to fill the
 * chart, and shares are of the papers inside the cutoff.
 */
export default function FlagHistogram({histogram,ai,cut='',unscored,onSelect,help,controls}:{histogram:number[];ai:string;cut?:string;unscored:number;onSelect:(ai:string)=>void;help:string;controls?:ReactNode}){
 const [hover,setHover]=useState<number|null>(null);
const cutRange=parseFlagCut(cut),inside=(i:number)=>!cutRange||(i>=cutRange[0]&&i<cutRange[1]);
 const range=parseFlagRange(ai),selected=Array.isArray(range)&&range[1]-range[0]===1?range[0]:null;
 const max=Math.max(1,...histogram),scored=histogram.reduce((a,b)=>a+b,0);
 if(histogram.length!==20||!scored)return null;
 const key=(i:number)=>`${i*5}-${i*5+5}`;
 const tip=hover??selected;
 return <figure className="pb-hist" aria-label="Papers by share of sentences flagged as AI">
  <figcaption><span title={help}>Flagged as AI · share of sentences</span><small>{scored.toLocaleString()} scored paper{scored===1?'':'s'}{cutRange?` ${flagCutLabel(cut).toLowerCase()} flagged`:''}{unscored?` · ${unscored.toLocaleString()} not scored`:''} · {selected===null?'click a bar to filter':'click the bar again to show all'}</small>{controls}</figcaption>
  <div className="pb-hist-plot" onPointerLeave={()=>setHover(null)}>
   {histogram.map((n,i)=>{
    const on=selected===i,state=on?' is-on':selected!==null?' is-off':'',out=!inside(i);
    return <button key={i} type="button" className={`pb-hist-col tone-${tone(i)}${state}${out?' is-out':''}${hover===i?' is-hover':''}`} aria-pressed={on} disabled={out}
     aria-label={`${i*5}–${i*5+5}% flagged: ${n.toLocaleString()} paper${n===1?'':'s'} (${(n/scored*100).toFixed(1)}% of scored results${cutRange?' in the cutoff':''})`}
     onPointerEnter={()=>{if(!out)setHover(i);}} onFocus={()=>setHover(i)} onBlur={()=>setHover(null)}
     onClick={()=>onSelect(on?'':key(i))}>
     <i style={{'--h':n?`max(2px, ${n/max*100}%)`:'0'} as CSSProperties}/>
    </button>;
   })}
   {tip!==null&&<span className={`pb-hist-tip${tip<4?' at-start':tip>15?' at-end':''}`} role="status" style={{'--x':`${(tip+.5)*5}%`} as CSSProperties}><b>{histogram[tip].toLocaleString()}</b> paper{histogram[tip]===1?'':'s'} · {tip*5}–{tip*5+5}% flagged<small>{(histogram[tip]/scored*100).toFixed(1)}% of scored results{cutRange?` ${flagCutLabel(cut).toLowerCase()}`:''}</small></span>}
  </div>
  <div className="pb-hist-axis" aria-hidden><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
 </figure>;
}
