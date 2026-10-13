"use client";
import {useState} from 'react';
import {TABS,OUTCOME,HYPOTHESES,CATEGORIES,MIXES,COLLECTIONS,DEAD_ENDS,OVERRIDES,EFFORT,COMPUTE,OPEN_PROBLEMS,TIMELINE} from '../lib/transparency-data';

/** Overview of the pages in the Transparency menu, in menu order: a summary and what each page covers (Oct 12 2026).
 * Shown as a two-column index (one column on narrow screens). */
const PAGES:{href:string;title:string;body:string;covers:string[]}[]=[
 {href:'/?view=architecture',title:'Architecture',body:'What happens between a paper’s PDF and the highlighted sentences you see.',
  covers:['Text clean-up rules, on a sample page you can toggle','510-token windows, and why each is fed to the model twice','The classifier on top of Qwen3.6, and how token scores become sentence flags','The training loss and full training setup']},
 {href:'/?view=data',title:'Data',body:'Everything the model learned from and was tested on, down to individual rows.',
  covers:['Human sources, and AI text by writing model and kind of edit','Browse training examples and the prompts used for data generation','The 72,000-window training mix','The 8 held-out evaluation sets']},
 {href:'/?view=calibration',title:'Calibration',body:'Why a sentence is flagged where it is, and how often that wrongly flags human writing.',
  covers:['Cutoffs set on 2,400 human papers from 2019–2022','Checks on held-out human papers, with intervals','ROC curves for AI edits and sections','A switch for counting appendices']},
 {href:'/?view=results',title:'Evaluation',body:'How much AI text the model actually catches, on text it never saw in training.',
  covers:['Paragraphs, sections, humanized and co-written text','Small edits by kind, writing model and set','Results at 1% and 0.1% false-positive rates','What has not been measured']},
 {href:'/?view=limitations',title:'Limitations',body:'The caveats to keep in mind before trusting any single score.',
  covers:['How often human papers get flags','The hardest kinds of AI help to catch','Where the site’s setup differs from the evaluations','My mistakes, and what time and money did not cover']},
];

export function Transparency(){
 const [tab,setTab]=useState('timeline'),[f,setF]=useState('all');
 const counts:Record<string,number>={all:HYPOTHESES.length,c:0,r:0,i:0,o:0};HYPOTHESES.forEach(h=>counts[h[2]]++);
 return <>
  <div className="sp-col sp-col-wide">
   <section className="sp-hero sp-measure">
    <h1>Transparency</h1>
    <p>How the detector was built, trained, calibrated and tested, and where it falls short. Each page below covers one part. After them is the full story of the project: every experiment, decision and mistake that got us here.</p>
   </section>

   <section className="sp-overview" aria-labelledby="sp-pages">
    <h2 id="sp-pages" className="sp-label">In this section</h2>
    <div className="sp-pindex">
     {PAGES.map(p=><div key={p.href}><h3><a href={p.href}>{p.title} →</a></h3><p>{p.body}</p><ul>{p.covers.map(x=><li key={x}>{x}</li>)}</ul></div>)}
    </div>
   </section>

   <section className="sp-story" aria-labelledby="sp-story">
    <div className="sp-measure"><h2 id="sp-story">The full story</h2><p>Abridged from the project logs: what was tried, what held up, what didn’t, and what it cost. Read as much or as little as you like.</p></div>
    <div role="tablist" aria-label="Story sections" className="sp-tabs">
     {TABS.map(([k,l,n])=><button key={k} type="button" role="tab" aria-selected={k===tab} onClick={()=>setTab(k)}>{l} {n&&<span>{n}</span>}</button>)}
    </div>
    <div role="tabpanel" className="sp-panel">
     {tab==='timeline'&&<>
      <p className="sp-lede">From one general recipe to a calibrated multi-writer model in two weeks. Each phase was a reaction to what the last one exposed.</p>
      <ol className="sp-timeline">{TIMELINE.map(([d,n,b],i)=><li key={d+n} className={i===TIMELINE.length-1?'is-now':undefined}><span className="sp-tl-date">{d}</span><span className="sp-tl-dot" aria-hidden/><div><b>{n}</b><span>{b}</span></div></li>)}</ol>
     </>}
     {tab==='hyp'&&<>
      <p className="sp-lede">21 hypotheses. The biggest lesson: small-edit detection was limited by training data, not by recipe or model size.</p>
      <div role="group" aria-label="Filter by outcome" className="sp-chips">
       {['all','c','r','i','o'].map(k=><button key={k} type="button" aria-pressed={k===f} onClick={()=>setF(k)}>{k==='all'?'All':OUTCOME[k]} · {counts[k]}</button>)}
      </div>
      <div className="sp-hyps">{HYPOTHESES.map((h,i)=>({h,i})).filter(({h})=>f==='all'||h[2]===f).map(({h,i})=><details key={i}>
       <summary><span className="site-num">{i+1}</span><span>{h[0]}</span><span className={`sp-pill sp-pill-${h[2]}`}>{OUTCOME[h[2]]}</span></summary>
       <dl><dt>Test</dt><dd>{h[1]}</dd><dt>Result</dt><dd>{h[3]}</dd><dt>When</dt><dd>{h[4]}</dd></dl>
      </details>)}</div>
     </>}
     {tab==='data'&&<>
      <p className="sp-lede">Five training mixes. Each bar shows what one mix was made of; hover a segment for its share.</p>
      <div className="sp-mixes">{MIXES.map(([n,w,why,p])=>{const tot=Object.values(p).reduce((a,b)=>a+b,0);return <div key={n} className="sp-mix">
       <div className="sp-mix-name"><b>{n}</b><span>{w}</span></div>
       <div className="sp-mix-bar">{Object.entries(p).map(([k,v])=><i key={k} title={`${CATEGORIES[k][0]}: ${(100*v/tot).toFixed(1)}%`} style={{flex:v,background:CATEGORIES[k][1]}}/>)}</div>
       <p>{why}</p>
      </div>;})}</div>
      <div className="sp-legend">{Object.values(CATEGORIES).map(([l,c])=><span key={l}><i style={{background:c}}/>{l}</span>)}</div>
      <h3 className="sp-h3">Collections</h3>
      <table className="sp-table"><tbody>{COLLECTIONS.map(([n,s,d])=><tr key={n}><td><b>{n}</b><span>{d}</span></td><td className="site-num sp-right">{s}</td></tr>)}</tbody></table>
      <p className="sp-fine">* Disclosed in full; checking is still under way.</p>
     </>}
     {tab==='dead'&&<>
      <p className="sp-lede">What went wrong, what it cost, and what changed so it wouldn’t happen again.</p>
      <div className="sp-scroll"><table className="sp-table sp-table-3"><thead><tr><th>What happened</th><th>Cost</th><th>Fix</th></tr></thead><tbody>{DEAD_ENDS.map(([a,b,c])=><tr key={a}><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>)}</tbody></table></div>
     </>}
     {tab==='over'&&<>
      <p className="sp-lede">Most of the engineering was done with AI coding agents. These are the times I went against their advice, and how it turned out. Abridged.</p>
      <div className="sp-scroll"><table className="sp-table sp-table-3"><thead><tr><th>The AI suggested</th><th>I chose</th><th>Outcome</th></tr></thead><tbody>{OVERRIDES.map(([a,b,c])=><tr key={a}><td>{a}</td><td><b>{b}</b></td><td>{c}</td></tr>)}</tbody></table></div>
     </>}
     {tab==='effort'&&<>
      <div className="sp-effort">{EFFORT.map(([n,l])=><div key={l}><b className="site-num">{n}</b><span>{l}</span></div>)}</div>
      <h3 className="sp-h3">Where the compute came from</h3>
      <div className="sp-scroll"><table className="sp-table sp-table-3"><tbody>{COMPUTE.map(([a,b,c])=><tr key={a+b}><td className="site-num">{a}</td><td><b>{b}</b></td><td>{c}</td></tr>)}</tbody></table></div>
     </>}
     {tab==='open'&&<>
      <p className="sp-lede">Known gaps. Read these before relying on any single score.</p>
      <div className="sp-grid2">{OPEN_PROBLEMS.map(([t,d])=><div key={t}><b>{t}</b><span>{d}</span></div>)}</div>
     </>}
    </div>
   </section>
  </div>
 </>;
}
