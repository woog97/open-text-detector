import {paperRate, summaryFor, modelLabel, defaultModelFor, tooShort, MIN_SENTENCES, type ScoredPaper} from '@/lib/paper-scores';
export default function PaperScore({paper,model,threshold}:{paper:ScoredPaper & {classified:boolean};model:string;threshold:number}) {
 const summary=summaryFor(paper,model),rate=paperRate(summary,threshold);
 const version=(model==='all'?defaultModelFor(paper):model)??model;
 if(summary&&tooShort(summary)){const note=`Too short to score reliably (${summary.total} sentence${summary.total===1?'':'s'}; fewer than ${MIN_SENTENCES})`;
  return <span className="atlas-score-short" role="note" aria-label={`${note}, ${modelLabel(version)}`} title={note}><span className="atlas-short-mark" aria-hidden="true">!</span><span>Too short to score reliably<small>{summary.total} sentence{summary.total===1?'':'s'} · {modelLabel(version)}</small></span></span>;}
 if(rate===null)return <span className="atlas-score-empty">{paper.classified?'Score unavailable':'Not classified'}</span>;
 const band=rate<=.1?'low':rate<=.5?'medium':'high';
 return <span className={`atlas-score-card ${band}`} aria-label={`${(rate*100).toFixed(1)}% of passages above ${threshold.toFixed(2)}, ${modelLabel(version)}`}>
   <span className="atlas-score-top"><strong>{(rate*100).toFixed(1)}<span>%</span></strong><span className="atlas-score-model">{modelLabel(version)}</span></span>
   <span className="atlas-score-track" aria-hidden="true"><span style={{width:`${rate*100}%`}}/></span>
   <span className="atlas-score-caption">passages above {threshold.toFixed(2)}</span>
 </span>;
}
