import {paperRate,summaryFor,tooShort,type ScoredPaper} from './paper-scores.ts';
import type {DiscoveryMetadata} from './atlas-archive.ts';
/** Browse rows carry `flags` instead of full summaries: per counting mode (paper-discovery FLAG_MODES), [sentences above
 * that mode's 1%-false-positive threshold, sentences counted]; `sections` = scored sentences in [body, bibliography, appendices]. */
export type AtlasPaper=ScoredPaper & DiscoveryMetadata & {id:string;title:string;filename:string;collection:string;bytes:number;classified:boolean;models?:string[];flags?:Partial<Record<import('./paper-discovery.ts').FlagMode|'all10'|'nobib10'|'noapp10'|'body10',[number,number]>>;sections?:[number,number,number];flag_model?:string;
 /** Mean sentence score per base counting mode (body: main text; nobib: main text and appendices); research/atlas-oct10-20261012/mean_scores_oct10.py. */
 mean_score?:Partial<Record<import('./paper-discovery.ts').BaseFlagMode,number|null>>};
export type AtlasCatalogue={items:AtlasPaper[];models?:{id:string;name:string;available:number;total:number;complete:boolean;sample?:boolean}[]};
export function validateCatalogue(value:unknown):AtlasCatalogue {
  const data=value as AtlasCatalogue;
  const fail=()=>{throw new Error('The collection data is invalid. Please retry shortly.');};
  if(!data||!Array.isArray(data.items))fail();
  const ids=new Set<string>();
  for(const p of data.items){
    if(!p||typeof p.id!=='string'||!/^[a-f0-9]{24}$/.test(p.id)||ids.has(p.id)||typeof p.title!=='string'||typeof p.filename!=='string'||typeof p.collection!=='string'||!Number.isFinite(p.bytes)||p.bytes<0||typeof p.classified!=='boolean'||(p.models!==undefined&&(!Array.isArray(p.models)||p.models.some(m=>typeof m!=='string'))))fail();
    ids.add(p.id);
    if(p.score_summaries!==undefined){
      if(!p.score_summaries||typeof p.score_summaries!=='object'||Array.isArray(p.score_summaries))fail();
      for(const s of Object.values(p.score_summaries))if(!s||!Array.isArray(s.histogram)||s.histogram.length!==21||s.histogram.some(n=>!Number.isSafeInteger(n)||n<0)||!Number.isSafeInteger(s.total)||s.total<0||s.histogram.reduce((a,b)=>a+b,0)!==s.total)fail();
    }
  }
  if(data.models!==undefined&&(!Array.isArray(data.models)||data.models.some(m=>!m||typeof m.id!=='string'||typeof m.name!=='string'||!Number.isSafeInteger(m.available)||!Number.isSafeInteger(m.total)||m.available<0||m.total<0||typeof m.complete!=='boolean'||(m.sample!==undefined&&typeof m.sample!=='boolean'))))fail();
  return data;
}
export function sortPapers<T extends AtlasPaper>(papers:T[],sort:string,descending:boolean,model:string,threshold:number):T[]{
  // Score once per paper, rather than for both sides of every O(n log n) comparison.
  // Papers too short to score reliably sort with the unscored ones.
  const rows=papers.map(paper=>{const s=sort==='share'?summaryFor(paper,model):undefined;return {paper,rate:s&&!tooShort(s)?paperRate(s,threshold):null};});
  rows.sort((a,b)=>{
    let comparison:number;
    if(sort==='share'){
      if(a.rate===null)return b.rate===null?a.paper.title.localeCompare(b.paper.title)||a.paper.id.localeCompare(b.paper.id):1;
      if(b.rate===null)return -1;
      comparison=a.rate-b.rate;
    }else comparison=sort==='title'?a.paper.title.localeCompare(b.paper.title):a.paper.bytes-b.paper.bytes;
    return (descending?-comparison:comparison)||a.paper.title.localeCompare(b.paper.title)||a.paper.id.localeCompare(b.paper.id);
  });
  return rows.map(r=>r.paper);
}
