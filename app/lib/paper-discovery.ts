import type {AtlasPaper} from './atlas-collection.ts';
import {MIN_SENTENCES} from './paper-scores.ts';
export type BrowseFilters={q:string;collection:string;area:string;tag:string;saved:boolean;sort:string;ai:string;mode?:string;cut?:string};
/** Which sentences a paper's flagged share counts, and at which false-positive rate. Each mode has its own thresholds,
 * calibrated on the human calibration papers with the same sentences left out (research/atlas-bib-exclusion-20261008,
 * final MoE). Base modes flag at 1% FPR; the `0.1` modes count the same sentences at 0.1% FPR. Thresholds are the
 * Oct 10 model's (final-moe), on the 2,400 human calibration papers (research/atlas-calibration-fix-20261011/recalibrate.py,
 * Oct 12, 2026). The modes that count the bibliography ('all', 'noapp') stay defined for the stored counts and the
 * Calibration page, but the Submissions page and the reader offer only the others (SHOWN_*): for the Oct 10 model the
 * bibliography-including 0.1% cutoff is 0.97 and misses its target. */
export type BaseFlagMode='all'|'nobib'|'noapp'|'body';
export type FlagMode=BaseFlagMode|'all0.1'|'nobib0.1'|'noapp0.1'|'body0.1'|'nobib0.01'|'body0.01';
export const BASE_FLAG_MODES:BaseFlagMode[]=['all','nobib','noapp','body'];
/** False-positive rates offered in both designs. */
export const FLAG_FPRS=[0.0001,0.001,0.01] as const;
export const LEGACY_FLAG_FPRS=FLAG_FPRS;
export type FlagFpr=typeof FLAG_FPRS[number];
/** Score at or above which a flagged sentence is red in every mode: about 0.015% of human calibration sentences with
 * the bibliography left out score this high under the Oct 10 model (its 99.985th percentile is 0.376; 0.8 was the Oct 7
 * model's, the same share there). */
export const STRONG_SCORE=0.38;
/** threshold: the mode's sentence threshold at its false-positive rate; a sentence is flagged when its score is at or
 * above it. strong: the red cutoff (STRONG_SCORE). At 0.01% (added Oct 12, 2026; bibliography-excluded modes only) the
 * cutoff is already above STRONG_SCORE, so every flag there is red. The 0.01% cutoffs rest on about 64-82 human
 * sentences and flag 2-3 times their target on held-out human papers. */
export const FLAG_MODES:Record<FlagMode,{bib:boolean;app:boolean;fpr:FlagFpr;base:BaseFlagMode;threshold:number;strong:number}>={
 all:{bib:true,app:true,fpr:0.01,base:'all',threshold:0.005114855085100446,strong:STRONG_SCORE},
 nobib:{bib:false,app:true,fpr:0.01,base:'nobib',threshold:0.0012564659118652344,strong:STRONG_SCORE},
 noapp:{bib:true,app:false,fpr:0.01,base:'noapp',threshold:0.006374359130859375,strong:STRONG_SCORE},
 body:{bib:false,app:false,fpr:0.01,base:'body',threshold:0.000994136873281227,strong:STRONG_SCORE},
 'all0.1':{bib:true,app:true,fpr:0.001,base:'all',threshold:0.9681396484375,strong:STRONG_SCORE},
 'nobib0.1':{bib:false,app:true,fpr:0.001,base:'nobib',threshold:0.02460369696983924,strong:STRONG_SCORE},
 'noapp0.1':{bib:true,app:false,fpr:0.001,base:'noapp',threshold:0.9790810032894737,strong:STRONG_SCORE},
 'body0.1':{bib:false,app:false,fpr:0.001,base:'body',threshold:0.021137782505580356,strong:STRONG_SCORE},
 'nobib0.01':{bib:false,app:true,fpr:0.0001,base:'nobib',threshold:0.5283203125,strong:0.5283203125},
 'body0.01':{bib:false,app:false,fpr:0.0001,base:'body',threshold:0.4599880642361111,strong:0.4599880642361111}};
/** Model whose report the flag counts and thresholds describe, and the only one the site offers: the Oct 10 MoE. */
export const FLAG_MODEL='qwen36-35b-a3b-oct10';
/** Main text only (bibliography and appendices excluded) at 0.1% FPR; appendices dropped from the default Oct 12, 2026
 * (bibliography excluded since Oct 11). */
export const DEFAULT_FLAG_MODE:FlagMode='body0.1';
/** Default paper order: least flagged first (Alice, Oct 11 2026). */
export const DEFAULT_SORT='flagged-asc';
export const flagModeOf=(bib:boolean,app:boolean,fpr:FlagFpr=0.01)=>(Object.keys(FLAG_MODES) as FlagMode[]).find(m=>FLAG_MODES[m].bib===bib&&FLAG_MODES[m].app===app&&FLAG_MODES[m].fpr===fpr)!;
export const asFlagMode=(v:string|null|undefined):FlagMode=>v&&Object.hasOwn(FLAG_MODES,v)?v as FlagMode:DEFAULT_FLAG_MODE;
/** The mode the Submissions page and reader use: a bibliography-counting mode (old links) becomes the same mode without it. */
export const asSiteFlagMode=(v:string|null|undefined):FlagMode=>{const m=asFlagMode(v),f=FLAG_MODES[m];return f.bib?flagModeOf(false,f.app,f.fpr):m;};
export const fprLabel=(fpr:number)=>`${+(fpr*100).toFixed(3)}%`;
export const thresholdText=(t:number)=>t.toFixed(t<0.001?5:4);
/** Share-of-sentences-flagged bands for the browse filter: [key, label, lower bound (inclusive), upper bound (exclusive)]. */
export const FLAG_BANDS:[string,string,number,number][]=[['0-10','Under 10%',0,.1],['10-25','10–25%',.1,.25],['25-50','25–50%',.25,.5],['50-75','50–75%',.5,.75],['75-90','75–90%',.75,.9],['90-100','90% or more',.9,Infinity]];
/** [flagged, counted sentences] under a mode, or undefined when the paper has no counts for it. */
export function flagCounts(p:Pick<AtlasPaper,'flags'>,mode:FlagMode=DEFAULT_FLAG_MODE){return p.flags?.[mode];}
/** Share of counted sentences above the mode's threshold; null when unscored or too short to score reliably. */
export function flagShare(p:Pick<AtlasPaper,'flags'>,mode:FlagMode=DEFAULT_FLAG_MODE){const c=flagCounts(p,mode);return c&&c[1]>=MIN_SENTENCES?c[0]/c[1]:null;}
/** 5%-wide histogram bin (0..19) of a flagged share; 100% falls in the last bin. */
export function flagBin(r:number){return Math.min(19,Math.floor(r*20+1e-9));}
/** `ai` filter value -> 'none' (unscored or too short), a [first, last) range of 5% bins, or null (no filter). Accepts "lo-hi" in percent, multiples of 5. */
export function parseFlagRange(ai:string):'none'|[number,number]|null{
 if(ai==='none')return 'none';
 const m=/^(\d{1,3})-(\d{1,3})$/.exec(ai);if(!m)return null;
 const lo=Number(m[1]),hi=Number(m[2]);
 return lo%5===0&&hi%5===0&&lo<hi&&hi<=100?[lo/5,hi/5]:null;
}
/** `cut` filter value -> [first, last) 5% bins: 'below95' keeps shares under 95%, 'above95' shares of 95% or more.
 * Cutoffs run 5-95% in 5% steps. Unlike `ai`, the cutoff also limits the histogram, so its bars rescale to the papers kept. */
export function parseFlagCut(cut:string|undefined):[number,number]|null{
 const m=/^(below|above)(\d{1,2})$/.exec(cut||'');if(!m)return null;
 const v=Number(m[2]);if(v%5||v<5||v>95)return null;
 return m[1]==='below'?[0,v/5]:[v/5,20];
}
export function flagCutLabel(cut:string){const r=parseFlagCut(cut);return r===null?'':r[0]===0?`Under ${r[1]*5}%`:`${r[0]*5}% or more`;}
export function inFlagRange(r:number|null,range:'none'|[number,number]){return range==='none'?r===null:r!==null&&flagBin(r)>=range[0]&&flagBin(r)<range[1];}
/** Band key for a paper, or 'none' when it has no reliable flagged share. */
export function flagBand(p:Pick<AtlasPaper,'flags'>,mode:FlagMode=DEFAULT_FLAG_MODE){const r=flagShare(p,mode);return r===null?'none':FLAG_BANDS.find(([k])=>inFlagRange(r,parseFlagRange(k)!))![0];}
/** Label for an `ai` filter value. */
export function flagRangeLabel(ai:string){const r=parseFlagRange(ai);return r===null?'':r==='none'?'Not scored':FLAG_BANDS.find(([k])=>k===ai)?.[1]??`${r[0]*5}–${r[1]*5}%`;}
export function searchText(p:AtlasPaper){return [p.title,p.filename,p.forum_id,p.number,p.primary_area,p.tldr,p.abstract_preview,...(p.keywords||[])].join(' ').toLowerCase();}
/** Compares titles exactly as String#localeCompare does, without per-call locale setup. */
export const titleOrder=new Intl.Collator().compare;
/** Papers matching every filter except the flagged-share band (the band facet counts these). */
export function matchDiscovery(papers:AtlasPaper[],filters:BrowseFilters,saved:Set<string>,index?:Map<string,string>){
 const terms=filters.q.toLowerCase().trim().split(/\s+/).filter(Boolean);
 return papers.filter(p=>(!filters.collection||p.collection===filters.collection)&&(!filters.area||p.primary_area===filters.area)&&(!filters.tag||p.keywords?.some(k=>k.toLowerCase()===filters.tag.toLowerCase()))&&(!filters.saved||saved.has(p.id))&&terms.every(t=>(index?.get(p.id)||searchText(p)).includes(t)));
}
/** `titleSorted` means `papers` is already in title order, so a title sort can be skipped. `matched`: matchDiscovery's result, if already computed. */
export function filterDiscovery(papers:AtlasPaper[],filters:BrowseFilters,saved:Set<string>,index?:Map<string,string>,titleSorted=false,matched?:AtlasPaper[]){
 let matches=matched?[...matched]:matchDiscovery(papers,filters,saved,index);
 const mode=asSiteFlagMode(filters.mode),range=parseFlagRange(filters.ai);if(range)matches=matches.filter(p=>inFlagRange(flagShare(p,mode),range));
 if(filters.sort==='flagged'||filters.sort==='flagged-asc'){
  // Unscored and too-short papers go last in both directions; ties keep title order.
  const dir=filters.sort==='flagged'?-1:1,rows=matches.map(p=>({p,r:flagShare(p,mode)}));
  rows.sort((a,b)=>a.r===null?(b.r===null?titleOrder(a.p.title,b.p.title)||a.p.id.localeCompare(b.p.id):1):b.r===null?-1:dir*(a.r-b.r)||titleOrder(a.p.title,b.p.title)||a.p.id.localeCompare(b.p.id));
  return rows.map(x=>x.p);
 }
 if(filters.sort==='title'&&titleSorted)return matches;
 return matches.sort((a,b)=>filters.sort==='title'?titleOrder(a.title,b.title)||a.id.localeCompare(b.id):filters.sort==='number'?(a.number??Infinity)-(b.number??Infinity)||a.id.localeCompare(b.id):(b.updated||0)-(a.updated||0)||a.title.localeCompare(b.title));
}
export function relatedPapers(paper:AtlasPaper,papers:AtlasPaper[]){
 const keys=new Set(paper.keywords?.map(k=>k.toLowerCase()));
 return papers.filter(p=>p.id!==paper.id&&p.collection===paper.collection).map(p=>({p,score:(p.primary_area&&p.primary_area===paper.primary_area?1:0)+(p.keywords||[]).filter(k=>keys.has(k.toLowerCase())).length*3})).filter(r=>r.score>0).sort((a,b)=>b.score-a.score||a.p.title.localeCompare(b.p.title)).slice(0,5).map(r=>r.p);
}
