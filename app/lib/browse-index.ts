import type {AtlasPaper} from './atlas-collection.ts';
import {BASE_FLAG_MODES,DEFAULT_SORT,FLAG_BANDS,FLAG_MODES,asSiteFlagMode,flagBin,flagShare,parseFlagCut,parseFlagRange,searchText,titleOrder,type BrowseFilters,type FlagMode} from './paper-discovery.ts';
const MODES=Object.keys(FLAG_MODES) as FlagMode[];
/**
 * Compact browse index for the Worker (128 MB memory limit). Holds only what filtering, sorting and counting need,
 * in title order; the displayed rows ("cards") live in one pack object, fetched by byte range for the 20 rows of a page.
 * The full row list parsed into objects took ~200 MB of heap for 80k papers; this takes about a quarter of that.
 * `flags[mode]` is each paper's flagged share under that counting mode (-1: unscored or too short). `flag` repeats the
 * 'nobib' column for Workers deployed before counting modes existed (Oct 8, 2026); drop it once none serve traffic.
 */
export type SearchIndex={v:2;ids:string[];text:string[];col:number[];cols:string[];area:number[];areas:string[];kw:number[][];kws:string[];num:(number|null)[];upd:number[];flag:number[];flags:Record<FlagMode,number[]>;card:number[]};
/** Search covers titles, TL;DRs, keywords, research areas, submission numbers, forum IDs and filenames (not abstracts). */
export function indexText(p:AtlasPaper){return searchText({...p,abstract_preview:undefined});}
const keyOf=(k:string)=>k.trim().toLowerCase();
/** Rows -> index (title order) and the card pack; `card` holds [byte offset, byte length] pairs into the pack. */
export function buildIndex(rows:AtlasPaper[]):{index:SearchIndex;pack:Uint8Array}{
 const items=[...rows].sort((a,b)=>titleOrder(a.title,b.title)||a.id.localeCompare(b.id));
 const cols=new Map<string,number>(),areas=new Map<string,number>(),kws=new Map<string,number>();
 const code=(m:Map<string,number>,v:string)=>{let c=m.get(v);if(c===undefined){c=m.size;m.set(v,c);}return c;};
 const enc=new TextEncoder(),parts:Uint8Array[]=[];let offset=0;
 const index:SearchIndex={v:2,ids:[],text:[],col:[],cols:[],area:[],areas:[],kw:[],kws:[],num:[],upd:[],flag:[],flags:Object.fromEntries(MODES.map(m=>[m,[]])) as unknown as Record<FlagMode,number[]>,card:[]};
 for(const p of items){
  index.ids.push(p.id);index.text.push(indexText(p));index.col.push(code(cols,p.collection));
  index.area.push(p.primary_area?code(areas,p.primary_area):-1);
  index.kw.push([...new Set((p.keywords||[]).map(keyOf).filter(Boolean))].map(k=>code(kws,k)));
  index.num.push(p.number??null);index.upd.push(p.updated||0);
  for(const m of MODES){const r=flagShare(p,m);index.flags[m].push(r===null?-1:r);}
  index.flag.push(index.flags.nobib.at(-1)!);
  const bytes=enc.encode(JSON.stringify(p)+'\n');parts.push(bytes);index.card.push(offset,bytes.length);offset+=bytes.length;
 }
 index.cols=[...cols.keys()];index.areas=[...areas.keys()];index.kws=[...kws.keys()];
 const pack=new Uint8Array(offset);let at=0;for(const b of parts){pack.set(b,at);at+=b.length;}
 return {index,pack};
}
/** JSON with non-ASCII escaped: the Worker then holds the raw text as a one-byte string while parsing (half the memory). */
export function indexJson(index:SearchIndex){return JSON.stringify(index).replace(/[\u0080-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));}
export function validateIndex(x:SearchIndex){
 const n=x?.ids?.length;
 if(x?.v!==2||!Number.isSafeInteger(n)||[x.text,x.col,x.area,x.kw,x.num,x.upd,...BASE_FLAG_MODES.map(m=>x.flags?.[m])].some(a=>!Array.isArray(a)||a.length!==n)||!Array.isArray(x.card)||x.card.length!==2*n)throw new Error('Invalid browse index');
 // Snapshots that lack a mode (e.g. published before the 0.1%-FPR modes, Oct 9, 2026): every paper counts as unscored there.
 for(const m of MODES)if(!Array.isArray(x.flags[m])||x.flags[m].length!==n)x.flags[m]=new Array(n).fill(-1);
 return x;
}
/** Query result in index positions; the caller turns positions into cards. */
export type IndexPage={total:number;page:number;pages:number;scopeTotal:number;areas:[string,number][];topics:[string,number][];collections:string[];collectionCounts:[string,number][];items:number[];selected:number;position:number;previous:number;next:number;related:number[];flagBands:[string,number][];histogram:number[]};
const SORTS=['number','updated','flagged','flagged-asc'] as const;
/** Parsed once per isolate; sort orders are built on first use. */
export class BrowseIndex{
 readonly x:SearchIndex;readonly n:number;readonly byId=new Map<string,number>();private orders=new Map<string,Int32Array>();private bin:Record<FlagMode,Int8Array>;
 // Typed copies of the numeric columns; the JSON arrays are released after loading (a plain array of small
 // arrays costs several times the memory).
 private col:Uint16Array;private area:Int32Array;private kwOff:Int32Array;private kwAll:Int32Array;private flag:Record<FlagMode,Float64Array>;
 constructor(x:SearchIndex){
  this.x=validateIndex(x);this.n=x.ids.length;x.ids.forEach((id,i)=>{if(!this.byId.has(id))this.byId.set(id,i);});
  // 5% histogram bin per paper (0..19), or -1 when unscored or too short.
  this.bin=Object.fromEntries(MODES.map(m=>[m,Int8Array.from(x.flags[m],r=>r<0?-1:flagBin(r))])) as Record<FlagMode,Int8Array>;
  this.flag=Object.fromEntries(MODES.map(m=>[m,Float64Array.from(x.flags[m])])) as Record<FlagMode,Float64Array>;
  this.col=Uint16Array.from(x.col);this.area=Int32Array.from(x.area);
  this.kwOff=new Int32Array(this.n+1);for(let i=0;i<this.n;i++)this.kwOff[i+1]=this.kwOff[i]+x.kw[i].length;
  this.kwAll=new Int32Array(this.kwOff[this.n]);for(let i=0;i<this.n;i++)this.kwAll.set(x.kw[i],this.kwOff[i]);
  x.col=[];x.area=[];x.kw=[];x.flag=[];for(const m of MODES)x.flags[m]=[];
 }
 private hasKw(i:number,k:number){for(let j=this.kwOff[i];j<this.kwOff[i+1];j++)if(this.kwAll[j]===k)return true;return false;}
 private order(sort:string,mode:FlagMode){
  const key=sort.startsWith('flagged')?`${sort}:${mode}`:sort;let o=this.orders.get(key);if(o)return o;
  const x=this.x,idx=Array.from({length:this.n},(_,i)=>i);
  // Positions are title order, so `a-b` breaks ties exactly as the full-row sort's title/id tie-break did.
  if(sort==='number')idx.sort((a,b)=>(x.num[a]??Infinity)-(x.num[b]??Infinity)||x.ids[a].localeCompare(x.ids[b]));
  else if(sort==='updated')idx.sort((a,b)=>x.upd[b]-x.upd[a]||a-b);
  else{const d=sort==='flagged'?-1:1,flag=this.flag[mode];idx.sort((a,b)=>{const ra=flag[a],rb=flag[b];return ra<0?(rb<0?a-b:1):rb<0?-1:d*(ra-rb)||a-b;});}
  o=Int32Array.from(idx);this.orders.set(key,o);return o;
 }
 query(f:BrowseFilters,saved:Set<string>,id:string|null,rawPage:number):IndexPage{
  const x=this.x,n=this.n,terms=f.q.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const col=f.collection?x.cols.indexOf(f.collection):-2,area=f.area?x.areas.indexOf(f.area):-2,tag=f.tag?x.kws.indexOf(keyOf(f.tag)):-2;
  const range=parseFlagRange(f.ai),cut=parseFlagCut(f.cut),mode=asSiteFlagMode(f.mode),bins=this.bin[mode];
  const match=new Uint8Array(n),histogram=new Array(20).fill(0);let unscored=0;
  const colCounts=new Int32Array(x.cols.length),areaCounts=new Int32Array(x.areas.length),topicCounts=new Int32Array(x.kws.length);let scopeTotal=0,total=0;
  const cols=this.col,areas=this.area,kwOff=this.kwOff,kwAll=this.kwAll;
  for(let i=0;i<n;i++){
   const c=cols[i];colCounts[c]++;
   if(col!==-2&&c!==col)continue;
   scopeTotal++;const a=areas[i];
   if(a>=0)areaCounts[a]++;
   if(area===-2||a===area)for(let j=kwOff[i];j<kwOff[i+1];j++)topicCounts[kwAll[j]]++;
   if(area!==-2&&a!==area)continue;
   if(tag!==-2&&!this.hasKw(i,tag))continue;
   if(f.saved&&!saved.has(x.ids[i]))continue;
   if(terms.length&&!terms.every(t=>x.text[i].includes(t)))continue;
   // The cutoff limits everything, histogram included (its bars rescale to the papers kept); the bin filter does not
   // limit the histogram and band counts, so every bin stays selectable.
   const b=bins[i];if(cut&&(b<cut[0]||b>=cut[1]))continue;if(b<0)unscored++;else histogram[b]++;
   if(range&&(range==='none'?b>=0:b<range[0]||b>=range[1]))continue;
   match[i]=1;total++;
  }
  const sort=(SORTS as readonly string[]).includes(f.sort)?f.sort:'title',results:number[]=[];
  if(sort==='title'){for(let i=0;i<n;i++)if(match[i])results.push(i);}
  else for(const i of this.order(sort,mode))if(match[i])results.push(i);
  const pages=Math.max(1,Math.ceil(total/20)),page=Math.min(pages-1,Number.isSafeInteger(rawPage)&&rawPage>0?rawPage:0);
  const selected=id?this.byId.get(id)??-1:-1,position=selected>=0&&match[selected]?results.indexOf(selected):-1;
  // Only entries at or above the limit-th largest count can make the cut, so the name sort stays small (~100k keywords).
  const top=(counts:Int32Array,names:string[],limit=Infinity)=>{
   let floor=1;if(limit<counts.length){const sorted=counts.slice().sort();floor=Math.max(1,sorted[sorted.length-limit]);}
   const out:[string,number][]=[];counts.forEach((v,k)=>{if(v>=floor)out.push([names[k],v]);});
   return out.sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,limit);
  };
  const collectionCounts=top(colCounts,x.cols);
  return {total,page,pages,scopeTotal,areas:top(areaCounts,x.areas),topics:top(topicCounts,x.kws,12),
   collections:collectionCounts.map(c=>c[0]).sort().reverse(),collectionCounts:collectionCounts.sort((a,b)=>a[0].localeCompare(b[0])),
   items:results.slice(page*20,page*20+20),selected,position,previous:position>0?results[position-1]:-1,next:position>=0&&position<results.length-1?results[position+1]:-1,
   related:selected>=0?this.related(selected):[],flagBands:[...FLAG_BANDS.map(([k])=>{const [lo,hi]=parseFlagRange(k) as [number,number];return [k,histogram.slice(lo,hi).reduce((a,c)=>a+c,0)] as [string,number];}),['none',unscored]],histogram};
 }
 /** Same collection; one point for a shared research area, three per shared keyword; top five, ties in title order. */
 related(s:number){
  const keys=new Set(this.kwAll.subarray(this.kwOff[s],this.kwOff[s+1])),out:{i:number;score:number}[]=[];
  for(let i=0;i<this.n;i++){
   if(i===s||this.col[i]!==this.col[s])continue;
   let score=this.area[s]>=0&&this.area[i]===this.area[s]?1:0;for(let j=this.kwOff[i];j<this.kwOff[i+1];j++)if(keys.has(this.kwAll[j]))score+=3;
   if(score>0)out.push({i,score});
  }
  return out.sort((a,b)=>b.score-a.score||a.i-b.i).slice(0,5).map(r=>r.i);
 }
}
export function parseFilters(params:URLSearchParams):BrowseFilters{
 const ai=params.get('ai')||'';
 return {q:(params.get('q')||'').slice(0,300),collection:params.get('collection')??'iclr/2027',area:params.get('area')||'',tag:params.get('tag')||'',sort:params.get('sort')||DEFAULT_SORT,saved:params.get('saved')==='1',ai:parseFlagRange(ai)?ai:'',mode:asSiteFlagMode(params.get('mode')),cut:parseFlagCut(params.get('cut')||'')?params.get('cut')!:''};
}
export function parseSaved(params:URLSearchParams){return new Set((params.get('ids')||'').split(',').filter(id=>/^[a-f0-9]{24}$/.test(id)));}
