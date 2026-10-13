import type {AtlasPaper} from './atlas-collection.ts';
import {BrowseIndex,buildIndex,parseFilters,parseSaved,type IndexPage} from './browse-index.ts';
export type BrowsePage={items:AtlasPaper[];total:number;page:number;pages:number;scopeTotal:number;areas:[string,number][];topics:[string,number][];collections:string[];collectionCounts:[string,number][];selected?:AtlasPaper;position:number;previous?:string;next?:string;related:AtlasPaper[];flagBands:[string,number][];histogram:number[];revision?:string};
export function queryIndex(ix:BrowseIndex,params:URLSearchParams):IndexPage{
 const raw=Number(params.get('page'));
 return ix.query(parseFilters(params),parseSaved(params),params.get('paper'),raw);
}
/** Index positions whose cards a page displays. */
export function neededCards(r:IndexPage){return [...new Set([...r.items,...r.related,...(r.selected>=0?[r.selected]:[])])];}
export function resolvePage(ix:BrowseIndex,r:IndexPage,cards:Map<number,AtlasPaper>):BrowsePage{
 const card=(i:number)=>{const c=cards.get(i);if(!c)throw new Error('Missing browse card');return c;};
 return {items:r.items.map(card),total:r.total,page:r.page,pages:r.pages,scopeTotal:r.scopeTotal,areas:r.areas,topics:r.topics,collections:r.collections,collectionCounts:r.collectionCounts,
  selected:r.selected>=0?card(r.selected):undefined,position:r.position,previous:r.previous>=0?ix.x.ids[r.previous]:undefined,next:r.next>=0?ix.x.ids[r.next]:undefined,related:r.related.map(card),flagBands:r.flagBands,histogram:r.histogram};
}
/** Whole-list version (publisher home snapshot, tests): same index and query code as the Worker. */
export function browsePage(papers:AtlasPaper[],params:URLSearchParams):BrowsePage{
 const ix=new BrowseIndex(buildIndex(papers).index),byId=new Map(papers.map(p=>[p.id,p]));
 const r=queryIndex(ix,params);
 return resolvePage(ix,r,new Map(neededCards(r).map(i=>[i,byId.get(ix.x.ids[i])!])));
}
