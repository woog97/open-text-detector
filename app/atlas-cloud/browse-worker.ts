import {neededCards,queryIndex,resolvePage,type BrowsePage} from '../lib/browse-page.ts';
import {BrowseIndex,buildIndex,type SearchIndex} from '../lib/browse-index.ts';
import type {AtlasPaper} from '../lib/atlas-collection.ts';
/** Part of the edge cache key: bump it whenever the same snapshot gives different results (a sort or filter change), or
 * pages cached by the previous Worker keep their old order for up to a day. v2: average-score sorts and tie-break (Oct 13, 2026). */
const QUERY_VERSION='v2';
import {DEFAULT_SORT} from '../lib/paper-discovery.ts';
type BrowseManifest={revision:string;index_key:string;home_key:string;search_key?:string;cards_key?:string};
/** Cards fetched within this many bytes of each other are read in one range request. */
const MERGE_GAP=65536,MAX_SPAN=1<<20,CARD_CACHE=5000;
/**
 * Requests never await I/O another request started: in Workers that I/O belongs to the starting request, and if that
 * request is cancelled (client disconnect) the promise never settles, hanging every request awaiting it. So the manifest
 * is fetched per request, and a request that finds an index load in progress polls on its own timer for the result,
 * taking the load over if it has run longer than LOAD_STALE (normal loads take 0.5-3 s).
 */
const LOAD_STALE=15000,WAIT_MAX=25000,POLL=50;
type Loaded={revision:string;ix:BrowseIndex;cards:Map<number,AtlasPaper>;pack?:Uint8Array;cardsKey?:string};
/** Immutable publications let the default page bypass index parsing entirely. */
export function createBrowseHandler({loadStale=LOAD_STALE,waitMax=WAIT_MAX}:{loadStale?:number;waitMax?:number}={}){
 let manifest:{data:BrowseManifest;until:number}|undefined;
 let parsed:Loaded|undefined;
 /** The index load in progress in this isolate: which revision, and when it started. */
 let loading:{revision:string;started:number}|undefined;
 async function current(bucket:R2Bucket){
  if(manifest&&manifest.until>Date.now())return manifest.data;
  const o=await bucket.get('indexes/browse/current.json');if(!o)throw new Error('Browse publication unavailable');const data=await o.json<BrowseManifest>();const base=`indexes/browse/${data.revision}`;if(!/^[a-f0-9]{64}$/.test(data.revision)||data.index_key!==`${base}/papers.json`||data.home_key!==`${base}/home.json`||(data.search_key!==undefined&&data.search_key!==`${base}/search.json`)||(data.cards_key!==undefined&&data.cards_key!==`${base}/cards.jsonl`)||!data.search_key!==!data.cards_key)throw new Error('Invalid browse publication');manifest={data,until:Date.now()+15000};return data;
 }
 async function load(bucket:R2Bucket,m:BrowseManifest):Promise<Loaded>{
  if(m.search_key&&m.cards_key){
   const o=await bucket.get(m.search_key);if(!o)throw new Error('Missing browse index');
   return {revision:m.revision,ix:new BrowseIndex(await o.json<SearchIndex>()),cards:new Map(),cardsKey:m.cards_key};
  }
  // Older publication without the compact index: build it from the full rows (heavy; kept only as a fallback).
  const o=await bucket.get(m.index_key);if(!o)throw new Error('Missing browse index');
  const {index,pack}=buildIndex((await o.json<{items:AtlasPaper[]}>()).items);
  return {revision:m.revision,ix:new BrowseIndex(index),cards:new Map(),pack};
 }
 async function cards(bucket:R2Bucket,l:Loaded,positions:number[]){
  const card=l.ix.x.card,missing=positions.filter(i=>!l.cards.has(i)).sort((a,b)=>card[2*a]-card[2*b]);
  const decoder=new TextDecoder(),read=(bytes:Uint8Array,start:number,i:number)=>JSON.parse(decoder.decode(bytes.subarray(card[2*i]-start,card[2*i]-start+card[2*i+1]))) as AtlasPaper;
  if(l.pack)for(const i of missing)l.cards.set(i,read(l.pack,0,i));
  else{
   // Group nearby cards so a title-sorted page (adjacent in the pack) is a single read.
   const groups:number[][]=[];
   for(const i of missing){const g=groups.at(-1);if(g&&card[2*i]-(card[2*g.at(-1)!]+card[2*g.at(-1)!+1])<=MERGE_GAP&&card[2*i]+card[2*i+1]-card[2*g[0]]<=MAX_SPAN)g.push(i);else groups.push([i]);}
   await Promise.all(groups.map(async g=>{
    const start=card[2*g[0]],end=card[2*g.at(-1)!]+card[2*g.at(-1)!+1];
    const o=await bucket.get(l.cardsKey!,{range:{offset:start,length:end-start}});if(!o)throw new Error('Missing browse cards');
    const bytes=new Uint8Array(await o.arrayBuffer());for(const i of g)l.cards.set(i,read(bytes,start,i));
   }));
  }
  while(l.cards.size>CARD_CACHE)l.cards.delete(l.cards.keys().next().value!);
  return l.cards;
 }
 return async function(request:Request,bucket:R2Bucket,ctx?:ExecutionContext){
  const url=new URL(request.url),params=url.searchParams,started=Date.now();
  const m=await current(bucket);
  const canonical=new URLSearchParams();
  for(const key of ['collection','q','area','tag','ai','cut','mode','sort','page','paper','saved','ids']){const value=params.get(key);if(value)canonical.set(key,value);}
  // Empty collection means all collections and must remain distinct from the default.
  if(params.has('collection')&&params.get('collection')==='')canonical.set('collection','');
  canonical.sort();
  const isHome=(!params.has('collection')||params.get('collection')==='iclr/2027')&&(!params.get('sort')||params.get('sort')===DEFAULT_SORT)&&[...params.keys()].every(k=>['collection','sort','page'].includes(k))&&(!params.get('page')||params.get('page')==='0');
  const privateList=params.get('saved')==='1'||params.has('ids');
  const key=new Request(`${url.origin}/backend/v1/browse-cache/${m.revision}/${QUERY_VERSION}?${canonical}`);
  const cache=typeof caches!=='undefined'?(caches as CacheStorage & {default:Cache}).default:undefined;
  if(!privateList&&cache){const hit=await cache.match(key);if(hit){const response=new Response(request.method==='HEAD'?null:hit.body,hit);response.headers.set('Cache-Control','public, max-age=15');response.headers.set('Server-Timing',`edge;desc="edge cache hit";dur=${Date.now()-started}`);return response;}}
  let response:Response,timing='home';
  const headers={'Content-Type':'application/json','Cache-Control':privateList?'private, no-store':'public, max-age=15','X-Content-Type-Options':'nosniff','X-Browse-Revision':m.revision};
  if(isHome){const o=await bucket.get(m.home_key);if(!o)throw new Error('Missing homepage');response=new Response(o.body,{headers});}
  else{
   const cold=parsed?.revision!==m.revision,loadStart=Date.now();
   if(cold){
    // One index load per isolate (two copies of the index would not fit in memory): others wait for its result.
    while(parsed?.revision!==m.revision){
     if(!loading||Date.now()-loading.started>loadStale){
      const mine={revision:m.revision,started:Date.now()};
      loading=mine;parsed=undefined; // drop the old index first
      try{const l=await load(bucket,m);if(loading===mine||(parsed as Loaded|undefined)?.revision!==m.revision)parsed=l; /* parsed may change while awaiting */}
      finally{if(loading===mine)loading=undefined;}
     }else{
      if(Date.now()-loadStart>waitMax)return new Response('Index loading; retry',{status:503,headers:{'Retry-After':'1'}});
      await new Promise(r=>setTimeout(r,POLL));
      // A load of another (older) revision finishing leaves parsed stale; the loop then starts this revision's load.
     }
    }
   }
   const l=parsed!,computeStart=Date.now(),r=queryIndex(l.ix,params),queried=Date.now();
   const page:BrowsePage=resolvePage(l.ix,r,await cards(bucket,l,neededCards(r)));page.revision=m.revision;response=Response.json(page,{headers});
   timing=`${cold?`index;desc="index load";dur=${computeStart-loadStart}, `:''}compute;dur=${queried-computeStart}, cards;dur=${Date.now()-queried}`;
  }
  response.headers.set('Server-Timing',`${timing==='home'?'home;desc="home snapshot"':timing}, total;dur=${Date.now()-started}`);
  if(!privateList&&cache&&ctx){const copy=response.clone();copy.headers.delete('Server-Timing');copy.headers.set('Cache-Control','public, max-age=86400');ctx.waitUntil(cache.put(key,copy));}
  return request.method==='HEAD'?new Response(null,response):response;
 };
}
