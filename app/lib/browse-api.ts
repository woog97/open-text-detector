import type {BrowsePage} from './browse-page.ts';
import {DEFAULT_FLAG_MODE,type BrowseFilters} from './paper-discovery.ts';
export function browseParams(filters:BrowseFilters,page:number,selected:string,saved:Set<string>){
 const p=new URLSearchParams({collection:filters.collection,sort:filters.sort,page:String(page)});
 for(const key of ['q','area','tag','ai','cut'] as const)if(filters[key])p.set(key,filters[key]);
 // The default counting mode is left out, so the default URL stays the cached homepage snapshot.
 if(filters.mode&&filters.mode!==DEFAULT_FLAG_MODE)p.set('mode',filters.mode);
 if(selected)p.set('paper',selected);
 if(filters.saved){p.set('saved','1');p.set('ids',[...saved].sort().join(','));}
 return p;
}
export async function loadBrowse(params:string,signal:AbortSignal,onPreview?:(data:BrowsePage)=>void):Promise<BrowsePage>{
 const query=new URLSearchParams(params),privateList=query.get('saved')==='1';
 const home=query.get('collection')==='iclr/2027'&&query.get('sort')==='title'&&query.get('page')==='0'&&![...query.keys()].some(k=>!['collection','sort','page'].includes(k));
 const key=home?'atlas-browse-home-v1':'atlas-browse-last-v1';
 if(!privateList&&onPreview&&typeof localStorage!=='undefined'){
  try{const entry=JSON.parse(localStorage.getItem(key)||'null');if(entry?.params===params&&Array.isArray(entry.data?.items)&&Array.isArray(entry.data?.areas)&&Array.isArray(entry.data?.topics)&&Array.isArray(entry.data?.collections)&&Number.isSafeInteger(entry.data?.total))onPreview(entry.data);}catch{/* Browser storage is optional. */}
 }

 // A cold Worker or a publication switch can fail one request; retry twice before reporting an error.
 let response:Response|undefined;
 for(let attempt=0;;attempt++){
  try{response=await fetch('/backend/v1/browse?'+params,{signal});}catch(error){if(signal.aborted||attempt>=2)throw error;}
  if(response&&(response.ok||response.status<500||attempt>=2))break;
  await new Promise<void>((resolve,reject)=>{const t=setTimeout(resolve,400*3**attempt);signal.addEventListener('abort',()=>{clearTimeout(t);reject(new DOMException('Aborted','AbortError'));},{once:true});});
 }
 if(!response!.ok)throw new Error('The paper library is unavailable. Please retry.');
 const data=await response!.json() as BrowsePage;
 if(!Array.isArray(data.items)||!Array.isArray(data.areas)||!Array.isArray(data.topics)||!Array.isArray(data.collections)||!Number.isSafeInteger(data.total))throw new Error('Invalid browse page');
 if(!privateList&&typeof localStorage!=='undefined'){try{localStorage.setItem(key,JSON.stringify({params,data}));}catch{/* Storage failure must not break browsing. */}}
 return data;
}
