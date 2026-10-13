"use client";
import {lazy,Suspense,useDeferredValue,useEffect,useMemo,useRef,useState,type CSSProperties,type FormEvent,type ReactNode} from 'react';
import {Bookmark,ArrowLeft,ArrowRight,Search,ExternalLink,BookOpen,X,FileText,Link2,Quote} from 'lucide-react';
import {loadBrowse,browseParams} from '../lib/browse-api';
import type {BrowsePage} from '../lib/browse-page';
import type {AtlasPaper} from '../lib/atlas-collection';
import {DEFAULT_SORT,FLAG_BANDS,FLAG_FPRS,LEGACY_FLAG_FPRS,FLAG_MODES,DEFAULT_FLAG_MODE,asSiteFlagMode,flagCounts,flagCutLabel,flagModeOf,flagRangeLabel,flagShare,fprLabel,parseFlagCut,parseFlagRange,thresholdText,type BrowseFilters,type FlagMode} from '../lib/paper-discovery';
import FlagHistogram from './flag-histogram';
import {MIN_SENTENCES,modelLabel} from '../lib/paper-scores';
import ReaderErrorBoundary from './reader-error-boundary';
import {SiteHeader,SiteFooter,SUBMISSIONS_HREF} from './site-chrome';
import {BETA} from '../lib/ui-mode';
import './paper-browser.css';
/** OpenReview abstracts keep author markdown (**bold**) and hard line breaks; show bold as bold and rejoin wrapped lines. */
function cleanAbstract(t:string){
 const text=t.replace(/\r/g,'').replace(/([^\n])\n(?!\n)/g,'$1 ').replace(/[ \t]{2,}/g,' ');
 return text.split(/\*\*([^*\n]+)\*\*/).map((part,i)=>i%2?<b key={i}>{part}</b>:part);
}
const PDFReader=lazy(()=>import('./pdf-reader'));
const defaults:BrowseFilters={q:'',collection:'iclr/2027',area:'',tag:'',saved:false,sort:DEFAULT_SORT,ai:'',mode:DEFAULT_FLAG_MODE,cut:''};
const collectionName=(s:string)=>s.replace('/', ' ').toUpperCase();
async function api(path:string,init?:RequestInit){const r=await fetch('/backend'+path,init);if(!r.ok)throw new Error('Unable to load this paper. Please try again.');return r.json();}
type Detail={abstract?:string;bibtex?:string;license?:string;venue?:string};
// Shared across views: hovering a row starts these loads, so opening a paper rarely waits.
const details=new Map<string,Promise<Detail>>();
// The reader's score data (~300 KB gzipped) is requested as soon as a paper link opens, in parallel with the paper
// list, instead of after the list arrives and the reader mounts. The reader takes the request once.
const readerData=new Map<string,Promise<unknown>>();
function prefetchReader(id:string){if(!/^[a-f0-9]{24}$/.test(id)||readerData.has(id))return;const p=api(`/v1/pdf-reader/${id}`);readerData.set(id,p);if(readerData.size>3)readerData.delete(readerData.keys().next().value!);p.catch(()=>readerData.delete(id));}
function readerApi(path:string,init?:RequestInit){const m=path.match(/^\/v1\/pdf-reader\/([a-f0-9]{24})$/),p=m&&readerData.get(m[1]);if(p){readerData.delete(m![1]);return p;}return api(path,init);}
function loadDetail(id:string){let p=details.get(id);if(!p){p=api(`/v1/pdf-reader/${id}/metadata`) as Promise<Detail>;details.set(id,p);p.catch(()=>details.delete(id));}return p;}
let readerChunk:Promise<unknown>|undefined;
const warmReader=()=>readerChunk??=import('./pdf-reader').catch(()=>{readerChunk=undefined;});
export default function PaperBrowser(){
 const [data,setData]=useState<BrowsePage|null>(null),[busy,setBusy]=useState(false),[initialized,setInitialized]=useState(false);
 const [loaded,setLoaded]=useState(false),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
 const [filters,setFilters]=useState<BrowseFilters>(defaults),[selected,setSelected]=useState(''),[page,setPage]=useState(0);
 const [summarySlot,setSummarySlot]=useState<HTMLElement|null>(null);
 const [density,setDensity]=useState<'standard'|'compact'>(()=>{try{return BETA&&localStorage.getItem('atlas-density')==='compact'?'compact':'standard';}catch{return 'standard';}});
 function pickDensity(d:'standard'|'compact'){setDensity(d);try{localStorage.setItem('atlas-density',d);}catch{/* per-visit only */}}
 const [saved,setSaved]=useState<Set<string>>(new Set()),[storageNotice,setStorageNotice]=useState('');
 const [detail,setDetail]=useState<Detail|null>(null),[detailError,setDetailError]=useState(''),[detailAttempt,setDetailAttempt]=useState(0),[copied,setCopied]=useState('');
 const deferred=useDeferredValue(filters),typing=useRef(false),[searchQ,setSearchQ]=useState('');
 // Typing waits for a short pause; any other change to the query applies immediately.
 useEffect(()=>{if(!typing.current){setSearchQ(deferred.q);return;}typing.current=false;const t=setTimeout(()=>setSearchQ(deferred.q),220);return()=>clearTimeout(t);},[deferred.q]);
 const query=useMemo(()=>({...deferred,q:searchQ}),[deferred,searchQ]);
 useEffect(()=>{const sync=()=>{const p=new URLSearchParams(location.search);setFilters({q:p.get('q')||'',collection:p.get('collection')??'iclr/2027',area:p.get('area')||'',tag:p.get('tag')||'',saved:p.get('saved')==='1',sort:p.get('sort')||DEFAULT_SORT,ai:p.get('ai')||'',mode:asSiteFlagMode(p.get('mode')),cut:parseFlagCut(p.get('cut')||'')?p.get('cut')!:''});setSelected(p.get('paper')||'');if(p.get('paper')){prefetchReader(p.get('paper')!);warmReader();loadDetail(p.get('paper')!).catch(()=>{});}setPage(Math.max(0,Number(p.get('page'))||0));};sync();window.addEventListener('popstate',sync);try{const s=JSON.parse(localStorage.getItem('atlas-reading-list')||'[]');if(Array.isArray(s))setSaved(new Set(s.filter(v=>typeof v==='string')));}catch{setStorageNotice('Reading list storage is unavailable in this browser. Saves will last for this visit.');}setInitialized(true);return()=>window.removeEventListener('popstate',sync);},[]);
 const queryParams=useMemo(()=>browseParams(query,page,selected,saved).toString(),[query,page,selected,saved]);
 const prefetched=useRef(new Set<string>());
 function prefetch(id:string){warmReader();loadDetail(id).catch(()=>{});const url=browseParams(query,data?.page||0,id,saved).toString();if(query.saved||prefetched.current.has(url))return;prefetched.current.add(url);fetch('/backend/v1/browse?'+url,{priority:'low'} as RequestInit).catch(()=>{});}
 // Sweeping the pointer down the list should not start a load per row: wait until it rests on one.
 const hoverTimer=useRef<ReturnType<typeof setTimeout>>(undefined);
 const cancelPrefetch=()=>clearTimeout(hoverTimer.current);
 const hoverPrefetch=(id:string)=>{cancelPrefetch();hoverTimer.current=setTimeout(()=>prefetch(id),150);};
 // Wait for the deferred filters to catch up, so a link's own sort or filters are fetched first, not the defaults.
 const ready=deferred===filters;
 useEffect(()=>{if(!initialized||!ready)return;const abort=new AbortController();let active=true;setBusy(true);const timer=setTimeout(()=>abort.abort(),30000);loadBrowse(queryParams,abort.signal,preview=>{if(active){setData(preview);setLoaded(true);}}).then(next=>{if(active){setData(next);setLoaded(true);setError('');}}).catch(()=>{if(active)setError('Could not load the library. Please retry.');}).finally(()=>{clearTimeout(timer);if(active)setBusy(false);});return()=>{active=false;abort.abort();clearTimeout(timer);};},[queryParams,attempt,initialized,ready]);
 // Refresh the list now and then, but never under an open paper: a new page object re-renders the reader.
 const reading=useRef(false);useEffect(()=>{reading.current=!!selected;},[selected]);
 useEffect(()=>{const refresh=setInterval(()=>{if(document.visibilityState==='visible'&&!reading.current)setAttempt(a=>a+1);},60000);return()=>clearInterval(refresh);},[]);
 useEffect(()=>{setDetail(null);setDetailError('');setCopied('');if(!selected)return;let live=true;if(detailAttempt)details.delete(selected);loadDetail(selected).then(data=>{if(live)setDetail(data);}).catch(()=>{if(live)setDetailError('The full abstract could not be loaded.');});return()=>{live=false;};},[selected,detailAttempt]);
 function navigate(next:BrowseFilters,id='',n=0,replace=false){if(id)prefetchReader(id);setFilters(next);setSelected(id);setPage(n);const p=new URLSearchParams();p.set('collection',next.collection);for(const k of ['q','area','tag','sort','ai','cut'] as const)if(next[k])p.set(k,next[k]);if(next.mode&&next.mode!==DEFAULT_FLAG_MODE)p.set('mode',next.mode);if(next.saved)p.set('saved','1');if(id)p.set('paper',id);if(n)p.set('page',String(n));history[replace?'replaceState':'pushState']({},'',`/?${p}`);if(!replace)window.scrollTo(0,0);}
 function change(patch:Partial<BrowseFilters>){navigate({...filters,...patch});}
 function toggle(id:string){const next=new Set(saved);if(next.has(id))next.delete(id);else next.add(id);setSaved(next);try{localStorage.setItem('atlas-reading-list',JSON.stringify([...next]));}catch{setStorageNotice('Could not save to this browser. Your reading list will last for this visit.');}}
 const wide=typeof window==='undefined'||window.matchMedia('(min-width: 901px)').matches;
 // While reading, warm the next paper so stepping through results is instant.
 useEffect(()=>{if(selected&&data?.next)prefetch(data.next);},[selected,data?.next]);// eslint-disable-line react-hooks/exhaustive-deps
 const searchInput=useRef<HTMLInputElement>(null);
 useEffect(()=>{const onKey=(e:KeyboardEvent)=>{const t=e.target as HTMLElement|null;if(e.key!=='/'||e.metaKey||e.ctrlKey||e.altKey||t?.closest('input,textarea,select,[contenteditable="true"]'))return;if(!searchInput.current)return;e.preventDefault();searchInput.current.focus();searchInput.current.select();};window.addEventListener('keydown',onKey);return()=>window.removeEventListener('keydown',onKey);},[]);
 const results=data?.items||[],areas=data?.areas||[],topics=data?.topics||[];
 const active=data?.selected?.id===selected?data.selected:undefined,position=data?.position??-1,lastPage=(data?.pages||1)-1,currentPage=data?.page||0;
 const related=data?.related||[];
 // The reader reloads its paper list when this array changes identity, so only rebuild it when the contents change.
 const paperKey=active?.id+'|'+results.map(p=>p.id).join(',');
 const papers=useMemo(()=>active?[...results.filter(p=>p.id!==active.id),active]:results,[paperKey]);// eslint-disable-line react-hooks/exhaustive-deps
 const total=data?.total||0,scopeTotal=data?.scopeTotal||0,largestArea=areas[0]?.[1]||1;
 const mode=asSiteFlagMode(filters.mode),bands=new Map(data?.flagBands||[]),bandCount=(k:string)=>bands.get(k)||0,flagTotal=FLAG_BANDS.reduce((n,[k])=>n+bandCount(k),0),largestBand=Math.max(1,...FLAG_BANDS.map(([k])=>bandCount(k)));
 const terms=useMemo(()=>[...new Set(searchQ.toLowerCase().split(/\s+/).filter(t=>t.length>1))].sort((a,b)=>b.length-a.length),[searchQ]);
 const mark=(text:string)=>highlight(text,terms);
 async function copy(value:string,label:string){try{await navigator.clipboard.writeText(value);setCopied(label);}catch{setCopied('Copy unavailable. Select and copy the text below.');}}
 function saveButton(p:AtlasPaper,label=true){const on=saved.has(p.id);return <button className={`pb-save${on?' is-saved':''}`} aria-label={`${on?'Remove from':'Add to'} reading list: ${p.title}`} aria-pressed={on} onClick={()=>toggle(p.id)}><Bookmark size={16}/>{label&&<span>{on?'Saved':'Save'}</span>}</button>;}
 const paperHref=(id:string)=>`/?${new URLSearchParams({...Object.fromEntries(new URLSearchParams(location.search)),paper:id})}`;
 const openreview=(p:AtlasPaper)=>p.forum_id?`https://openreview.net/forum?id=${encodeURIComponent(p.forum_id)}`:'';
 const heading=filters.saved?'Your reading list':filters.collection?`${collectionName(filters.collection)} ${filters.collection.startsWith('iclr/')?'submissions':'papers'}`:'All papers';
 return <div className="site pb">
 <SiteHeader page={filters.saved?'saved':'discover'} savedCount={saved.size} discover={{href:SUBMISSIONS_HREF,onClick:()=>navigate({...defaults})}} saved={{href:'/?saved=1',onClick:()=>navigate({...filters,saved:true},'')}}/>
 {storageNotice&&<p role="status" className="pb-notice">{storageNotice}</p>}{error&&<p role="alert" className="pb-notice">{error} <button onClick={()=>setAttempt(a=>a+1)}>Retry</button></p>}
 {selected?<main className="pb-reader">
  <aside className="pb-dossier" aria-label="Paper details">
   <div className="pb-dossier-nav"><button onClick={()=>navigate(filters,'',page)}><ArrowLeft size={15}/> All results</button><div className="pb-stepper"><button aria-label="Previous paper" disabled={busy||!data?.previous} onClick={()=>navigate(filters,data!.previous!,page)}><ArrowLeft size={15}/></button><span>{position>=0?<><b>{(position+1).toLocaleString()}</b> / {total.toLocaleString()}</>:'Paper'}</span><button aria-label="Next paper" disabled={busy||!data?.next} onClick={()=>navigate(filters,data!.next!,page)}><ArrowRight size={15}/></button></div></div>
   {(!loaded||busy&&!active)?<DossierSkeleton/>:!active?<div className="pb-empty"><h3>This paper isn’t in the uploaded collection.</h3><p>It may have been withdrawn or not uploaded yet.</p><button className="site-primary" onClick={()=>navigate(filters)}>Browse papers</button></div>:<>
    <p className="site-kicker">{collectionName(active.collection)}{active.number?<> · <span className="site-num">#{active.number}</span></>:null}</p>
    <h1>{tidy(active.title)}</h1>
    <p className="pb-status">{detail?.venue||'Research paper'}{active.collection==='iclr/2027'?' · Anonymous · Under review':''}</p>
    {!BETA&&<FlagSummary p={active} mode={mode}/>}
    <div className="pb-actions">{saveButton(active)}<a className="site-primary" href={`/backend/v1/pdf-reader/${active.id}/file`} target="_blank" rel="noreferrer"><FileText size={15}/> Open PDF</a>{active.forum_id&&<a href={openreview(active)} target="_blank" rel="noreferrer">OpenReview <ExternalLink size={13}/></a>}</div>
    {BETA&&<div className="pb-model-slot" ref={setSummarySlot}/>}
    {active.tldr?.trim()&&<p className="pb-tldr"><span>TL;DR</span>{active.tldr}</p>}
    <details className="pb-fold" open={wide}><summary>Abstract</summary><p className="pb-abstract">{cleanAbstract(detail?.abstract||active.abstract_preview||'No abstract available.')}{!detail&&active.abstract_preview?'…':''}</p>{detailError&&<p role="status" className="pb-muted">{detailError} <button className="site-link" onClick={()=>setDetailAttempt(a=>a+1)}>Retry</button></p>}</details>
    {(active.primary_area||!!active.keywords?.length)&&<div className="pb-fold-static"><h2>Filed under</h2><div className="pb-chips">{active.primary_area&&<button className="pb-chip is-area" onClick={()=>navigate({...filters,area:active.primary_area!,tag:''})}>{active.primary_area}</button>}{active.keywords?.map(k=><button className="pb-chip" key={k} onClick={()=>navigate({...filters,tag:k})}>{k}</button>)}</div></div>}
    <div className="pb-tools"><button onClick={()=>copy(location.href,'Link copied')}><Link2 size={14}/> Copy link</button></div>
    {detail?.bibtex&&<details className="pb-fold"><summary><Quote size={13}/> Cite this submission</summary><pre>{detail.bibtex}</pre><button className="site-link" onClick={()=>copy(detail.bibtex!,'Citation copied')}>Copy BibTeX</button></details>}
    <p role="status" className="pb-copied">{copied}</p>
    {related.length>0&&<section className="pb-related"><h2>Keep exploring</h2><p className="pb-muted">Shares keywords or a research area with this paper.</p><ol>{related.map(p=><li key={p.id}><a href={paperHref(p.id)} onClick={e=>{e.preventDefault();navigate(filters,p.id,page);}}>{p.number&&<span className="site-num">#{p.number}</span>}<span>{tidy(p.title)}</span></a></li>)}</ol></section>}
   </>}
  </aside>
  <section className="pb-sheet" aria-label="PDF">{active&&<ReaderErrorBoundary key={selected} onBack={()=>navigate(filters)}><Suspense fallback={<p className="pb-sheet-wait">Opening PDF…</p>}><PDFReader availablePapers={papers} api={readerApi} initialPaperId={selected} embedded hosted beta={BETA} summarySlot={BETA?summarySlot:null} flagMode={mode} onFlagModeChange={m=>navigate({...filters,mode:m},selected,page,true)}/></Suspense></ReaderErrorBoundary>}</section>
 </main>:<main className="pb-main">
  <section className="pb-hero">
   <p className="site-kicker">{filters.saved?`${saved.size} saved in this browser`:loaded?`${scopeTotal.toLocaleString()} papers · public OpenReview metadata`:'Opening the library…'}</p>
   <h1>{heading}</h1>
   <p className="pb-lede">{filters.saved?'Papers you saved in this browser. No account needed.':'Search, filter by research area, and read the PDF.'}</p>
   <div className="pb-search"><Search size={20} aria-hidden/><input ref={searchInput} type="search" aria-label="Search papers" placeholder="Search titles, keywords, TL;DRs or submission numbers" value={filters.q} onChange={e=>{typing.current=true;navigate({...filters,q:e.target.value},'',0,true);}}/>{filters.q?<button aria-label="Clear search" onClick={()=>change({q:''})}><X size={18}/></button>:<kbd aria-hidden>/</kbd>}</div>
   {topics.length>0&&<div className="pb-topics"><span>{filters.area?'Common in this area':'Popular keywords'}</span>{topics.map(([tag,count])=><button aria-pressed={filters.tag===tag} className={filters.tag===tag?'is-on':''} key={tag} onClick={()=>change({tag:filters.tag===tag?'':tag})}>{tag}<small>{count.toLocaleString()}</small></button>)}</div>}
  </section>
  <div className="pb-layout">
   <aside className="pb-sidebar" aria-label="Filters">
    <label className="pb-field">Collection<select aria-label="Collection" value={filters.collection} onChange={e=>change({collection:e.target.value,area:'',tag:''})}><option value="">All collections</option>{(data?.collections||[]).map(c=><option key={c} value={c}>{collectionName(c)} ({(data?.collectionCounts?.find(([name])=>name===c)?.[1]||0).toLocaleString()})</option>)}</select></label>
    <label className="pb-field pb-mobile-area">Research area<select aria-label="Research area" value={filters.area} onChange={e=>change({area:e.target.value,tag:''})}><option value="">All research areas</option>{areas.map(([area,count])=><option key={area} value={area}>{area} ({count.toLocaleString()})</option>)}</select></label>
    <label className="pb-field pb-mobile-area">Flagged as AI<select aria-label="Share of sentences flagged as AI" value={filters.ai} onChange={e=>change({ai:e.target.value})}><option value="">Any share</option>{FLAG_BANDS.map(([k,label])=><option key={k} value={k}>{label} ({bandCount(k).toLocaleString()})</option>)}<option value="none">Not scored ({bandCount('none').toLocaleString()})</option></select></label>
    {flagTotal>0&&<><h2 title={flagHelp(mode)}>Flagged as AI <span className="pb-h2-note">share of sentences</span></h2>
    <div className="pb-areas pb-bands">
     <button aria-pressed={!filters.ai} className={!filters.ai?'is-on':''} onClick={()=>change({ai:''})}><span>Any share</span><b>{(flagTotal+bandCount('none')).toLocaleString()}</b></button>
     {FLAG_BANDS.map(([k,label])=><button aria-pressed={filters.ai===k} className={`${filters.ai===k?'is-on':''} band-${bandTone(k)}`} key={k} onClick={()=>change({ai:filters.ai===k?'':k})} style={{'--share':`${Math.max(2,bandCount(k)/largestBand*100)}%`} as CSSProperties}><span>{label}</span><b>{bandCount(k).toLocaleString()}</b><i aria-hidden/></button>)}
     <button aria-pressed={filters.ai==='none'} className={filters.ai==='none'?'is-on':''} onClick={()=>change({ai:filters.ai==='none'?'':'none'})}><span>Not scored</span><b>{bandCount('none').toLocaleString()}</b></button>
    </div></>}
    <h2>Research areas</h2>
    <div className="pb-areas">
     <button aria-pressed={!filters.area} className={!filters.area?'is-on':''} onClick={()=>change({area:'',tag:''})}><span>All areas</span><b>{scopeTotal.toLocaleString()}</b></button>
     {areas.map(([area,count])=><button aria-pressed={filters.area===area} className={filters.area===area?'is-on':''} key={area} onClick={()=>change({area,tag:''})} style={{'--share':`${Math.max(2,count/largestArea*100)}%`} as CSSProperties}><span>{area}</span><b>{count.toLocaleString()}</b><i aria-hidden/></button>)}
    </div>
    {!areas.length&&loaded&&<p className="pb-muted">No research areas are available for this collection.</p>}
    <p className="pb-sidebar-note"><BookOpen size={15}/> Your reading list stays in this browser.</p>
   </aside>
   <section className="pb-results" aria-busy={busy} aria-label="Results">
    <div className="pb-results-bar"><h2>{filters.saved?'Saved':'Papers'} <span className="site-num">{loaded?total.toLocaleString():'—'}</span></h2>
     {(filters.area||filters.tag||filters.saved||filters.q||filters.ai||filters.cut)&&<div className="pb-active">{filters.q&&<button onClick={()=>change({q:''})}>“{filters.q}” <X size={12}/></button>}{filters.area&&<button onClick={()=>change({area:''})}>{filters.area} <X size={12}/></button>}{filters.tag&&<button onClick={()=>change({tag:''})}>{filters.tag} <X size={12}/></button>}{filters.ai&&<button onClick={()=>change({ai:''})}>{filters.ai==='none'?'Not scored':`${flagRangeLabel(filters.ai)} flagged`} <X size={12}/></button>}{filters.cut&&<button onClick={()=>change({cut:''})}>{flagCutLabel(filters.cut)} flagged <X size={12}/></button>}{filters.saved&&<button onClick={()=>change({saved:false})}>Saved only <X size={12}/></button>}<button className="site-link" onClick={()=>navigate({...defaults,collection:filters.collection})}>Clear all</button></div>}
     {BETA&&<div className="pb-density" role="group" aria-label="List density">{(["standard","compact"] as const).map(d=><button key={d} type="button" aria-pressed={density===d} onClick={()=>pickDensity(d)}>{d==="standard"?"Standard":"Compact"}</button>)}</div>}
     <label className="pb-sort">Sort<select aria-label="Sort papers" value={filters.sort} onChange={e=>change({sort:e.target.value})}><option value="title">Title A–Z</option><option value="updated">Recently updated</option><option value="number">Submission number</option><option value="flagged">Most flagged as AI</option><option value="flagged-asc">Least flagged as AI</option></select></label>
     {busy&&loaded&&<span className="pb-progress" role="status" aria-label="Updating results"/>}
    </div>
    {loaded&&data?.histogram&&<FlagHistogram histogram={data.histogram} ai={filters.ai} cut={filters.cut||''} unscored={bandCount('none')} onSelect={ai=>change({ai})} help={flagHelp(mode)} controls={<><ModeToggles mode={mode} onChange={m=>change({mode:m})}/><CutControl cut={filters.cut||''} onChange={cut=>change({cut,ai:aiInsideCut(filters.ai,cut)?filters.ai:''})}/></>}/>}
    {!loaded?<ol className="pb-list" aria-hidden>{Array.from({length:6},(_,i)=><li key={i} className="pb-row is-ghost"><span className="pb-gutter"/><div><i/><i/><i/></div></li>)}</ol>:!total?<div className="pb-empty"><BookOpen size={28}/><h3>{filters.saved?(saved.size?'No saved papers match these filters':'Nothing saved yet'):'No papers match'}</h3><p>{filters.saved?'Use Save on any paper to keep it here.':'Try fewer words, or another research area.'}</p><button className="site-primary" onClick={()=>navigate({...defaults,collection:filters.saved?filters.collection:''})}>{filters.saved?'Browse papers':'Search all collections'}</button></div>:
    <ol className={`pb-list${density==="compact"?" is-compact":""}`} start={currentPage*20+1}>{results.map(p=><li key={p.id} className="pb-row" onPointerEnter={()=>hoverPrefetch(p.id)} onPointerLeave={cancelPrefetch} onTouchStart={()=>prefetch(p.id)} onFocus={()=>prefetch(p.id)}>
     <span className="pb-gutter" aria-label={p.number?`Submission ${p.number}`:undefined}>{p.number?String(p.number).padStart(5,'0'):'—'}</span>
     <div className="pb-row-body">
      <h3><a href={paperHref(p.id)} onClick={e=>{e.preventDefault();navigate(filters,p.id,currentPage);}}>{mark(tidy(p.title))}</a></h3>
      <p className={`pb-summary${p.tldr?.trim()?" is-tldr":""}`}>{(p.tldr?.trim()||p.abstract_preview)&&<span className="pb-src">{p.tldr?.trim()?'TL;DR':'From the abstract'}</span>}{mark(p.tldr?.trim()||p.abstract_preview||'Open the PDF to read this paper.')}{!p.tldr?.trim()&&p.abstract_preview?'…':''}</p>
      <div className="pb-row-meta">{p.primary_area&&<button className="pb-area" onClick={()=>change({area:p.primary_area!,tag:''})}>{p.primary_area}</button>}{p.keywords?.slice(0,4).map(k=><button key={k} className="pb-kw" onClick={()=>change({tag:k})}>{mark(k)}</button>)}</div>
     </div>
     <div className="pb-row-side"><FlagBadge p={p} mode={mode}/>{saveButton(p,false)}{p.forum_id&&<a href={openreview(p)} target="_blank" rel="noreferrer" aria-label={`OpenReview forum: ${p.title}`}>OpenReview <ExternalLink size={12}/></a>}<span>{(p.bytes/1e6).toFixed(1)} MB{p.classified?' · scored':''}</span></div>
    </li>)}</ol>}
    {total>0&&lastPage>0&&<nav className="pb-pagination" aria-label="Pages"><button disabled={!currentPage} onClick={()=>navigate(filters,'',currentPage-1)}><ArrowLeft size={15}/> Previous</button>{BETA?<PageJump current={currentPage} last={lastPage} onGo={n=>navigate(filters,'',n)}/>:<span>Page <b>{(currentPage+1).toLocaleString()}</b> of {(lastPage+1).toLocaleString()}</span>}<button disabled={currentPage>=lastPage} onClick={()=>navigate(filters,'',currentPage+1)}>Next <ArrowRight size={15}/></button></nav>}
   </section>
  </div>
 </main>}
 {!selected&&<SiteFooter/>}
 </div>;
}
/** What the flagged share counts under each mode, for tooltips and the paper panel. */
function modeText(mode:FlagMode){const m=FLAG_MODES[mode];return m.bib&&m.app?'all sentences':m.app?'bibliography excluded':m.bib?'appendices excluded':'main text only (bibliography and appendices excluded)';}
function flagHelp(mode:FlagMode){return `Share of the paper’s sentences (${modeText(mode)}) that the Qwen3.6 35B-A3B MoE detector scores at or above its ${fprLabel(FLAG_MODES[mode].fpr)} false-positive threshold for this counting mode (${thresholdText(FLAG_MODES[mode].threshold)}), calibrated on 2,400 human-written papers from 2019–2022 with the same sentences left out. A flag means the text resembles AI-generated text; it is not proof of authorship.`;}
const bandTone=(k:string)=>k==='0-10'?'low':k==='10-25'||k==='25-50'?'mid':'high';
function pct(r:number){return r<.1&&r>0?`${(r*100).toFixed(1)}%`:`${Math.round(r*100)}%`;}
/** Per-row indicator: share of sentences flagged as AI, or a warning for papers too short to score. Unscored papers show nothing. */
function FlagBadge({p,mode}:{p:AtlasPaper;mode:FlagMode}){
 const r=flagShare(p,mode),c=flagCounts(p,mode);
 if(r===null)return c?<span className="pb-flag is-short" title={`Too short to score reliably: ${c[1]} counted sentence${c[1]===1?'':'s'}, fewer than ${MIN_SENTENCES}`}><b>!</b> too short</span>:null;
 const tone=r<.1?'low':r<.5?'mid':'high';
 return <span className={`pb-flag is-${tone}`} title={flagHelp(mode)} aria-label={`${pct(r)} of sentences flagged as AI`}><b>{pct(r)}</b><small>flagged AI</small><i aria-hidden style={{'--w':`${r*100}%`} as CSSProperties}/></span>;
}
function FlagSummary({p,mode}:{p:AtlasPaper;mode:FlagMode}){
 const r=flagShare(p,mode),c=flagCounts(p,mode);
 if(!c)return null;
 const model=p.flag_model?modelLabel(p.flag_model):'';
 if(r===null)return <p className="pb-flag-summary is-short"><b>!</b> Too short to score reliably ({c[1]} counted sentence{c[1]===1?'':'s'}; fewer than {MIN_SENTENCES}).</p>;
 const tone=r<.1?'low':r<.5?'mid':'high';
 const sec=p.sections?` · paper has ${p.sections[0].toLocaleString()} main-text, ${p.sections[1].toLocaleString()} bibliography and ${p.sections[2].toLocaleString()} appendix sentences`:'';
 return <p className={`pb-flag-summary is-${tone}`} title={flagHelp(mode)}><b>{pct(r)}</b> of sentences flagged as AI <small>{c[0].toLocaleString()} of {c[1].toLocaleString()} counted ({modeText(mode)}){sec} · {fprLabel(FLAG_MODES[mode].fpr)} human false-positive threshold{model?` · ${model}`:''}</small><i aria-hidden style={{'--w':`${r*100}%`} as CSSProperties}/></p>;
}
/** Pick the counting mode: whether the bibliography and the appendices count toward a paper's share, and the
 * false-positive rate a sentence must clear to be flagged. */
function ModeToggles({mode,onChange}:{mode:FlagMode;onChange:(m:FlagMode)=>void}){
 const m=FLAG_MODES[mode];
 return <div className="pb-modes" role="group" aria-label="Sentences counted in the flagged share"><span>Count</span>
  <label><input type="checkbox" checked={m.app} onChange={e=>onChange(flagModeOf(m.bib,e.target.checked,m.fpr))}/> Appendices</label>
  <label title="Human false-positive rate of the sentence threshold: the share of sentences in human-written papers that would be flagged">FPR <select aria-label="False-positive rate" value={m.fpr} onChange={e=>onChange(flagModeOf(m.bib,m.app,Number(e.target.value) as typeof m.fpr))}>{(BETA?FLAG_FPRS:LEGACY_FLAG_FPRS).map(f=><option key={f} value={f}>{fprLabel(f)}</option>)}</select></label>
 </div>;
}
/** Cutoff on the flagged share (5% steps): keep only papers below it, or at or above it. Bars outside it drop out and
 * the rest rescale, so the shape below (say) 95% stays readable when most papers sit above it. */
/** Whether a bin/band filter still selects something inside a cutoff (a selection outside it would show no papers). */
function aiInsideCut(ai:string,cut:string){const r=parseFlagRange(ai),c=parseFlagCut(cut);return !r||!c?true:r!=='none'&&r[0]<c[1]&&r[1]>c[0];}
function CutControl({cut,onChange}:{cut:string;onChange:(cut:string)=>void}){
 const r=parseFlagCut(cut),side=r===null?'':r[0]===0?'below':'above',value=r===null?95:r[0]===0?r[1]*5:r[0]*5;
 return <div className="pb-modes pb-cut" role="group" aria-label="Flagged-share cutoff"><span>Show</span>
  <select aria-label="Cutoff direction" value={side} onChange={e=>onChange(e.target.value?`${e.target.value}${value}`:'')}><option value="">All papers</option><option value="below">Under</option><option value="above">At or above</option></select>
  {side&&<select aria-label="Cutoff" value={value} onChange={e=>onChange(`${side}${e.target.value}`)}>{Array.from({length:19},(_,i)=>(i+1)*5).map(v=><option key={v} value={v}>{v}%</option>)}</select>}
 </div>;
}
function DossierSkeleton(){return <div className="pb-ghost" aria-hidden><i/><i/><i/><i/><i/></div>;}
/** OpenReview titles often carry LaTeX or Markdown markup; show the words, not the markup. */
function tidy(title:string){return title.replace(/\\(?:textsc|texttt|textit|textbf|emph|mathcal|mathbf|mathrm|text)\s*\{([^{}]*)\}/g,'$1').replace(/\*\*([^*]+)\*\*/g,'$1').replace(/\$([^$]*)\$/g,'$1').replace(/\\\(|\\\)/g,'').replace(/\\([a-zA-Z]+)/g,(m,name)=>GREEK[name]??m).replace(/\{([^{}]*)\}/g,'$1').replace(/\^(\d+)/g,(_,n:string)=>[...n].map(d=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join('')).replace(/\s+/g,' ').trim();}
const GREEK:Record<string,string>={alpha:'α',beta:'β',gamma:'γ',delta:'δ',epsilon:'ε',lambda:'λ',mu:'μ',pi:'π',sigma:'σ',tau:'τ',phi:'φ',psi:'ψ',omega:'ω',infty:'∞'};
function escapeRe(s:string){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');}
/** Marks search terms like a highlighter on a printed page. */
function highlight(text:string,terms:string[]):ReactNode{
 if(!terms.length)return text;
 const parts=text.split(new RegExp(`(${terms.map(escapeRe).join('|')})`,'gi'));
 return parts.map((part,i)=>i%2?<mark key={i}>{part}</mark>:part);
}

/** "Page [n] of N": type a page number and press Enter to jump there. */
function PageJump({current,last,onGo}:{current:number;last:number;onGo:(n:number)=>void}){
 const [v,setV]=useState(String(current+1));
 useEffect(()=>setV(String(current+1)),[current]);
 function go(e:FormEvent){e.preventDefault();const n=Math.min(last,Math.max(0,Math.round(Number(v))-1));if(Number.isFinite(n)&&n!==current)onGo(n);else setV(String(current+1));}
 return <form className="pb-jump" onSubmit={go}><label htmlFor="pb-jump">Page</label><input id="pb-jump" inputMode="numeric" pattern="[0-9]*" value={v} onChange={e=>setV(e.target.value.replace(/\D/g,''))} onBlur={()=>setV(String(current+1))} aria-label={`Page number, 1 to ${last+1}`} style={{width:`${String(last+1).length+2}ch`}}/><span>of {(last+1).toLocaleString()}</span></form>;
}
