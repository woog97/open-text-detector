"use client";
import { lazy, Suspense, useDeferredValue, useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowLeft, ArrowRight, ChevronLeft, ChevronRight, Search, X } from "lucide-react";
// Full-page links also work in the standalone Cloudflare build.
/* eslint-disable @next/next/no-html-link-for-pages */
import ReaderErrorBoundary from "./reader-error-boundary";
const PDFReader = lazy(()=>import("./pdf-reader"));
import {SiteHeader,SiteFooter,useSavedCount} from "./site-chrome";
import "./paper-atlas.css";
import PaperScore from "./paper-score";
import ScoreDashboard from "./score-dashboard";
import { scoreThresholds } from "@/lib/paper-scores";
import {sortPapers,type AtlasPaper as Paper} from "@/lib/atlas-collection";
import {loadCatalogue} from "@/lib/atlas-api";

async function api(path: string, init?:RequestInit): Promise<unknown> {
  const response = await fetch(`/backend${path}`,init);
  if (!response.ok) throw new Error("The paper collection is unavailable. Please try again shortly.");
  return response.json();
}
const collectionName = (name: string) => name.replace(/^(\w+)\/(\d{4})$/, (_, venue, year) => `${venue.toUpperCase()} ${year}`).replaceAll("_", " ").replaceAll("-", " ");
export default function PaperAtlas({ hosted = false, privateLink = false, browse = false }: { hosted?: boolean; privateLink?: boolean; browse?: boolean }) {
  const savedCount=useSavedCount();
  const [papers, setPapers] = useState<Paper[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const deferredQuery=useDeferredValue(query);
  const [refreshError,setRefreshError]=useState("");
  const [collection, setCollection] = useState("all");
  const [model, setModel] = useState("all");
  const [modelStats, setModelStats] = useState<{id:string;name:string;available:number;total:number;complete:boolean;sample?:boolean}[]>([]);
  const [filter, setFilter] = useState("all");
  const [sort, setSort] = useState("share");
  const [descending,setDescending] = useState(true);
  const [page, setPage] = useState(0);
  const [threshold,setThreshold] = useState(.8);
  const [pageInput,setPageInput] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [about, setAbout] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    const abort=new AbortController();
    const deadline=window.setTimeout(()=>abort.abort(),30000);
    loadCatalogue(abort.signal).then(data => {
      if (!active) return;
      setPapers(data.items); setModelStats(data.models || []); setStatus("ready");setRefreshError("");
    }).catch(e => { if (active) {
      const message=abort.signal.aborted?"The collection request timed out. Please retry.":e.message;
      setError(message);setStatus(previous=>previous==="ready"?previous:"error");setRefreshError(message);
    } }).finally(()=>window.clearTimeout(deadline));
    const refresh=window.setInterval(()=>{if(document.visibilityState==='visible')setAttempt(a=>a+1);},60000);
    return () => { active = false; abort.abort();window.clearTimeout(deadline);window.clearInterval(refresh); };
  }, [attempt]);
  useEffect(() => {
    const sync = () => {const params=new URLSearchParams(location.search);setSelected(params.get("paper"));setModel(params.get("model")||"all");setCollection(params.get("collection")||"all");setAbout(false);};
    sync(); window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);
  function openPaper(id: string | null) {
    const params=new URLSearchParams();params.set("view","classifications");if(collection!=="all")params.set("collection",collection);if(id)params.set("paper",id);if(model!=="all")params.set("model",model);setSelected(id); history.pushState({}, "", params.size ? `/?${params}` : "/"); window.scrollTo(0, 0);
  }
  const collections = useMemo(()=>[...new Set(papers.map(p=>p.collection))].sort(),[papers]);
  const classified = useMemo(()=>papers.filter(p=>p.classified).length,[papers]);
  const searchIndex=useMemo(()=>papers.map(p=>({paper:p,text:`${p.id} ${p.title} ${p.filename}`.toLowerCase()})),[papers]);
  const matching=useMemo(()=>{
    const term=deferredQuery.trim().toLowerCase();
    return searchIndex.filter(({paper:p,text})=>text.includes(term)&&
      (collection==="all"||p.collection===collection)&&
      (model==="all"||p.models?.includes(model))&&
      (filter==="all"||(filter==="classified"?p.classified:!p.classified))).map(r=>r.paper);
  },[searchIndex,deferredQuery,collection,model,filter]);
  const filtered=useMemo(()=>sortPapers(matching,sort,descending,model,threshold),[matching,sort,descending,model,threshold]);
  const totalPages = Math.max(1, Math.ceil(filtered.length / 15));
  const visiblePage = Math.min(page, totalPages - 1);
  const visible = filtered.slice(visiblePage * 15, (visiblePage + 1) * 15);
  function reset() { setQuery(""); setCollection("all"); setFilter("all"); setModel("all"); setPage(0); const url=new URL(location.href);url.searchParams.delete("model");url.searchParams.delete("collection");history.replaceState({},"",url); }
  const showList={href:"/?view=classifications",onClick:()=>{setAbout(false);openPaper(null);}};
  const howTo=()=>{setAbout(true);window.scrollTo(0,0);};
  const setCollectionParam=(value:string)=>{setCollection(value);setPage(0);const url=new URL(location.href);if(value==="all")url.searchParams.delete("collection");else url.searchParams.set("collection",value);history.replaceState({},"",url);};
  const setModelParam=(value:string)=>{setModel(value);setPage(0);const url=new URL(location.href);if(value==="all")url.searchParams.delete("model");else url.searchParams.set("model",value);history.replaceState({},"",url);};
  return <div className="site atlas">
    <SiteHeader page="classifications" savedCount={savedCount} browse={browse} classifications={showList} extra={!hosted&&<a href="/workbench">Workbench <ArrowRight size={14}/></a>}/>
    {about ? <main className="atlas-about">
      <button className="atlas-back" onClick={()=>setAbout(false)}><ArrowLeft size={15}/> Back to results</button>
      <p className="site-kicker">How to read these scores</p>
      <h1>Signals to check, not verdicts.</h1>
      <p className="atlas-lede">Each paper is split into text passages, and a detection model scores every passage. The paper score is the share of passages above the threshold you choose. Open a paper to see each scored passage on the original page.</p>
      <div className="atlas-about-grid">
        <section><h2>What’s in the collection</h2><p>Uploaded research PDFs, including ICLR 2027 submissions as they arrive. Every paper can be read before its classification results are ready.</p></section>
        <section><h2>What a highlight means</h2><p>Highlights are model output, not proof of authorship. A score is not the percentage of AI-written text. Compare passages in context and check which model produced them.</p></section>
        <section><h2>Faithful to the source</h2><p>The reader shows the original PDF pages. Published results are checked against the exact PDF and its extracted text. Saved word positions place highlights on the page; older reports use conservative text matching.</p></section>
        <section><h2>Models</h2><p>MELD v5, MELD v8 and the experimental Laya sample can be compared with the model selector. Availability counts update as verified results are published.</p></section>
      </div>
      <button className="atlas-primary" onClick={()=>setAbout(false)}>Back to results <ArrowRight size={15}/></button>
    </main> : selected ? <main className="atlas-reader">
      <div className="atlas-reader-bar"><button className="atlas-back" onClick={() => openPaper(null)}><ArrowLeft size={15}/> All results</button><button className="site-link" onClick={howTo}>How to read these scores</button></div>
      <ReaderErrorBoundary key={selected} onBack={()=>openPaper(null)}><Suspense fallback={<p role="status" className="atlas-wait">Opening the reader…</p>}>{status==="ready"?<PDFReader availablePapers={papers} key={selected} api={api} initialPaperId={selected} embedded hosted={hosted} preferredModel={model === "all" ? undefined : model} scoreThreshold={threshold}/>:<div role="status" className="atlas-wait">{status==="error"?<><p>{error}</p><button className="atlas-primary" onClick={()=>{setStatus("loading");setAttempt(a=>a+1);}}>Try again</button></>:"Opening the collection…"}</div>}</Suspense></ReaderErrorBoundary>
    </main> : <main className="atlas-main">
      <section className="atlas-hero">
        <p className="site-kicker">{status === "ready" ? <>{papers.length.toLocaleString()} papers · {classified.toLocaleString()} with classifier results</> : "Opening the collection…"}</p>
        <h1>Classifier results</h1>
        <p className="atlas-lede">Papers ranked by the share of passages a detection model scored above the threshold. Open one to see every scored passage on the page. <button className="site-link" onClick={howTo}>How to read these scores</button></p>
        {modelStats.length > 0 && <ul className="atlas-models" aria-live="polite">{modelStats.map(m=><li key={m.id}><b>{m.name}</b><span className="site-num">{m.available.toLocaleString()} / {m.total.toLocaleString()}</span><em>{m.sample ? "Sample complete" : m.complete ? "Complete" : "In progress"}</em></li>)}</ul>}
      </section>
      <div className="atlas-controls">
        <label className="atlas-search"><Search size={18} aria-hidden/><input type="search" aria-label="Search papers" placeholder="Search by paper title or ID" value={query} onChange={e => { setQuery(e.target.value); setPage(0); }}/>{query && <button aria-label="Clear search" onClick={() => { setQuery(""); setPage(0); }}><X size={16}/></button>}</label>
        <div className="atlas-selects">
          <label>Collection<select aria-label="Collection" value={collection} onChange={e=>setCollectionParam(e.target.value)}><option value="all">All collections</option>{collections.map(c => <option key={c} value={c}>{collectionName(c)}</option>)}</select></label>
          <label>Model<select aria-label="Classification model" value={model} onChange={e=>setModelParam(e.target.value)}><option value="all">Default per paper (final MoE first)</option>{modelStats.map(m=><option key={m.id} value={m.id}>{m.name} ({m.available.toLocaleString()})</option>)}</select></label>
          <label title="Paper score = share of text passages scoring strictly above this value.">Threshold<select aria-label="Passage score threshold" value={threshold} onChange={e=>{setThreshold(Number(e.target.value));setPage(0);}}>{scoreThresholds.map(t=><option key={t} value={t}>{t.toFixed(2)}</option>)}</select></label>
          <label>Sort<span className="atlas-sort"><select aria-label="Sort papers" value={sort} onChange={e => { setSort(e.target.value);setDescending(e.target.value!=="title");setPage(0); }}><option value="share">Classification share</option><option value="title">Paper title</option><option value="size">PDF size</option></select><button type="button" aria-label={descending?"Sort ascending":"Sort descending"} title={descending?"Switch to ascending order":"Switch to descending order"} onClick={()=>{setDescending(v=>!v);setPage(0);}}>{descending?<ArrowDown size={15}/>:<ArrowUp size={15}/>}<span>{sort==="title"?(descending?"Z–A":"A–Z"):descending?"High first":"Low first"}</span></button></span></label>
        </div>
        <p className="atlas-note">Paper score = share of text passages scoring strictly above {threshold.toFixed(2)}.</p>
      </div>
      {model === "laya" && <p className="atlas-callout">Laya experimental sample: five papers per conference and year. Scores use the MELD-alignment recipe, which showed weak agreement with MELD v8. This sample is not an estimate of AI use across the collection.</p>}
      {status === "ready" && <ScoreDashboard papers={matching} model={model} threshold={threshold}/>}
      {status==="ready"&&refreshError&&<p role="status" className="atlas-callout">Showing the last loaded collection. Refresh failed; it will retry automatically.</p>}
      <section className="atlas-results" aria-label="Results">
      <div className="atlas-results-bar"><h2>Papers <span className="site-num">{status === "ready" ? filtered.length.toLocaleString() : "—"}</span></h2><div className="atlas-tabs" role="group" aria-label="Result availability">{[["all", "All"], ["classified", "With results"], ["pending", "PDF only"]].map(([value, name]) => <button key={value} aria-pressed={filter === value} className={filter === value ? "is-on" : ""} onClick={() => { setFilter(value); setPage(0);if(value==="pending")setModelParam("all"); }}>{name}</button>)}</div></div>
      {status === "loading" ? <ol className="atlas-list" aria-hidden>{Array.from({length:6},(_,i)=><li key={i} className="atlas-ghost"><span/><i/><i/></li>)}</ol> : status === "error" ? <div className="atlas-empty" role="alert"><h3>Collection unavailable</h3><p>{error}</p><button className="atlas-primary" onClick={() => { setStatus("loading"); setAttempt(a => a + 1); }}>Try again</button></div> : !visible.length ? <div className="atlas-empty"><Search size={28}/><h3>{papers.length ? "No papers match" : "The collection is getting ready"}</h3><p>{papers.length ? "Try a different title, or broaden the filters." : "Papers appear here as they are added."}</p>{papers.length > 0 && <button onClick={reset} className="atlas-primary">Clear filters</button>}</div> : <ol className="atlas-list">{visible.map((paper, i) => <li key={paper.id}><a className="atlas-row" href={`/?view=classifications&paper=${paper.id}`} onClick={e=>{e.preventDefault();openPaper(paper.id);}}>
        <span className="atlas-gutter" aria-hidden>{String(visiblePage * 15 + i + 1).padStart(3, "0")}</span>
        <span className="atlas-row-body"><strong>{paper.title}</strong><small>{collectionName(paper.collection)} · <span className="site-num">{paper.filename.replace(/\.pdf$/i, "")}</span> · {(paper.bytes / 1e6).toFixed(1)} MB</small></span>
        <PaperScore paper={paper} model={model} threshold={threshold}/>
        <ArrowRight className="atlas-row-arrow" size={17} aria-hidden/>
      </a></li>)}</ol>}
      {status === "ready" && filtered.length > 0 && <nav className="atlas-pagination" aria-label="Pages"><span>Showing <b>{(visiblePage * 15 + 1).toLocaleString()}–{Math.min((visiblePage + 1) * 15, filtered.length).toLocaleString()}</b> of {filtered.length.toLocaleString()}</span><div><button aria-label="Previous page" disabled={visiblePage === 0} onClick={() => setPage(visiblePage - 1)}><ChevronLeft size={16}/></button><form className="atlas-page-jump" onSubmit={e=>{e.preventDefault();const n=Number(pageInput);if(Number.isInteger(n)&&n>=1&&n<=totalPages){setPage(n-1);setPageInput("");}}}><label>Page <input aria-label="Collection page number" type="number" min={1} max={totalPages} required placeholder={String(visiblePage+1)} value={pageInput} onChange={e=>setPageInput(e.target.value)}/></label><span>of {totalPages.toLocaleString()}</span><button type="submit">Go</button></form><button aria-label="Next page" disabled={visiblePage + 1 === totalPages} onClick={() => setPage(visiblePage + 1)}><ChevronRight size={16}/></button></div></nav>}
      </section>
    </main>}
    <SiteFooter>{privateLink && <a href="/private">Sign in</a>}</SiteFooter>
  </div>;
}
