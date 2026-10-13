"use client";
import { useEffect, useState } from "react";
import { Archive, ArrowRight, FileText, Loader2, Search } from "lucide-react";

type Model = { id: string; name: string; task: string; enabled: boolean };
type Item = { id: string; title: string; preview: string; word_count: number; characters?: number; excerpt?: boolean; generator?: string; source_url?: string; created_at?: string; model?: Model; result?: { label: string; score_type?: string }; };
type Dataset = { id: string; name: string; description: string; count: number };
const labels: Record<string, string> = { human: "Human", ai: "AI generated", ai_assisted: "AI assisted", ai_evidence: "AI evidence", below_threshold: "Below AI threshold", uncertain: "Uncertain" };

export default function CachedFindings({ models, defaultModel, api, open }: {
  models: Model[]; defaultModel: string; api: (path: string, init?: RequestInit) => Promise<any>; open: (id: string) => void;
}) {
  const [mode, setMode] = useState("saved");
  const [dataset, setDataset] = useState("iclr_2023");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [model, setModel] = useState(defaultModel);
  const [filter, setFilter] = useState("");
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<Item[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { if (!model && defaultModel) setModel(defaultModel); }, [defaultModel, model]);
  useEffect(() => { let active = true; api("/v1/finding-datasets").then(d => { if (active) setDatasets(d.items); }).catch(e => { if (active) setError(e.message); }); return () => { active = false; }; }, [api]);
  useEffect(() => {
    let active = true;
    setLoading(true); setError("");
    const timer = setTimeout(() => {
      const path = mode === "saved" ? `/v1/findings?model_id=${encodeURIComponent(filter)}&` : `/v1/finding-datasets/${dataset}?`;
      api(`${path}q=${encodeURIComponent(query)}&offset=${offset}&limit=20`)
        .then(d => { if (active) { setItems(d.items); setTotal(d.total); } })
        .catch(e => { if (active) { setItems([]); setTotal(0); setError(e.message); } })
        .finally(() => { if (active) setLoading(false); });
    }, 180);
    return () => { active = false; clearTimeout(timer); };
  }, [api, mode, dataset, filter, query, offset]);
  async function compute(id: string) {
    setBusy(id); setError("");
    try {
      const data = await api(`/v1/finding-datasets/${dataset}/${id}/compute`, { method: "POST", body: JSON.stringify({ model_id: model }) });
      open(data.scan.id);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(null); }
  }
  const source = datasets.find(d => d.id === dataset);
  return <section className="content-page findings-page">
    <div className="page-heading"><h1>Cached findings</h1><span className="findings-storage"><Archive size={16} /> Saved on this workspace</span></div>
    <p className="findings-intro">Compute once. Revisit the full report, model results and highlighted passages anytime.</p>
    <div className="findings-tabs" role="tablist" aria-label="Findings views">
      {[['saved', 'Saved results'], ['datasets', 'Browse datasets']].map(([id, label]) => <button key={id} role="tab" aria-selected={mode === id} onClick={() => { setMode(id); setOffset(0); setQuery(""); }}>{label}</button>)}
    </div>
    <div className="history-toolbar findings-toolbar">
      <div className="search-field"><Search size={17} /><input aria-label="Search findings" placeholder={mode === 'saved' ? "Search saved reports" : "Search titles or source IDs"} value={query} onChange={e => { setQuery(e.target.value); setOffset(0); }} /></div>
      {mode === "saved" ? <select aria-label="Filter findings by model" value={filter} onChange={e => { setFilter(e.target.value); setOffset(0); }}><option value="">All models</option>{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select> : <>
        <select aria-label="Dataset" value={dataset} onChange={e => { setDataset(e.target.value); setOffset(0); }} >{datasets.map(d => <option key={d.id} value={d.id}>{d.name} ({d.count.toLocaleString()})</option>)}</select>
        <select aria-label="Model for dataset analysis" value={model} onChange={e => setModel(e.target.value)}><option value="" disabled>Choose a model</option>{models.filter(m => m.task === "text").map(m => <option key={m.id} value={m.id} disabled={!m.enabled}>{m.name}{!m.enabled ? " (unavailable)" : ""}</option>)}</select>
      </>}
    </div>
    {mode === "datasets" && <div className="findings-note"><strong>{source?.name}</strong><p>{source?.description} Papers and AI responses use their full collected text. Existing results for identical text and model settings open immediately; new analyses are saved automatically.</p></div>}
    {error && <p role="alert" className="findings-error">{error}</p>}
    {loading ? <div className="findings-empty"><Loader2 className="spin" size={24} /> Loading findings…</div> : items.length === 0 ? <div className="findings-empty"><Archive size={28} /><h2>{query || filter ? "No matching findings" : "No saved findings yet"}</h2><p>{mode === 'saved' ? "Browse a dataset and compute a result, or run a check from Text Detection. Completed reports appear here automatically." : "No examples are available for this selection."}</p></div> : <div className="findings-grid">{items.map(item => <article key={item.id} className="finding-card">
      <div className="finding-card-top"><FileText size={19} /><span>{item.word_count.toLocaleString()} words{item.excerpt ? " · excerpt" : ""}</span>{item.created_at && <time>{new Date(item.created_at).toLocaleDateString()}</time>}</div>
      <h2><button onClick={() => mode === "saved" ? open(item.id) : compute(item.id)} disabled={busy !== null || (mode !== 'saved' && (!model || (item.characters || 0) > 500000 || item.word_count < 50 || item.word_count > 100000))}>{item.title}</button></h2>
      <p className="finding-preview">{item.preview}</p>
      {item.model && <div className="finding-model"><span>{item.model.name}</span><span className={`result-badge ${item.result?.label || ''}`}>{labels[item.result?.label || ''] || item.result?.label || 'Saved'}</span></div>}
      {item.generator && <p className="finding-generator">Generated by {item.generator}</p>}
      <div className="finding-footer">{mode === 'saved' ? <><span>Saved report · no recomputation</span><button onClick={() => open(item.id)}>Open report <ArrowRight size={16} /></button></> : <><span>{(item.characters || 0) > 500000 || item.word_count < 50 || item.word_count > 100000 ? "Outside scan size limits" : item.excerpt ? "First 2,000 words" : "Full collected text"}</span><button disabled={busy !== null || !model || (item.characters || 0) > 500000 || item.word_count < 50 || item.word_count > 100000} onClick={() => compute(item.id)}>{busy === item.id ? <Loader2 size={16} className="spin" /> : null}Compute / open <ArrowRight size={16} /></button></>}</div>
    </article>)}</div>}
    <div className="findings-pagination"><span>{total ? `${offset + 1}–${Math.min(offset + 20, total)} of ${total.toLocaleString()}` : "0 results"}</span><button disabled={loading || offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>Previous</button><button disabled={loading || offset + 20 >= total} onClick={() => setOffset(offset + 20)}>Next</button></div>
  </section>;
}
