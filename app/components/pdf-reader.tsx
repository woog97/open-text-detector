"use client";
import "../lib/pdfjs-compat.js";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReaderSummary from "./reader-summary";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  EyeOff,
  FileText,
  Loader2,
  Maximize2,
  Minimize2,
  Minus,
  Plus,
  Search,
} from "lucide-react";
import { hasTextContent, scoreBand, summarizePassages, paperRate, scoreThresholds, DEFAULT_MODELS, MIN_SENTENCES, modelLabel } from "@/lib/paper-scores";
import { fprReadout } from "@/lib/score-fpr";
import { FLAG_FPRS, LEGACY_FLAG_FPRS, FLAG_MODES, FLAG_MODEL, DEFAULT_FLAG_MODE, flagModeOf, fprLabel, thresholdText, type FlagMode } from "@/lib/paper-discovery";
import { segmentSections, type Section } from "@/lib/paper-sections";
import { validateReaderDetail, type ReaderDetail as Detail } from "@/lib/pdf-reader-data";
import { acquirePdfRender, rasterSize, abortable } from "@/lib/pdf-render-budget";
import { mergeHighlightBoxes } from "@/lib/pdf-highlight-boxes";
import legacyAliases from "@/lib/legacy-paper-aliases.json";
import type { PDFDocumentProxy } from "pdfjs-dist";
import {
  alignPassages,
  buildIndex,
  
  type Passage,
} from "@/lib/pdf-alignment";
import { positionPassages, supportsSavedPositions, type PositionedPassage as AlignedPassage } from "@/lib/pdf-positions";
import "./pdf-text-layer.css";
import "./pdf-reader.css";
// Screens where the inspector becomes a sheet instead of a column: narrow windows and phones in landscape.
const COMPACT="(max-width: 800px), (max-height: 500px)";
/** Most passages one feedback message can mark; the Worker enforces the same (atlas-cloud/feedback.ts MAX_PASSAGES). */
const MARK_LIMIT=2000;
// The beta layout keeps its column panel (with its own toggle) down to phone widths, also in fullscreen.
const COMPACT_BETA="(max-width: 640px), (max-height: 500px)";

type Paper = {
  id: string;
  title: string;
  filename: string;
  bytes: number;
  classified: boolean;
  collection: string;
};
type PDFLib = typeof import("pdfjs-dist");
type Index = ReturnType<typeof buildIndex>;
const labels: Record<string,string> = {
  ai:"AI", ai_assisted:"AI assisted", ai_evidence:"Elevated evidence",
  low_evidence:"Lower evidence", human:"Human", uncertain:"Uncertain", below_threshold:"Below threshold",
};
const label = (s:Passage) => Object.hasOwn(labels,s.label)?labels[s.label]:s.label;

const EMPTY_PASSAGES: AlignedPassage[] = [];

function PageCanvas({
  pdf,
  lib,
  number,
  width,
  passages,
  selected,
  highlights,
  onSelect,
  focusId, onFocused,
  bandOf,
  marking = null,
  shownMarks = null,
}: {
  pdf: PDFDocumentProxy;
  lib: PDFLib;
  number: number;
  width: number;
  passages: AlignedPassage[];
  selected: number | null;
  focusId: number | null;
  onFocused: (id:null)=>void;
  highlights: boolean;
  onSelect: (id: number) => void;
  bandOf: (id: number) => string;
  /** Marking mode (feedback): clicks and drags over highlights mark passages instead of inspecting them. */
  marking?: {marked: Set<number>; paint: (id: number, start: boolean) => void} | null;
  /** Passages the reader marked, drawn over the result until the feedback is sent or discarded (null: hidden). */
  shownMarks?: Set<number> | null;
}) {
  const host = useRef<HTMLDivElement>(null);
  const hoverFrame = useRef(0), hovered = useRef<Element | null>(null);
  const [pageImage, setPageImage] = useState("");
  const [boxes, setBoxes] = useState<
    {
      id: number;
      x: number;
      y: number;
      w: number;
      h: number;
    }[]
  >([]);
  const [retry,setRetry] = useState(0);
  const [busy, setBusy] = useState(true),
    [error, setError] = useState("");
  useEffect(() => {
    const target = host.current!;
    let stopped = false;
    const abort = new AbortController();
    let imageUrl: string | undefined;
    let canvas: HTMLCanvasElement | undefined;
    let renderedPage: Awaited<ReturnType<PDFDocumentProxy["getPage"]>> | undefined;
    let release: (()=>void) | undefined;
    let deadline: ReturnType<typeof setTimeout> | undefined;
    let render:
      | ReturnType<Awaited<ReturnType<PDFDocumentProxy["getPage"]>>["render"]>
      | undefined;
    let layer: InstanceType<PDFLib["TextLayer"]> | undefined;
    // Reset transient UI when the external PDF render task changes.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setBoxes([]);
    setPageImage("");
    setBusy(true);
    setError("");
    target.querySelector(".textLayer")?.remove();
    (async () => {
      release = await acquirePdfRender(abort.signal);
      if(stopped)return;
      deadline=setTimeout(()=>{render?.cancel();layer?.cancel();abort.abort();},45000);
      const page = await abortable(pdf.getPage(number),abort.signal);
      renderedPage=page;
      if (stopped) return;
      const viewport = page.getViewport({
        scale: width / page.getViewport({ scale: 1 }).width,
      });
      target.style.width = `${viewport.width}px`;
      target.style.height = `${viewport.height}px`;
      target.style.setProperty("--scale-factor", String(viewport.scale));
      target.style.setProperty("--total-scale-factor", String(viewport.scale));
      canvas = document.createElement("canvas");
      const budget=rasterSize(viewport.width,viewport.height,window.devicePixelRatio);
      const ratio=budget.ratio;
      canvas.width = budget.width;
      canvas.height = budget.height;
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      render = page.render({
        canvas,
        viewport,
        transform: [ratio, 0, 0, ratio, 0, 0],
      });
      await render.promise;
      if (stopped) return;
      const blob=await abortable(new Promise<Blob>((resolve,reject)=>canvas!.toBlob(value=>value?resolve(value):reject(new Error("Unable to render PDF image")),"image/png")),abort.signal);
      if(stopped)return;
      imageUrl=URL.createObjectURL(blob);
      setPageImage(imageUrl);
      canvas.width = 0; canvas.height = 0;
      const content = await abortable(page.getTextContent(),abort.signal);
      if (stopped) return;
      const text = document.createElement("div");
      text.className = "textLayer";
      target.appendChild(text);
      // Dense plots may encode thousands of dot glyphs. Avoid a huge invisible
      // text layer when saved positions already provide the highlight geometry.
      if(content.items.length<=3000) {
      layer = new lib.TextLayer({
        textContentSource: content,
        container: text,
        viewport,
      });
      await abortable(layer.render(),abort.signal);
      }
      if (stopped) return;
      const bounds = target.getBoundingClientRect();
      const rects: typeof boxes = [];
      for (const passage of passages) {
        for (const rect of passage.rectangles || []) {
          if(rect.page===number)rects.push({id:passage.id,x:rect.x*viewport.width,y:rect.y*viewport.height,w:rect.w*viewport.width,h:rect.h*viewport.height});
        }
        for (const piece of passage.pieces) {
          if (piece.page !== number) continue;
          const span = layer?.textDivs[piece.item];
          if (!span?.firstChild || piece.end > (span.textContent?.length || 0))
            continue;
          const range = document.createRange();
          range.setStart(span.firstChild, piece.start);
          range.setEnd(span.firstChild, piece.end);
          for (const rect of Array.from(range.getClientRects()))
            if (rect.width > 0 && rect.height > 0) {
              rects.push({
                id: passage.id,
                x: rect.left - bounds.left,
                y: rect.top - bounds.top,
                w: rect.width,
                h: rect.height,
              });
            }
        }
      }
      setBoxes(mergeHighlightBoxes(rects));
      setBusy(false);
    })().catch((e) => {
      if (!stopped) {
        setError(abort.signal.aborted ? "This page took too long to render. Try again." : e.message || "This page could not be rendered.");
        setBusy(false);
      }
    }).finally(()=>{
      clearTimeout(deadline);
      if(canvas){canvas.width=0;canvas.height=0;}
      try { renderedPage?.cleanup(); } catch { /* Document may already be closed. */ }
      release?.();
    });
    return () => {
      stopped = true;
      abort.abort();
      render?.cancel();
      layer?.cancel();
      target.querySelector(".textLayer")?.remove();
      if(imageUrl)URL.revokeObjectURL(imageUrl);
    };
  }, [pdf, lib, number, width, passages,retry]);
  useEffect(() => {
    if (focusId === null || focusId !== selected) return;
    const box = boxes.find((b) => b.id === focusId);
    const scroller = host.current?.closest(".pr-viewport");
    if (box && scroller) {
      onFocused(null);
      // Sheet layouts: bring the reading pane up under the sticky bars (site nav, paper nav) unless fullscreen, and keep
      // the passage clear of the inspector sheet (a bottom sheet covers the lower ~55% of the screen).
      let room = scroller.clientHeight;
      const ws = scroller.closest(".pr-workspace"), bottom = ws?.classList.contains("pr-sheet-bottom");
      if (ws?.classList.contains("pr-sheet")) {
        if (!ws.classList.contains("pr-full")) {
          const inset = [...document.querySelectorAll<HTMLElement>(".site-nav,.pb-dossier-nav")].reduce((a, e) => a + e.offsetHeight, 0);
          window.scrollBy({ top: scroller.getBoundingClientRect().top - inset, behavior: "instant" });
        }
        room = Math.max(120, Math.min(scroller.clientHeight, innerHeight * (bottom ? 0.45 : 1) - Math.max(0, scroller.getBoundingClientRect().top)));
      }
      scroller.scrollTo({
        top: Math.max(0, scroller.scrollTop + host.current!.getBoundingClientRect().top - scroller.getBoundingClientRect().top + box.y - room / 3),
        behavior: "instant",
      });
    }
  }, [boxes, selected, focusId, onFocused]);
  return (
    <div className="pr-page-wrap" style={{ width }} onMouseMove={e => {
      // Hover outline for the highlight under the text layer; at most one hit test per frame.
      const {clientX, clientY} = e;
      if (hoverFrame.current) return;
      hoverFrame.current = requestAnimationFrame(() => {
        hoverFrame.current = 0;
        const hit = document.elementsFromPoint(clientX, clientY).find(el => el.classList.contains("pr-highlight")) || null;
        if (hit !== hovered.current) { hovered.current?.classList.remove("is-hover"); hit?.classList.add("is-hover"); hovered.current = hit; }
      });
    }} onMouseLeave={() => { cancelAnimationFrame(hoverFrame.current); hoverFrame.current = 0; hovered.current?.classList.remove("is-hover"); hovered.current = null; }} onClick={e => {
      // The PDF text layer sits above the highlights so text can be drag-selected; a plain click still inspects
      // the highlight underneath, found by hit-testing through the layers.
      if (marking || (e.target as HTMLElement).closest(".pr-highlight") || String(window.getSelection() || "")) return;
      const hit = document.elementsFromPoint(e.clientX, e.clientY).find(el => el.classList.contains("pr-highlight")) as HTMLElement | undefined;
      if (hit?.dataset.id) onSelect(Number(hit.dataset.id));
    }}>
      {busy && (
        <div className="pr-page-status">
          <Loader2 className="pr-spin" size={18} /> Rendering page {number}
        </div>
      )}
      {error && (
        <div role="alert" className="pr-error">
          {error} <button onClick={()=>setRetry(n=>n+1)}>Retry page</button>
        </div>
      )}
      <div className="pr-page" ref={host}>
        {/* A single rendered page; PDF text and annotations remain separate layers. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {pageImage && <img src={pageImage} alt={`PDF page ${number}`} style={{display:"block",width:"100%"}} />}
      </div>
      <div className="pr-overlays">
        {boxes
          .filter(() => highlights || !!marking)
          .map((b, i) => (
            <button
              key={i}
              tabIndex={-1}
              aria-label={marking ? `Mark passage ${b.id + 1}` : `Inspect passage ${b.id + 1}`}
              title={`Passage ${b.id + 1}`}
              data-id={b.id}
              className={`pr-highlight ${bandOf(b.id)} ${selected === b.id && !marking ? "selected" : ""} ${shownMarks?.has(b.id) ? "marked" : ""}`}
              style={{ left: b.x, top: b.y, width: b.w, height: b.h }}
              onClick={() => { if (!marking) onSelect(b.id); }}
              onPointerDown={marking ? e => { e.preventDefault(); marking.paint(b.id, true); } : undefined}
              onPointerEnter={marking ? e => { if (e.buttons & 1) marking.paint(b.id, false); } : undefined}
            />
          ))}
      </div>
    </div>
  );
}

// Keep page-sized placeholders so scrolling stays continuous while only nearby pages render.
const BAND_COLORS: Record<string,string> = {low:"#8ebd95", medium:"#ddbc52", high:"#d97669", unknown:"#cfdacc", skip:"#d9dbe0"};
/** One vertical strip per page: each passage gets an equal slice, in reading order, colored by its band. */
function pageStrip(local: {id:number}[], bandOf: (id:number)=>string) {
  if (!local.length) return "transparent";
  const step = 100 / local.length;
  return `linear-gradient(to bottom, ${local.map((p, k) => `${BAND_COLORS[bandOf(p.id)]} ${(k*step).toFixed(2)}% ${((k+1)*step).toFixed(2)}%`).join(", ")})`;
}
const SECTION_NAMES: Record<Section,string> = {body:"main text", bib:"bibliography", app:"appendix"};
/** Same rounding as the Discover flagged share. */
function sharePct(r: number) { return r < .1 && r > 0 ? `${(r*100).toFixed(1)}%` : `${Math.round(r*100)}%`; }
function modeName(m: FlagMode) { const b = FLAG_MODES[m].base; return b==="all" ? "all" : b==="nobib" ? "non-bibliography" : b==="noapp" ? "non-appendix" : "main-text"; }

function ScrollingPage({ratios,frozen,...props}: React.ComponentProps<typeof PageCanvas> & {ratios:Map<number,number>;frozen:boolean}) {
  const slot = useRef<HTMLDivElement>(null);
  // While the Text view is shown the stack is hidden; keep rendered pages instead of dropping them.
  const frozenRef = useRef(frozen); frozenRef.current = frozen;
  const [nearby, setNearby] = useState(false);
  const [ratio, setRatio] = useState(()=>ratios.get(props.number)||792 / 612);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => { if (!frozenRef.current) setNearby(entry.isIntersecting); }, {
      root: slot.current!.closest(".pr-viewport"), rootMargin: "1000px 0px",
    });
    observer.observe(slot.current!);
    return () => observer.disconnect();
  }, [props.pdf, props.number]);
  useEffect(()=>{
    if(!nearby)return;
    let active=true;
    props.pdf.getPage(props.number).then(page=>{
      const size=page.getViewport({scale:1});
      if(active && Number.isFinite(size.height/size.width)){ratios.set(props.number,size.height/size.width);setRatio(size.height/size.width);}
    }).catch(()=>{});
    return ()=>{active=false;};
  },[nearby,props.pdf,props.number,ratios]);
  return <div ref={slot} className="pr-page-slot" data-page={props.number}
    aria-label={`PDF page ${props.number}`} style={{width: props.width, minHeight: props.width * ratio}}>
    {nearby ? <PageCanvas {...props} /> : <div className="pr-page-placeholder">Page {props.number}</div>}
  </div>;
}

export default function PDFReader({
  api,
  availablePapers,
  initialPaperId = "",
  embedded = false,
  hosted = false,
  readingOnly = false,
  preferredModel,
  scoreThreshold = .8,
  fileBasePath = "/backend/v1/pdf-reader",
  flagMode: flagModeProp,
  onFlagModeChange,
  beta = false,
  summarySlot = null,
}: {
  /** Beta layout: passages panel collapsed by default; model results rendered into `summarySlot`. */
  beta?: boolean;
  summarySlot?: HTMLElement | null;
  availablePapers?: Paper[];
  initialPaperId?: string;
  embedded?: boolean;
  hosted?: boolean;
  readingOnly?: boolean;
  preferredModel?: string;
  scoreThreshold?: number;
  fileBasePath?: string;
  /** Counting mode shared with Discover (bibliography/appendix toggles); the reader keeps its own when not given. */
  flagMode?: FlagMode;
  onFlagModeChange?: (mode: FlagMode) => void;
  api: (path: string, init?:RequestInit) => Promise<unknown>;
}) {
  const [papers, setPapers] = useState<Paper[]>([]),
    [paperId, setPaperId] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null),
    [reportId, setReportId] = useState("");
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null),
    [lib, setLib] = useState<PDFLib | null>(null);
  const [index, setIndex] = useState<Index | null>(null),
    [page, setPage] = useState(1);
  const [selected, setSelected] = useState<number | null>(null),
    [highlights, setHighlights] = useState(true);
  // Feedback: passages the reader marks as written without AI, sent with "Think this result is wrong?" (reader-summary.tsx).
  const [marked, setMarked] = useState<Set<number>>(()=>new Set()),
    [feedbackOpen, setFeedbackOpen] = useState(false),
    [markMode, setMarkMode] = useState(false),
    [showMarks, setShowMarks] = useState(true);
  const stroke = useRef<"mark"|"unmark"|null>(null);
  const [mode, setMode] = useState<"pdf" | "text">("pdf"),
    [filter, setFilter] = useState("all");
  const [passageSort,setPassageSort] = useState("order");
  const [threshold,setThreshold] = useState(scoreThreshold);
  const [ownFlagMode,setOwnFlagMode] = useState<FlagMode>(DEFAULT_FLAG_MODE);
  const flagMode = flagModeProp ?? ownFlagMode;
  const setFlagMode = (m: FlagMode) => { setOwnFlagMode(m); onFlagModeChange?.(m); };
  const [focusId,setFocusId] = useState<number|null>(null);
  const [pageInput,setPageInput] = useState("");
  const [loadAttempt,setLoadAttempt] = useState(0);
  const [textSection,setTextSection] = useState(0);
  const [sidebarPage,setSidebarPage] = useState(0);
  const [panel,setPanel] = useState(false);
  const [search, setSearch] = useState(""),
    [error, setError] = useState(""),
    [progress, setProgress] = useState("");
  const [width, setWidth] = useState(720),
    [zoom, setZoom] = useState(1);
  const viewport = useRef<HTMLDivElement>(null);
  // Trackpad pinch (ctrl+wheel in Chrome/Firefox, gesture events in Safari) zooms the PDF to any level from 50% to 300%.
  // While pinching, the page stack is scaled with a CSS transform around the pointer; 180 ms after the last event the
  // new zoom is committed (pages re-render sharp) and the scroll position keeps the same point under the pointer.
  const zoomState = useRef({zoom: 1, mode: "pdf"});
  zoomState.current = {zoom, mode};
  useEffect(() => {
    const el = viewport.current;
    if (!el) return;
    let pending = 0, timer = 0, start = 1, client = {x: 0, y: 0};
    const stack = () => el.querySelector<HTMLElement>(".pr-page-stack:not([hidden])");
    const preview = (next: number, x: number, y: number) => {
      const s = stack();
      if (!s) return;
      if (!pending) { const r = s.getBoundingClientRect(); s.style.transformOrigin = `${x - r.left}px ${y - r.top}px`; client = {x, y}; }
      pending = Math.min(3, Math.max(0.5, next));
      s.style.transform = `scale(${pending / zoomState.current.zoom})`;
      clearTimeout(timer);
      timer = window.setTimeout(commit, 180);
    };
    const commit = () => {
      const s = stack(), base = zoomState.current.zoom, next = pending, r = next / base, box = el.getBoundingClientRect();
      pending = 0;
      if (!s || !next) return;
      const cx = client.x - box.left, cy = client.y - box.top, left = (el.scrollLeft + cx) * r - cx, top = (el.scrollTop + cy) * r - cy;
      s.style.transform = ""; s.style.transformOrigin = "";
      setZoom(next);
      requestAnimationFrame(() => requestAnimationFrame(() => el.scrollTo({left, top, behavior: "instant"})));
    };
    const wheel = (e: WheelEvent) => {
      if (!e.ctrlKey || zoomState.current.mode !== "pdf") return;
      e.preventDefault();
      preview((pending || zoomState.current.zoom) * Math.exp(-e.deltaY * 0.01), e.clientX, e.clientY);
    };
    type Gesture = UIEvent & {scale: number; clientX: number; clientY: number};
    const gStart = (e: Event) => { if (zoomState.current.mode !== "pdf") return; e.preventDefault(); start = pending || zoomState.current.zoom; };
    const gChange = (e: Event) => { if (zoomState.current.mode !== "pdf") return; e.preventDefault(); const g = e as Gesture; preview(start * g.scale, g.clientX, g.clientY); };
    el.addEventListener("wheel", wheel, {passive: false});
    el.addEventListener("gesturestart", gStart);
    el.addEventListener("gesturechange", gChange);
    return () => { clearTimeout(timer); el.removeEventListener("wheel", wheel); el.removeEventListener("gesturestart", gStart); el.removeEventListener("gesturechange", gChange); };
  }, [pdf]);
  const viewScroll=useRef({pdf:0,text:0});
  // Keep measurements across view switches, but discard them for a new PDF.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const pageRatios=useMemo(()=>new Map<number,number>(),[pdf]);
  function switchMode(next:"pdf"|"text"){
    if(next===mode)return;
    viewScroll.current[mode]=viewport.current?.scrollTop||0;
    setFocusId(null);
    if(next==="text" && selected!==null)setTextSection(Math.floor(selected/100));
    setMode(next);
  }
  useLayoutEffect(()=>{
    viewport.current?.scrollTo({top:viewScroll.current[mode],behavior:"instant"});
  },[mode]);
  useEffect(() => {
    let active = true;
    const abort=new AbortController();
    (availablePapers ? Promise.resolve({items:availablePapers}) : api("/v1/pdf-reader",{signal:abort.signal}))
      .then((payload) => {
        const data = payload as { items: Paper[] };
        if (active) {
          setPapers(data.items);
          const resolvedId=data.items.some(p=>p.id===initialPaperId) ? initialPaperId : (legacyAliases as Record<string,string>)[initialPaperId] || initialPaperId;
          if (resolvedId && !data.items.some(p => p.id === resolvedId)) {
            setError("This paper is no longer in the local collection.");
            return;
          }
          setPaperId(resolvedId || data.items[0]?.id || "");
          if (!data.items.length)
            setProgress("No local PDFs found in the research data folder.");
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
      abort.abort();
    };
  }, [api, initialPaperId,loadAttempt,availablePapers]);
  useEffect(() => {
    if (!paperId) return;
    let active = true;
    const abort=new AbortController();
    let task: ReturnType<PDFLib["getDocument"]> | undefined;
    let timedOut=false;
    const deadline=setTimeout(()=>{
      timedOut=true;abort.abort();void task?.destroy().catch(()=>{});
      if(active){setError("Loading this paper timed out. Please retry.");setProgress("");}
    },45000);
    // A new external document invalidates all page and alignment state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPdf(null);
    setIndex(null);
    setDetail(null);
    viewScroll.current={pdf:0,text:0};
    setMode("pdf");
    setTextSection(0);
    setSidebarPage(0);
    setFilter("all");
    setPage(1);
    setSelected(null);
    setError("");
    setProgress(hosted ? "Opening archived PDF…" : "Opening local PDF…");
    (async () => {
      const [payload, library] = await Promise.all([
        api(`/v1/pdf-reader/${paperId}`,{signal:abort.signal}),
        import("pdfjs-dist"),
      ]);
      const raw = validateReaderDetail(payload);
      // The public site offers only the live model (Oct 10 MoE); older reports in a detail object stay hidden there.
      const data = hosted ? {...raw, reports: raw.reports.filter(r=>r.id.startsWith(`${FLAG_MODEL}-`))} : raw;
      if (!active || timedOut) return;
      library.GlobalWorkerOptions.workerSrc = "/pdfjs/pdf.worker.compat.mjs";
      setDetail(data);
      // Without a chosen model, open the live MoE report when the paper has one.
      const want=preferredModel??DEFAULT_MODELS[0];
      setReportId(data.reports.find(r=>r.id.startsWith(`${want}-`) || r.model.name.toLowerCase().includes(want))?.id || data.reports[0]?.id || "");
      setLib(library);
      task = library.getDocument({
        url: `${fileBasePath}/${paperId}/file?v=${encodeURIComponent(data.version)}`,
        cMapUrl: "/pdfjs/cmaps/",
        cMapPacked: true,
        standardFontDataUrl: "/pdfjs/standard_fonts/",
        wasmUrl: "/pdfjs/wasm/",
        isEvalSupported: false,
        disableAutoFetch: true,
        disableStream: true,
        // 256 KB ranges: page 1 of a typical paper needs one or two requests instead of four 64 KB round trips.
        rangeChunkSize: 262144,
      });
      const document = await task.promise;
      if (!active || timedOut) return;
      setPdf(document);
      if (readingOnly || !data.reports.length || data.reports.every(r=>supportsSavedPositions(data.position_maps?.[r.text_sha256 || ""]))) {
        setProgress("");
        return;
      }
      const pages: string[][] = [];
      for (let n = 1; n <= document.numPages; n++) {
        if (!active || timedOut) return;
        setProgress(
          `Locating saved passages · page ${n} of ${document.numPages}`,
        );
        const sourcePage = await abortable(document.getPage(n),abort.signal);
        const content = await abortable(sourcePage.getTextContent(),abort.signal);
        sourcePage.cleanup();
        pages.push(
          content.items
            .filter((item) => "str" in item)
            .map((item) => ("str" in item ? item.str : "")),
        );
      }
      if (active && !timedOut) {
        setIndex(buildIndex(pages));
        setProgress("");
      }
    })().catch((e) => {
      if (active) {
        setError(timedOut ? "Loading this paper timed out. Please retry." : e.message || "Could not open PDF");
        setProgress("");
      }
    }).finally(()=>clearTimeout(deadline));
    return () => {
      active = false;
      clearTimeout(deadline);
      abort.abort();
      void task?.destroy().catch(()=>{});
    };
  }, [api, paperId, hosted, preferredModel,loadAttempt,fileBasePath,readingOnly]);
  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    // Pages are sized to this width. If a page's height makes the scrollbar appear, the width shrinks, the page
    // shrinks, the scrollbar goes and the width grows again: a visible shake. CSS reserves the scrollbar gutter;
    // where that is unsupported, ignore a change straight back to the width before last.
    const seen = { previous: -1, current: -1, at: 0 };
    const resize = new ResizeObserver((entries) => {
      const next = Math.round(Math.max(120, Math.min(900, entries[0].contentRect.width))), now = performance.now();
      if (next === seen.current || (next === seen.previous && now - seen.at < 600)) return;
      seen.previous = seen.current; seen.current = next; seen.at = now;
      setWidth(next);
    });
    resize.observe(element);
    return () => resize.disconnect();
  }, []);
  const paper = papers.find((p) => p.id === paperId);
  const report = detail?.reports.find((r) => r.id === reportId);
  const passages = useMemo<AlignedPassage[]>(
    () =>
      report && supportsSavedPositions(detail?.position_maps?.[report.text_sha256 || ""])
        ? positionPassages(report.text, report.result.segments, detail!.position_maps![report.text_sha256!])
        : report && index ? alignPassages(report.text, report.result.segments, index) : [],
    [report, index, detail],
  );
  // First passage located on each new PDF page, for soft page boundaries in text mode.
  const pageStarts=useMemo(()=>{const starts=new Map<number,number>();let last=0;
    for(const p of passages)if(p.page!==undefined&&p.page>last){starts.set(p.id,p.page);last=p.page;}
    return starts;},[passages]);
  const section=Math.min(textSection,Math.max(0,Math.ceil(passages.length/100)-1));
  const textStart=section*100, textEnd=Math.min(passages.length,textStart+100);
  useEffect(()=>{
    if(mode==="text" && selected!==null)viewport.current?.querySelector(`[data-passage="${selected}"]`)?.scrollIntoView({block:"center",behavior:"instant"});
  },[mode,selected,section]);
  // Page-map jumps in text mode: scroll the target passage to the top once its section has rendered.
  const [textTarget,setTextTarget]=useState<number|null>(null);
  useEffect(()=>{
    if(mode!=="text"||textTarget===null)return;
    const el=viewport.current?.querySelector<HTMLElement>(`[data-passage="${textTarget}"]`);
    // Land on the page boundary line when the passage starts a page.
    const target=el?.previousElementSibling?.classList.contains("pr-page-break")?el.previousElementSibling as HTMLElement:el;
    const scroller=viewport.current;
    if(target&&scroller){
      // Scroll only the viewer when it scrolls; otherwise let the page scroll.
      if(scroller.scrollHeight>scroller.clientHeight+1)scroller.scrollTo({top:scroller.scrollTop+target.getBoundingClientRect().top-scroller.getBoundingClientRect().top-16,behavior:"instant"});
      else target.scrollIntoView({block:"start",behavior:"instant"});
      setTextTarget(null);
    }
  },[mode,textTarget,section]);
  const current = selected === null ? undefined : passages[selected];
  const sourceChars = useMemo(()=>Array.from(report?.text || ""),[report?.text]);
  const reportSummary = useMemo(()=>summarizePassages(report?.result.segments || [],report?.text),[report]);
  // The final MoE report is calibrated: it is colored and counted exactly like the Discover flagged share (same
  // sentences, sections and per-mode false-positive threshold). Other models keep the fixed score bands.
  const calibrated = !!report?.id.startsWith(`${FLAG_MODEL}-`);
  const baseFlag = FLAG_MODES[flagMode];
  // Optional reader-only colour thresholds (yellow and red cutoffs); they recolour this paper's sentences in the browser.
  const [custom, setCustom] = useState<{on: boolean; y: number; r: number}>({on: false, y: baseFlag.threshold, r: baseFlag.strong});
  // A new FPR or counting mode snaps the sliders to that mode's calibrated cutoffs.
  useEffect(()=>{setCustom(c=>({...c, y: baseFlag.threshold, r: baseFlag.strong}));},[baseFlag.threshold, baseFlag.strong]);
  const flag = custom.on ? {...baseFlag, threshold: custom.y, strong: Math.max(custom.r, custom.y)} : baseFlag;
  const sectionList = useMemo(()=>report ? segmentSections(report.text, report.result.segments) : [],[report]);
  const segmentHasText = useMemo(()=>(report?.result.segments || []).map(s=>hasTextContent(sourceChars.slice(s.start,s.end).join(""))),[report,sourceChars]);
  const counted = (id: number) => { const sec = sectionList[id]; return sec === "body" || (sec === "bib" && flag.bib) || (sec === "app" && flag.app); };
  const bandOfScore = (id: number, score: number) => !segmentHasText[id] ? "unknown" : !calibrated ? scoreBand(score) :
    !counted(id) ? "skip" : score >= flag.strong ? "high" : score >= flag.threshold ? "medium" : "low";
  const bandOf = (id: number) => bandOfScore(id, report?.result.segments[id]?.score ?? NaN);
  const flagStats = useMemo(()=>{
    if (!calibrated || !report) return null;
    const stats = {counted: 0, flagged: 0, strong: 0, skipped: {bib: 0, app: 0}};
    report.result.segments.forEach((s, id) => {
      if (!segmentHasText[id]) return;
      const sec = sectionList[id];
      if (!(sec === "body" || (sec === "bib" && flag.bib) || (sec === "app" && flag.app))) { stats.skipped[sec as "bib"|"app"]++; return; }
      stats.counted++;
      if (s.score >= flag.threshold) { stats.flagged++; if (s.score >= flag.strong) stats.strong++; }
    });
    return stats;
  },[calibrated,report,segmentHasText,sectionList,flag]);
  // Sliders always show; unticked they are disabled at the calibrated cutoffs of the chosen FPR and counting mode.
  const shownY = custom.on ? custom.y : baseFlag.threshold, shownR = custom.on ? Math.max(custom.r, custom.y) : baseFlag.strong;
  const customControls = (<div className={`pr-custom${custom.on ? "" : " is-off"}`}>
                  <label><input type="checkbox" checked={custom.on} onChange={e=>setCustom({on: e.target.checked, y: baseFlag.threshold, r: baseFlag.strong})}/> Custom colour thresholds</label>
                  <label>Yellow from <input type="range" aria-label="Yellow threshold" disabled={!custom.on} min={-4} max={0} step={0.01} value={Math.log10(shownY)} onChange={e=>{const y=10**Number(e.target.value);setCustom(c=>({...c, y, r: Math.max(c.r, y)}));}}/><output>{thresholdText(shownY)}<small>{fprReadout(FLAG_MODES[flagMode].base,shownY)}</small></output></label>
                  <label>Red from <input type="range" aria-label="Red threshold" disabled={!custom.on} min={-4} max={0} step={0.01} value={Math.log10(shownR)} onChange={e=>{const r=10**Number(e.target.value);setCustom(c=>({...c, r, y: Math.min(c.y, r)}));}}/><output>{thresholdText(shownR)}<small>{fprReadout(FLAG_MODES[flagMode].base,shownR)}</small></output></label>
                  {custom.on && <>
                    <button type="button" className="site-link" onClick={()=>setCustom({on: true, y: baseFlag.threshold, r: baseFlag.strong})}>Reset to the calibrated values</button>
                    <p className="pr-mapping-note">Custom thresholds change only the colours and counts in this reader. The rate under each slider is the share of sentences in the 2,400 human calibration papers that score at or above it; see Choosing a cutoff on the <a href="/?view=calibration">Calibration page</a>.</p>
                  </>}
                </div>);
  const tierLabel = (id: number) => { const b = bandOf(id); return b === "skip" ? `Not counted: ${SECTION_NAMES[sectionList[id]]}` : b === "high" ? "Strongly flagged" : b === "medium" ? "Flagged" : b === "low" ? "Not flagged" : "Excluded fragment"; };
  const highlightPassages=useMemo(()=>passages.filter(p=>hasTextContent(p.text)),[passages]);
  const pagePassages=useMemo(()=>{
    const pages=new Map<number,AlignedPassage[]>();
    for(const passage of highlightPassages){
      const numbers=new Set([...passage.pieces.map(p=>p.page),...(passage.rectangles||[]).map(r=>r.page)]);
      for(const n of numbers){const group=pages.get(n)||[];group.push(passage);pages.set(n,group);}
    }
    return pages;
  },[highlightPassages]);
  const mapped = useMemo(()=>passages.filter((p) => p.mapped).length,[passages]);
  const flaggedList = useMemo(()=>!beta?[]:passages.filter(p=>hasTextContent(p.text)&&(calibrated?counted(p.id)&&p.score>=flag.threshold:p.score>threshold))
    .map(p=>({id:p.id,page:p.mapped?p.page:undefined,score:p.score,strong:calibrated?p.score>=flag.strong:p.score>.97})),[beta,passages,calibrated,flag,threshold,sectionList]); // eslint-disable-line react-hooks/exhaustive-deps
  const visible = useMemo(()=>passages.filter(
    (p) =>
      filter === "artifacts" ? !hasTextContent(p.text) : hasTextContent(p.text) && (filter === "all" || (filter === "elevated" ? (calibrated ? counted(p.id) && p.score >= flag.threshold : p.score > threshold) : filter === "skipped" ? calibrated && !counted(p.id) : !p.mapped)),
  ).sort((a,b)=>passageSort==="high"?b.score-a.score||a.id-b.id:passageSort==="low"?a.score-b.score||a.id-b.id:a.id-b.id),[passages,filter,threshold,passageSort,calibrated,flag,sectionList]); // eslint-disable-line react-hooks/exhaustive-deps
  // Passage list follows the reading position unless the reader scrolls the list themselves.
  const inspector=useRef<HTMLElement>(null);
  // Small screens and fullscreen: the inspector is a sheet, hidden until a passage is selected, rising from the bottom in
  // portrait and sliding in from the right in landscape; its top bar hides it again (tap, or swipe down/right).
  const workspace=useRef<HTMLDivElement>(null);
  const [compact,setCompact]=useState(()=>typeof matchMedia!=="undefined"&&matchMedia(COMPACT).matches);
  const [portrait,setPortrait]=useState(()=>typeof matchMedia!=="undefined"&&matchMedia("(orientation: portrait)").matches);
  const [sheet,setSheet]=useState(false),[full,setFull]=useState(false);
  useEffect(()=>{
    const c=matchMedia(beta?COMPACT_BETA:COMPACT),o=matchMedia("(orientation: portrait)"),on=()=>{setCompact(c.matches);setPortrait(o.matches);};
    on();c.addEventListener("change",on);o.addEventListener("change",on);
    return()=>{c.removeEventListener("change",on);o.removeEventListener("change",on);};
  },[]);
  const sheetMode=readingOnly?null:compact||(full&&!beta)?(portrait?"bottom":"side"):null;
  const narrow=sheetMode!==null;
  useEffect(()=>setSheet(false),[paperId]);
  useEffect(()=>{setMarked(new Set());setFeedbackOpen(false);setMarkMode(false);setShowMarks(true);},[paperId,reportId]);
  useEffect(()=>{if(markMode)setShowMarks(true);},[markMode]);
  // Marking mode is offered on wide screens only (Alice, Oct 13 2026); phones keep the plain message box.
  useEffect(()=>{if(narrow)setMarkMode(false);},[narrow]);
  useEffect(()=>{
    if(!markMode)return;
    const end=()=>{stroke.current=null;},key=(e:KeyboardEvent)=>{if(e.key==="Escape")setMarkMode(false);};
    addEventListener("pointerup",end);addEventListener("pointercancel",end);addEventListener("keydown",key);
    return ()=>{removeEventListener("pointerup",end);removeEventListener("pointercancel",end);removeEventListener("keydown",key);};
  },[markMode]);
  /** A press on a flagged highlight starts a stroke that marks (or, if it was marked, unmarks); dragging applies the same to
   * every flagged highlight the pointer passes over. Only flagged sentences can be marked. */
  function paint(id:number,start:boolean){
    const band=bandOf(id);if(band!=="medium"&&band!=="high")return;
    if(start)stroke.current=marked.has(id)?"unmark":"mark";
    const act=stroke.current;if(!act)return;
    // At most MARK_LIMIT marks per message (atlas-cloud/feedback.ts MAX_PASSAGES); further marks are ignored and the bar says so.
    setMarked(m=>{if((act==="mark")===m.has(id)||(act==="mark"&&m.size>=MARK_LIMIT))return m;const n=new Set(m);if(act==="mark")n.add(id);else n.delete(id);return n;});
  }
  const marking=markMode?{marked,paint}:null;
  const shownMarks=markMode||showMarks?marked:null;
  // Feedback in progress (marking, the form open, or marks not yet sent): result settings are locked so the message, the
  // marks and the recorded settings describe the same view (Alice, Oct 13 2026).
  const feedbackLocked=markMode||feedbackOpen||marked.size>0;
  function openFeedback(){setFeedbackOpen(true);setTimeout(()=>document.getElementById('rs-feedback')?.scrollIntoView({block:'center',behavior:'smooth'}),50);}
  // Layout changes (rotation, entering or leaving fullscreen) move the page: bring the selected passage back into view.
  // Wait for the pages to re-render at the new width (resizes settle after ~600 ms) before scrolling.
  useEffect(()=>{if(selected===null||mode!=="pdf")return;setFocusId(selected);const t=setTimeout(()=>setFocusId(selected),900);return()=>clearTimeout(t);},[sheetMode,full]); // eslint-disable-line react-hooks/exhaustive-deps
  // Fullscreen uses the Fullscreen API where it exists (desktop, Android); elsewhere (iPhone Safari) the same class
  // turns the workspace into a fixed full-window layer.
  async function toggleFull(){
    if(full){setFull(false);if(document.fullscreenElement)await document.exitFullscreen().catch(()=>{});return;}
    setFull(true);
    const el=workspace.current;
    if(el?.requestFullscreen)await el.requestFullscreen({navigationUI:"hide"}).catch(()=>{});
  }
  useEffect(()=>{
    const off=()=>{if(!document.fullscreenElement)setFull(false);};
    document.addEventListener("fullscreenchange",off);
    return()=>document.removeEventListener("fullscreenchange",off);
  },[]);
  useEffect(()=>{
    document.documentElement.classList.toggle("pr-full-lock",full);
    return()=>document.documentElement.classList.remove("pr-full-lock");
  },[full]);
  useEffect(()=>()=>{if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});},[]);
  useEffect(()=>{
    const key=(e:KeyboardEvent)=>{if(e.key!=="Escape")return;if(narrow&&sheet)setSheet(false);else if(full)void toggleFull();};
    window.addEventListener("keydown",key);
    return()=>window.removeEventListener("keydown",key);
  });
  useEffect(()=>{
    if(!narrow||!sheet||selected===null)return;
    const box=inspector.current,el=box?.querySelector<HTMLElement>(".pr-detail");
    if(box&&el)box.scrollTo({top:box.scrollTop+el.getBoundingClientRect().top-box.getBoundingClientRect().top-(box.querySelector<HTMLElement>(".pr-sheet-handle")?.offsetHeight??0)-8});
  },[narrow,sheet,selected]);
  const drag=useRef<{x:number;y:number;d:number}|null>(null);
  function dragTo(d:number|null){const el=inspector.current;if(!el)return;el.style.transition=d===null?"":"none";el.style.transform=d===null?"":sheetMode==="side"?`translateX(${d}px)`:`translateY(${d}px)`;}
  const [follow,setFollow]=useState(true);
  const [readingId,setReadingId]=useState<number|null>(null);
  const followActive=follow&&passageSort==="order"&&selected===null;
  const followIndex=followActive&&readingId!==null?visible.findIndex(p=>p.id>=readingId):-1;
  const sidebarSection=followIndex>=0?Math.floor(followIndex/100):Math.min(sidebarPage,Math.max(0,Math.ceil(visible.length/100)-1));
  const sidebarStart=sidebarSection*100, sidebarEnd=Math.min(visible.length,sidebarStart+100);
  // First rectangle of each passage per page, in document order, for locating the reading line in PDF mode.
  const pageAnchors=useMemo(()=>{const anchors=new Map<number,{y:number;id:number}[]>();
    for(const p of passages){const seen=new Set<number>();for(const r of p.rectangles||[]){if(seen.has(r.page))continue;seen.add(r.page);
      if(!anchors.has(r.page))anchors.set(r.page,[]);anchors.get(r.page)!.push({y:r.y,id:p.id});}}
    return anchors;},[passages]);
  const readingFrame=useRef(0);
  // The panel only starts following once the reader has moved through the paper, so its summary stays visible on open.
  const moved=useRef(false);
  function trackReading(){
    cancelAnimationFrame(readingFrame.current);
    readingFrame.current=requestAnimationFrame(()=>{
      const scroller=viewport.current;if(!scroller)return;
      // Reading line: 30% down whichever box is actually scrolling (the viewer, or the window below the site nav).
      const own=scroller.scrollHeight>scroller.clientHeight+1, rect=scroller.getBoundingClientRect();
      const nav=parseFloat(getComputedStyle(scroller).getPropertyValue("--nav"))||0;
      const top=own?rect.top:Math.max(rect.top,nav), bottom=own?rect.bottom:Math.min(rect.bottom,window.innerHeight);
      const line=top+(bottom-top)*.3;
      let id:number|undefined;
      if(mode==="text"){
        const el=Array.from(scroller.querySelectorAll<HTMLElement>("[data-passage]")).find(e=>e.getBoundingClientRect().bottom>line);
        if(el)id=Number(el.dataset.passage);
      }else{
        const slot=Array.from(scroller.querySelectorAll<HTMLElement>("[data-page]")).find(e=>e.getBoundingClientRect().bottom>line);
        if(slot){const n=Number(slot.dataset.page),r=slot.getBoundingClientRect(),rel=Math.min(1,Math.max(0,(line-r.top)/Math.max(1,r.height)));
          for(let pg=n;pg>=1&&id===undefined;pg--){const list=pageAnchors.get(pg);if(!list?.length)continue;
            if(pg<n){id=list[list.length-1].id;break;}
            const before=list.filter(a=>a.y<=rel);id=(before.length?before[before.length-1]:list[0]).id;}}
      }
      if(id!==undefined)setReadingId(id);
    });
  }
  useEffect(()=>{const onScroll=()=>{moved.current=true;trackReading();};window.addEventListener("scroll",onScroll,{passive:true});return()=>window.removeEventListener("scroll",onScroll);});
  // Locate the reading position on open and after switching views, before any scroll happens.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(()=>{const t=setTimeout(trackReading,400);return()=>clearTimeout(t);},[mode,passages,pdf,section]);
  useEffect(()=>{
    if(followIndex<0||!moved.current||narrow)return;
    const box=inspector.current,el=box?.querySelector<HTMLElement>(`[data-list-passage="${visible[followIndex].id}"]`);
    if(!box||!el||box.scrollHeight<=box.clientHeight+1)return;
    box.scrollTo({top:box.scrollTop+el.getBoundingClientRect().top-box.getBoundingClientRect().top-56,behavior:"smooth"});
  },[followIndex,visible,sidebarSection]);
  // Phones: the passage drawer covers the PDF, so live following is off; opening the drawer jumps once to the
  // passage at the current reading position instead.
  useEffect(()=>{
    // Fall back to the first listed passage on the current page when the reading line has not been located yet.
    const id=followIndex>=0?visible[followIndex].id:passageSort==="order"&&selected===null?visible.find(p=>(p.page??0)>=page)?.id:undefined;
    if(!narrow||!sheet||id===undefined)return;
    const t=setTimeout(()=>{
      const box=inspector.current,el=box?.querySelector<HTMLElement>(`[data-list-passage="${id}"]`);
      if(box&&el)box.scrollTo({top:box.scrollTop+el.getBoundingClientRect().top-box.getBoundingClientRect().top-56,behavior:"instant"});
    },60);
    return()=>clearTimeout(t);
  },[sheet,narrow]); // eslint-disable-line react-hooks/exhaustive-deps
  const readingListId=followIndex>=0?visible[followIndex].id:undefined;
  function select(id: number) {
    const p = passages[id];
    if(!p)return;
    setSelected(id);
    setSheet(true);
    setPanel(true);
    setFocusId(mode==="pdf"?id:null);
    if(!hasTextContent(p.text)) {setFocusId(null);if(mode==="text")setTextSection(Math.floor(id/100));return;}
    if (p.page && mode === "pdf") jumpToPage(p.page);
    if (mode === "text") setTextSection(Math.floor(id/100));
  }
  function jumpToPage(n: number) {
    if(!pdf || !Number.isInteger(n) || n<1 || n>pdf.numPages)return;
    const scroller = viewport.current;
    const target = scroller?.querySelector<HTMLElement>(`[data-page="${n}"]`);
    // Narrow layouts let the page itself scroll; then scroll whichever ancestor can.
    if (scroller && target && scroller.scrollHeight <= scroller.clientHeight + 1) target.scrollIntoView({block: "start", behavior: "instant"});
    else if (scroller && target) scroller.scrollTo({
      top: scroller.scrollTop + target.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 24,
      behavior: "instant",
    });
    setPage(n);
  }
  function changePage(n: number) {
    setSelected(null);
    setFocusId(null);
    if (mode === "text") {
      // Text mode has no page elements: jump to the first passage located on that page.
      const first = passages.find(p => p.page === n && hasTextContent(p.text)) ?? passages.find(p => (p.page ?? 0) >= n);
      setPage(n);
      if (first) { setTextSection(Math.floor(first.id/100)); setTextTarget(first.id); }
      return;
    }
    jumpToPage(n);
  }
  function trackPage() {
    const scroller = viewport.current;
    if (!scroller || mode !== "pdf") return;
    const threshold = scroller.getBoundingClientRect().top + scroller.clientHeight * .3;
    const slots = Array.from(scroller.querySelectorAll<HTMLElement>("[data-page]"));
    const active = slots.find(slot => slot.getBoundingClientRect().bottom > threshold);
    if (active) setPage(Number(active.dataset.page));
  }
  const options = embedded ? [] : papers.filter(
    (p) =>
      `${p.title} ${p.filename} ${p.collection}`
        .toLowerCase()
        .includes(search.toLowerCase()) || p.id === paperId,
  );
  return (
    <section className={`pr-shell ${embedded ? "pr-embedded" : ""}${beta ? " pr-beta" : ""}`}>
      {!embedded&&<header className="pr-heading">
        <div>
          <div className="pr-eyebrow">
            OPEN TEXT DETECTOR / READING ROOM <span>LOCAL PROTOTYPE</span>
          </div>
          <h1>Read the paper. See the evidence.</h1>
          <p>
            Original pages, saved classifications, and the context between them.
          </p>
        </div>
        <div className="pr-local">
          <span /> {papers.length} local PDFs
        </div>
      </header>}
      {!embedded && <div className="pr-picker">
        <label className="pr-search">
          <Search size={16} />
          <input
            aria-label="Find a local PDF"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Find a local paper…"
          />
        </label>
        <select
          aria-label="Local paper"
          value={paperId}
          onChange={(e) => setPaperId(e.target.value)}
        >
          {options.map((p) => (
            <option value={p.id} key={p.id}>
              {p.classified ? "● " : ""}
              {p.title} · {(p.bytes / 1e6).toFixed(1)} MB
            </option>
          ))}
        </select>
      </div>}
      {error && (
        <div role="alert" className="pr-error">
          {error} <button onClick={()=>setLoadAttempt(n=>n+1)}>Retry paper</button>
        </div>
      )}
      <div className="pr-document-heading">
        <div>
          <h2>{paper?.title || "Choose a paper"}</h2>
          <p>
            {paper?.collection.replace(/^(\w+)\/(\d{4})$/,(_,v:string,y:string)=>`${v.toUpperCase()} ${y}`).replaceAll("_", " ")} <span>·</span>{" "}
            {paper ? `${(paper.bytes / 1e6).toFixed(1)} MB` : ""}{" "}
            {pdf && (
              <>
                {" "}
                <span>·</span> {pdf.numPages} pages
              </>
            )}
          </p>
        </div>
        {paper && (
          <a
            href={`${fileBasePath}/${paper.id}/file`}
            download={paper.filename}
          >
            <Download size={15} /> Original PDF
          </a>
        )}
      </div>
      <div ref={workspace} className={`pr-workspace${readingOnly?" pr-reading-only":""}${sheetMode?` pr-sheet pr-sheet-${sheetMode}`:""}${full?" pr-full":""}${beta&&!panel&&!narrow?" pr-panel-closed":""}${markMode?" is-marking":""}`}>
        <nav className="pr-map" aria-label="Page overview">
          <span>PAGES</span>
          {pdf &&
            Array.from({ length: pdf.numPages }, (_, i) => {
              const local = pagePassages.get(i+1)||EMPTY_PASSAGES;
              return (
                <button
                  key={i}
                  aria-label={`Page ${i + 1}, ${local.length} mapped passages`}
                  aria-current={page === i + 1 ? "page" : undefined}
                  className={page === i + 1 ? "active" : ""}
                  onClick={() => changePage(i + 1)}
                >
                  <b>{pdf.numPages <= 30 || i === 0 || (i + 1) % 5 === 0 ? i + 1 : ""}</b>
                  <i style={{background: pageStrip(local, bandOf)}} />
                </button>
              );
            })}
        </nav>
        <div className="pr-reading">
          <div className="pr-toolbar">
            <div className="pr-toggle">
              <button
                className={mode === "pdf" ? "active" : ""}
                onClick={() => switchMode("pdf")}
              >
                <FileText size={14} /> PDF
              </button>
              <button
                className={mode === "text" ? "active" : ""}
                onClick={() => switchMode("text")}
                disabled={!report}
              >
                <BookOpen size={14} /> Text
              </button>
            </div>
            {!readingOnly&&<button
              aria-pressed={highlights}
              onClick={() => setHighlights(!highlights)}
            >
              {highlights ? <Eye size={15} /> : <EyeOff size={15} />} Highlights
            </button>}
            {beta&&!narrow&&!readingOnly&&calibrated&&report&&<button aria-pressed={markMode} title="Mark flagged sentences you wrote without AI, to send with your feedback" onClick={()=>setMarkMode(m=>!m)}>
              <Check size={15} /> Mark passages
            </button>}
            <div className="pr-zoom">
              <button
                aria-label="Zoom out"
                disabled={zoom <= 0.5}
                onClick={() => setZoom((z) => Math.max(0.5, z - 0.15))}
              >
                <Minus size={14} />
              </button>
              <span>{Math.round(zoom * 100)}%</span>
              <button
                aria-label="Zoom in"
                disabled={zoom >= 3}
                onClick={() => setZoom((z) => Math.min(3, z + 0.15))}
              >
                <Plus size={14} />
              </button>
            </div>
            {beta&&!readingOnly&&!narrow&&<button type="button" className="pr-panel-toggle" aria-expanded={panel} aria-controls="pr-passages-panel" onClick={()=>setPanel(v=>!v)}>{panel?"Hide passages":"Passages"}</button>}
            {(!full||beta)&&<button className="pr-full-toggle" aria-label={full?"Exit fullscreen":"Fullscreen"} title={full?"Exit fullscreen":"Fullscreen"} onClick={()=>void toggleFull()}>{full?<Minimize2 size={15}/>:<Maximize2 size={15}/>}</button>}
          </div>
          <div className="pr-viewport" ref={viewport} onScroll={()=>{moved.current=true;trackPage();trackReading();}}>
            {markMode&&<div className="pr-markbar" role="status">
              <span><b>Marking mode.</b> Click or drag across flagged sentences you wrote without AI. <i className="pr-markbar-key" aria-hidden/> = marked by you. <b>{marked.size.toLocaleString()}</b> marked{marked.size>=MARK_LIMIT?` (the limit of ${MARK_LIMIT.toLocaleString()} for one message)`:""}.</span>
              <button type="button" className="pr-markbar-done" onClick={()=>{setMarkMode(false);openFeedback();}}>Done</button>
              <button type="button" onClick={()=>{setMarked(new Set());setMarkMode(false);}}>Cancel</button>
            </div>}
            {progress && (
              <div className="pr-progress">
                <Loader2 size={14} className="pr-spin" />
                {progress}
              </div>
            )}
            {pdf && lib && (
              <div className="pr-page-stack" hidden={mode !== "pdf"}>
                {Array.from({length: pdf.numPages}, (_, i) => (
                  <ScrollingPage ratios={pageRatios} frozen={mode !== "pdf"} key={`${paperId}-${i}`} pdf={pdf} lib={lib} number={i + 1}
                    width={Math.round(width * zoom)} passages={pagePassages.get(i+1)||EMPTY_PASSAGES} focusId={focusId} onFocused={setFocusId}
                    selected={current?.page === i + 1 ? selected : null}
                    highlights={!readingOnly&&highlights} onSelect={select} bandOf={bandOf} marking={marking} shownMarks={shownMarks} />
                ))}
              </div>
            )}
            {mode === "text" && report && (
              <article className="pr-text">
                <small>CLASSIFIED SOURCE TEXT</small>
                <p className="pr-text-note">
                  This is the text the model received. Formatting can differ
                  from the PDF. Long reports are shown in sections of 100 passages.
                </p>
                {passages.length ? (
                  passages.slice(textStart,textEnd).map((p, i) => (
                    <span key={p.id}>
                      {sourceChars
                        .slice(i+textStart ? passages[i+textStart - 1].end : 0, p.start)
                        .join("")}
                      {pageStarts.has(p.id) &&
                        <span className="pr-page-break" data-page-break={pageStarts.get(p.id)} aria-label={`Page ${pageStarts.get(p.id)}`}><span>Page {pageStarts.get(p.id)}</span></span>}
                      <button
                        data-passage={p.id}
                        title={`Passage ${p.id+1} · score ${p.score.toFixed(calibrated ? 4 : 3)}${calibrated ? ` · ${tierLabel(p.id)}` : ""}`}
                        className={`${highlights && hasTextContent(p.text) ? bandOf(p.id) : ""} ${selected === p.id ? "selected" : ""}`}
                        onClick={() => select(p.id)}
                      >
                        {p.text}
                      </button>
                    </span>
                  ))
                ) : (
                  <p>{report.text}</p>
                )}
                {passages.length > 0 && textEnd===passages.length &&
                  sourceChars
                    .slice(passages[passages.length - 1].end)
                    .join("")}
              </article>
            )}
          </div>
          {mode==="text" ? <footer className="pr-pagination"><button disabled={section===0} aria-label="Previous text section" onClick={()=>{setSelected(null);setTextSection(section-1);viewport.current?.scrollTo({top:0});}}><ChevronLeft size={16}/>Previous</button><span>Passages {passages.length?textStart+1:0}–{textEnd} of {passages.length}</span><button disabled={textEnd===passages.length} aria-label="Next text section" onClick={()=>{setSelected(null);setTextSection(section+1);viewport.current?.scrollTo({top:0});}}>Next<ChevronRight size={16}/></button></footer> : <footer className="pr-pagination">
            <button
              disabled={page <= 1 || !pdf}
              onClick={() => changePage(page - 1)}
            >
              <ChevronLeft size={16} /> Previous
            </button>
            <div>
              <form className="pr-page-jump" onSubmit={e=>{e.preventDefault();const n=Number(pageInput);if(pdf&&Number.isInteger(n)&&n>=1&&n<=pdf.numPages){changePage(n);setPageInput("");}}}><label>Page <input aria-label="PDF page number" type="number" min={1} max={pdf?.numPages||1} required disabled={!pdf} placeholder={String(page)} value={pageInput} onChange={e=>setPageInput(e.target.value)}/></label> of {pdf?.numPages || "—"}<button type="submit" disabled={!pdf}>Go</button></form>
            </div>
            <button
              disabled={!pdf || page >= pdf.numPages}
              onClick={() => changePage(page + 1)}
            >
              Next <ChevronRight size={16} />
            </button>
          </footer>}
        </div>
        {!readingOnly&&<aside id="pr-passages-panel" className={`pr-inspector${sheet?" open":""}`} ref={inspector} inert={narrow&&!sheet} aria-hidden={narrow&&!sheet?true:undefined}
          onWheel={()=>setFollow(false)} onTouchMove={()=>setFollow(false)}
          onKeyDown={e=>{if(["ArrowUp","ArrowDown","PageUp","PageDown","Home","End"," "].includes(e.key))setFollow(false);}}>
          <button type="button" className="pr-sheet-handle" aria-label="Hide passage evidence"
            onPointerDown={e=>{drag.current={x:e.clientX,y:e.clientY,d:0};e.currentTarget.setPointerCapture(e.pointerId);}}
            onPointerMove={e=>{const d=drag.current;if(!d)return;d.d=Math.max(0,e.clientY-d.y,sheetMode==="side"?e.clientX-d.x:0);dragTo(d.d);}}
            onPointerUp={()=>{const d=drag.current;drag.current=null;dragTo(null);if(d&&(d.d>60||d.d<6))setSheet(false);}}
            onPointerCancel={()=>{drag.current=null;dragTo(null);}}
            onKeyDown={e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();setSheet(false);}}}><i/></button>
          {beta&&!narrow&&<button type="button" className="pr-panel-close" aria-label="Close passages panel" onClick={()=>setPanel(false)}>×</button>}
          <div className="pr-inspector-top">
            <div className="pr-eyebrow">MODEL OBSERVATIONS</div>
            <h3>Passage evidence</h3>
            <p>Experimental results, in context.</p>
            {detail && detail.reports.length > 0 && (
              <select
                aria-label="Saved classification"
                disabled={feedbackLocked}
                title={feedbackLocked?"Locked while you give feedback":undefined}
                value={reportId}
                onChange={(e) => {
                  setReportId(e.target.value);
                  setTextSection(0);
                  setSidebarPage(0);
                  setSelected(null);
                  setFocusId(null);
                }}
              >
                {detail.reports.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.model.name}
                  </option>
                ))}
              </select>
            )}
          </div>
          {!report && detail && (
            <div className="pr-empty">
              <BookOpen size={28} />
              <h4>A clean reading copy</h4>
              <p>
                This PDF has no saved classification. Browse the original pages,
                or choose a paper marked ● to explore real model results.
              </p>
            </div>
          )}
          {report && (
            <>
              {report.result.notice && !calibrated && <p className="pr-mapping-note">{report.result.notice}</p>}
              <div className="pr-coverage">
                <Check size={15} />
                <span>
                  <b>
                    {mapped} / {report.result.segments.length}
                  </b>{" "}
                  classified passages matched to PDF
                </span>
              </div>
              <p className="pr-mapping-note">
                {supportsSavedPositions(detail?.position_maps?.[report.text_sha256 || ""]) ? "Highlights use saved word positions from the verified extraction." : "Only unique text matches are highlighted."} All classified passages remain available below. {detail?.version_verified ? "PDF and classification input hashes verified." : "PDF version correspondence is unverified."}
              </p>
              {calibrated && flagStats ? <>
                {/* Beta: the score, counting mode, FPR and colour sliders live only in the left summary; this panel shows evidence. */}
                {!beta && <><div className="pr-paper-score">
                  <strong>{flagStats.counted < MIN_SENTENCES ? "—" : sharePct(flagStats.flagged / flagStats.counted)}</strong>
                  <span>{flagStats.counted < MIN_SENTENCES ? `Too short to score reliably (${flagStats.counted} counted sentences; fewer than ${MIN_SENTENCES})` : <>of {modeName(flagMode)} sentences flagged: <b>{flagStats.flagged.toLocaleString()}</b> of {flagStats.counted.toLocaleString()}</>}</span>
                </div>
                <div className="pr-modes" role="group" aria-label="Sentences counted"><span>Count</span>
                  <label><input type="checkbox" checked={flag.app} onChange={e=>setFlagMode(flagModeOf(flag.bib,e.target.checked,flag.fpr))}/> Appendices</label>
                  <label title="Human false-positive rate of the sentence threshold">FPR <select aria-label="False-positive rate" value={flag.fpr} onChange={e=>setFlagMode(flagModeOf(flag.bib,flag.app,Number(e.target.value) as typeof flag.fpr))}>{(beta?FLAG_FPRS:LEGACY_FLAG_FPRS).map(f=><option key={f} value={f}>{fprLabel(f)}</option>)}</select></label>
                </div>
                {customControls}</>}
                {!beta && reportSummary.excluded!>0 && <p className="pr-mapping-note">{reportSummary.excluded!.toLocaleString()} punctuation-only fragments are not scored. View them under “Punctuation / artifacts”.</p>}
                {!beta && <p className="pr-mapping-note">{modelLabel(FLAG_MODEL)}. Yellow and red are exactly the {flagStats.flagged.toLocaleString()} flagged sentences in this share{flagStats.strong ? ` (${flagStats.strong.toLocaleString()} red)` : ""}: scores at or above this counting mode’s {fprLabel(flag.fpr)} false-positive threshold ({thresholdText(flag.threshold)}, calibrated on 2,400 human-written papers). Red: {flag.strong.toFixed(2)} or above, a score about 0.015% of human sentences reach. {flagStats.skipped.bib + flagStats.skipped.app ? `Grey sentences (${[flagStats.skipped.bib && `${flagStats.skipped.bib.toLocaleString()} bibliography`, flagStats.skipped.app && `${flagStats.skipped.app.toLocaleString()} appendix`].filter(Boolean).join(", ")}) are left out of this count.` : "Every sentence is counted in this mode."} A flag means the text resembles AI-generated text, not proof of authorship.</p>}
                <div className="pr-legend"><span><i className="low"/> Below {thresholdText(flag.threshold)}</span><span><i className="medium"/> Flagged, below {flag.strong.toFixed(2)}</span><span><i className="high"/> {flag.strong.toFixed(2)} or above</span><span><i className="skip"/> Not counted</span></div>
              </> : <>
                <div className="pr-paper-score"><strong>{paperRate(reportSummary,threshold)===null?'—':`${(paperRate(reportSummary,threshold)!*100).toFixed(1)}%`}</strong><span>of passages above threshold</span><label>Threshold <select aria-label="Reader score threshold" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}>{scoreThresholds.map(t=><option key={t} value={t}>{t.toFixed(2)}</option>)}</select></label></div>
                {!beta && reportSummary.excluded!>0 && <p className="pr-mapping-note">{reportSummary.excluded!.toLocaleString()} punctuation-only fragments excluded from scores and highlights. View them under “Punctuation / artifacts”; selecting them will not navigate the PDF.</p>}
                <div className="pr-legend"><span><i className="low"/> Low ≤ 0.20</span><span><i className="medium"/> Mid ≤ 0.80</span><span><i className="high"/> High &gt; 0.80</span></div><p className="pr-mapping-note">This model has no calibrated threshold here: highlight colors use fixed score bands, and the chosen threshold controls counts and filtering.</p>
              </>}
              {current ? (
                <div className="pr-detail">
                  <div>
                    <span className={`pr-pill${calibrated ? ` ${bandOf(current.id)}` : ""}`}>{!hasTextContent(current.text) ? "Excluded fragment" : calibrated ? tierLabel(current.id) : label(current)}</span>
                    <button
                      aria-label="Clear selected passage"
                      onClick={() => setSelected(null)}
                    >
                      ×
                    </button>
                  </div>
                  <h4>
                    Passage {current.id + 1}{" "}
                    <small>
                      {current.mapped
                        ? `· Page ${current.page}`
                        : "· Not located"}
                    </small>
                  </h4>
                  <blockquote>{current.text}</blockquote>
                  <div className="pr-score">
                    <span>Model score</span>
                    <b>{current.score.toFixed(calibrated ? 4 : 3)}</b>
                  </div>
                  {calibrated && hasTextContent(current.text) && <p className="pr-mapping-note">{counted(current.id) ? `Counted (${SECTION_NAMES[sectionList[current.id]]}). ${custom.on ? `Flagged above ${thresholdText(flag.threshold)}, strongly flagged above ${flag.strong.toFixed(4)} (your custom thresholds).` : flag.threshold === flag.strong ? `Flagged above ${thresholdText(flag.threshold)} (0.1% false-positive rate).` : `Flagged above ${thresholdText(flag.threshold)} (${fprLabel(flag.fpr)} false-positive rate), strongly flagged above ${flag.strong.toFixed(4)}.`}` : `In the ${SECTION_NAMES[sectionList[current.id]]}, which this counting mode leaves out. Turn on ${sectionList[current.id] === "bib" ? "Bibliography" : "Appendices"} to count it.`}</p>}
                  <p>
                    Scores are experimental model outputs, not probabilities of
                    authorship.
                  </p>
                  <div className="pr-detail-nav">
                    <button
                      disabled={current.id === 0}
                      onClick={() => select(current.id - 1)}
                    >
                      <ArrowLeft size={14} /> Previous
                    </button>
                    <button
                      disabled={current.id === passages.length - 1}
                      onClick={() => select(current.id + 1)}
                    >
                      Next <ArrowRight size={14} />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="pr-hint">
                  Select a highlight on the page or a passage below to inspect
                  it.
                </div>
              )}
              <div className="pr-list-head">
                <h4>
                  Passages <span>{visible.length}</span>
                </h4>
                {beta ? <label className="pr-follow"><input type="checkbox" checked={filter==="elevated"} onChange={e=>{setFilter(e.target.checked?"elevated":"all");setSidebarPage(0);}}/> Only flagged</label> : <select
                  aria-label="Filter passages"
                  value={filter}
                  onChange={(e) => {setFilter(e.target.value);setSidebarPage(0);}}
                >
                  <option value="all">Text passages</option><option value="artifacts">Punctuation / artifacts</option>
                  <option value="elevated">{calibrated ? "Flagged (counted)" : "Above threshold"}</option>
                  {calibrated && <option value="skipped">Not counted</option>}
                  <option value="unmapped">Not located</option>
                </select>}
              </div>
              {!narrow&&<div className="pr-follow-row">
                <label className="pr-follow"><input type="checkbox" checked={follow} onChange={e=>setFollow(e.target.checked)}/> Follow reading position</label>
                {follow&&!followActive&&<span>{passageSort!=="order"?"Paused: sorted by score":"Paused while a passage is selected"}</span>}
              </div>}
              <label className="pr-sort">Sort passages by: <select aria-label="Sort passages" value={passageSort} onChange={e=>{setPassageSort(e.target.value);setSidebarPage(0);}}><option value="order">Document order</option><option value="high">Highest AI score first</option><option value="low">Lowest AI score first</option></select></label>
              <div className="pr-passages">
                {visible.slice(sidebarStart,sidebarEnd).map((p) => (
                  <button
                    key={p.id}
                    data-list-passage={p.id}
                    aria-current={readingListId===p.id?"location":undefined}
                    className={`${hasTextContent(p.text)?bandOf(p.id):"unknown"} ${selected === p.id ? "active" : ""} ${readingListId===p.id ? "reading" : ""}`}
                    onClick={() => select(p.id)}
                  >
                    <div>
                      <span
                        className={`pr-dot ${hasTextContent(p.text)?bandOf(p.id):"unknown"}`}
                      />
                      <b>{p.id + 1}</b>
                      <span>{p.mapped ? `p. ${p.page}` : "Not located"}{calibrated && sectionList[p.id] !== "body" ? ` · ${SECTION_NAMES[sectionList[p.id]]}` : ""}</span>
                      <strong>{calibrated ? p.score.toFixed(4) : p.score.toFixed(2)}</strong>
                    </div>
                    <p>{p.text}</p>
                  </button>
                ))}
                {visible.length>100 && <nav className="pr-sidebar-pages" aria-label="Passage list pages"><button aria-label="Previous passage list page" disabled={sidebarSection===0} onClick={()=>{setFollow(false);setSidebarPage(sidebarSection-1);}}>Previous</button><span>{sidebarStart+1}–{sidebarEnd} of {visible.length}</span><button aria-label="Next passage list page" disabled={sidebarEnd===visible.length} onClick={()=>{setFollow(false);setSidebarPage(sidebarSection+1);}}>Next</button></nav>}
                {!visible.length && !progress && (
                  <p className="pr-hint">No passages match this filter.</p>
                )}
              </div>
            </>
          )}
        </aside>}
        {beta&&summarySlot&&createPortal(<ReaderSummary paperId={paperId} reports={detail?.reports||[]} reportId={reportId} loading={!detail} calibrated={calibrated}
          onPick={id=>{setReportId(id);setTextSection(0);setSidebarPage(0);setSelected(null);setFocusId(null);}}
          flagMode={flagMode} onFlagMode={setFlagMode} threshold={threshold} flagged={flaggedList}
          onSelect={select} locked={feedbackLocked} showMarks={showMarks} onShowMarks={setShowMarks} custom={customControls} customYellow={custom.on?custom.y:undefined}
          marked={[...marked].sort((a,b)=>a-b)} canMark={!narrow&&calibrated} onStartMarking={()=>{setMode("pdf");setMarkMode(true);}} marking={markMode} onStopMarking={keep=>{if(!keep)setMarked(new Set());setMarkMode(false);if(keep)openFeedback();}} onClearMarks={()=>setMarked(new Set())} feedbackOpen={feedbackOpen} onFeedbackOpen={o=>{if(o)openFeedback();else setFeedbackOpen(false);}} onOpenPanel={()=>{setFilter("elevated");setSidebarPage(0);setPanel(true);setSheet(true);}}/>,summarySlot)}
        {full&&!beta&&<button className="pr-exit-full" aria-label="Exit fullscreen" title="Exit fullscreen" onClick={()=>void toggleFull()}><Minimize2 size={18}/></button>}
      </div>
    </section>
  );
}
