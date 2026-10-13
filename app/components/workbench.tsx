"use client";
import { useState, useEffect, useCallback } from "react";
import SegmentExplorer from "@/components/segment-explorer";
import {
  FileText,
  BookOpen,
  Image as ImageIcon,
  History,
  Mail,
  Code2,
  Settings2,
  Trash2,
  Upload,
  ChevronRight,
  ChevronDown,
  Plus,
  Download,
  Copy,
  ArrowLeft,
  Search,
  Loader2,
  CircleHelp,
  SlidersHorizontal,
  StickyNote,
  ChartPie,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Check,
  ExternalLink,
  X,
  ShieldCheck,
} from "lucide-react";
import {
  SidebarProvider,
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import {
  Table,
  TableHeader,
  TableHead,
  TableRow,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import RunPerformance, { type RunMetrics, type InferenceInfo } from "@/components/run-performance";
import PDFReader from "@/components/pdf-reader";
import PaperLibrary from "@/components/paper-library";
import CachedFindings from "@/components/cached-findings";
import { Toaster, toast } from "sonner";

type Model = {
  id: string;
  name: string;
  provider: string;
  task: string;
  model_id: string;
  base_model_id?: string;
  endpoint?: string;
  enabled: boolean;
  lower_threshold: number;
  upper_threshold: number;
  has_api_key?: boolean;
};
type Scan = {
  source_locked?: number;
  source?: string;
  id: string;
  title: string;
  text: string;
  status: string;
  created_at: string;
  word_count: number;
  kind: string;
  notes?: string;
  model: Model;
  error?: { message: string };
  result?: {
    completed_at?: string;
    performance?: RunMetrics;
    inference?: InferenceInfo;
    score: number;
    score_type?: string;
    raw_score?: number;
    tokens?: { start: number; end: number; score: number; raw_score: number; label: string }[];
    localization?: { notice: string };
    thresholds?: { raw_ai_above?: number; scope?: string };
    label: string;
    notice?: string;
    segments: { start: number; end: number; score: number; label: string }[];
    plagiarism?: {
      notice: string;
      matched_word_fraction: number;
      matches: { title: string; matched_words: number }[];
    };
  };
  sample?: boolean;
};
const SAMPLE = `The library on our street used to close at five. If you worked a regular shift, you almost never made it through the doors before the librarian turned the sign around. Last autumn, a few of us asked the council to try later hours on Thursdays. We expected a polite reply and nothing more. Instead, they agreed to a three-month trial.\n\nLibraries play a vital role in fostering vibrant, connected communities. Beyond their traditional function as repositories of knowledge, they serve as inclusive spaces where individuals can explore new perspectives and develop valuable skills. By extending opening hours, libraries can increase accessibility and ensure that their resources benefit a broader range of residents. This commitment to accessibility helps create a more equitable and engaged society.\n\nI went last Thursday after dinner. There were two teenagers working on a poster, a man reading a newspaper, and someone asleep beside a stack of cookbooks. I borrowed a novel I had been meaning to read for years. On the way out, the librarian said they had counted forty extra visitors that evening. That was enough for me. I wrote to the council again the next morning.`;
const sampleModel: Model = {
  id: "sample",
  name: "Illustrative sample",
  provider: "sample",
  task: "text",
  model_id: "sample",
  enabled: true,
  lower_threshold: 0.2,
  upper_threshold: 0.8,
};
function sampleReport(): Scan {
  const p = SAMPLE.indexOf("Libraries play"),
    end = SAMPLE.indexOf("\n\nI went");
  return {
    id: "sample",
    title: "A later evening at the library",
    text: SAMPLE,
    status: "completed",
    created_at: new Date().toISOString(),
    word_count: SAMPLE.split(/\s+/).length,
    kind: "text",
    model: sampleModel,
    sample: true,
    result: {
      score: 0.43,
      label: "ai_assisted",
      segments: [
        { start: 0, end: p, score: 0.06, label: "human" },
        { start: p, end, score: 0.92, label: "ai" },
        { start: end, end: SAMPLE.length, score: 0.08, label: "human" },
      ],
    },
  };
}
async function api(path: string, init: RequestInit = {}) {
  const res = await fetch("/backend" + path, {
    ...init,
    headers: {
      ...(init.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init.headers,
    },
  });
  const data: any = await res
    .json()
    .catch(() => ({ detail: "Unexpected server response" }));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(
      typeof d === "string"
        ? d
        : d?.message ||
            (Array.isArray(d)
              ? d.map((e: { msg: string }) => e.msg).join(", ")
              : "Request failed"),
    );
  }
  return data;
}
const countWords = (s: string) => (s.trim() ? s.trim().split(/\s+/).length : 0);
function labelName(label?: string) {
  return (
    (
      { human: "Human", ai: "AI", ai_assisted: "AI-assisted", ai_evidence: "AI evidence", low_evidence: "Lower evidence", below_threshold: "Below threshold", uncertain: "Inconclusive" } as Record<
        string,
        string
      >
    )[label || ""] || "Pending"
  );
}
function ModelSelect({
  value,
  onChange,
  models,
}: {
  value: string;
  onChange: (v: string) => void;
  models: Model[];
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger aria-label="Detection model">
        <SelectValue placeholder="Choose a model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((m) => (
          <SelectItem key={m.id} value={m.id}>
            {m.name}
            {!m.enabled ? " · Pending setup" : ""}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
export default function Home() {
  const [view, setView] = useState("dashboard"),
    [text, setText] = useState(""),
    [inputTab, setInputTab] = useState("text"),
    [url, setUrl] = useState(""),
    [files, setFiles] = useState<File[]>([]),
    [plag, setPlag] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [models, setModels] = useState<Model[]>([]),
    [modelId, setModelId] = useState(""),
    [defaults, setDefaults] = useState<Record<string, string>>({}),
    [scan, setScan] = useState<Scan | null>(null),
    [samples, setSamples] = useState(false),
    [info, setInfo] = useState(""),
    [history, setHistory] = useState<Scan[]>([]),
    [total, setTotal] = useState(0),
    [query, setQuery] = useState(""),
    [status, setStatus] = useState("all"),
    [trash, setTrash] = useState(false),
    [offset, setOffset] = useState(0),
    [historyLoading, setHistoryLoading] = useState(false),
    [connected, setConnected] = useState(false),
    [modelDialog, setModelDialog] = useState<Model | null>(null),
    [keys, setKeys] = useState<
      { id: string; name: string; scopes: string[]; revoked_at?: string }[]
    >([]),
    [keyName, setKeyName] = useState(""),
    [newKey, setNewKey] = useState("");
  const navigate = useCallback((v: string) => {
    setView(v);
    setScan(null);
    setError("");
    window.history.pushState({}, "", v === "dashboard" ? "/workbench" : "/" + v);
  }, []);
  useEffect(() => {
    const context = (
      document as Document & {
        modelContext?: {
          registerTool: (
            tool: unknown,
            options: { signal: AbortSignal },
          ) => void | Promise<void>;
        };
      }
    ).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    try {
      Promise.resolve(
        context.registerTool(
          {
            name: "stage_text_check",
            title: "Prepare a text check",
            description:
              "Place text in the visible detector without submitting it or running a model.",
            inputSchema: {
              type: "object",
              properties: { text: { type: "string", maxLength: 500000 } },
              required: ["text"],
              additionalProperties: false,
            },
            annotations: { readOnlyHint: false, untrustedContentHint: true },
            async execute(input: unknown) {
              if (
                !input ||
                typeof input !== "object" ||
                Object.keys(input).some((k) => k !== "text") ||
                typeof (input as { text: unknown }).text !== "string" ||
                (input as { text: string }).text.length > 500000
              )
                throw new Error("Provide text of at most 500,000 characters");
              const value = (input as { text: string }).text;
              navigate("dashboard");
              setText(value);
              setInputTab("text");
              await new Promise<void>((resolve) =>
                requestAnimationFrame(() => resolve()),
              );
              return {
                staged: true,
                words: countWords(value),
                submitted: false,
              };
            },
          },
          { signal: lifecycle.signal },
        ),
      ).catch(() => {});
    } catch {}
    return () => lifecycle.abort();
  }, [navigate]);
  const refreshModels = useCallback(async () => {
    try {
      const d = await api("/v1/models");
      setModels(d.items);
      setDefaults(d.defaults);
      setModelId((v) => v || d.defaults.default_text_model || "");
      setConnected(true);
    } catch (e) {
      setConnected(false);
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    refreshModels();
    const route = () => {
      const v = (window.location.pathname.split("/")[1] === "workbench" ? "dashboard" : window.location.pathname.split("/")[1]) || "dashboard";
      setView(
        [
          "dashboard",
          "history",
          "cached-findings",
          "pdf-reader",
          "papers",
          "models",
          "apikey",
          "gmail",
          "image-detection",
        ].includes(v)
          ? v
          : "dashboard",
      );
      setScan(null);
    };
    route();
    window.addEventListener("popstate", route);
    return () => window.removeEventListener("popstate", route);
  }, [refreshModels]);
  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const d = await api(
        `/v1/scans?limit=20&offset=${offset}&q=${encodeURIComponent(query)}${status === "all" ? "" : "&status=" + status}&trash=${trash}`,
      );
      setHistory(d.items);
      setTotal(d.total);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setHistoryLoading(false);
    }
  }, [offset, query, status, trash]);
  useEffect(() => {
    if (view === "history") {
      const timer = setTimeout(loadHistory, 200);
      return () => clearTimeout(timer);
    }
    if (view === "apikey")
      api("/v1/keys")
        .then((d) => setKeys(d.items))
        .catch((e) => toast.error(e.message));
  }, [view, loadHistory]);
  useEffect(() => {
    if (!scan || scan.sample || !["queued", "running"].includes(scan.status))
      return;
    let active = true;
    const t = setInterval(
      () =>
        api("/v1/scans/" + scan.id)
          .then((s) => {
            if (active) setScan(s);
          })
          .catch((e) => {
            if (active) setError(e.message);
          }),
      1800,
    );
    return () => {
      active = false;
      clearInterval(t);
    };
  }, [scan]);
  async function run() {
    setError("");
    setBusy(true);
    try {
      let result;
      if (view === "image-detection" || inputTab === "upload") {
        if (!files.length) throw new Error("Choose a file first.");
        const form = new FormData();
        files.forEach((f) =>
          form.append(view === "image-detection" ? "file" : "files", f),
        );
        form.append(
          "model_id",
          view === "image-detection"
            ? defaults.default_image_model || ""
            : modelId,
        );
        if (view !== "image-detection")
          form.append("check_plagiarism", String(plag));
        const d = await api(
          view === "image-detection" ? "/v1/images" : "/v1/uploads",
          { method: "POST", body: form },
        );
        if (d.errors?.length)
          toast.warning(`${d.errors.length} file(s) could not be read.`);
        if (d.items?.length > 1) {
          toast.success(`${d.items.length} documents queued`);
          navigate("history");
          loadHistory();
          return;
        }
        result = d.items?.[0] || d;
      } else {
        result = await api(inputTab === "url" ? "/v1/scans/url" : "/v1/scans", {
          method: "POST",
          body: JSON.stringify({
            ...(inputTab === "url" ? { url } : { text }),
            model_id: modelId,
            check_plagiarism: plag,
          }),
        });
      }
      setScan(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function openScan(id: string) {
    try {
      setScan(await api("/v1/scans/" + id));
    } catch (e) {
      toast.error((e as Error).message);
    }
  }
  function useSample() {
    setSamples(false);
    setScan(sampleReport());
  }
  async function saveModel(m: Model, key: string) {
    const {
      name,
      provider,
      task,
      model_id,
      base_model_id,
      endpoint,
      enabled,
      lower_threshold,
      upper_threshold,
    } = m;
    await api("/v1/models" + (m.id ? "/" + m.id : ""), {
      method: m.id ? "PUT" : "POST",
      body: JSON.stringify({
        name,
        provider,
        task,
        model_id,
        base_model_id: base_model_id || null,
        endpoint: endpoint || null,
        enabled,
        lower_threshold,
        upper_threshold,
        ...(key ? { api_key: key } : {}),
      }),
    });
    setModelDialog(null);
    await refreshModels();
    toast.success("Model settings saved");
  }
  const nav = (
    v: string,
    title: string,
    Icon: typeof FileText,
    badge?: string,
  ) => (
    <SidebarMenuItem>
      <SidebarMenuButton
        isActive={view === v && !scan}
        onClick={() => navigate(v)}
      >
        <Icon />
        <span>{title}</span>
        {badge && <span className="new-tag">{badge}</span>}
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
  const selected = models.find((m) => m.id === modelId);
  return (
    <SidebarProvider
      style={{ "--sidebar-width": "216px" } as React.CSSProperties}
    >
      <Toaster position="bottom-right" richColors />
      <Sidebar>
        <SidebarHeader>
          <button
            className="fox-home"
            aria-label="Open Text Detector home"
            onClick={() => navigate("dashboard")}
          >
            <img src="/favicon.svg" alt="Open Text Detector" />
          </button>
        </SidebarHeader>
        <SidebarContent>
          <SidebarMenu>
            {nav("dashboard", "Text Detection", FileText)}
            {nav("image-detection", "Image Detection", ImageIcon)}
            <div className="nav-section">HISTORY</div>
            {nav("history", "All Checks", History)}
            {nav("cached-findings", "Cached findings", FileText)}
            {nav("papers", "Paper library", FileText)}
            {nav("pdf-reader", "PDF reader", BookOpen, "Local")}
            <div className="nav-section">MORE FEATURES</div>
            {nav("gmail", "Gmail Integration", Mail, "NEW")}
            {nav("apikey", "API", Code2)}
          </SidebarMenu>
        </SidebarContent>
        <SidebarFooter>
          <button className="whats-new" onClick={() => setInfo("What’s new")}>
            <img src="/reference/fox.webp" alt="" />
            <span>✧ &nbsp; See what’s new</span>
          </button>
          <button className="side-more" onClick={() => navigate("models")}>
            <SlidersHorizontal size={17} /> Models & settings
          </button>
          <button className="org-card" onClick={() => setInfo("Integrations")}>
            <span>▣</span>
            <div>
              Open Text Detector for Organizations
              <small>LMS, API & custom solutions</small>
            </div>
          </button>
          <button className="plan-card" onClick={() => navigate("models")}>
            <Settings2 size={18} />
            <div>
              Detection model
              <small>
                {selected?.enabled
                  ? selected.name
                  : "No model enabled"}
              </small>
            </div>
            <ChevronRight size={15} />
          </button>
        </SidebarFooter>
      </Sidebar>
      <main className={"workspace " + (scan ? "has-report" : "")}>
        <header className="topbar">
          <SidebarTrigger />
          {scan ? (
            <>
              <button
                className="icon-btn"
                aria-label="Back to detector"
                onClick={() => setScan(null)}
              >
                <ArrowLeft size={19} />
              </button>
              <img
                className="report-wordmark"
                src="/reference/wordmark.svg"
                alt="Open Text Detector"
              />
              <button
                className="new-scan"
                title="New scan"
                onClick={() => {
                  setScan(null);
                  setText("");
                  navigate("dashboard");
                }}
              >
                <Plus size={18} />
              </button>
            </>
          ) : (
            <span className="top-spacer" />
          )}
          <div className="top-actions">
            <span className="backend-state">
              {connected ? "Local workspace" : "Connecting…"}
            </span>
            <button className="orange-small" onClick={() => navigate("models")}>
              Model settings
            </button>
            <button
              className="account"
              onClick={() => setInfo("Your workspace")}
            >
              <span>
                A<small>Local</small>
              </span>
              Alice
            </button>
          </div>
        </header>
        {scan ? (
          <Report
            scan={scan}
            setScan={setScan}
            close={() => setScan(null)}
            edit={() => {
              setText(scan.text);
              navigate("dashboard");
              setInputTab("text");
            }}
          />
        ) : (
          <>
            {(view === "dashboard" || view === "image-detection") && (
              <>
                <section className="detection-surface">
                  <button
                    className="image-promo"
                    onClick={() =>
                      navigate(
                        view === "dashboard" ? "image-detection" : "dashboard",
                      )
                    }
                  >
                    <span>{view === "dashboard" ? "NEW" : "TEXT"}</span>
                    {view === "dashboard"
                      ? "Try image detection"
                      : "Try text detection"}
                    <ImageIcon size={14} />
                  </button>
                  <div className="headline">
                    {view === "dashboard" ? (
                      <img
                        src="/reference/headline-text.svg"
                        alt="Detect AI in text with Open Text Detector."
                      />
                    ) : (
                      <h1>
                        Detect AI <em>in images</em> with{" "}
                        <img src="/favicon.svg" alt="" />
                        Open Text Detector.
                      </h1>
                    )}
                  </div>
                  <div className="editor-card">
                    {view === "dashboard" ? (
                      <Tabs
                        value={inputTab}
                        onValueChange={(v) => {
                          setInputTab(v);
                          setError("");
                        }}
                      >
                        <div className="editor-top">
                          <TabsList>
                            <TabsTrigger value="text">Text</TabsTrigger>
                            <TabsTrigger value="upload">Upload</TabsTrigger>
                            <TabsTrigger value="url">URL</TabsTrigger>
                          </TabsList>
                          <span>50 words minimum</span>
                        </div>
                        <TabsContent value="text">
                          <div
                            className="text-input"
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={(e) => {
                              e.preventDefault();
                              setFiles(Array.from(e.dataTransfer.files));
                              setInputTab("upload");
                            }}
                          >
                            <textarea
                              aria-label="Text to check for AI"
                              placeholder="Enter at least 50 words or drop a file to check for AI."
                              value={text}
                              onChange={(e) => setText(e.target.value)}
                            />
                            <div className="text-bottom">
                              <span>
                                {text ? `${countWords(text)} words` : ""}
                              </span>
                              <button
                                className="icon-btn"
                                aria-label="Clear text"
                                onClick={() => setText("")}
                              >
                                <Trash2 size={18} />
                              </button>
                            </div>
                          </div>
                        </TabsContent>
                        <TabsContent value="upload">
                          <FileDrop files={files} setFiles={setFiles} />
                        </TabsContent>
                        <TabsContent value="url">
                          <div className="url-input">
                            <label htmlFor="article-url">
                              Enter a website URL
                            </label>
                            <input
                              id="article-url"
                              className="field"
                              placeholder="https://example.com/article"
                              value={url}
                              onChange={(e) => setUrl(e.target.value)}
                            />
                            <small>
                              Check the text of a publicly accessible article.
                            </small>
                          </div>
                        </TabsContent>
                      </Tabs>
                    ) : (
                      <FileDrop files={files} setFiles={setFiles} image />
                    )}
                    {view === "dashboard" && (
                      <button
                        className="sample-link"
                        onClick={() => setSamples(true)}
                      >
                        Try a sample text &gt;
                      </button>
                    )}
                    <div className="scan-options">
                      {view === "dashboard" && (
                        <label className="plagiarism-choice">
                          <span className="cyan-icon">▧</span>Check for
                          plagiarism{" "}
                          <span
                            className="muted-chip"
                            title="Matches against documents in your private reference corpus"
                          >
                            Reference corpus
                          </span>
                          <Checkbox
                            checked={plag}
                            onCheckedChange={(v) => setPlag(v === true)}
                          />
                        </label>
                      )}
                    </div>
                    <div className="submit-row">
                      <button
                        className="primary"
                        disabled={
                          busy ||
                          (view === "dashboard"
                            ? inputTab === "text"
                              ? countWords(text) < 50
                              : inputTab === "upload"
                                ? !files.length
                                : !url
                            : !files.length)
                        }
                        onClick={run}
                      >
                        {busy ? (
                          <>
                            <Loader2 className="spin" size={16} />
                            Checking…
                          </>
                        ) : (
                          "Check for AI"
                        )}
                      </button>
                    </div>
                    {error && (
                      <div className="inline-error" role="alert">
                        {error}{" "}
                        <button onClick={() => navigate("models")}>
                          Model settings
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="under-editor">
                    <span>Detection model</span>
                    <button onClick={() => navigate("models")}>
                      {view === "dashboard"
                        ? selected?.name || (connected ? "Choose a detector" : "Loading models…")
                        : "Configure image model"}
                      <ChevronDown size={13} />
                    </button>
                    {view === "dashboard" && selected && !selected.enabled && (
                      <span className="pending-note">Access pending</span>
                    )}
                  </div>
                </section>
                <div className="landscape" aria-hidden="true" />
              </>
            )}
            {view === "pdf-reader" && <PDFReader api={api} />}
            {view === "papers" && <PaperLibrary models={models} defaultModel={defaults.default_text_model || modelId} api={api} open={openScan} />}
            {view === "cached-findings" && <CachedFindings models={models} defaultModel={defaults.default_text_model || modelId} api={api} open={openScan} />}
            {view === "history" && (
              <section className="content-page">
                <div className="page-heading">
                  <h1>All Checks</h1>
                  <button
                    className="primary"
                    onClick={() => navigate("dashboard")}
                  >
                    <Plus size={17} />
                    New scan
                  </button>
                </div>
                <div className="history-toolbar">
                  <div className="search-field">
                    <Search size={17} />
                    <input
                      aria-label="Search checks"
                      placeholder="Search your checks"
                      value={query}
                      onChange={(e) => {
                        setQuery(e.target.value);
                        setOffset(0);
                      }}
                    />
                  </div>
                  <Select
                    value={status}
                    onValueChange={(v) => {
                      setStatus(v);
                      setOffset(0);
                    }}
                  >
                    <SelectTrigger aria-label="Filter status">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {[
                        "all",
                        "completed",
                        "queued",
                        "running",
                        "failed",
                        "cancelled",
                      ].map((v) => (
                        <SelectItem key={v} value={v}>
                          {v === "all" ? "All statuses" : v}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <button
                    className={"secondary " + (trash ? "chosen" : "")}
                    onClick={() => {
                      setTrash(!trash);
                      setOffset(0);
                    }}
                  >
                    <Trash2 size={15} />
                    {trash ? "Show active" : "Trash"}
                  </button>
                  <a className="secondary" href="/backend/v1/export">
                    <Download size={15} />
                    Export
                  </a>
                </div>
                <div className="history-table">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Document</TableHead>
                        <TableHead>Result</TableHead>
                        <TableHead>Words</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead>
                          <span className="sr-only">Actions</span>
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {history.map((s) => (
                        <TableRow key={s.id}>
                          <TableCell>
                            <button
                              className="document-link"
                              onClick={() => !trash && openScan(s.id)}
                            >
                              <FileText size={17} />
                              {s.title}
                            </button>
                          </TableCell>
                          <TableCell>
                            <span
                              className={"badge " + (s.result?.label || "")}
                            >
                              {s.result ? labelName(s.result.label) : s.status}
                            </span>
                          </TableCell>
                          <TableCell>{s.word_count}</TableCell>
                          <TableCell>
                            {new Date(s.created_at).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            <button
                              className="icon-btn"
                              aria-label={
                                trash ? "Restore check" : "Move check to trash"
                              }
                              onClick={async () => {
                                try {
                                  await api(
                                    "/v1/scans/" +
                                      s.id +
                                      (trash ? "/restore" : ""),
                                    { method: trash ? "POST" : "DELETE" },
                                  );
                                  loadHistory();
                                } catch (e) {
                                  toast.error((e as Error).message);
                                }
                              }}
                            >
                              {trash ? (
                                <RotateCcw size={16} />
                              ) : (
                                <Trash2 size={16} />
                              )}
                            </button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  {!history.length && (
                    <div className="empty-state">
                      {historyLoading ? (
                        <Loader2 className="spin" />
                      ) : (
                        <History />
                      )}
                      <h2>
                        {historyLoading
                          ? "Loading checks…"
                          : query
                            ? "No matching checks"
                            : "Your checks will appear here"}
                      </h2>
                      <p>
                        {query
                          ? "Try another search."
                          : "Scan text or upload a document to get started."}
                      </p>
                      {!query && (
                        <button
                          className="secondary"
                          onClick={() => navigate("dashboard")}
                        >
                          Check text
                        </button>
                      )}
                    </div>
                  )}
                </div>
                <div className="pagination">
                  <span>{total} checks</span>
                  <button
                    disabled={!offset}
                    onClick={() => setOffset(offset - 20)}
                  >
                    Previous
                  </button>
                  <span>{Math.floor(offset / 20) + 1}</span>
                  <button
                    disabled={offset + 20 >= total}
                    onClick={() => setOffset(offset + 20)}
                  >
                    Next
                  </button>
                </div>
              </section>
            )}
            {view === "models" && (
              <section className="content-page">
                <div className="page-heading">
                  <h1>Models & settings</h1>
                  <button
                    className="primary"
                    onClick={() =>
                      setModelDialog({
                        id: "",
                        name: "",
                        provider: "http",
                        task: "text",
                        model_id: "",
                        endpoint: "",
                        enabled: false,
                        lower_threshold: 0.2,
                        upper_threshold: 0.8,
                      })
                    }
                  >
                    <Plus size={16} />
                    Add model
                  </button>
                </div>
                <p className="page-description">
                  Choose the detector used for new scans. Previous reports keep
                  their original model settings.
                </p>
                <div className="settings-card">
                  <h2>Default text detector</h2>
                  <ModelSelect
                    value={defaults.default_text_model || ""}
                    models={models.filter((m) => m.task === "text")}
                    onChange={async (v) => {
                      try {
                        await api("/v1/settings/default-model", {
                          method: "PUT",
                          body: JSON.stringify({ model_id: v }),
                        });
                        setModelId(v);
                        await refreshModels();
                        toast.success("Default model updated");
                      } catch (e) {
                        toast.error((e as Error).message);
                      }
                    }}
                  />
                </div>
                {models.filter((m) => m.task === "image").length > 0 && (
                  <div className="settings-card">
                    <h2>Default image detector</h2>
                    <ModelSelect
                      value={defaults.default_image_model || ""}
                      models={models.filter((m) => m.task === "image")}
                      onChange={async (v) => {
                        try {
                          await api("/v1/settings/default-model", {
                            method: "PUT",
                            body: JSON.stringify({ model_id: v }),
                          });
                          await refreshModels();
                          toast.success("Image model updated");
                        } catch (e) {
                          toast.error((e as Error).message);
                        }
                      }}
                    />
                  </div>
                )}
                {models.map((m) => (
                  <div className="model-card" key={m.id}>
                    <span className="model-symbol">
                      <Settings2 size={21} />
                    </span>
                    <div>
                      <h2>{m.name}</h2>
                      <code>{m.model_id}</code>
                      <p>
                        {m.provider === "editlens"
                          ? "EditLens · Runs locally on your inference host"
                          : m.provider === "laya" ? "Laya · Experimental contextual phrases" : m.provider === "meld" ? "MELD v5 · Local token and sentence evidence" : "Connected HTTP classifier"}{" "}
                        · {m.task} detection
                      </p>
                    </div>
                    <span className={"badge " + (m.enabled ? "human" : "")}>
                      {m.enabled ? "Enabled" : "Setup pending"}
                    </span>
                    <button
                      className="secondary"
                      onClick={() => setModelDialog(m)}
                    >
                      Configure
                    </button>
                  </div>
                ))}
                <div className="notice">
                  <CircleHelp size={19} />
                  <div>
                    {models.some((m) => m.enabled && m.task === "text")
                      ? "Your local detector is ready."
                      : "Choose and enable a detector to start scanning."}
                    <p>
                      EditLens measures the extent of AI intervention. Scores
                      are not proof of authorship. The original Llama checkpoint
                      still requires separate access approval.
                    </p>
                    <button
                      className="text-button"
                      onClick={() => {
                        navigate("dashboard");
                        useSample();
                      }}
                    >
                      View a sample report
                    </button>
                  </div>
                </div>
                {error && <p className="inline-error">{error}</p>}
              </section>
            )}
            {view === "apikey" && (
              <section className="content-page">
                <div className="page-heading">
                  <h1>API</h1>
                  <a
                    className="secondary"
                    href="http://127.0.0.1:8000/docs"
                    target="_blank"
                    rel="noreferrer"
                  >
                    API documentation
                    <ExternalLink size={15} />
                  </a>
                </div>
                <p className="page-description">
                  Connect your applications to your chosen detection model.
                </p>
                <div className="settings-card">
                  <h2>Create an API key</h2>
                  <form
                    className="key-form"
                    onSubmit={async (e) => {
                      e.preventDefault();
                      try {
                        const key = await api("/v1/keys", {
                          method: "POST",
                          body: JSON.stringify({
                            name: keyName,
                            scopes: ["read", "scan"],
                          }),
                        });
                        setNewKey(key.key);
                        setKeyName("");
                        setKeys((await api("/v1/keys")).items);
                      } catch (e) {
                        toast.error((e as Error).message);
                      }
                    }}
                  >
                    <input
                      className="field"
                      required
                      placeholder="Key name"
                      value={keyName}
                      onChange={(e) => setKeyName(e.target.value)}
                    />
                    <button className="primary">Create key</button>
                  </form>
                  {newKey && (
                    <div className="new-key">
                      <p>Copy this key now. It will only be shown once.</p>
                      <code>{newKey}</code>
                      <button
                        className="secondary"
                        onClick={() =>
                          navigator.clipboard
                            .writeText(newKey)
                            .then(() => toast.success("Key copied"))
                        }
                      >
                        <Copy size={15} />
                        Copy key
                      </button>
                      <button
                        className="text-button"
                        onClick={() => setNewKey("")}
                      >
                        Dismiss
                      </button>
                    </div>
                  )}
                </div>
                <div className="settings-card">
                  <h2>Your API keys</h2>
                  {keys.map((k) => (
                    <div className="key-row" key={k.id}>
                      <div>
                        <b>{k.name}</b>
                        <small>
                          {k.scopes.join(", ")}{" "}
                          {k.revoked_at ? "· Revoked" : ""}
                        </small>
                      </div>
                      {k.id !== "bootstrap" && !k.revoked_at && (
                        <button
                          className="secondary"
                          onClick={async () => {
                            await api("/v1/keys/" + k.id, { method: "DELETE" });
                            setKeys((await api("/v1/keys")).items);
                            toast.success("Key revoked");
                          }}
                        >
                          Revoke
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}
            {view === "gmail" && (
              <section className="content-page integration-page">
                <Mail size={42} />
                <h1>Gmail Integration</h1>
                <p>Check the authenticity of the messages in your inbox.</p>
                <div className="notice">
                  <CircleHelp size={20} />
                  <div>
                    Gmail is not connected in this workspace.
                    <p>
                      You can paste an email into Text Detection now. Connecting
                      a mailbox requires a separate integration.
                    </p>
                    <button
                      className="primary"
                      onClick={() => navigate("dashboard")}
                    >
                      Check email text
                    </button>
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </main>
      <Dialog open={samples} onOpenChange={setSamples}>
        <DialogContent className="sample-dialog">
          <DialogHeader>
            <DialogTitle>Try Sample Texts</DialogTitle>
            <DialogDescription>
              Explore the report layout with an illustrative sample. No model is
              run.
            </DialogDescription>
          </DialogHeader>
          <button className="sample-card" onClick={useSample}>
            <p>{SAMPLE.slice(0, 380)}…</p>
            <span className="badge ai_assisted">
              AI + Human · Sample report
            </span>
          </button>
          <button
            className="secondary"
            onClick={() => {
              setText(SAMPLE);
              setSamples(false);
              setInputTab("text");
            }}
          >
            Use this text in the editor
          </button>
        </DialogContent>
      </Dialog>
      <Dialog open={!!info} onOpenChange={() => setInfo("")}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{info}</DialogTitle>
            <DialogDescription>
              {info === "Your workspace"
                ? "Your private local workspace is connected to the backend on this computer."
                : info === "What’s new"
                  ? "Model selection, document uploads, saved reports, API keys, and scan history are available in this workspace."
                  : "The scan API is available for custom integrations. Native Gmail and LMS connections are not configured."}
            </DialogDescription>
          </DialogHeader>
          <button
            className="primary"
            onClick={() => {
              setInfo("");
              navigate("models");
            }}
          >
            Open model settings
          </button>
        </DialogContent>
      </Dialog>
      {modelDialog && (
        <ModelEditor
          model={modelDialog}
          close={() => setModelDialog(null)}
          save={saveModel}
        />
      )}
    </SidebarProvider>
  );
}
function FileDrop({
  files,
  setFiles,
  image = false,
}: {
  files: File[];
  setFiles: (f: File[]) => void;
  image?: boolean;
}) {
  return (
    <div
      className="drop-zone"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        setFiles(Array.from(e.dataTransfer.files).slice(0, image ? 1 : 100));
      }}
    >
      <Upload size={28} />
      <b>
        {image
          ? "Upload an image to check for AI"
          : "Drag and drop your files here"}
      </b>
      <span>
        or{" "}
        <label className="file-browse">
          browse files
          <input
            type="file"
            multiple={!image}
            accept={
              image ? ".jpg,.jpeg,.png,.webp" : ".txt,.md,.csv,.pdf,.docx,.rtf"
            }
            onChange={(e) => setFiles(Array.from(e.target.files || []))}
          />
        </label>
      </span>
      <small>
        {image
          ? "JPG, PNG, WebP · min 512 × 512px · up to 20MB"
          : "PDF, DOCX, RTF, TXT · up to 100 files"}
      </small>
      {files.length > 0 && (
        <div className="selected-files">
          {files.map((f, i) => (
            <span key={i}>
              <FileText size={14} />
              {f.name}
              <button
                aria-label={"Remove " + f.name}
                onClick={() => setFiles(files.filter((_, n) => i !== n))}
              >
                <X size={14} />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
function Report({
  scan,
  setScan,
  close,
  edit,
}: {
  scan: Scan;
  setScan: (s: Scan) => void;
  close: () => void;
  edit: () => void;
}) {
  const [tab, setTab] = useState("overview"),
    [highlight, setHighlight] = useState(true),
    [resolution, setResolution] = useState("sentence"),
    [notes, setNotes] = useState(scan.notes || ""),
    [saving, setSaving] = useState(false),
    [share, setShare] = useState("");
  useEffect(() => setNotes(scan.notes || ""), [scan.id, scan.notes]);
  const result = scan.result,
    percent = Math.round((result?.score || 0) * 100),
    evidence = result?.score_type === "ai_evidence",
    experimental = result?.score_type === "experimental_ai_intervention";
  async function action(path: string) {
    try {
      setScan(
        await api("/v1/scans/" + scan.id + "/" + path, { method: "POST" }),
      );
    } catch (e) {
      toast.error((e as Error).message);
    }
  }
  async function saveNotes() {
    setSaving(true);
    try {
      setScan(
        await api("/v1/scans/" + scan.id, {
          method: "PATCH",
          body: JSON.stringify({ notes }),
        }),
      );
      toast.success("Notes saved");
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  useEffect(() => { setSelectedIndex(null); }, [scan.id, resolution]);
  const units = (resolution === "token" && evidence ? result?.tokens : result?.segments) || [];
  function selectSegment(index: number | null, scrollToText = false) {
    setSelectedIndex(index);
    setHighlight(true);
    setTab("details");
    if (scrollToText && index !== null) requestAnimationFrame(() => document.getElementById("report-segment-" + index)?.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "center"}));
  }
  const chars = Array.from(scan.text);
  const spans = [];
  let cursor = 0;
  for (const [index, s] of units.entries()) {
    if (cursor < s.start)
      spans.push(
        <span key={"gap" + cursor}>
          {chars.slice(cursor, s.start).join("")}
        </span>,
      );
    spans.push(
      <mark
        key={s.start}
        id={"report-segment-" + index}
        role="button"
        tabIndex={0}
        aria-pressed={selectedIndex === index}
        aria-label={`${resolution === "token" && evidence ? "Token" : evidence ? "Sentence" : experimental ? "Phrase" : "Passage"} ${index + 1}: ${labelName(s.label)}. Show details.`}
        onClick={() => selectSegment(index)}
        onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectSegment(index); } }}
        className={(highlight ? s.label : "plain") + (selectedIndex === index ? " segment-selected" : "")}
        style={evidence && highlight ? { backgroundColor: `rgba(255, 139, 75, ${0.04 + 0.48 * s.score})` } : undefined}
        data-tooltip={`${labelName(s.label)} · ${evidence ? "evidence" : "score"} ${s.score.toFixed(3)}`}
      >
        {chars.slice(s.start, s.end).join("")}
      </mark>,
    );
    cursor = s.end;
  }
  if (cursor < chars.length)
    spans.push(<span key="tail">{chars.slice(cursor).join("")}</span>);
  return (
    <div className="report-view">
      <div className="report-actions">
        {scan.sample && (
          <span className="sample-indicator">
            Sample report · illustrative, not a live prediction
          </span>
        )}
        <button
          className="secondary"
          disabled={scan.sample || scan.status !== "completed"}
          onClick={async () => {
            try {
              const d = await api("/v1/scans/" + scan.id + "/shares", {
                method: "POST",
                body: "{}",
              });
              setShare(window.location.origin + "/backend" + d.path);
            } catch (e) {
              toast.error((e as Error).message);
            }
          }}
        >
          Share
        </button>
        <a
          className={"icon-btn " + (scan.sample ? "disabled" : "")}
          aria-label="Download PDF report"
          href={
            scan.sample
              ? undefined
              : "/backend/v1/reports/" + scan.id + "?format=pdf"
          }
        >
          <Download size={18} />
        </a>
        <button className="icon-btn" aria-label="Close report" onClick={close}>
          <X size={18} />
        </button>
      </div>
      <div className="report-columns">
        <article className="document-pane">
          <div className="document-heading">
            {scan.source_locked || scan.source?.startsWith("dataset:") || scan.source?.startsWith("reviewbench:") ? <div><h1>{scan.title}</h1><small>Read-only source text</small></div> : <button title="Edit text" onClick={edit}><h1>{scan.title}</h1></button>}
            <span className={"badge " + (result?.label || "")}>
              {result ? labelName(result.label) : scan.status}
            </span>
          </div>
          <div className="document-meta">
            <span>{new Date(scan.created_at).toLocaleString()}</span>
            <span>
              <FileText size={12} />
              {scan.word_count} words
            </span>
          </div>
          <div className="document-tools">
            <button
              className="icon-btn"
              aria-label="Copy document text"
              onClick={() =>
                navigator.clipboard
                  .writeText(scan.text)
                  .then(() => toast.success("Text copied"))
              }
            >
              <Copy size={15} />
            </button>
            <button
              className={"highlight-toggle " + (highlight ? "active" : "")}
              onClick={() => setHighlight(!highlight)}
            >
              <span>A</span>
              {highlight ? (experimental ? "Phrase scores" : evidence ? "Evidence highlight" : "AI Highlight") : "No highlight"}
              <ChevronDown size={13} />
            </button>
          </div>
          {evidence && (
            <div className="localization-controls">
              <div role="group" aria-label="Highlight resolution">
                {["sentence", "token"].map((value) => <button key={value} className={resolution === value ? "secondary active" : "secondary"} aria-pressed={resolution === value} onClick={() => setResolution(value)}>{value === "sentence" ? "Sentences" : "Tokens"}</button>)}
              </div>
              <p>Faint → strong AI evidence · click a highlight to explore its details.</p>
              <p>{result?.localization?.notice}</p>
            </div>
          )}
          {experimental && <p className="muted">{result?.localization?.notice}</p>}
          <div className="document-text">
            {spans.length
              ? spans
              : scan.kind === "image"
                ? "Image submitted for analysis."
                : "No text available."}
          </div>
        </article>
        <aside className="analysis-pane">
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList className="report-tabs" variant="line">
              <TabsTrigger value="overview">
                <ChartPie />
                Overview
              </TabsTrigger>
              <TabsTrigger value="details">
                <SlidersHorizontal />
                Details
              </TabsTrigger>
              <TabsTrigger value="notes">
                <StickyNote />
                Notes
              </TabsTrigger>
            </TabsList>
            <TabsContent value="overview">
              <div className="analysis-card">
                <div className="analysis-heading">
                  <span className="pink-icon">▧</span>
                  <div>
                    <h2>
                      {result
                        ? experimental ? "Experimental phrase assessment" : evidence ? labelName(result.label) : result.label === "human"
                          ? "Human-written content"
                          : "AI Detected"
                        : "Analysis"}
                    </h2>
                    <small>{scan.word_count} words scanned</small>
                  </div>
                  <span className="model-tag">
                    <img src="/favicon.svg" alt="" />
                    {scan.sample ? "Sample" : scan.model.name}
                  </span>
                </div>
                {result ? (
                  <>
                    <p className="verdict">
                      {evidence
                        ? result.label === "uncertain" ? "This text is too short for a reliable document verdict." : result.label === "ai_evidence" ? "The document exceeds MELD’s reference threshold for AI evidence." : "The document is below MELD’s reference threshold. This does not establish human authorship."
                        : result.label === "human"
                        ? "This model classified the text as human-written."
                        : result.label === "ai"
                          ? "This model classified the text as AI-generated."
                          : "This model classified the text as AI-assisted."}
                    </p>
                    <div
                      className="score-ring"
                      style={{
                        background: `conic-gradient(#ff640b 0 ${percent}%,#0e604d ${percent}% 100%)`,
                      }}
                    >
                      <div>
                        <span className="pink-icon">▧</span>
                        <strong>
                          {percent}
                          <small>/100</small>
                        </strong>
                        <span>{evidence ? "AI evidence score" : "AI intervention score"}</span>
                      </div>
                    </div>
                    <div className="score-legend">
                      <span>
                        <i />
                        {evidence ? "AI evidence" : "AI intervention"}
                      </span>
                      <span>
                        <i />
                        {evidence ? "Lower evidence" : "Human end of scale"}
                      </span>
                    </div>
                    <p className="score-disclaimer">
                      {result.notice || "A model score, not a percentage of AI-written words."}
                    </p>
                    <button
                      className="pattern-card"
                      onClick={() => setTab("details")}
                    >
                      <span className="cyan-icon">▧</span>Explore the passage
                      analysis
                      <ChevronRight size={15} />
                    </button>
                    <div className="feedback">
                      Was this result helpful?
                      {[true, false].map((up) => (
                        <button
                          key={String(up)}
                          disabled={scan.sample}
                          aria-label={
                            up ? "Result was helpful" : "Result was unhelpful"
                          }
                          onClick={() =>
                            api("/v1/scans/" + scan.id, {
                              method: "PATCH",
                              body: JSON.stringify({
                                feedback: up ? "helpful" : "unhelpful",
                              }),
                            })
                              .then(() => toast.success("Feedback saved"))
                              .catch((e) => toast.error(e.message))
                          }
                        >
                          {up ? (
                            <ThumbsUp size={14} />
                          ) : (
                            <ThumbsDown size={14} />
                          )}
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="processing-state">
                    {["queued", "running"].includes(scan.status) ? (
                      <>
                        <Loader2 className="spin" size={32} />
                        <h3>
                          {scan.status === "queued"
                            ? "Your scan is queued"
                            : "Analyzing your document…"}
                        </h3>
                        <p>This report updates automatically.</p>
                        <button
                          className="secondary"
                          onClick={() => action("cancel")}
                        >
                          Cancel scan
                        </button>
                      </>
                    ) : (
                      <>
                        <CircleHelp size={30} />
                        <h3>
                          {scan.status === "failed"
                            ? "The scan could not be completed"
                            : "Scan cancelled"}
                        </h3>
                        <p>{scan.error?.message}</p>
                        <button
                          className="primary"
                          onClick={() => action("retry")}
                        >
                          Retry scan
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
              <div className="analysis-card plagiarism-card">
                <h3>Plagiarism Analysis</h3>
                {result?.plagiarism ? (
                  <>
                    <p>
                      {Math.round(
                        result.plagiarism.matched_word_fraction * 100,
                      )}
                      % matched in your reference corpus.
                    </p>
                    <small>{result.plagiarism.notice}</small>
                    {result.plagiarism.matches.map((m, i) => (
                      <p key={i}>
                        {m.title}: {m.matched_words} words
                      </p>
                    ))}
                  </>
                ) : (
                  <p className="muted">
                    {scan.sample
                      ? "Not included in the illustrative sample."
                      : "Enable “Check for plagiarism” before scanning to compare with your reference documents."}
                  </p>
                )}
              </div>
            </TabsContent>
            <TabsContent value="details">
              <SegmentExplorer units={units} text={scan.text} evidence={evidence} token={resolution === "token" && evidence} selected={selectedIndex} onSelect={selectSegment}/>
              <div className="analysis-card">
                <h2>Detection details</h2>
                <dl>
                  <dt>Model</dt>
                  <dd>{scan.model.name}</dd>
                  <dt>Classification</dt>
                  <dd>{labelName(result?.label)}</dd>
                  <dt>{evidence ? "Raw evidence" : "Raw score"}</dt>
                  <dd>{(evidence ? result?.raw_score : result?.score)?.toFixed(4) || "Pending"}</dd>
                  {evidence ? <><dt>Document reference threshold</dt><dd>Above {result?.thresholds?.raw_ai_above?.toFixed(4)}</dd><dt>Localization</dt><dd>Token evidence with sentence aggregation</dd></> : <><dt>Human threshold</dt><dd>Below {scan.model.lower_threshold}</dd><dt>AI threshold</dt><dd>{scan.model.upper_threshold} and above</dd></>}
                </dl>
                <p className="muted">
                  {evidence ? "The reference threshold targets 1% false positives on the publisher’s validation documents. It is not calibrated for individual highlights or this corpus." : "Thresholds are workspace settings, not Pangram’s calibrated production thresholds."}
                </p>
              </div>
              {!scan.sample && <RunPerformance result={result} words={scan.word_count} />}

            </TabsContent>
            <TabsContent value="notes">
              <div className="analysis-card">
                <h2>Notes</h2>
                <p className="muted">Private notes for this document.</p>
                <textarea
                  className="notes-input"
                  aria-label="Report notes"
                  placeholder="Add a note…"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
                <button
                  className="primary"
                  disabled={saving || scan.sample}
                  onClick={saveNotes}
                >
                  {saving ? "Saving…" : "Save notes"}
                </button>
                {scan.sample && (
                  <p className="muted">Notes are saved for real scans only.</p>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </aside>
      </div>
      <Dialog open={!!share} onOpenChange={() => setShare("")}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Share report</DialogTitle>
            <DialogDescription>
              Anyone with this link can read the document and report for 24
              hours. Private notes are excluded.
            </DialogDescription>
          </DialogHeader>
          <input className="field" readOnly value={share} />
          <button
            className="primary"
            onClick={() =>
              navigator.clipboard
                .writeText(share)
                .then(() => toast.success("Link copied"))
            }
          >
            <Copy size={16} />
            Copy link
          </button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
function ModelEditor({
  model,
  close,
  save,
}: {
  model: Model;
  close: () => void;
  save: (m: Model, key: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState(model),
    [key, setKey] = useState(""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  return (
    <Dialog open onOpenChange={close}>
      <DialogContent className="model-dialog">
        <DialogHeader>
          <DialogTitle>
            {model.id ? "Configure model" : "Add model"}
          </DialogTitle>
          <DialogDescription>
            Settings apply to new scans. Credentials stay on your backend.
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            try {
              await save(draft, key);
            } catch (e) {
              setError((e as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          <label>
            Name
            <input
              className="field"
              required
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            />
          </label>
          <label>
            Provider
            <Select
              value={draft.provider}
              onValueChange={(v) => setDraft({ ...draft, provider: v, ...(v === "laya" ? { model_id: "convaiinnovations/laya", task: "text", base_model_id: undefined } : {}), ...(v === "meld" ? { model_id: "anon-review-meld-2026/meld", task: "text", base_model_id: undefined } : {}) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="editlens">
                  EditLens
                </SelectItem>
                <SelectItem value="laya">Laya · Experimental contextual phrases</SelectItem>
                <SelectItem value="meld">MELD v5 · Local evidence</SelectItem>
                <SelectItem value="http">Custom HTTP endpoint</SelectItem>
              </SelectContent>
            </Select>
          </label>
          <label>
            Model ID
            <input
              className="field"
              required
              value={draft.model_id}
              onChange={(e) => setDraft({ ...draft, model_id: e.target.value })}
            />
          </label>
          {draft.provider === "http" && (
            <>
              <label>
                Endpoint
                <input
                  type="url"
                  required
                  className="field"
                  value={draft.endpoint || ""}
                  placeholder="https://your-host.example/classify"
                  onChange={(e) =>
                    setDraft({ ...draft, endpoint: e.target.value })
                  }
                />
              </label>
              <label>
                API key
                <input
                  type="password"
                  className="field"
                  autoComplete="new-password"
                  value={key}
                  placeholder={
                    draft.has_api_key
                      ? "Leave blank to keep saved key"
                      : "Optional bearer token"
                  }
                  onChange={(e) => setKey(e.target.value)}
                />
              </label>
              <label>
                Detection type
                <Select
                  value={draft.task}
                  onValueChange={(v) => setDraft({ ...draft, task: v })}
                  disabled={!!draft.id}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="text">Text</SelectItem>
                    <SelectItem value="image">Image</SelectItem>
                  </SelectContent>
                </Select>
              </label>
            </>
          )}
          {draft.provider !== "meld" && <div className="threshold-fields">
            <label>
              Human below
              <input
                type="number"
                className="field"
                min="0"
                max="1"
                step="0.01"
                value={draft.lower_threshold}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    lower_threshold: Number(e.target.value),
                  })
                }
              />
            </label>
            <label>
              AI at or above
              <input
                type="number"
                className="field"
                min="0"
                max="1"
                step="0.01"
                value={draft.upper_threshold}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    upper_threshold: Number(e.target.value),
                  })
                }
              />
            </label>
          </div>
          }
          {draft.provider === "laya" && <p className="muted">Local English zero-shot baseline. Scores each phrase with surrounding context. Not calibrated for AI-writing detection; thresholds are exploratory.</p>}
          {draft.provider === "meld" && <p className="muted">Uses the checkpoint’s document threshold. Token and sentence highlights show exploratory evidence, not AI-assisted authorship labels.</p>}
          <label className="enabled-choice">
            <Checkbox
              checked={draft.enabled}
              onCheckedChange={(v) =>
                setDraft({ ...draft, enabled: v === true })
              }
            />
            Enable this model
          </label>
          {draft.provider === "editlens" && (
            <p className="muted">
              Leave disabled until Hugging Face access and model dependencies
              are ready.
            </p>
          )}
          {error && <p className="inline-error">{error}</p>}
          <div className="dialog-actions">
            <button type="button" className="secondary" onClick={close}>
              Cancel
            </button>
            <button className="primary" disabled={busy}>
              {busy ? "Saving…" : "Save settings"}
            </button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
