/** Histogram buckets use ceil(score * 20), preserving strict > at 0.05 thresholds. */
export type ScoreSummary = { histogram: number[]; total: number; excluded?: number; policy?: number };
export type ScoredPaper = { score_summaries?: Record<string, ScoreSummary> };
export function modelLabel(id: string) {
  return id === "laya" ? "Laya · Experimental" : id === "qwen35-4b" ? "Qwen3.5-4B · Experimental" : id === "qwen36-35b-a3b-oct10" ? "Qwen3.6 35B-A3B MoE · Oct 10" : id === "qwen36-35b-a3b" ? "Qwen3.6 35B-A3B MoE · Oct 7" : id === "v5" || id === "v8" ? `MELD ${id}` : id;
}
export const scoreThresholds = Array.from({ length: 21 }, (_, i) => i / 20);
/** The default ("all") view shows the first of these a paper has, else its first model. Only the Oct 10 MoE is offered (Oct 12, 2026). */
export const DEFAULT_MODELS = ["qwen36-35b-a3b-oct10"];
/** Papers with fewer scored sentences than this get no paper-level rate (stub or blank-template PDFs). */
export const MIN_SENTENCES = 50;
export function defaultModelFor(paper: ScoredPaper) {
  const summaries = paper.score_summaries;
  if (!summaries) return undefined;
  return DEFAULT_MODELS.find(m => summaries[m]) ?? Object.keys(summaries)[0];
}
export function summaryFor(paper: ScoredPaper, model: string) {
  const summaries = paper.score_summaries;
  if (!summaries) return undefined;
  const key = model === "all" ? defaultModelFor(paper) : model;
  return key === undefined ? undefined : summaries[key];
}
export function tooShort(summary: ScoreSummary | undefined) {
  return summary !== undefined && summary.total < MIN_SENTENCES;
}
export function aboveCount(summary: ScoreSummary, threshold: number) {
  return summary.histogram.slice(Math.round(threshold * 20) + 1).reduce((a, b) => a + b, 0);
}
export function paperRate(summary: ScoreSummary | undefined, threshold: number): number | null {
  return summary && summary.total > 0 ? aboveCount(summary, threshold) / summary.total : null;
}
export function scoreBand(score: number) {
  return !Number.isFinite(score) ? "unknown" : score <= .2 ? "low" : score <= .8 ? "medium" : "high";
}
export function hasTextContent(text:string) { return /[\p{L}\p{N}]/u.test(text); }
export function summarizePassages(passages: { score: number; start?:number; end?:number }[], text?:string): ScoreSummary {
  const chars=text===undefined?undefined:Array.from(text);
  let excluded=0;
  const histogram = Array<number>(21).fill(0);
  for (const p of passages) {
    if(chars && !hasTextContent(chars.slice(p.start,p.end).join(""))){excluded++;continue;}
    if (Number.isFinite(p.score) && p.score >= 0 && p.score <= 1) histogram[Math.ceil(p.score * 20)]++;
  }
  return { histogram, total: histogram.reduce((a,b) => a+b,0), ...(chars?{excluded,policy:2}:{}) };
}

export function collectionBreakdown(papers: (ScoredPaper & {collection: string})[], model: string, threshold: number) {
  const groups = new Map<string, {collection:string;papers:number;scored:number;above:number;below:number;bins:number[]}>();
  for(const paper of papers) {
    let group=groups.get(paper.collection);
    if(!group){group={collection:paper.collection,papers:0,scored:0,above:0,below:0,bins:[0,0,0,0]};groups.set(paper.collection,group);}
    group.papers++;
    const summary=summaryFor(paper,model),rate=paperRate(summary,threshold);
    if(!summary||rate===null||tooShort(summary)){group.bins[3]++;continue;}
    group.scored++;
    const above=aboveCount(summary,threshold);
    group.above+=above;group.below+=summary.total-above;
    group.bins[rate<=.1?0:rate<=.5?1:2]++;
  }
  return [...groups.values()].sort((a,b)=>a.collection.localeCompare(b.collection,undefined,{numeric:true}));
}
