export type RunMetrics = {
  started_at?: string; classification_seconds?: number; processing_seconds?: number;
  queue_seconds?: number; elapsed_since_submission_seconds?: number;
  words_per_second?: number; attempt?: number; stored_result_bytes?: number;
  uncompressed_result_bytes?: number; storage_format?: string; input_utf8_bytes?: number; timing_scope?: string;
};
export type InferenceInfo = { head_dtype?: string; batch_size?: number; phase_seconds?: Record<string,number>; memory_after_run?: {allocated_bytes?:number;driver_bytes?:number}; device?: string; dtype?: string; revision?: string; model_id?: string; max_length?: number; windows?: number; source_tokens?: number; overlap?: number };
const duration = (n?: number) => n == null ? 'Not recorded' : n < 60 ? `${n.toFixed(2)} seconds` : `${Math.floor(n / 60)}m ${(n % 60).toFixed(1)}s`;
const date = (s?: string) => s ? new Date(s).toLocaleString(undefined, { timeZoneName: 'short' }) : 'Not recorded';
const bytes = (n: number) => n < 1000000 ? `${(n / 1000).toFixed(1)} KB` : `${(n / 1000000).toFixed(2)} MB`;
export default function RunPerformance({ result, words }: { result?: { completed_at?: string; performance?: RunMetrics; inference?: InferenceInfo; segments?: unknown[] }; words: number }) {
  if (!result) return null;
  const p = result.performance || {}, info = result.inference || {};
  const rows: [string, string][] = [
    ['Completed', date(result.completed_at)], ['Classification runtime', duration(p.classification_seconds)],
  ];
  if (p.started_at) rows.push(['Started', date(p.started_at)]);
  if (p.queue_seconds != null) rows.push(['Initial queue wait', duration(p.queue_seconds)]);
  if (p.processing_seconds != null) rows.push(['Total processing time', duration(p.processing_seconds)]);
  if (p.elapsed_since_submission_seconds != null) rows.push(['Submission to completion', duration(p.elapsed_since_submission_seconds)]);
  if (p.words_per_second != null) rows.push(['Classification throughput', `${p.words_per_second.toLocaleString()} words / second`]);
  rows.push(['Input size', `${words.toLocaleString()} words${p.input_utf8_bytes != null ? ' · '+bytes(p.input_utf8_bytes) : ''}`]);
  if (info.source_tokens != null) rows.push(['Source tokens', info.source_tokens.toLocaleString()]);
  if (info.windows != null) rows.push(['Inference windows', info.windows.toLocaleString()]);
  if (info.max_length != null) rows.push(['Context per window', `${info.max_length.toLocaleString()} tokens`]);
  if (info.overlap != null) rows.push(['Window overlap', `${info.overlap.toLocaleString()} tokens`]);
  if (info.device) rows.push(['Compute device', info.device]);
  if (info.dtype) rows.push(['Precision', info.dtype]);
  if (info.head_dtype) rows.push(['Evidence head precision', info.head_dtype]);
  if (info.batch_size != null) rows.push(['Windows per batch', String(info.batch_size)]);
  if (info.phase_seconds) for (const [phase, seconds] of Object.entries(info.phase_seconds)) rows.push([`${phase.charAt(0).toUpperCase()+phase.slice(1)} time`,duration(seconds)]);
  if (info.memory_after_run?.allocated_bytes != null) rows.push(['GPU tensor memory after run', bytes(info.memory_after_run.allocated_bytes)]);
  if (info.memory_after_run?.driver_bytes != null) rows.push(['GPU driver allocation after run', bytes(info.memory_after_run.driver_bytes)]);
  if (p.attempt != null) rows.push(['Processing attempt', String(p.attempt)]);
  if (p.stored_result_bytes != null) rows.push(['Stored result size', bytes(p.stored_result_bytes)]);
  if (p.uncompressed_result_bytes != null) rows.push(['Uncompressed result size', bytes(p.uncompressed_result_bytes)]);
  if (p.storage_format) rows.push(['Storage format', 'Lossless compressed']);
  if (info.revision) rows.push(['Model revision', info.revision]);
  return <div className="analysis-card run-performance"><h2>Run performance</h2><dl>{rows.map(([name,value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl><p className="muted">{p.classification_seconds == null ? 'This older report has no separate runtime measurement. Submission-to-completion includes queueing and any retries.' : 'Classification runtime includes model loading when needed, preprocessing and inference. Total processing also includes optional plagiarism checks.'} Opening a cached report preserves the original measurements.</p></div>;
}
