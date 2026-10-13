/** Open Text Detector website lookups shared by the background worker, popup and viewer. */
import {FLAG_MODEL, DEFAULT_FLAG_MODE, FLAG_MODES, FLAG_FPRS, fprLabel, flagModeOf, type FlagMode, type FlagFpr} from '../../app/lib/paper-discovery.ts';
import {MIN_SENTENCES, hasTextContent} from '../../app/lib/paper-scores.ts';
import {segmentSections} from '../../app/lib/paper-sections.ts';
import type {ReaderDetail} from '../../app/lib/pdf-reader-data.ts';

export const SITE = 'https://open-text-detector.woog09.workers.dev';
export {FLAG_MODEL, DEFAULT_FLAG_MODE, FLAG_MODES, FLAG_FPRS, MIN_SENTENCES, fprLabel, flagModeOf, type FlagMode, type FlagFpr};

/** The viewer's chosen false-positive rate (0.1% or 1%), shared by the card, badge, popup and viewer; the website default until changed. */
export async function savedFpr(): Promise<FlagFpr> {
  const {fpr} = await chrome.storage.local.get('fpr');
  return (FLAG_FPRS as readonly number[]).includes(fpr as number) ? fpr as FlagFpr : FLAG_MODES[DEFAULT_FLAG_MODE].fpr;
}
export const saveFpr = (fpr: FlagFpr) => chrome.storage.local.set({fpr});
/** The default sections (references excluded) counted at `fpr`. */
export const modeAt = (fpr: FlagFpr) => {const d = FLAG_MODES[DEFAULT_FLAG_MODE]; return flagModeOf(d.bib, d.app, fpr);};

/** One browse card: the fields the extension reads. flags[mode] = [flagged, counted] sentences of FLAG_MODEL. */
export type PaperHit = {id: string; title: string; collection: string; forum_id: string; flags?: Record<string, [number, number]>};

/** OpenReview forum IDs are short URL-safe strings. */
export const isForumId = (v: string | null | undefined): v is string => !!v && /^[A-Za-z0-9_-]{6,20}$/.test(v);

/** The paper whose forum ID is exactly `forum`, or null when the website does not hold it. Search spans all collections. */
export async function fetchHit(forum: string): Promise<PaperHit | null> {
  const r = await fetch(`${SITE}/backend/v1/browse?q=${encodeURIComponent(forum)}&collection=`);
  if (!r.ok) throw new Error(`Lookup failed (${r.status})`);
  const data = await r.json() as {items: PaperHit[]};
  return data.items.find(p => p.forum_id === forum) ?? null;
}

/**
 * [flagged, counted] per mode, recomputed from the FLAG_MODEL sentence scores with the cutoffs in this build (the
 * reader's rule: score >= threshold), so the card always agrees with the viewer even before the website recounts its
 * browse snapshot. Undefined when the paper has no FLAG_MODEL report.
 */
export function countFlags(detail: ReaderDetail): Record<string, [number, number]> | undefined {
  const r = detail.reports.find(x => x.id.startsWith(`${FLAG_MODEL}-`));
  if (!r) return undefined;
  const sections = segmentSections(r.text, r.result.segments), chars = Array.from(r.text);
  const out: Record<string, [number, number]> = {};
  for (const mode of Object.keys(FLAG_MODES)) out[mode] = [0, 0];
  r.result.segments.forEach((seg, i) => {
    if (!hasTextContent(chars.slice(seg.start, seg.end).join(''))) return;
    const sec = sections[i];
    for (const [mode, f] of Object.entries(FLAG_MODES)) {
      if (!(sec === 'body' || (sec === 'bib' && f.bib) || (sec === 'app' && f.app))) continue;
      out[mode][1]++;
      if (seg.score >= f.threshold) out[mode][0]++;
    }
  });
  return out;
}

/** Flagged share in the default mode, or null when the paper is unscored or too short (same rule as the website's list). */
export function share(hit: PaperHit, mode: FlagMode = DEFAULT_FLAG_MODE) {
  const c = hit.flags?.[mode];
  if (!c) return {state: 'unscored' as const};
  if (c[1] < MIN_SENTENCES) return {state: 'short' as const, counted: c[1]};
  return {state: 'scored' as const, flagged: c[0], counted: c[1], rate: c[0] / c[1]};
}

/** Which sentences a mode counts, in words. */
export function countedText(mode: FlagMode) {
  const f = FLAG_MODES[mode];
  return f.bib && f.app ? 'sentences' : !f.bib && !f.app ? 'main-text sentences' : !f.bib ? 'sentences (references excluded)' : 'sentences (appendix excluded)';
}

export const pct = (r: number) => r < .1 && r > 0 ? `${(r * 100).toFixed(1)}%` : `${Math.round(r * 100)}%`;
export const paperUrl = (hit: PaperHit) => `${SITE}/?collection=${encodeURIComponent(hit.collection)}&paper=${hit.id}`;
