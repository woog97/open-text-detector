/**
 * PDF viewer with the website's flagged sentences drawn over the pages. It renders the website's copy of the PDF, the file the
 * saved word positions were computed on, and counts and colours sentences the way the website's reader does
 * (FLAG_MODEL report, segmentSections, FLAG_MODES thresholds).
 */
import * as pdfjs from 'pdfjs-dist';
import type {PDFDocumentProxy} from 'pdfjs-dist';
import {SITE, FLAG_MODEL, FLAG_MODES, DEFAULT_FLAG_MODE, MIN_SENTENCES, fprLabel, isForumId, pct, paperUrl, countedText, FLAG_FPRS, savedFpr, saveFpr, type FlagFpr, type PaperHit, type FlagMode} from './site.ts';
import {validateReaderDetail, type ReaderDetail} from '../../app/lib/pdf-reader-data.ts';
import {positionPassages, supportsSavedPositions, type PositionedPassage} from '../../app/lib/pdf-positions.ts';
import {mergeHighlightBoxes} from '../../app/lib/pdf-highlight-boxes.ts';
import {segmentSections, type Section} from '../../app/lib/paper-sections.ts';
import {hasTextContent} from '../../app/lib/paper-scores.ts';

pdfjs.GlobalWorkerOptions.workerSrc = chrome.runtime.getURL('pdfjs/pdf.worker.min.mjs');

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector(s) as T;
const forum = new URLSearchParams(location.search).get('forum');
const original = (id: string) => `https://openreview.net/pdf?id=${encodeURIComponent(id)}&otd=0`;
const SECTION_NAMES: Record<Section, string> = {body: 'main text', bib: 'references', app: 'appendix'};

type Item = PositionedPassage & {section: Section; hasText: boolean};
type Box = {id: number; x: number; y: number; w: number; h: number};
const state: {pdf?: PDFDocumentProxy; items: Item[]; mode: FlagMode; scale: number; pages: HTMLElement[]} = {items: [], mode: DEFAULT_FLAG_MODE, scale: 1, pages: []};

function status(text: string) {$('#status').textContent = text;}

/** Flag state of one sentence in the current mode: skip (not counted), low, medium (>= threshold) or high (>= strong). */
function band(p: Item) {
  const f = FLAG_MODES[state.mode];
  if (!p.hasText || !(p.section === 'body' || (p.section === 'bib' && f.bib) || (p.section === 'app' && f.app))) return 'skip';
  return p.score >= f.strong ? 'high' : p.score >= f.threshold ? 'medium' : 'low';
}

function summarize() {
  let counted = 0, flagged = 0, strong = 0;
  for (const p of state.items) {const b = band(p); if (b === 'skip') continue; counted++; if (b !== 'low') flagged++; if (b === 'high') strong++;}
  const f = FLAG_MODES[state.mode];
  $('#share').textContent = !state.items.length ? '—' : counted < MIN_SENTENCES ? 'Too short' : `${pct(flagged / counted)} AI`;
  $('#share-detail').textContent = state.items.length
    ? `${flagged.toLocaleString()} of ${counted.toLocaleString()} ${countedText(state.mode)} flagged at ${fprLabel(f.fpr)} human false-positive rate${strong ? ` · ${strong.toLocaleString()} at 0.80 or above (red)` : ''}`
    : '';
}

function drawHighlights(page: HTMLElement) {
  page.querySelector('.hl-layer')?.remove();
  const n = Number(page.dataset.page), w = page.clientWidth, h = page.clientHeight;
  if (!page.dataset.rendered) return;
  const boxes: Box[] = [];
  for (const p of state.items) {
    const b = band(p);
    if (b !== 'medium' && b !== 'high') continue;
    for (const r of p.rectangles || []) if (r.page === n) boxes.push({id: p.id, x: r.x * w, y: r.y * h, w: r.w * w, h: r.h * h});
  }
  const layer = document.createElement('div');
  layer.className = 'hl-layer';
  for (const b of mergeHighlightBoxes(boxes)) {
    const el = document.createElement('div');
    el.className = `hl ${band(state.items[b.id])}`;
    el.dataset.id = String(b.id);
    Object.assign(el.style, {left: `${b.x}px`, top: `${b.y}px`, width: `${b.w}px`, height: `${b.h}px`});
    layer.appendChild(el);
  }
  page.querySelector('canvas')!.after(layer);
}

function drawList() {
  const list = $('#flagged');
  list.textContent = '';
  const flagged = state.items.filter(p => band(p) === 'medium' || band(p) === 'high').sort((a, b) => b.score - a.score || a.id - b.id);
  $('#flagged-count').textContent = flagged.length ? `(${flagged.length.toLocaleString()}, highest first)` : '';
  for (const p of flagged.slice(0, 400)) {
    const li = document.createElement('li');
    li.className = band(p);
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = `${p.score.toFixed(4)} · ${p.mapped ? `p. ${p.page}` : 'not located'} · ${SECTION_NAMES[p.section]}`;
    const text = document.createElement('span');
    text.textContent = p.text.length > 220 ? p.text.slice(0, 220) + '…' : p.text;
    li.append(meta, text);
    if (p.mapped) li.onclick = () => jump(p);
    list.appendChild(li);
  }
}

function redraw() {summarize(); drawList(); state.pages.forEach(drawHighlights);}

function jump(p: Item) {
  const page = state.pages[p.page! - 1], r = p.rectangles![0];
  $('#pages').scrollTo({top: page.offsetTop + r.y * page.clientHeight - 120, behavior: 'smooth'});
  setTimeout(() => page.querySelectorAll<HTMLElement>(`.hl[data-id="${p.id}"]`).forEach(el => {el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');}), 400);
}

async function renderPage(page: HTMLElement) {
  if (page.dataset.rendered || page.dataset.busy) return;
  page.dataset.busy = '1';
  const p = await state.pdf!.getPage(Number(page.dataset.page));
  const viewport = p.getViewport({scale: state.scale});
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = document.createElement('canvas');
  canvas.width = Math.floor(viewport.width * ratio);
  canvas.height = Math.floor(viewport.height * ratio);
  Object.assign(page.style, {width: `${viewport.width}px`, height: `${viewport.height}px`});
  page.style.setProperty('--scale-factor', String(viewport.scale));
  page.style.setProperty('--total-scale-factor', String(viewport.scale));
  page.textContent = '';
  page.appendChild(canvas);
  await p.render({canvas, viewport, transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : undefined}).promise;
  page.dataset.rendered = '1';
  delete page.dataset.busy;
  drawHighlights(page);
  // Selectable text above the highlights; dense plots can carry thousands of glyph items, so skip those.
  const content = await p.getTextContent();
  if (content.items.length <= 3000) {
    const text = document.createElement('div');
    text.className = 'textLayer';
    page.appendChild(text);
    await new pdfjs.TextLayer({textContentSource: content, container: text, viewport}).render();
  }
}

async function layoutPages() {
  const host = $('#pages');
  host.textContent = '';
  const first = await state.pdf!.getPage(1);
  const base = first.getViewport({scale: 1});
  state.scale = state.scale || 1;
  state.pages = [];
  const observer = new IntersectionObserver(entries => entries.forEach(e => e.isIntersecting && renderPage(e.target as HTMLElement)), {root: host, rootMargin: '1200px 0px'});
  for (let n = 1; n <= state.pdf!.numPages; n++) {
    const page = document.createElement('div');
    page.className = 'page';
    page.dataset.page = String(n);
    Object.assign(page.style, {width: `${base.width * state.scale}px`, height: `${base.height * state.scale}px`});
    host.appendChild(page);
    state.pages.push(page);
    observer.observe(page);
  }
}

function zoom(factor: number) {
  const host = $('#pages'), at = host.scrollTop / host.scrollHeight;
  state.scale = Math.max(.5, Math.min(3, state.scale * factor));
  layoutPages().then(() => {host.scrollTop = at * host.scrollHeight;});
}

/** Hover a highlight to see its score; highlights sit under the text layer, so hit-test by position. */
function tooltips() {
  const tip = $('#tip');
  $('#pages').addEventListener('mousemove', e => {
    const page = (e.target as HTMLElement).closest('.page');
    const hit = page && [...page.querySelectorAll<HTMLElement>('.hl')].find(el => {const r = el.getBoundingClientRect(); return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;});
    if (!hit) {tip.hidden = true; return;}
    const p = state.items[Number(hit.dataset.id)];
    tip.textContent = `Score ${p.score.toFixed(4)} · flagged at ${fprLabel(FLAG_MODES[state.mode].fpr)} FPR${band(p) === 'high' ? ', 0.80 or above' : ''} · ${SECTION_NAMES[p.section]}`;
    Object.assign(tip.style, {left: `${e.clientX + 12}px`, top: `${e.clientY + 14}px`});
    tip.hidden = false;
  });
}

async function main() {
  if (!isForumId(forum)) {status('No OpenReview paper ID in the address.'); return;}
  $<HTMLAnchorElement>('#forum').href = `https://openreview.net/forum?id=${forum}`;
  $<HTMLAnchorElement>('#original').href = original(forum);
  const res = await chrome.runtime.sendMessage({type: 'lookup', forum}) as {hit?: PaperHit | null; error?: string};
  // Not on the website (or the website is down): show the ordinary OpenReview PDF instead, without a history entry.
  if (!res.hit) {location.replace(original(forum)); return;}
  const hit = res.hit;
  document.title = `${hit.title} · Open Text Detector`;
  $('#title').textContent = hit.title;
  $<HTMLAnchorElement>('#site').href = paperUrl(hit);

  status('Loading scores…');
  const r = await fetch(`${SITE}/backend/v1/pdf-reader/${hit.id}`);
  if (!r.ok) throw new Error(`Results failed to load (${r.status})`);
  const detail: ReaderDetail = validateReaderDetail(await r.json());
  const report = detail.reports.find(x => x.id.startsWith(`${FLAG_MODEL}-`));
  const map = report && detail.position_maps?.[report.text_sha256 || ''];
  if (!report) status('Not scored by the final model yet; showing the PDF without highlights.');
  else if (!supportsSavedPositions(map)) status('No saved word positions for this paper; highlights are unavailable. Flagged sentences are listed on the right.');
  else status('');
  if (report) {
    const sections = segmentSections(report.text, report.result.segments);
    const placed = supportsSavedPositions(map) ? positionPassages(report.text, report.result.segments, map)
      : report.result.segments.map((s, id) => ({...s, id, text: Array.from(report.text).slice(s.start, s.end).join(''), mapped: false, page: undefined, pieces: [], rectangles: []}));
    state.items = placed.map((p, i) => ({...p, section: sections[i], hasText: hasTextContent(p.text)}));
  }
  summarize();
  drawList();

  const task = pdfjs.getDocument({
    url: `${SITE}/backend/v1/pdf-reader/${hit.id}/file?v=${encodeURIComponent(detail.version)}`,
    cMapUrl: chrome.runtime.getURL('pdfjs/cmaps/'), cMapPacked: true,
    standardFontDataUrl: chrome.runtime.getURL('pdfjs/standard_fonts/'),
    wasmUrl: chrome.runtime.getURL('pdfjs/wasm/'),
    isEvalSupported: false,
  });
  state.pdf = await task.promise;
  const first = (await state.pdf.getPage(1)).getViewport({scale: 1});
  state.scale = Math.min(1.6, ($('#pages').clientWidth - 48) / first.width);
  await layoutPages();
}

for (const [value, label] of [['body', 'Main text'], ['noapp', 'Main text + references'], ['nobib', 'Main text + appendix'], ['all', 'Everything']]) $<HTMLSelectElement>('#sections').add(new Option(label, value));
$<HTMLSelectElement>('#sections').value = FLAG_MODES[DEFAULT_FLAG_MODE].base;
for (const f of FLAG_FPRS) $<HTMLSelectElement>('#fpr').add(new Option(fprLabel(f), String(f)));
const syncMode = () => {
  const base = $<HTMLSelectElement>('#sections').value, fpr = Number($<HTMLSelectElement>('#fpr').value) as FlagFpr;
  state.mode = (Object.keys(FLAG_MODES) as FlagMode[]).find(m => FLAG_MODES[m].base === base && FLAG_MODES[m].fpr === fpr)!;
  redraw();
};
savedFpr().then(fpr => {$<HTMLSelectElement>('#fpr').value = String(fpr); syncMode();});
$('#sections').onchange = syncMode;
$('#fpr').onchange = () => {saveFpr(Number($<HTMLSelectElement>('#fpr').value) as FlagFpr); syncMode();};
$('#zoom-in').onclick = () => zoom(1.2);
$('#zoom-out').onclick = () => zoom(1 / 1.2);
tooltips();
main().catch(e => status(`Could not load this paper: ${e?.message || e}`));
