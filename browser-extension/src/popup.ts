import {share, pct, paperUrl, isForumId, countedText, fprLabel, FLAG_FPRS, savedFpr, saveFpr, modeAt, type PaperHit, type FlagFpr} from './site.ts';

const $ = (s: string) => document.querySelector(s) as HTMLElement;
const toggle = $('#auto') as HTMLInputElement;
chrome.storage.local.get('autoOpen').then(({autoOpen = true}) => {toggle.checked = autoOpen as boolean;});
toggle.onchange = () => chrome.storage.local.set({autoOpen: toggle.checked});

const fprSelect = $('#fpr') as HTMLSelectElement;
for (const f of FLAG_FPRS) fprSelect.add(new Option(fprLabel(f), String(f)));
fprSelect.onchange = async () => {await saveFpr(Number(fprSelect.value) as FlagFpr); show();};

async function show() {
  const fpr = await savedFpr(), mode = modeAt(fpr);
  fprSelect.value = String(fpr);
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  const url = tab?.url ? new URL(tab.url) : null;
  const viewer = url?.protocol === 'chrome-extension:' ? url.searchParams.get('forum') : null;
  const forum = viewer ?? (url?.hostname === 'openreview.net' && ['/forum', '/pdf'].includes(url.pathname) ? url.searchParams.get('id') : null);
  if (!isForumId(forum)) {$('#result').textContent = 'Open an OpenReview paper to see its result.'; return;}
  const res = await chrome.runtime.sendMessage({type: 'lookup', forum}) as {hit?: PaperHit | null; error?: string};
  const hit = res.hit, s = hit && share(hit, mode);
  $('#result').textContent = res.error ? `Open Text Detector unavailable: ${res.error}` : !hit ? 'Not in the collection.'
    : s!.state === 'scored' ? `${pct(s!.rate)} AI · ${s!.flagged} of ${s!.counted} ${countedText(mode)} flagged at ${fprLabel(fpr)} FPR`
    : s!.state === 'short' ? 'Too short to score.' : 'Collected, not scored yet.';
  if (!hit) return;
  const open = $('#open') as HTMLAnchorElement, site = $('#site') as HTMLAnchorElement;
  open.hidden = !!viewer;
  open.onclick = e => {e.preventDefault(); chrome.tabs.create({url: `${chrome.runtime.getURL('viewer.html')}?forum=${forum}`});};
  site.href = paperUrl(hit);
  site.hidden = false;
}
show();
