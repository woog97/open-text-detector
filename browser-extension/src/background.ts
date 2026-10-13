/**
 * Looks papers up on the website (cached for the browser session), keeps the toolbar badge on OpenReview tabs, and
 * owns the redirect rule that opens OpenReview PDFs in the highlighting viewer.
 */
import {SITE, countFlags, fetchHit, isForumId, share, pct, savedFpr, modeAt, type PaperHit} from './site.ts';

const TTL = 30 * 60 * 1000;
const REDIRECT_RULE = 1;

async function lookup(forum: string): Promise<PaperHit | null> {
  const key = `hit:${forum}`;
  const cached = (await chrome.storage.session.get(key))[key] as {hit: PaperHit | null; at: number} | undefined;
  if (cached && Date.now() - cached.at < TTL) return cached.hit;
  const hit = await fetchHit(forum);
  // Exact counts from the sentence scores; the browse snapshot's counts can lag a cutoff change.
  if (hit?.flags) {
    try {
      const r = await fetch(`${SITE}/backend/v1/pdf-reader/${hit.id}`);
      const flags = r.ok ? countFlags(await r.json()) : undefined;
      if (flags) hit.flags = flags;
    } catch {/* keep the snapshot counts */}
  }
  await chrome.storage.session.set({[key]: {hit, at: Date.now()}});
  return hit;
}

async function badge(tabId: number, hit: PaperHit | null) {
  const s = hit && share(hit, modeAt(await savedFpr()));
  const text = !s ? '' : s.state === 'scored' ? pct(s.rate).replace('.0%', '%') : '–';
  const color = !s || s.state !== 'scored' ? '#777' : s.rate > .5 ? '#b2412b' : s.rate > .1 ? '#a3690c' : '#3f7a55';
  chrome.action.setBadgeText({tabId, text}).catch(() => {});
  chrome.action.setBadgeBackgroundColor({tabId, color}).catch(() => {});
}

/** Forum ID of an OpenReview forum or PDF page. */
function forumOf(url: string | undefined) {
  if (!url) return null;
  const u = new URL(url);
  if (u.hostname !== 'openreview.net' || !['/forum', '/pdf'].includes(u.pathname)) return null;
  const id = u.searchParams.get('id');
  return isForumId(id) ? id : null;
}

/**
 * Main-frame requests for openreview.net/pdf?id=X (nothing else in the query) go to the viewer, which falls back to
 * the original with `&otd=0` appended when the website has no highlights for X. A network redirect leaves no extra
 * history entry, so Back still works.
 */
async function setRedirect(on: boolean) {
  await chrome.declarativeNetRequest.updateDynamicRules({
    removeRuleIds: [REDIRECT_RULE],
    addRules: on ? [{
      id: REDIRECT_RULE, priority: 1,
      action: {type: chrome.declarativeNetRequest.RuleActionType.REDIRECT,
        redirect: {regexSubstitution: `${chrome.runtime.getURL('viewer.html')}?forum=\\1`}},
      condition: {regexFilter: '^https://openreview\\.net/pdf\\?id=([A-Za-z0-9_-]+)$',
        resourceTypes: [chrome.declarativeNetRequest.ResourceType.MAIN_FRAME]},
    }] : [],
  });
}

async function syncRedirect() {
  const {autoOpen = true} = await chrome.storage.local.get('autoOpen');
  await setRedirect(autoOpen as boolean);
}
chrome.runtime.onInstalled.addListener(syncRedirect);
chrome.runtime.onStartup.addListener(syncRedirect);
chrome.storage.onChanged.addListener((changes, area) => {if (area === 'local' && 'autoOpen' in changes) syncRedirect();});

chrome.tabs.onUpdated.addListener((tabId, change, tab) => {
  if (change.status !== 'loading') return;
  const forum = forumOf(tab.url);
  if (!forum) return;
  lookup(forum).then(hit => badge(tabId, hit), () => badge(tabId, null)).catch(() => {});
});

chrome.runtime.onMessage.addListener((msg: {type: string; forum: string}, sender, reply) => {
  if (msg.type !== 'lookup' || !isForumId(msg.forum)) return;
  lookup(msg.forum).then(
    hit => {if (sender.tab?.id !== undefined) badge(sender.tab.id, hit); reply({hit});},
    err => reply({error: String(err?.message || err)}),
  );
  return true;
});
