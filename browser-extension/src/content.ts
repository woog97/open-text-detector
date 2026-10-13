/** OpenReview forum pages: a card under the paper title with the website's flagged share and links. */
import {share, pct, paperUrl, fprLabel, countedText, FLAG_FPRS, savedFpr, saveFpr, modeAt, type PaperHit, type FlagFpr} from './site.ts';

const HOST_ID = 'otd-card';

function card(forum: string, hit: PaperHit | null, fpr: FlagFpr, error?: string) {
  const mode = modeAt(fpr);
  const host = document.createElement('div');
  host.id = HOST_ID;
  const root = host.attachShadow({mode: 'open'});
  const s = hit && share(hit, mode);
  const viewer = `${chrome.runtime.getURL('viewer.html')}?forum=${encodeURIComponent(forum)}`;
  let tone = 'none', head: string, detail: string;
  if (error) {head = 'Open Text Detector unavailable'; detail = error;}
  else if (!hit) {head = 'Not in the collection'; detail = 'Open Text Detector has not collected this paper.';}
  else if (s!.state === 'unscored') {head = 'Not scored yet'; detail = 'Collected, not yet classified by the final model.';}
  else if (s!.state === 'short') {head = 'Too short to score'; detail = `${s!.counted} ${countedText(mode)} (needs 50).`;}
  else {
    const r = s!.rate;
    tone = r > .5 ? 'high' : r > .1 ? 'mid' : 'low';
    head = `${pct(r)} AI`;
    detail = `${s!.flagged.toLocaleString()} of ${s!.counted.toLocaleString()} ${countedText(mode)} flagged at ${fprLabel(fpr)} human false-positive rate`;
  }
  const fprs = hit?.flags ? `<span class="f" title="Human false-positive rate of the sentence cutoff">FPR ${FLAG_FPRS.map(f => `<button data-fpr="${f}" class="${f === fpr ? 'on' : ''}">${fprLabel(f)}</button>`).join('')}</span>` : '';
  const links = hit ? `<a href="${viewer}" target="_blank">Highlighted PDF</a><a href="${paperUrl(hit)}" target="_blank">Paper page</a>` : '';
  root.innerHTML = `<style>
    :host{all:initial;display:block;margin:10px 0 14px}
    .c{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font:13px/1.4 system-ui,sans-serif;color:#222;
       border:1px solid #d8d4cc;border-left:4px solid var(--t);background:#fbfaf7;border-radius:6px;padding:8px 12px}
    .low{--t:#3f7a55}.mid{--t:#a3690c}.high{--t:#b2412b}.none{--t:#999}
    b{font-size:18px;color:var(--t);white-space:nowrap}
    .d{flex:1;min-width:200px;color:#555}
    .n{display:block;font-size:11px;color:#888;letter-spacing:.02em;text-transform:uppercase}
    .f{color:#888;font-size:12px;white-space:nowrap}
    .f button{font:inherit;color:#555;background:#fff;border:1px solid #d8d4cc;padding:1px 7px;margin-left:3px;border-radius:4px;cursor:pointer}
    .f button.on{background:#8c1b13;border-color:#8c1b13;color:#fff}
    a{color:#8c1b13;text-decoration:none;font-weight:600;margin-left:10px}a:hover{text-decoration:underline}
  </style><div class="c ${tone}"><b></b><span class="d"><span class="n">Open Text Detector · Qwen3.6 35B MoE</span><span class="t"></span></span>${fprs}<span>${links}</span></div>`;
  root.querySelector('b')!.textContent = head;
  root.querySelector('.t')!.textContent = detail;
  root.querySelectorAll<HTMLButtonElement>('.f button').forEach(b => b.onclick = () => saveFpr(Number(b.dataset.fpr) as FlagFpr));
  return host;
}

async function run(forum: string) {
  const res = await chrome.runtime.sendMessage({type: 'lookup', forum}) as {hit?: PaperHit | null; error?: string};
  // OpenReview renders the title client-side; wait for it (up to ~15 s).
  for (let i = 0; i < 60 && location.search.includes(forum); i++) {
    const title = document.querySelector('.forum-title') || document.querySelector('h2.citation_title')?.parentElement;
    if (title) {
      document.getElementById(HOST_ID)?.remove();
      title.after(card(forum, res.hit ?? null, await savedFpr(), res.error));
      return;
    }
    await new Promise(r => setTimeout(r, 250));
  }
}

// A switch here, in the popup or in the viewer redraws the card (and the badge, through a fresh lookup).
chrome.storage.onChanged.addListener((changes, area) => {if (area === 'local' && 'fpr' in changes && shown) run(shown);});

// OpenReview navigates between forums without page loads, so follow the URL.
let shown = '';
setInterval(() => {
  const forum = new URLSearchParams(location.search).get('id') || '';
  if (location.pathname === '/forum' && forum && forum !== shown) {shown = forum; run(forum);}
}, 500);
