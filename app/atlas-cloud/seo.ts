/** Search and link-preview metadata. The site is one HTML shell, so the Worker writes each page's title, description
 * and canonical address into it before it is sent: crawlers and chat link previews read these without running scripts. */
export const ORIGIN='https://opentextdetector.com';
const NAME='Open Text Detector';
type Page={title:string;description:string};
/** Keys are the `?view=` values in atlas-cloud/main.tsx; "" is the home page. */
export const PAGES:Record<string,Page>={
  '':{title:'Open Text Detector: AI-written text in ICLR 2027 submissions',
    description:'A free, open-source detector that checks every public ICLR 2027 submission for AI-written text and shows, passage by passage, where the writing looks generated or edited by a language model.'},
  transparency:{title:'Transparency',
    description:'How the Open Text Detector was built, trained, calibrated and evaluated, with the data, code and model weights released.'},
  architecture:{title:'How the detector works',
    description:'A frozen language model with LoRA adapters and a classifier that gives every token a probability of being AI-written, with cutoffs set on human writing.'},
  data:{title:'Training and evaluation data',
    description:'The human and AI-written text the detector was trained and evaluated on: research papers, sentence and paragraph edits, and fully AI-written documents from several language models.'},
  results:{title:'Evaluation',
    description:'Recall at fixed false-positive rates on held-out papers and writers, and the runs compared to choose the backbone and training recipe.'},
  calibration:{title:'Calibration',
    description:'How the detector’s cutoffs are set from human-written papers, with ROC curves and the false-positive rate at each cutoff.'},
  limitations:{title:'Limitations',
    description:'What the Open Text Detector cannot do reliably, mistakes made while building it, and what time and budget did not cover.'},
  manifesto:{title:'Why I built this',
    description:'A manifesto by alice rigg // woog'},
  about:{title:'About',
    description:'Credits and release status of the Open Text Detector: code, model weights, training data and evaluation sets.'},
};
const BROWSE:Page={title:'ICLR 2027 submissions',
  description:'Browse public ICLR 2027 submissions with the share of each paper flagged as AI-written at a calibrated false-positive rate.'};
const escape=(s:string)=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

/** The metadata for one page address, or null for an unknown `?view=` (left to the app). */
export function pageMeta(url:URL):{title:string;description:string;canonical:string|null;noindex?:boolean}|null {
  const view=url.searchParams.get('view');
  if(view!==null&&!(view in PAGES)&&view!=='classifications')return null;
  if(view&&view in PAGES){const p=PAGES[view];return {title:`${p.title} · ${NAME}`,description:p.description,canonical:`${ORIGIN}/?view=${view}`};}
  // A paper in the reader: kept out of search results, so a search for a paper's title doesn't surface its flags (Alice, Oct 13).
  if(url.searchParams.has('paper'))return {title:`${BROWSE.title} \u00b7 ${NAME}`,description:BROWSE.description,canonical:null,noindex:true};
  // Submission lists and the reader: many filter combinations, so no canonical address of their own.
  if([...url.searchParams.keys()].length)return {title:`${BROWSE.title} · ${NAME}`,description:BROWSE.description,canonical:null};
  return {title:PAGES[''].title,description:PAGES[''].description,canonical:`${ORIGIN}/`};
}

/** Writes the page's metadata into the HTML shell (built from atlas-cloud/index.html). */
export function withMeta(html:string,url:URL):string {
  const m=pageMeta(url);if(!m)return html;
  const t=escape(m.title),d=escape(m.description);
  const tags=[
    m.noindex&&`<meta name="robots" content="noindex"/>`,
    m.canonical&&`<link rel="canonical" href="${escape(m.canonical)}"/>`,
    `<meta property="og:site_name" content="${NAME}"/>`,`<meta property="og:type" content="website"/>`,
    `<meta property="og:title" content="${t}"/>`,`<meta property="og:description" content="${d}"/>`,
    m.canonical&&`<meta property="og:url" content="${escape(m.canonical)}"/>`,
    `<meta name="twitter:card" content="summary"/>`,
    m.canonical===`${ORIGIN}/`&&`<script type="application/ld+json">${JSON.stringify({'@context':'https://schema.org','@type':'WebSite',name:NAME,url:`${ORIGIN}/`,description:m.description})}</script>`,
  ].filter(Boolean).join('');
  return html.replace(/<title>[^<]*<\/title>/,`<title>${t}</title>`)
    .replace(/<meta name="description" content="[^"]*"\/>/,`<meta name="description" content="${d}"/>`)
    .replace('</head>',`${tags}</head>`);
}

export function robotsTxt():string {
  return `User-agent: *\nDisallow: /workbench\nDisallow: /private\n\nSitemap: ${ORIGIN}/sitemap.xml\n`;
}
export function sitemapXml():string {
  const urls=['',...Object.keys(PAGES).filter(Boolean).map(v=>`?view=${v}`),'?collection=iclr%2F2027'];
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.map(u=>`  <url><loc>${escape(`${ORIGIN}/${u}`)}</loc></url>`).join('\n')}\n</urlset>\n`;
}
