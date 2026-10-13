/** Usage statistics (Alice, Oct 13 2026): one Analytics Engine data point per meaningful request, written by the Worker.
 * No cookies and no script in the browser. IP addresses are never stored: a visitor is counted by a hash of the address,
 * browser and date with a secret salt, which changes every day, so the same person cannot be followed across days.
 * Read with scripts/usage_report.py. Disclosed on the About page.
 *
 * Data point layout (Analytics Engine: index1, blob1..blob12, double1..double2):
 *   index1  event: page | paper | browse | feedback
 *   blob1   event (repeated for SQL convenience)     blob7   browser family
 *   blob2   page (view name) or paper id              blob8   referrer site (pages only, other sites only)
 *   blob3   country                                   blob9   first browser language
 *   blob4   region / city                             blob10  daily visitor hash
 *   blob5   institution or network (AS organization)  blob11  detail: sort, counting mode, search used
 *   blob6   device: mobile, tablet, desktop or bot    blob12  site host (opentextdetector.com, workers.dev, …)
 *   double1 1 (count)                                 double2 HTTP status
 */
export type UsageSink={writeDataPoint(p:{indexes?:string[];blobs?:string[];doubles?:number[]}):void};
type Cf={country?:string;region?:string;city?:string;asOrganization?:string;verifiedBotCategory?:string};

const BOT=/bot|crawl|spider|slurp|fetch|preview|scrape|curl|wget|python|httpx|axios|node-fetch|go-http|java\/|headless|lighthouse|monitor/i;
export function device(ua:string,cf:Cf={}):string{
  if(cf.verifiedBotCategory||!ua||BOT.test(ua))return 'bot';
  if(/iPad|Tablet|Android(?!.*Mobile)/i.test(ua))return 'tablet';
  return /Mobi|iPhone|Android/i.test(ua)?'mobile':'desktop';
}
export function browser(ua:string):string{
  for(const [name,re] of [['Edge',/Edg\//],['Opera',/OPR\//],['Brave',/Brave/],['Firefox',/Firefox\//],['Chrome',/Chrome\//],['Safari',/Safari\//]] as const)if(re.test(ua))return name;
  return 'other';
}

/** Which event a request is, or null for requests that are not counted (PDF bytes, metadata, assets, robots.txt…). */
export function classify(url:URL,method:string):{event:string;item:string;detail:string}|null{
  if(method!=='GET'&&!(method==='POST'&&url.pathname==='/backend/v1/feedback'))return null;
  const p=url.searchParams;
  if(url.pathname==='/'){
    const view=p.get('view');
    const item=p.has('paper')?'reader':view||([...p.keys()].length?'submissions':'home');
    return {event:'page',item,detail:''};
  }
  const reader=url.pathname.match(/^\/backend\/v1\/pdf-reader\/([a-f0-9]{24})$/);
  if(reader)return {event:'paper',item:reader[1],detail:''};
  if(url.pathname==='/backend/v1/browse'&&!p.has('paper')){
    // Search text is not stored (it can contain names); only whether a search or filter was used.
    const parts=[`sort=${p.get('sort')||'default'}`,`mode=${p.get('mode')||'default'}`,p.get('q')?'search':'',p.get('area')?'area':'',p.get('ai')||p.get('cut')?'flag-filter':'',p.get('saved')==='1'?'reading-list':''].filter(Boolean);
    return {event:'browse',item:p.get('collection')??'iclr/2027',detail:parts.join(' ')};
  }
  if(url.pathname==='/backend/v1/feedback')return {event:'feedback',item:'',detail:''};
  return null;
}

async function visitorHash(salt:string,ip:string,ua:string,day:string){
  const d=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(`${salt}|${day}|${ip}|${ua}`));
  return [...new Uint8Array(d).slice(0,8)].map(b=>b.toString(16).padStart(2,'0')).join('');
}

/** Records one data point when the request is a counted event. Never throws: statistics must not break the site. */
export async function recordUsage(sink:UsageSink|undefined,salt:string|undefined,request:Request,status:number,now=new Date()){
  try{
    if(!sink)return;
    const url=new URL(request.url),e=classify(url,request.method);if(!e)return;
    const cf=((request as unknown as {cf?:Cf}).cf)||{},ua=request.headers.get('User-Agent')||'';
    const ip=request.headers.get('CF-Connecting-IP')||'';
    let ref='';
    if(e.event==='page'){try{const r=new URL(request.headers.get('Referer')||'');if(r.hostname!==url.hostname)ref=r.hostname;}catch{}}
    const lang=(request.headers.get('Accept-Language')||'').split(',')[0].trim().slice(0,16);
    const visitor=salt&&ip?await visitorHash(salt,ip,ua,now.toISOString().slice(0,10)):'';
    sink.writeDataPoint({indexes:[e.event],doubles:[1,status],blobs:[e.event,e.item,cf.country||'',[cf.region,cf.city].filter(Boolean).join(' / '),
      (cf.asOrganization||'').slice(0,96),device(ua,cf),browser(ua),ref,lang,visitor,e.detail,url.hostname]});
  }catch{}
}
