/** Reader feedback ("Think this result is wrong?"): free-text messages about one paper's result, stored privately in R2
 * under feedback/ (nothing on the site reads that prefix back; scripts/read_feedback.py lists them for Alice).
 * No IP address or user agent is stored; visitors may choose to leave contact details in the message itself. Added Oct 13, 2026. */
export const FEEDBACK_PREFIX='feedback/';
export const MAX_COMMENT=4000;
/** Minimum message length, also when passages are marked (Alice, Oct 13 2026). */
export const MIN_COMMENT=100;
/** Passages a reader marked as written without AI (indices into the report's segments; Oct 13, 2026). */
export const MAX_PASSAGES=2000;
/** Request size cap in bytes: a 4,000-character message in any script (up to 4 bytes per character) plus 2,000 marks. */
const MAX_BODY=32768;
/** Settings shown when the message was sent, so the result the reader saw can be reproduced. */
const CONTEXT_KEYS=['report','mode','shown','cutoff'] as const;
type Limiter={limit(o:{key:string}):Promise<{success:boolean}>};
type Deps={bucket:Pick<R2Bucket,'put'>;perSender?:Limiter;sitewide?:Limiter;paperExists:(id:string)=>Promise<boolean>;now?:()=>Date};
type Reply=(value:unknown,status?:number)=>Response;

export async function handleFeedback(request:Request,deps:Deps,json:Reply):Promise<Response>{
  if(request.method!=='POST')return json({detail:'Method not allowed'},405);
  // Only this site's own pages may send messages (blocks posts from other sites' forms and scripts).
  if(request.headers.get('Origin')!==new URL(request.url).origin)return json({detail:'Forbidden'},403);
  if(!(request.headers.get('Content-Type')||'').startsWith('application/json'))return json({detail:'Expected JSON'},415);
  if(Number(request.headers.get('Content-Length')||0)>MAX_BODY)return json({detail:'Message too long'},413);
  const raw=await request.text();if(raw.length>MAX_BODY)return json({detail:'Message too long'},413);
  let body:Record<string,unknown>;try{body=JSON.parse(raw);}catch{return json({detail:'Invalid JSON'},400);}
  if(!body||typeof body!=='object'||Array.isArray(body))return json({detail:'Invalid message'},400);
  // Hidden field that people never fill in; form-filling bots often do. Pretend success and store nothing.
  if(typeof body.website==='string'&&body.website)return json({ok:true},201);
  // Two kinds: about one paper's result (the reader), or about the site in general (the corner button on other pages).
  const site=body.kind==='site',paper=body.paper,comment=typeof body.comment==='string'?body.comment.trim():'';
  if(!site&&(typeof paper!=='string'||!/^[a-f0-9]{24}$/.test(paper)))return json({detail:'Unknown paper'},400);
  const rawPassages=body.passages===undefined||site?[]:body.passages;
  if(!Array.isArray(rawPassages)||rawPassages.length>MAX_PASSAGES||!rawPassages.every(n=>Number.isSafeInteger(n)&&(n as number)>=0&&(n as number)<1_000_000))return json({detail:'Invalid marked passages'},400);
  const passages=[...new Set(rawPassages as number[])].sort((a,b)=>a-b);
  if(comment.length<MIN_COMMENT)return json({detail:`Please write at least ${MIN_COMMENT} characters`},400);
  if(comment.length>MAX_COMMENT)return json({detail:`Please keep it under ${MAX_COMMENT.toLocaleString('en-US')} characters`},400);
  const ctxIn=body.context&&typeof body.context==='object'?body.context as Record<string,unknown>:{},context:Record<string,string>={};
  for(const k of site?(['page'] as const):CONTEXT_KEYS){const v=ctxIn[k];if(typeof v==='string'||typeof v==='number')context[k]=String(v).slice(0,120);}
  // Rate limits (wrangler.atlas.json): per sender, keyed by IP address only inside the limiter (never stored), and site-wide.
  const busy=()=>json({detail:'Too many messages right now; please try again in a few minutes'},429);
  if(deps.perSender&&!(await deps.perSender.limit({key:request.headers.get('CF-Connecting-IP')||'unknown'})).success)return busy();
  if(deps.sitewide&&!(await deps.sitewide.limit({key:'all'})).success)return busy();
  if(!site&&!(await deps.paperExists(paper as string)))return json({detail:'Unknown paper'},400);
  // Site messages may carry optional contact details, given in their own fields so the sender chooses to (Oct 13, 2026).
  const contact:Record<string,string>={};
  if(site&&body.contact&&typeof body.contact==='object')for(const k of ['name','reach'] as const){const v=(body.contact as Record<string,unknown>)[k];if(typeof v==='string'&&v.trim())contact[k]=v.trim().slice(0,200);}
  const now=(deps.now||(()=>new Date()))(),day=now.toISOString().slice(0,10);
  const key=`${FEEDBACK_PREFIX}${site?'site/':''}${day}/${now.getTime()}-${crypto.randomUUID()}.json`;
  await deps.bucket.put(key,JSON.stringify(site?{received_at:now.toISOString(),kind:'site',comment,context,...(Object.keys(contact).length?{contact}:{})}:{received_at:now.toISOString(),paper,comment,context,...(passages.length?{passages}:{})}),{httpMetadata:{contentType:'application/json'}});
  return json({ok:true},201);
}
