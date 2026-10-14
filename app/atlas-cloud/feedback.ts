/** Reader feedback ("Think this result is wrong?"): free-text messages about one paper's result, stored privately in R2
 * under feedback/ (nothing on the site reads that prefix back; scripts/read_feedback.py lists them for Alice).
 * No IP address or user agent is stored; visitors may choose to leave contact details in the message itself. Added Oct 13, 2026.
 *
 * Editing (Alice, Oct 14 2026): a paper message is stored at feedback/<day>/<ts>-<uuid>.json and the reply gives the browser its
 * id and a random edit key, which the browser keeps (reader-summary.tsx). Only a SHA-256 hash of the key is stored. An edit sends
 * the id and key back and is saved as a new file feedback/<id>/edit-<ts>.json; earlier versions are never overwritten.
 * Exact repeats (same paper, same text, same UTC day) are stored once: an empty marker feedback/_seen/<day>/<hash> records it. */
export const FEEDBACK_PREFIX='feedback/';
/** Answers to "Are you an author of this paper?" in the reader's feedback form (reader-summary.tsx). */
export const AUTHOR_ROLES=['no','first','second','other'] as const;
/** Answers to "Do you consent to these labels being used for hard negative mining?" in the same form. */
export const CONSENT_ANSWERS=['yes','no'] as const;
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
type Deps={bucket:Pick<R2Bucket,'put'|'get'|'head'>;perSender?:Limiter;sitewide?:Limiter;paperExists:(id:string)=>Promise<boolean>;now?:()=>Date};
type Reply=(value:unknown,status?:number)=>Response;
/** A paper message's id: its key under feedback/ without the .json ending. */
const MESSAGE_ID=/^\d{4}-\d{2}-\d{2}\/\d{13}-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
async function sha256(text:string){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text)))].map(b=>b.toString(16).padStart(2,'0')).join('');}

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
  // Optional contact details, given in their own fields so the sender chooses to (Oct 13, 2026).
  const contact:Record<string,string>={};
  if(body.contact&&typeof body.contact==='object')for(const k of ['name','reach'] as const){const v=(body.contact as Record<string,unknown>)[k];if(typeof v==='string'&&v.trim())contact[k]=v.trim().slice(0,200);}
  // Paper feedback only: whether the sender says they are an author of the paper (Alice, Oct 14 2026).
  const author=!site&&typeof body.author==='string'&&(AUTHOR_ROLES as readonly string[]).includes(body.author)?body.author:'';
  const consent=!site&&typeof body.consent==='string'&&(CONSENT_ANSWERS as readonly string[]).includes(body.consent)?body.consent:'';
  const now=(deps.now||(()=>new Date()))(),day=now.toISOString().slice(0,10);
  const put=(key:string,value:unknown)=>deps.bucket.put(key,JSON.stringify(value),{httpMetadata:{contentType:'application/json'}});
  const optional={...(Object.keys(contact).length?{contact}:{})};
  if(site){
    await put(`${FEEDBACK_PREFIX}site/${day}/${now.getTime()}-${crypto.randomUUID()}.json`,{received_at:now.toISOString(),kind:'site',comment,context,...optional});
    return json({ok:true},201);
  }
  const message={received_at:now.toISOString(),paper,comment,context,...(passages.length?{passages}:{}),...(author?{author}:{}),...(consent?{consent}:{}),...optional};
  // An edit of an earlier message: needs that message's id and edit key, and the same paper.
  if(body.edit!==undefined){
    const edit=body.edit as Record<string,unknown>|null,id=edit?.id,editKey=edit?.key;
    if(typeof id!=='string'||!MESSAGE_ID.test(id)||typeof editKey!=='string'||editKey.length>128)return json({detail:'This message can no longer be edited'},403);
    const first=await deps.bucket.get(`${FEEDBACK_PREFIX}${id}.json`);
    const original=first?await first.json() as {paper?:string;edit_key_sha256?:string}:null;
    if(!original?.edit_key_sha256||original.paper!==paper||original.edit_key_sha256!==await sha256(editKey))return json({detail:'This message can no longer be edited'},403);
    await put(`${FEEDBACK_PREFIX}${id}/edit-${now.getTime()}.json`,{...message,edit_of:id});
    return json({ok:true,id},201);
  }
  const seen=`${FEEDBACK_PREFIX}_seen/${day}/${await sha256(`${paper}\n${comment}`)}`;
  if(await deps.bucket.head(seen))return json({ok:true,duplicate:true},201);
  const id=`${day}/${now.getTime()}-${crypto.randomUUID()}`,editKey=[...crypto.getRandomValues(new Uint8Array(24))].map(b=>b.toString(16).padStart(2,'0')).join('');
  await put(`${FEEDBACK_PREFIX}${id}.json`,{...message,edit_key_sha256:await sha256(editKey)});
  await deps.bucket.put(seen,'');
  return json({ok:true,id,key:editKey},201);
}
