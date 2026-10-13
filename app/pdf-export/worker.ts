// Temporary read-only export of PDFs from R2 for calibration (not deployed until approved).
// POST /pack with a JSON list of up to 200 "papers/<sha256>.pdf" keys streams one bundle:
// per object [u32 key length][key][u64 body length][body]; missing objects get length 0.
interface Env { BUCKET: R2Bucket; TOKEN_SHA256: string; EXPIRES_AT: string }
const KEY = /^papers\/[a-f0-9]{64}\.pdf$/;
export default { async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  if (Date.now() > Number(env.EXPIRES_AT)*1000) return new Response('Expired',{status:403});
  const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode((request.headers.get('Authorization')||'').replace(/^Bearer /,'')));
  if([...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('')!==env.TOKEN_SHA256) return new Response('Unauthorized',{status:401});
  const url=new URL(request.url);
  // GET /list?cursor=...: one page (up to 1000) of "papers/" PDF keys with sizes, for inventory.
  if(request.method==='GET'&&url.pathname==='/list'){
    const page=await env.BUCKET.list({prefix:'papers/',limit:1000,cursor:url.searchParams.get('cursor')||undefined});
    return Response.json({keys:page.objects.filter(o=>KEY.test(o.key)).map(o=>({key:o.key,size:o.size})),cursor:page.truncated?page.cursor:null});
  }
  if(request.method!=='POST'||url.pathname!=='/pack') return new Response('Not found',{status:404});
  const keys:unknown=await request.json();
  if(!Array.isArray(keys)||keys.length>200||!keys.every(k=>typeof k==='string'&&KEY.test(k))) return new Response('Invalid keys',{status:400});
  const {readable,writable}=new TransformStream<Uint8Array,Uint8Array>();
  ctx.waitUntil((async()=>{
    const w=writable.getWriter(), enc=new TextEncoder();
    try{
      for(const key of keys as string[]){
        const o=await env.BUCKET.get(key), k=enc.encode(key), head=new Uint8Array(12+k.length), dv=new DataView(head.buffer);
        dv.setUint32(0,k.length); head.set(k,4); dv.setBigUint64(4+k.length,BigInt(o?o.size:0)); await w.write(head);
        if(o){const r=o.body.getReader(); for(;;){const {done,value}=await r.read(); if(done)break; await w.write(value);}}
      }
      await w.close();
    }catch(e){await w.abort(e);}
  })());
  return new Response(readable,{headers:{'Content-Type':'application/octet-stream'}});
}} satisfies ExportedHandler<Env>;
