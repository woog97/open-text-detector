interface Env { BUCKET: R2Bucket; TOKEN_SHA256: string; EXPIRES_AT: string }
export default { async fetch(request: Request, env: Env): Promise<Response> {
 if(Date.now()>Number(env.EXPIRES_AT)*1000)return new Response('Expired',{status:403});
 const h=await crypto.subtle.digest('SHA-256',new TextEncoder().encode((request.headers.get('Authorization')||'').replace(/^Bearer /,'')));
 if([...new Uint8Array(h)].map(x=>x.toString(16).padStart(2,'0')).join('')!==env.TOKEN_SHA256)return new Response('Unauthorized',{status:401});
 const key=new URL(request.url).pathname.slice(1);
 if(!/^(papers\/[a-f0-9]{64}\.pdf|backups\/iclr2027\/objects\/[a-f0-9]{64}|backups\/iclr2027\/manifests\/[a-f0-9]{64}\.json)$/.test(key))return new Response('Not found',{status:404});
 if(request.method==='PUT'){
  const sha=request.headers.get('X-Content-SHA256');
  if(!sha||!key.includes(sha))return new Response('Bad checksum',{status:400});
  const old=await env.BUCKET.head(key);
  if(old)return Response.json({size:old.size,etag:old.etag,sha256:old.checksums.toJSON().sha256||old.customMetadata?.sha256,existing:true});
  const o=await env.BUCKET.put(key,request.body,{sha256:sha,customMetadata:{sha256:sha},httpMetadata:{contentType:key.endsWith('.pdf')?'application/pdf':'application/octet-stream'}});
  return Response.json({size:o.size,etag:o.etag,sha256:o.checksums.toJSON().sha256});
 }
 if(request.method==='GET' && new URL(request.url).searchParams.has('metadata')){
  const o=await env.BUCKET.head(key);return o?Response.json({size:o.size,etag:o.etag,sha256:o.checksums.toJSON().sha256||o.customMetadata?.sha256,existing:true}):new Response('Not found',{status:404});
 }
 if(request.method==='GET'){
  const o=await env.BUCKET.get(key);return o?new Response(o.body,{headers:{ETag:o.httpEtag}}):new Response('Not found',{status:404});
 }
 return new Response('Method not allowed',{status:405});
}} satisfies ExportedHandler<Env>;
