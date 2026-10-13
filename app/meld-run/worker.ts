interface Env { BUCKET:R2Bucket; TOKEN_SHA256:string; EXPIRES_AT:string; RUN_PREFIX:string }
export default {async fetch(request:Request,env:Env):Promise<Response>{
 if(Date.now()>Number(env.EXPIRES_AT)*1000)return new Response('Expired',{status:403});
 const auth=request.headers.get('Authorization')||'';
 const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(auth.replace(/^Bearer /,'')));
 if([...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('')!==env.TOKEN_SHA256)return new Response('Unauthorized',{status:401});
 const key=new URL(request.url).pathname.slice(1);
 const output=key.startsWith(env.RUN_PREFIX+'/')&&/\/(v5|v8)\/[a-f0-9]{64}\.pgf$/.test(key)&&key.split('/').length===4;
 const readable=output||/^model-cache\/meld\/[a-f0-9]{64}\/(part-\d{4}|manifest\.json)$/.test(key);
 if(!readable)return new Response('Not found',{status:404});
 if(request.method==='PUT'&&output){
  const old=await env.BUCKET.head(key);if(old)return Response.json({size:old.size,etag:old.etag});
  const obj=await env.BUCKET.put(key,request.body,{onlyIf:{etagDoesNotMatch:'*'},httpMetadata:{contentType:'application/octet-stream'}});
  return obj?Response.json({size:obj.size,etag:obj.etag}):new Response('Retry',{status:409});
 }
 if(request.method==='HEAD'){const obj=await env.BUCKET.head(key);return obj?new Response(null,{headers:{'Content-Length':String(obj.size),'ETag':obj.httpEtag,'Cache-Control':'private, no-store'}}):new Response(null,{status:404});}
 if(request.method==='GET'){const obj=await env.BUCKET.get(key);return obj?new Response(obj.body,{headers:{'Content-Length':String(obj.size),'ETag':obj.httpEtag,'Cache-Control':'private, no-store'}}):new Response(null,{status:404});}
 return new Response('Method not allowed',{status:405});
}} satisfies ExportedHandler<Env>;
