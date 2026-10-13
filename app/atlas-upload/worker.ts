interface Env { BUCKET: R2Bucket; TOKEN_SHA256: string; EXPIRES_AT: string }
export default { async fetch(request: Request, env: Env): Promise<Response> {
  if (Date.now() > Number(env.EXPIRES_AT)*1000) return new Response('Expired',{status:403});
  const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode((request.headers.get('Authorization')||'').replace(/^Bearer /,'')));
  if([...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('')!==env.TOKEN_SHA256) return new Response('Unauthorized',{status:401});
  const key=new URL(request.url).pathname.slice(1);
  // Bundle: [u32 header length][header JSON: [{key,offset,length}]][bodies]; written to R2 inside Cloudflare.
  if(request.method==='POST'&&key==='bundle'){
    const buf=await request.arrayBuffer(),n=new DataView(buf).getUint32(0);
    const entries:{key:string;offset:number;length:number}[]=JSON.parse(new TextDecoder().decode(new Uint8Array(buf,4,n)));
    if(!Array.isArray(entries)||entries.length>45)return new Response('Too many entries',{status:400});
    const base=4+n;
    for(const e of entries)if(!/^atlas-public\/objects\/[a-f0-9]{64}\.json\.gz$/.test(e.key)||!Number.isInteger(e.offset)||!Number.isInteger(e.length)||e.offset<0||base+e.offset+e.length>buf.byteLength)return new Response('Invalid entry',{status:400});
    const written=await Promise.all(entries.map(async e=>{const o=await env.BUCKET.put(e.key,new Uint8Array(buf,base+e.offset,e.length),{httpMetadata:{contentType:'application/octet-stream'}});return {key:e.key,size:o.size,etag:o.etag};}));
    return Response.json(written);
  }
  if(!/^atlas-public\/(objects\/[a-f0-9]{64}\.json\.gz|catalogue\.json)$/.test(key))return new Response('Not found',{status:404});
  if(request.method==='PUT'){
    const object=await env.BUCKET.put(key,request.body,{onlyIf:request.headers,httpMetadata:{contentType:key.endsWith('.gz')?'application/octet-stream':'application/json'}});
    if(!object)return new Response('Publication changed; retry from the current catalogue',{status:412});
    return Response.json({size:object.size,etag:object.etag});
  }
  if(request.method==='GET'){
    const object=await env.BUCKET.get(key);
    return object?new Response(object.body,{headers:{ETag:object.httpEtag}}):new Response('Not found',{status:404});
  }
  return new Response('Method not allowed',{status:405});
}} satisfies ExportedHandler<Env>;
