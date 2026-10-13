interface Env { BUCKET: R2Bucket; TOKEN_SHA256: string; EXPIRES_AT: string }
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (Date.now() > Number(env.EXPIRES_AT) * 1000) return new Response('Expired', {status:403});
    const auth = request.headers.get('Authorization') || '';
    const digest = await crypto.subtle.digest('SHA-256',new TextEncoder().encode(auth.replace(/^Bearer /,'')));
    const hash = [...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('');
    if(hash!==env.TOKEN_SHA256)return new Response('Unauthorized',{status:401});
    const key = new URL(request.url).pathname.slice(1);
    if(!/^classified-archive\/(results|extractions|indexes)\/[a-f0-9]{64}\.(pgf|json)$/.test(key))return new Response('Not found',{status:404});
    const headers = {'Cache-Control':'private, no-store'};
    if(request.method==='HEAD'){
      const obj=await env.BUCKET.head(key);
      return obj ? new Response(null,{headers:{...headers,'Content-Length':String(obj.size),'ETag':obj.httpEtag}}) : new Response(null,{status:404});
    }
    if(request.method==='GET'){
      const obj=await env.BUCKET.get(key);
      return obj ? new Response(obj.body,{headers:{...headers,'Content-Length':String(obj.size),'ETag':obj.httpEtag}}) : new Response(null,{status:404});
    }
    if(request.method==='PUT'){
      const previous=await env.BUCKET.head(key);
      if(previous)return Response.json({size:previous.size,etag:previous.etag,existing:true},{headers});
      const obj=await env.BUCKET.put(key,request.body,{onlyIf:{etagDoesNotMatch:'*'},httpMetadata:{contentType:key.endsWith('.json')?'application/json':'application/octet-stream'}});
      if(!obj)return new Response('Concurrent write; retry',{status:409});
      return Response.json({size:obj.size,etag:obj.etag},{headers});
    }
    return new Response('Method not allowed',{status:405});
  }
} satisfies ExportedHandler<Env>;
