interface Env { BUCKET: R2Bucket; TOKEN_SHA256: string; EXPIRES_AT: string }
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (Date.now() > Number(env.EXPIRES_AT) * 1000) return new Response("Expired", {status:403});
    const auth = request.headers.get("Authorization") || "";
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(auth.replace(/^Bearer /, "")));
    const hex = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2,"0")).join("");
    if (hex !== env.TOKEN_SHA256) return new Response("Unauthorized", {status:401});
    const key = new URL(request.url).pathname.slice(1);
    const allowed = /^model-cache\/meld\/[a-f0-9]{64}\/(part-\d{4}|manifest\.json)$/.test(key)
      || /^extractions\/positioned\/[a-f0-9]{64}\/[a-f0-9]{64}\.pgf$/.test(key)
      || /^queues\/meld-v5\/[a-f0-9]{64}\.json$/.test(key)
      || /^meld-batches\/objects\/[a-f0-9]{64}\.tar$/.test(key)
      || /^classification-indexes\/meld-v5\/[a-f0-9]{64}\.json$/.test(key)
      || /^runtime-profiles\/meld-v5\/[a-f0-9]{64}\.json$/.test(key);
    if (!allowed) return new Response("Not found", {status:404});
    if (!["GET","HEAD"].includes(request.method)) return new Response("Method not allowed", {status:405});
    const object = await env.BUCKET.get(key);
    if (!object) return new Response("Not found", {status:404});
    return new Response(request.method === "HEAD" ? null : object.body, {headers:{
      "Content-Type":"application/octet-stream", "Content-Length":String(object.size),
      "Cache-Control":"private, no-store", "ETag":object.httpEtag}});
  }
} satisfies ExportedHandler<Env>;
