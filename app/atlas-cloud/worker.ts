import legacyAliases from "../lib/legacy-paper-aliases.json" with {type:"json"};
import type {PublishedPaper} from "../lib/atlas-archive.ts";
import {createBrowseHandler} from "./browse-worker.ts";
import {ORIGIN,robotsTxt,sitemapXml,withMeta} from "./seo.ts";
interface Env { ASSETS: Fetcher; BUCKET: R2Bucket }
/** Written by scripts/publish_public_catalogue.py: the public catalogue and reader shards of one revision. */
type Publication = {revision:string;catalogue_key:string;shards_prefix:string};
export const PUBLICATION_KEY='atlas-public/public/current.json';
const security = {
  "X-Content-Type-Options":"nosniff", "X-Frame-Options":"DENY",
  "Referrer-Policy":"no-referrer", "Cross-Origin-Resource-Policy":"same-origin",
  "Permissions-Policy":"camera=(), microphone=(), geolocation=()",
  "Content-Security-Policy":"default-src 'none'; frame-ancestors 'none'; sandbox",
};
/** PDFs above this size are served from R2 on every request rather than copied into the edge cache. */
const EDGE_PDF_LIMIT=64<<20;
function headers(extra:HeadersInit={}) {const result=new Headers(security);new Headers(extra).forEach((v,k)=>result.set(k,v));return result;}
function json(value:unknown,status=200) {return Response.json(value,{status,headers:headers({"Cache-Control":"no-store"})});}
function matchesEtag(value:string|null,etag:string) {return !!value && value.split(',').some(v=>v.trim()==='*'||v.trim().replace(/^W\//,'')===etag.replace(/^W\//,''));}
/** Single byte ranges only. Reject unsafe integers and empty suffix ranges. */
export function byteRange(value:string,size:number):{offset:number;length:number}|null {
  const parts=/^bytes=(\d*)-(\d*)$/.exec(value);
  if(!parts||(!parts[1]&&!parts[2])||size<=0)return null;
  const first=parts[1]?Number(parts[1]):undefined, last=parts[2]?Number(parts[2]):undefined;
  if((first!==undefined&&!Number.isSafeInteger(first))||(last!==undefined&&!Number.isSafeInteger(last)))return null;
  if(first===undefined){if(!last)return null;const length=Math.min(last,size);return {offset:size-length,length};}
  const end=last===undefined?size-1:Math.min(last,size-1);
  return first>=size||first>end?null:{offset:first,length:end-first+1};
}
/** Factory keeps caches isolated in tests; production has one instance per isolate. */
export function createAtlasWorker() {
  const browse=createBrowseHandler();
  let publication:{data:Publication|null;until:number}|undefined;
  let publicationLoading:Promise<Publication|null>|undefined;
  /** The live publication, re-read every 30 s; null when none has been published. */
  async function current(env:Env):Promise<Publication|null>{
    if(publication&&publication.until>Date.now())return publication.data;
    publicationLoading??=(async()=>{
      const o=await env.BUCKET.get(PUBLICATION_KEY);
      const data=o?await o.json<Publication>():null;
      const base=data&&`atlas-public/public/${data.revision}`;
      if(data&&(!/^[a-f0-9]{64}$/.test(data.revision)||data.catalogue_key!==`${base}/catalogue.json.gz`||data.shards_prefix!==`${base}/reader/`))throw new Error('Invalid publication');
      publication={data,until:Date.now()+30000};return data;
    })().finally(()=>{publicationLoading=undefined;});
    return publicationLoading;
  }
  // Per-prefix id -> (PDF hash, detail hash) shards; keyed by revision, so a new publication never mixes with an old one.
  const shards=new Map<string,Record<string,[string,string]>>();
  /** A paper's object keys; null when the publication does not list it; undefined when there is no publication. */
  async function lookupPaper(env:Env,id:string):Promise<PublishedPaper|null|undefined>{
    const p=await current(env);if(!p)return undefined;
    const key=`${p.shards_prefix}${id.slice(0,2)}.json`;let shard=shards.get(key);
    if(!shard){
      const o=await env.BUCKET.get(key);shard=o?await o.json<Record<string,[string,string]>>():{};
      shards.set(key,shard);while(shards.size>64)shards.delete(shards.keys().next().value!);
    }
    const entry=shard[id];if(!entry)return null;
    const [pdf,detail]=entry;
    if(!/^[a-f0-9]{64}$/.test(pdf)||(detail&&!/^[a-f0-9]{64}$/.test(detail)))throw new Error('Invalid published object');
    return {id,pdf_key:`papers/${pdf}.pdf`,detail_key:detail?`atlas-public/objects/${detail}.json.gz`:undefined,classified:!!detail,models:[],title:id,filename:id+'.pdf',collection:'',bytes:0} as PublishedPaper;
  }
  async function readerPaper(env:Env,id:string){
    const listed=await lookupPaper(env,id);
    if(listed)return listed;
    // PDFs uploaded after the last publication: IDs are public PDF hash prefixes; never scan the full archive.
    // PDF.js sends many range requests per paper, so remember recent answers (PDF objects are content-addressed and immutable).
    const seen=unlisted.get(id);if(seen)return seen;
    const found:{key:string;size:number}[]=[];let cursor:string|undefined;
    do{const page=await env.BUCKET.list({prefix:`papers/${id}`,limit:1000,cursor});
      found.push(...page.objects.filter(o=>/^papers\/[a-f0-9]{64}\.pdf$/.test(o.key)&&o.key.slice(7,31)===id));
      cursor=page.truncated?page.cursor:undefined;
    }while(cursor);
    if(found.length>1)throw new Error('PDF identifier collision');
    const item=found[0];if(!item)return undefined;
    const paper={id,pdf_key:item.key,classified:false,models:[],title:id,filename:id+'.pdf',collection:'Other uploads',bytes:item.size} as PublishedPaper;
    unlisted.set(id,paper);while(unlisted.size>2000)unlisted.delete(unlisted.keys().next().value!);
    return paper;
  }
  const unlisted=new Map<string,PublishedPaper>();
  // Abstract/BibTeX shards (~0.3 MB each): keep a few parsed, refreshed after a minute.
  const details=new Map<string,{data:Record<string,unknown>;until:number}>();
  async function detailShard(env:Env,key:string){
    const hit=details.get(key);if(hit&&hit.until>Date.now())return hit.data;
    const o=await env.BUCKET.get(key),data=o?await o.json<Record<string,unknown>>():{};
    details.delete(key);details.set(key,{data,until:Date.now()+60000});while(details.size>24)details.delete(details.keys().next().value!);
    return data;
  }
  const edgeFor=()=>typeof caches!=='undefined'?(caches as CacheStorage&{default:Cache}).default:undefined;
  // Objects this isolate is already copying into the edge cache, so a burst of range requests copies a PDF once.
  const copying=new Set<string>();
  /**
   * Serves an immutable gzip JSON object (catalogue or per-paper results), from the edge cache when this data
   * centre has a copy. Content-addressed keys make the hash a strong ETag known before any storage read.
   */
  async function gzipObject(request:Request,env:Env,ctx:ExecutionContext|undefined,key:string,hash:string,cacheControl:string){
    const etag=`"${hash}"`,head=request.method==='HEAD';
    const h=headers({'Content-Type':'application/json','Content-Encoding':'gzip','ETag':etag,'Cache-Control':cacheControl});
    if(matchesEtag(request.headers.get('if-none-match'),etag))return new Response(null,{status:304,headers:h});
    const edge=edgeFor(),edgeKey=`${new URL(request.url).origin}/backend/v1/object-cache/gz2/${hash}.json.gz`;
    const hit=edge?await edge.match(edgeKey):undefined;
    if(hit){h.set('X-Edge-Cache','hit');return new Response(head?null:hit.body,{headers:h,encodeBody:'manual'});}
    const object=await env.BUCKET.get(key);
    if(!object)return json({detail:'Results not found'},404);
    let body=object.body;
    if(edge&&ctx&&!head){
      const [client,copy]=object.body.tee();body=client;
      // Stored as opaque bytes: a cached Content-Encoding: gzip response came back with an empty body.
      ctx.waitUntil(edge.put(edgeKey,new Response(copy,{headers:{'Content-Type':'application/octet-stream','ETag':etag,'Cache-Control':'public, max-age=31536000, immutable'}})).catch(()=>{}));
    }
    h.set('X-Edge-Cache','miss');
    return new Response(head?null:body,{headers:h,encodeBody:'manual'});
  }
  /** PDF bytes with single-range support. The edge cache serves ranges from a full copy once one is stored. */
  async function pdf(request:Request,env:Env,ctx:ExecutionContext|undefined,id:string,key:string){
    const hash=key.slice(7,-4),etag=`"${hash}"`,head=request.method==='HEAD';
    const h=headers({'Content-Type':'application/pdf','ETag':etag,'Cache-Control':'public, max-age=86400','Accept-Ranges':'bytes','Content-Disposition':`inline; filename="${id}.pdf"`});
    if(matchesEtag(request.headers.get('if-none-match'),etag))return new Response(null,{status:304,headers:h});
    const ifRange=request.headers.get('if-range'),requested=(!ifRange||ifRange===etag)?request.headers.get('range'):null;
    const edge=edgeFor(),edgeKey=`${new URL(request.url).origin}/backend/v1/object-cache/${hash}.pdf`;
    // A stored copy answers a Range request with 206 itself; only well-formed single ranges reach it.
    if(edge&&!head&&(!requested||/^bytes=(\d+-\d*|-\d+)$/.test(requested))){
      const hit=await edge.match(new Request(edgeKey,{headers:requested?{Range:requested}:{}}));
      if(hit&&(hit.status===200||hit.status===206)){
        for(const name of ['Content-Length','Content-Range'])if(hit.headers.has(name))h.set(name,hit.headers.get(name)!);
        h.set('X-Edge-Cache','hit');return new Response(hit.body,{status:hit.status,headers:h});
      }
    }
    const meta=await env.BUCKET.head(key);
    if(!meta)return json({detail:'PDF not found'},404);
    let range={offset:0,length:meta.size},status=200;
    if(!head&&requested){
      const parsed=byteRange(requested,meta.size);
      if(!parsed){h.set('Content-Range',`bytes */${meta.size}`);h.set('Cache-Control','no-store');return new Response(null,{status:416,headers:h});}
      range=parsed;status=206;h.set('Content-Range',`bytes ${range.offset}-${range.offset+range.length-1}/${meta.size}`);
    }
    h.set('Content-Length',String(range.length));
    if(head)return new Response(null,{headers:h});
    const object=await env.BUCKET.get(key,{range});
    if(!object)return json({detail:'Object not found'},404);
    if(edge&&ctx&&meta.size<=EDGE_PDF_LIMIT&&!copying.has(hash)){
      copying.add(hash);
      ctx.waitUntil((async()=>{
        const full=await env.BUCKET.get(key);if(!full)return;
        await edge.put(edgeKey,new Response(full.body,{headers:{'Content-Type':'application/pdf','Content-Length':String(full.size),'ETag':etag,'Cache-Control':'public, max-age=31536000, immutable'}}));
      })().catch(()=>{}).finally(()=>copying.delete(hash)));
    }
    h.set('X-Edge-Cache','miss');
    return new Response(object.body,{status,headers:h});
  }
  async function serve(request:Request,env:Env,ctx?:ExecutionContext):Promise<Response> {
    const url=new URL(request.url);
    if(url.pathname==='/backend/v1/browse'&&['GET','HEAD'].includes(request.method))return browse(request,env.BUCKET,ctx);
    if(url.pathname==='/private'||url.pathname.startsWith('/private/'))return json({detail:'Not found'},404);
    if(!['GET','HEAD'].includes(request.method))return new Response(null,{status:405,headers:headers({Allow:'GET, HEAD','Cache-Control':'no-store'})});
    if(url.pathname==='/backend/v1/transparency/ai-examples'){
      // Fixed sample of AI-text training rows for the Data page (built offline; contains paper text, so it lives in R2, not git).
      const object=await env.BUCKET.get('indexes/transparency/ai-examples.json');
      if(!object)return json({detail:'Not found'},404);
      return new Response(request.method==='HEAD'?null:object.body,{headers:headers({'Content-Type':'application/json','Cache-Control':'public, max-age=300','ETag':object.httpEtag})});
    }
    if(url.pathname==='/backend/v1/discovery'){
      const object=await env.BUCKET.get('indexes/iclr2027-discovery.json');
      if(!object)return json({papers:[]});
      return new Response(request.method==='HEAD'?null:object.body,{headers:headers({'Content-Type':'application/json','Cache-Control':'public, max-age=60','ETag':object.httpEtag})});
    }
    const list=url.pathname==='/backend/v1/pdf-reader';
    const match=url.pathname.match(/^\/backend\/v1\/pdf-reader\/([a-f0-9]{24})(\/file|\/metadata)?$/);
    if(!list&&!match){
      if(url.pathname.startsWith('/backend/')||url.pathname==='/workbench')return json({detail:'Not found'},404);
      // One public address for search engines: www and workers.dev pages move permanently to opentextdetector.com.
      if(url.hostname==='www.opentextdetector.com'||url.hostname.endsWith('.workers.dev'))return Response.redirect(ORIGIN+url.pathname+url.search,301);
      if(url.pathname==='/robots.txt')return new Response(robotsTxt(),{headers:headers({'Content-Type':'text/plain; charset=utf-8','Cache-Control':'public, max-age=3600'})});
      if(url.pathname==='/sitemap.xml')return new Response(sitemapXml(),{headers:headers({'Content-Type':'application/xml; charset=utf-8','Cache-Control':'public, max-age=3600'})});
      const asset=await env.ASSETS.fetch(request);
      if(!asset.ok||!(asset.headers.get('Content-Type')||'').includes('text/html'))return asset;
      const h=new Headers(asset.headers);h.delete('Content-Length');h.delete('ETag');h.set('Cache-Control','public, max-age=0, must-revalidate');
      return new Response(withMeta(await asset.text(),url),{status:asset.status,headers:h});
    }
    if(list){
      // The whole public catalogue, prepared offline; browsers revalidate it every minute with its ETag.
      const p=await current(env);
      if(!p)return json({detail:'The collection is temporarily unavailable.'},503);
      return gzipObject(request,env,ctx,p.catalogue_key,p.revision,'public, max-age=60');
    }
    const id=(legacyAliases as Record<string,string>)[match![1]]||match![1];
    const paper=await readerPaper(env,id);
    if(!paper)return json({detail:'Paper not found'},404);
    if(match![2]==='/metadata'){
      const hash=paper.pdf_key.slice(7,-4);
      for(const prefix of ['indexes/paper-details','indexes/iclr2027-details']){
        const data=await detailShard(env,`${prefix}/${hash.slice(0,2)}.json`);
        if(data[hash.slice(0,24)])return json(data[hash.slice(0,24)]);
      }
      return json({});
    }
    if(match![2]==='/file')return pdf(request,env,ctx,id,paper.pdf_key);
    if(!paper.detail_key)return json({version:paper.pdf_key.slice(7,-4),reports:[]});
    // A republication can point this ID at new results, so browsers keep the mapping for one minute only.
    return gzipObject(request,env,ctx,paper.detail_key,paper.detail_key.slice(21,-8),'public, max-age=60');
  }
  return {async fetch(request:Request,env:Env,ctx?:ExecutionContext):Promise<Response>{
    try {const response=await serve(request,env,ctx);return request.method==='HEAD'?new Response(null,response):response;}
    catch(error) {console.error("Catalogue failure", error instanceof Error?error.message:"Storage failure");const response=json({detail:'The collection is temporarily unavailable. Please retry.'},503);return request.method==='HEAD'?new Response(null,response):response;}
  }};
}
export default createAtlasWorker() satisfies ExportedHandler<Env>;
