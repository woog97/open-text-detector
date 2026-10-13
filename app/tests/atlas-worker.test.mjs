import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createAtlasWorker,byteRange,PUBLICATION_KEY} from '../atlas-cloud/worker.ts';
import {buildPublicCatalogue} from '../lib/atlas-archive.ts';
const id='a'.repeat(24),pdfHash='b'.repeat(64),detailHash='c'.repeat(64),pdfKey=`papers/${pdfHash}.pdf`,detailKey=`atlas-public/objects/${detailHash}.json.gz`;
const revision='1'.repeat(64),base=`atlas-public/public/${revision}`;
function fixture({manifest={revision,catalogue_key:`${base}/catalogue.json.gz`,shards_prefix:`${base}/reader/`},shards={aa:{[id]:[pdfHash,detailHash]}}}={}){
 const calls=[];
 const env={ASSETS:{fetch:async()=>new Response('asset')},BUCKET:{
   list:async options=>{calls.push(['list',options.prefix]);return {objects:[],truncated:false};},
   head:async key=>{calls.push(['head',key]);return {size:100,httpEtag:'"obj"'};},
   get:async(key,options)=>{
    calls.push(['get',key,options]);
    if(key===PUBLICATION_KEY)return manifest===null?null:{json:async()=>manifest};
    if(manifest&&key.startsWith(manifest.shards_prefix)){const s=shards[key.slice(-7,-5)];return s?{json:async()=>s}:null;}
    if(key.startsWith('indexes/'))return null;
    return {body:new Response('bytes').body,size:100};
   }
 }};
 const worker=createAtlasWorker();
 const fetch=(path='',init={},ctx)=>worker.fetch(new Request('https://example.test/backend/v1/pdf-reader'+path,init),env,ctx);
 return {fetch,calls,env,worker};
}
const reads=(calls,key)=>calls.filter(c=>c[0]==='get'&&(!key||c[1]===key)).length;
test('publication allowlist rejects unknown IDs, object traversal and write methods',async()=>{
 const {fetch,calls}=fixture();
 for(const path of ['/'+ 'd'.repeat(24),'/__proto__','/%2e%2e/secret','/'+id+'/file/extra'])assert.equal((await fetch(path)).status,404);
 for(const method of ['POST','PUT','DELETE','PATCH','OPTIONS'])assert.equal((await fetch('/'+id,{method})).status,405);
 assert.ok(!calls.some(c=>c[0]==='get'&&!c[1].startsWith('atlas-public/public/')),'only the publication and its shards are read');
});
test('missing or unsafe publications fail closed and do not expose keys',async()=>{
 assert.equal((await fixture({manifest:null}).fetch()).status,503);
 for(const manifest of [{revision,catalogue_key:'private/secret',shards_prefix:`${base}/reader/`},{revision:'x',catalogue_key:'atlas-public/public/x/catalogue.json.gz',shards_prefix:'atlas-public/public/x/reader/'}]){
  const r=await fixture({manifest}).fetch();assert.equal(r.status,503);assert.doesNotMatch(await r.text(),/secret|stack/);
 }
 const r=await fixture({shards:{aa:{[id]:['../secret','']}}}).fetch('/'+id);assert.equal(r.status,503);
});
test('the catalogue is the prepared object: one manifest read for concurrent requests, ETag is the revision',async()=>{
 const {fetch,calls}=fixture();
 const responses=await Promise.all(Array.from({length:50},()=>fetch()));
 assert.equal(reads(calls,PUBLICATION_KEY),1);assert.ok(!calls.some(c=>c[0]==='list'),'never lists the archive');
 const r=responses[0];assert.equal(r.status,200);assert.equal(r.headers.get('etag'),`"${revision}"`);assert.equal(r.headers.get('content-encoding'),'gzip');
 assert.equal(reads(calls,`${base}/catalogue.json.gz`),50);
 const before=calls.length,cached=await fetch('',{headers:{'If-None-Match':`"other", W/"${revision}"`}});
 assert.equal(cached.status,304);assert.equal(await cached.text(),'');assert.equal(calls.length,before,'a revalidation reads nothing');
 const head=await fetch('',{method:'HEAD'});assert.equal(await head.text(),'');assert.equal(head.headers.get('etag'),`"${revision}"`);
});
test('a new publication replaces the old one within 30 seconds',async t=>{
 let now=1000;t.mock.method(Date,'now',()=>now);
 const {fetch,env}=fixture();assert.equal((await fetch()).headers.get('etag'),`"${revision}"`);
 const next='2'.repeat(64),get=env.BUCKET.get;
 env.BUCKET.get=async(k,o)=>k===PUBLICATION_KEY?{json:async()=>({revision:next,catalogue_key:`atlas-public/public/${next}/catalogue.json.gz`,shards_prefix:`atlas-public/public/${next}/reader/`})}:get(k,o);
 now+=10000;assert.equal((await fetch()).headers.get('etag'),`"${revision}"`);
 now+=21000;assert.equal((await fetch()).headers.get('etag'),`"${next}"`);
});
test('PDF ranges, If-Range, validators, HEAD and response security are consistent',async()=>{
 const {fetch,calls}=fixture(),etag=`"${pdfHash}"`;
 let r=await fetch('/'+id+'/file',{headers:{Range:'bytes=10-19'}});assert.equal(r.status,206);assert.equal(r.headers.get('content-range'),'bytes 10-19/100');assert.deepEqual(calls.at(-1)[2],{range:{offset:10,length:10}});
 r=await fetch('/'+id+'/file',{headers:{Range:'bytes=10-19','If-Range':'"stale"'}});assert.equal(r.status,200);assert.equal(r.headers.get('content-length'),'100');
 r=await fetch('/'+id+'/file',{headers:{Range:'bytes=10-19','If-Range':etag}});assert.equal(r.status,206);
 for(const range of ['bytes=100-','bytes=-0','bytes=9-1','bytes=0-1,4-5','bytes=9007199254740993-','bytes=-9007199254740993']){
  r=await fetch('/'+id+'/file',{headers:{Range:range}});assert.equal(r.status,416);assert.equal(r.headers.get('content-range'),'bytes */100');assert.equal(r.headers.get('x-content-type-options'),'nosniff');
 }
 const before=reads(calls);
 r=await fetch('/'+id+'/file',{method:'HEAD',headers:{Range:'bytes=0-1'}});assert.equal(r.status,200);assert.equal(r.headers.get('content-length'),'100');assert.equal(await r.text(),'');
 r=await fetch('/'+id+'/file',{headers:{'If-None-Match':'W/'+etag}});assert.equal(r.status,304);
 r=await fetch('/'+id,{headers:{'If-None-Match':`W/"${detailHash}"`}});assert.equal(r.status,304);
 assert.equal(reads(calls),before);
 r=await fetch('/'+id);assert.equal(r.headers.get('content-encoding'),'gzip');assert.equal(r.headers.get('etag'),`"${detailHash}"`);assert.equal(r.headers.get('x-frame-options'),'DENY');assert.match(r.headers.get('content-security-policy'),/sandbox/);assert.equal(r.headers.get('access-control-allow-origin'),null);
});
test('storage failures return sanitized retryable errors, never raw internal exceptions',async()=>{
 const {fetch,env}=fixture();env.BUCKET.get=async()=>{throw new Error('secret internal path');};const r=await fetch();assert.equal(r.status,503);assert.doesNotMatch(await r.text(),/secret/);assert.equal(r.headers.get('cache-control'),'no-store');
});
test('10,000 generated ranges never escape object bounds',()=>{
 let seed=113;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
 for(let n=0;n<10000;n++){
  const size=rnd()%100000+1,a=rnd()%200000,b=rnd()%200000;
  const range=byteRange(n%2?`bytes=${a}-${b}`:`bytes=-${a}`,size);
  if(range){assert.ok(Number.isSafeInteger(range.offset)&&range.offset>=0);assert.ok(range.length>0);assert.ok(range.offset+range.length<=size);}
 }
});
test('PDFs uploaded after the last publication open unclassified through one prefix listing',async()=>{
 const {fetch,env,calls}=fixture(),hash='e'.repeat(64),key=`papers/${hash}.pdf`;
 env.BUCKET.list=async options=>{calls.push(['list',options.prefix]);assert.equal(options.prefix,`papers/${hash.slice(0,24)}`);return {objects:[{key,size:123}],truncated:false};};
 assert.deepEqual(await (await fetch('/'+hash.slice(0,24))).json(),{version:hash,reports:[]});
 assert.equal((await fetch('/'+hash.slice(0,24)+'/file')).status,200);
 assert.equal(calls.filter(c=>c[0]==='list').length,1,'the prefix listing is remembered across range requests');
 const none=fixture({manifest:null});none.env.BUCKET.list=env.BUCKET.list;
 assert.deepEqual(await (await none.fetch('/'+hash.slice(0,24))).json(),{version:hash,reports:[]},'still readable before any publication');
});
test('paper metadata comes from the detail shards of the paper hash',async()=>{
 const {fetch,env}=fixture();const original=env.BUCKET.get;
 env.BUCKET.get=async(k,o)=>k==='indexes/iclr2027-details/bb.json'?{json:async()=>({['b'.repeat(24)]:{abstract:'Full abstract',bibtex:'Citation'}})}:original(k,o);
 assert.equal((await (await fetch('/'+id+'/metadata')).json()).abstract,'Full abstract');
 assert.equal((await fetch('/'+'f'.repeat(24)+'/metadata')).status,404);
});
test('general metadata supports older collections and takes precedence over legacy shards',async()=>{
 const {fetch,env}=fixture();const original=env.BUCKET.get;
 env.BUCKET.get=async(k,o)=>k.startsWith('indexes/paper-details/')?{json:async()=>({['b'.repeat(24)]:{abstract:'Older collection abstract',venue:'COLM'}})}:original(k,o);
 const detail=await (await fetch('/'+id+'/metadata')).json();assert.equal(detail.abstract,'Older collection abstract');assert.equal(detail.venue,'COLM');
});
function edgeCache(t){
 const store=new Map(),matched=[];
 Object.defineProperty(globalThis,'caches',{configurable:true,writable:true,value:undefined});t.after(()=>delete globalThis.caches);
 t.mock.property(globalThis,'caches',{default:{
  match:async req=>{const url=typeof req==='string'?req:req.url,range=typeof req==='string'?null:req.headers.get('range');matched.push([url,range]);const hit=store.get(url);if(!hit)return undefined;
   return range?new Response('part',{status:206,headers:{'Content-Range':'bytes 0-3/100','Content-Length':'4'}}):hit.clone();},
  put:async(req,res)=>{store.set(typeof req==='string'?req:req.url,new Response(await res.arrayBuffer(),res));}
 }});
 const pending=[];return {store,matched,pending,ctx:{waitUntil:p=>pending.push(p)}};
}
test('catalogue and results are copied to the edge once, then served without storage reads',async t=>{
 const edge=edgeCache(t);
 const first=fixture();assert.equal((await first.fetch('',{},edge.ctx)).headers.get('x-edge-cache'),'miss');
 assert.equal((await first.fetch('/'+id,{},edge.ctx)).headers.get('x-edge-cache'),'miss');
 await Promise.all(edge.pending);assert.equal(edge.store.size,2);
 const second=fixture();
 const r=await second.fetch('',{},edge.ctx);assert.equal(r.headers.get('x-edge-cache'),'hit');assert.equal(await r.text(),'bytes');assert.equal(r.headers.get('etag'),`"${revision}"`);
 assert.equal((await second.fetch('/'+id,{},edge.ctx)).headers.get('x-edge-cache'),'hit');
 assert.deepEqual(second.calls.map(c=>c[1]),[PUBLICATION_KEY,`${base}/reader/aa.json`],'only the manifest and one shard');
});
test('PDF range requests copy the whole file to the edge once, then the edge answers ranges',async t=>{
 const edge=edgeCache(t),{fetch,calls}=fixture();
 const misses=await Promise.all([0,1,2].map(i=>fetch('/'+id+'/file',{headers:{Range:`bytes=${i}-${i+3}`}},edge.ctx)));
 assert.deepEqual(misses.map(r=>r.status),[206,206,206]);
 await Promise.all(edge.pending);assert.equal(calls.filter(c=>c[0]==='get'&&c[1]===pdfKey&&!c[2]).length,1,'one full copy per isolate');
 const before=calls.length,hit=await fetch('/'+id+'/file',{headers:{Range:'bytes=0-3'}},edge.ctx);
 assert.equal(hit.status,206);assert.equal(hit.headers.get('content-range'),'bytes 0-3/100');assert.equal(hit.headers.get('x-edge-cache'),'hit');assert.equal(hit.headers.get('x-frame-options'),'DENY');
 assert.equal(calls.length,before);assert.deepEqual(edge.matched.at(-1),[`https://example.test/backend/v1/object-cache/${pdfHash}.pdf`,'bytes=0-3']);
 assert.equal((await fetch('/'+id+'/file',{headers:{Range:'bytes=0-1,4-5'}},edge.ctx)).status,416,'multi-range never reaches the edge copy');
});
test('the public catalogue merges archive PDFs and index metadata, excludes non-PDF keys and hides storage keys',()=>{
 const hash='f'.repeat(64),key=`papers/${hash}.pdf`;
 const source={items:[{id,title:'A paper',filename:'a.pdf',collection:'iclr/2025',bytes:100,classified:true,models:['v8'],pdf_key:pdfKey,detail_key:detailKey}],models:[{id:'v8',total:1,available:1,complete:true}],published_at:123};
 const {catalogue,shards}=buildPublicCatalogue(source,[{key,size:25},{key:'private/test.pdf',size:20},{key:pdfKey,size:100}],[{key,filename:'forum.pdf',conference:'iclr',year:2027,title:'Indexed title'},{key:pdfKey,forum_id:'forum',primary_area:'Robotics',abstract:'Full abstract'}]);
 assert.equal(catalogue.items.length,2);assert.equal(catalogue.items[1].title,'Indexed title');assert.equal(catalogue.items[1].collection,'iclr/2027');assert.equal(catalogue.items[1].classified,false);
 assert.equal(catalogue.items[0].primary_area,'Robotics');assert.equal(catalogue.items[0].abstract,undefined);
 for(const p of catalogue.items){assert.equal(p.pdf_key,undefined);assert.equal(p.detail_key,undefined);}
 assert.deepEqual(catalogue.models,[{id:'v8',total:2,available:1,complete:false}]);
 assert.deepEqual(shards,{aa:{[id]:[pdfHash,detailHash]},ff:{[hash.slice(0,24)]:[hash,'']}});
 assert.throws(()=>buildPublicCatalogue({...source,items:[source.items[0],source.items[0]]},[],[]),/Invalid published object/);
 assert.throws(()=>buildPublicCatalogue({...source,items:[{...source.items[0],detail_key:'private/secret'}]},[],[]),/Invalid published object/);
});
test('pages get their own title, description and canonical address; other hosts redirect to opentextdetector.com',async()=>{
 const shell='<!doctype html><html lang="en"><head><meta charset="UTF-8"/><meta name="description" content="old"/><title>Open Text Detector</title></head><body><div id="root"></div></body></html>';
 const env={ASSETS:{fetch:async()=>new Response(shell,{headers:{'Content-Type':'text/html','ETag':'"x"'}})},BUCKET:{}};
 const worker=createAtlasWorker(),get=url=>worker.fetch(new Request(url),env);
 let html=await (await get('https://opentextdetector.com/?view=calibration')).text();
 assert.match(html,/<title>Calibration · Open Text Detector<\/title>/);
 assert.match(html,/<link rel="canonical" href="https:\/\/opentextdetector.com\/\?view=calibration"\/>/);
 assert.doesNotMatch(html,/content="old"/);
 html=await (await get('https://opentextdetector.com/')).text();
 assert.match(html,/application\/ld\+json/);assert.match(html,/canonical" href="https:\/\/opentextdetector.com\/"/);
 const paper=await get('https://opentextdetector.com/?paper=abc');html=await paper.text();
 assert.doesNotMatch(html,/canonical/);assert.match(html,/ICLR 2027 submissions/);
 assert.match(html,/<meta name="robots" content="noindex"\/>/);assert.equal(paper.headers.get('X-Robots-Tag'),'noindex');
 const list=await get('https://opentextdetector.com/?collection=iclr%2F2027');assert.doesNotMatch(await list.text(),/noindex/);assert.equal(list.headers.get('X-Robots-Tag'),null);
 assert.equal(await (await get('https://opentextdetector.com/?view=<script>')).text(),shell,'unknown views are left alone');
 for(const host of ['www.opentextdetector.com','open-text-detector.woog09.workers.dev']){
  const r=await get(`https://${host}/?view=data`);assert.equal(r.status,301);assert.equal(r.headers.get('Location'),'https://opentextdetector.com/?view=data');
 }
 const robots=await get('https://opentextdetector.com/robots.txt');assert.match(await robots.text(),/Sitemap: https:\/\/opentextdetector.com\/sitemap.xml/);
 const sitemap=await (await get('https://opentextdetector.com/sitemap.xml')).text();
 assert.match(sitemap,/<loc>https:\/\/opentextdetector.com\/\?view=limitations<\/loc>/);
});
