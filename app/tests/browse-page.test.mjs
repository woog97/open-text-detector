import {test} from 'node:test';import assert from 'node:assert/strict';
import {browsePage} from '../lib/browse-page.ts';import {browseParams} from '../lib/browse-api.ts';import {createBrowseHandler} from '../atlas-cloud/browse-worker.ts';
const papers=Array.from({length:47},(_,i)=>({id:i.toString(16).padStart(24,'0'),title:`Paper ${String(i).padStart(2,'0')}`,filename:`${i}.pdf`,collection:i===46?'iclr/2026':'iclr/2027',bytes:100,classified:false,keywords:i%2?['Robotics']:['Control'],primary_area:i%2?'Robotics':'Theory',updated:i}));
test('server pages preserve counts, filters, facets and navigation without transferring all records',()=>{
 let p=browsePage(papers,new URLSearchParams());assert.equal(p.items.length,20);assert.equal(p.total,46);assert.equal(p.pages,3);assert.equal(p.scopeTotal,46);
 p=browsePage(papers,new URLSearchParams({page:'2',paper:papers[43].id}));assert.equal(p.items.length,6);assert.equal(p.previous,papers[42].id);assert.equal(p.next,papers[44].id);assert.equal(p.position,43);
 p=browsePage(papers,new URLSearchParams({q:'paper',tag:'robotics',area:'Robotics',page:'900'}));assert.equal(p.total,23);assert.equal(p.page,1);assert.equal(p.items.length,3);
 p=browsePage(papers,new URLSearchParams({saved:'1',ids:papers[30].id}));assert.deepEqual(p.items.map(p=>p.id),[papers[30].id]);
 assert.equal(browsePage(papers,new URLSearchParams({collection:''})).total,47);
 assert.equal(browsePage(papers,new URLSearchParams({page:'Infinity'})).page,0);
});
function fixture(){
 const calls=[];let revision='a'.repeat(64);
 const bucket={get:async key=>{calls.push(key);if(key==='indexes/browse/current.json')return {json:async()=>({revision,index_key:`indexes/browse/${revision}/papers.json`,home_key:`indexes/browse/${revision}/home.json`})};if(key.endsWith('/home.json'))return {body:JSON.stringify({...browsePage(papers,new URLSearchParams()),revision})};if(key.endsWith('/papers.json'))return {json:async()=>({items:papers})};throw new Error('Unexpected bucket access');},list:()=>{throw new Error('Page load must never scan bucket');}};
 const handler=createBrowseHandler();return {calls,bucket,revision,next:()=>{revision='b'.repeat(64)},fetch:(q='',ctx)=>handler(new Request('https://example.test/backend/v1/browse'+q),bucket,ctx)};
}
test('cold homepage needs two small reads and never loads the full search index',async()=>{
 const f=fixture();const r=await f.fetch('?collection=iclr%2F2027&sort=flagged-asc&page=0');assert.equal(r.status,200);assert.equal((await r.json()).items.length,20);assert.equal(f.calls.length,2);assert.ok(f.calls.every(k=>!k.endsWith('/papers.json')));
});
test('concurrent searches share index loading; saved filters are private and never cached publicly',async t=>{
 const f=fixture(),puts=[],store=new Map(),pending=[];
 Object.defineProperty(globalThis,'caches',{configurable:true,writable:true,value:undefined});t.after(()=>delete globalThis.caches);
 t.mock.property(globalThis,'caches',{default:{match:async req=>store.get(req.url)?.clone(),put:async(req,res)=>{puts.push(req.url);store.set(req.url,res);}}});
 const ctx={waitUntil:p=>pending.push(p)};
 await Promise.all([f.fetch('?q=paper',ctx),f.fetch('?tag=Control',ctx)]);await Promise.all(pending);
 assert.equal(f.calls.filter(k=>k.endsWith('/papers.json')).length,1);assert.equal(puts.length,2);
 const before=f.calls.length;const hit=await f.fetch('?q=paper',ctx);assert.equal(f.calls.length,before);assert.equal(hit.headers.get('Cache-Control'),'public, max-age=15');
 const saved=await f.fetch('?saved=1&ids='+papers[0].id,ctx);assert.equal(saved.headers.get('Cache-Control'),'private, no-store');assert.equal(puts.length,2);
});
test('new publication uses a different edge cache key and cannot reuse old homepage',async t=>{
 let now=100;t.mock.method(Date,'now',()=>now);const f=fixture();const before=await f.fetch();f.next();now+=16000;const after=await f.fetch();assert.notEqual(before.headers.get('X-Browse-Revision'),after.headers.get('X-Browse-Revision'));
});
test('browser request contains only active saved IDs when viewing the reading list',()=>{
 const filters={q:'',collection:'iclr/2027',area:'',tag:'',sort:'title',saved:false};assert.equal(browseParams(filters,0,'',new Set(['a'])).has('ids'),false);assert.equal(browseParams({...filters,saved:true},0,'',new Set([papers[0].id])).get('ids'),papers[0].id);
});
test('last successful preview is delivered before refresh and survives a failed refresh',async t=>{
 const {loadBrowse}=await import('../lib/browse-api.ts');const values=new Map(),params='collection=iclr%2F2027&sort=title&page=0';
 Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:k=>values.get(k)||null,setItem:(k,v)=>values.set(k,v)}});t.after(()=>delete globalThis.localStorage);
 let step=0;t.mock.method(globalThis,'fetch',async()=>{if(++step>=2)return new Response('Unavailable',{status:503});return Response.json(browsePage(papers,new URLSearchParams()));});
 const data=await loadBrowse(params,new AbortController().signal);let preview;
 await assert.rejects(loadBrowse(params,new AbortController().signal,p=>{preview=p;}));assert.deepEqual(preview,data);assert.equal(values.size,1);assert.equal(step,4,'three attempts after the first success');
});
test('flagged-share filter, sort and band counts use the default mode counts and skip too-short papers',()=>{
 // shares: 0 -> 5%, 1 -> 80%, 2 -> 30%, 3 -> too short (10 sentences), 4 -> unscored
 const rows=papers.slice(0,5).map((p,i)=>({...p,...[{flags:{'nobib0.1':[5,100]}},{flags:{'nobib0.1':[80,100]}},{flags:{'nobib0.1':[30,100]}},{flags:{'nobib0.1':[9,10]}},{}][i]}));
 let p=browsePage(rows,new URLSearchParams({sort:'flagged'}));assert.deepEqual(p.items.map(x=>x.id),[1,2,0,3,4].map(i=>rows[i].id));
 p=browsePage(rows,new URLSearchParams({sort:'flagged-asc'}));assert.deepEqual(p.items.map(x=>x.id),[0,2,1,3,4].map(i=>rows[i].id));
 p=browsePage(rows,new URLSearchParams({ai:'75-90'}));assert.deepEqual(p.items.map(x=>x.id),[rows[1].id]);
 assert.deepEqual(Object.fromEntries(p.flagBands),{'0-10':1,'10-25':0,'25-50':1,'50-75':0,'75-90':1,'90-100':0,none:2});
 assert.equal(browsePage(rows,new URLSearchParams({ai:'none'})).total,2);
 assert.equal(browsePage(rows,new URLSearchParams({ai:'bogus'})).total,5);
});
test('histogram counts 5% bins over the query, ignoring the bin filter; any 5%-step range filters',()=>{
 const rows=papers.slice(0,6).map((p,i)=>({...p,...[{flags:{'nobib0.1':[5,100]}},{flags:{'nobib0.1':[80,100]}},{flags:{'nobib0.1':[30,100]}},{flags:{'nobib0.1':[100,100]}},{flags:{'nobib0.1':[9,10]}},{}][i]}));
 let p=browsePage(rows,new URLSearchParams({ai:'30-35'}));
 assert.deepEqual(p.items.map(x=>x.id),[rows[2].id]);
 const h=Array(20).fill(0);h[1]=1;h[16]=1;h[6]=1;h[19]=1;assert.deepEqual(p.histogram,h);
 assert.equal(browsePage(rows,new URLSearchParams({ai:'95-100'})).items[0].id,rows[3].id);
 assert.equal(browsePage(rows,new URLSearchParams({ai:'0-50'})).total,2);
 assert.equal(browsePage(rows,new URLSearchParams({ai:'3-7'})).total,6);
 assert.deepEqual(browsePage(rows,new URLSearchParams({q:'paper 01'})).histogram.reduce((a,b)=>a+b),1);
});
test('compact index: pages read only their cards, by grouped byte ranges, and match the full-list result',async()=>{
 const {buildIndex}=await import('../lib/browse-index.ts');
 const rows=papers.map((p,i)=>({...p,flags:{body:[i%10,100]},abstract_preview:'x'.repeat(5000)}));
 const {index,pack}=buildIndex(rows),rev='c'.repeat(64),gets=[];
 const bucket={get:async(key,opts)=>{gets.push([key,opts?.range]);
  if(key==='indexes/browse/current.json')return {json:async()=>({revision:rev,index_key:`indexes/browse/${rev}/papers.json`,home_key:`indexes/browse/${rev}/home.json`,search_key:`indexes/browse/${rev}/search.json`,cards_key:`indexes/browse/${rev}/cards.jsonl`})};
  if(key.endsWith('/search.json'))return {json:async()=>JSON.parse(JSON.stringify(index))};
  if(key.endsWith('/cards.jsonl')){const {offset,length}=opts.range;return {arrayBuffer:async()=>pack.slice(offset,offset+length).buffer};}
  throw new Error('Unexpected bucket access '+key);}};
 const handler=createBrowseHandler();
 for(const q of ['?q=paper&sort=title','?sort=flagged&ai=5-10','?sort=number&paper='+papers[7].id]){
  gets.length=0;const r=await handler(new Request('https://example.test/backend/v1/browse'+q),bucket);assert.equal(r.status,200);
  const got=await r.json(),want=browsePage(rows,new URL('https://x'+q).searchParams);delete got.revision;
  assert.deepEqual(got,JSON.parse(JSON.stringify(want)));
  assert.ok(gets.every(([k])=>!k.endsWith('/papers.json')));
 }
 gets.length=0;await handler(new Request('https://example.test/backend/v1/browse?q=paper&page=1'),bucket);
 assert.equal(gets.filter(([k])=>k.endsWith('cards.jsonl')).length,1,'a title-ordered page is one range read');
});
test('counting modes: each mode filters, sorts and bins on its own counts; missing modes count as unscored',()=>{
 const rows=papers.slice(0,3).map((p,i)=>({...p,flags:[{nobib:[90,100],all:[60,100],body:[95,100]},{nobib:[10,100],all:[70,100]},{all:[50,100]}][i]}));
 assert.deepEqual(browsePage(rows,new URLSearchParams({sort:'flagged'})).items.map(x=>x.id),[0,1,2].map(i=>rows[i].id));
 assert.deepEqual(browsePage(rows,new URLSearchParams({sort:'flagged',mode:'all'})).items.map(x=>x.id),[1,0,2].map(i=>rows[i].id));
 const body=browsePage(rows,new URLSearchParams({mode:'body'}));assert.equal(body.histogram[19],1);assert.deepEqual(Object.fromEntries(body.flagBands).none,2);
 assert.equal(browsePage(rows,new URLSearchParams({mode:'all',ai:'60-65'})).items[0].id,rows[0].id);
 assert.equal(browsePage(rows,new URLSearchParams({mode:'bogus',sort:'flagged'})).items[0].id,rows[0].id,'unknown modes fall back to the default');
});
test('0.1% FPR modes: own counts, own sort; snapshots without them load with those modes unscored',async()=>{
 const rows=papers.slice(0,2).map((p,i)=>({...p,flags:[{body:[5,100],'body0.1':[20,100]},{body:[8,100],'body0.1':[12,100]}][i]}));
 assert.deepEqual(browsePage(rows,new URLSearchParams({sort:'flagged',mode:'body'})).items.map(x=>x.id),[1,0].map(i=>rows[i].id));
 assert.deepEqual(browsePage(rows,new URLSearchParams({sort:'flagged',mode:'body0.1'})).items.map(x=>x.id),[0,1].map(i=>rows[i].id));
 const {buildIndex,BrowseIndex}=await import('../lib/browse-index.ts');
 const {index}=buildIndex(rows);for(const m of ['all0.1','nobib0.1','noapp0.1','body0.1'])delete index.flags[m];
 const loaded=new BrowseIndex(JSON.parse(JSON.stringify(index)));
 assert.equal(loaded.x.flags['body0.1'].length,0,'columns are released after loading');
 const {flagModeOf,asFlagMode,FLAG_MODES,DEFAULT_FLAG_MODE}=await import('../lib/paper-discovery.ts');
 assert.equal(flagModeOf(false,false,0.001),'body0.1');assert.equal(flagModeOf(true,true),'all');assert.equal(asFlagMode('nobib10'),DEFAULT_FLAG_MODE,'removed 10% modes fall back to the default');
 assert.equal(DEFAULT_FLAG_MODE,'nobib0.1');
 for(const m of Object.keys(FLAG_MODES))assert.ok(FLAG_MODES[m].threshold>0&&FLAG_MODES[m].threshold<=FLAG_MODES[m].strong,m);
});
test('a cancelled index load does not hang later searches: waiters poll, then take over a stale load; concurrent searches share one load',async()=>{
 const {buildIndex}=await import('../lib/browse-index.ts');
 const {index,pack}=buildIndex(papers),rev='d'.repeat(64);let searchGets=0,hang=true;
 const bucket={get:async(key,opts)=>{
  if(key==='indexes/browse/current.json')return {json:async()=>({revision:rev,index_key:`indexes/browse/${rev}/papers.json`,home_key:`indexes/browse/${rev}/home.json`,search_key:`indexes/browse/${rev}/search.json`,cards_key:`indexes/browse/${rev}/cards.jsonl`})};
  if(key.endsWith('/search.json')){searchGets++;if(hang){hang=false;return new Promise(()=>{});} // the first load's request was cancelled: it never settles
   await new Promise(r=>setTimeout(r,20));return {json:async()=>JSON.parse(JSON.stringify(index))};}
  if(key.endsWith('/cards.jsonl')){const {offset,length}=opts.range;return {arrayBuffer:async()=>pack.slice(offset,offset+length).buffer};}
  throw new Error('Unexpected bucket access '+key);}};
 const handler=createBrowseHandler({loadStale:150,waitMax:2000}),get=q=>handler(new Request('https://example.test/backend/v1/browse'+q),bucket);
 void get('?q=paper'); // stuck forever
 const started=Date.now(),rs=await Promise.all(['?q=paper','?q=x','?sort=flagged'].map(get));
 assert.deepEqual(rs.map(r=>r.status),[200,200,200]);
 assert.ok(Date.now()-started<1000,'waiters take over after loadStale, not after waitMax');
 assert.equal(searchGets,2,'one stuck load, then one shared replacement');
 const stuck=createBrowseHandler({loadStale:10000,waitMax:100});hang=true;void stuck(new Request('https://example.test/backend/v1/browse?q=a'),bucket);
 assert.equal((await stuck(new Request('https://example.test/backend/v1/browse?q=b'),bucket)).status,503,'a waiter gives up with a retryable 503 rather than hanging');
});
test('cutoff filter: keeps papers below or at/above it, limits the histogram so the remaining bars rescale',()=>{
 const shares=[[2,100],[50,100],[94,100],[95,100],[100,100]];
 const rows=papers.slice(0,6).map((p,i)=>({...p,flags:i<5?{'nobib0.1':shares[i]}:undefined}));
 const below=browsePage(rows,new URLSearchParams({cut:'below95'}));
 assert.equal(below.total,3);assert.equal(below.histogram.reduce((a,b)=>a+b,0),3);assert.equal(below.histogram[19],0);assert.equal(Object.fromEntries(below.flagBands).none,0,'unscored papers have no share to compare');
 const above=browsePage(rows,new URLSearchParams({cut:'above95'}));assert.equal(above.total,2);assert.equal(above.histogram[19],2);
 assert.equal(browsePage(rows,new URLSearchParams({cut:'below95',ai:'0-5'})).total,1,'bin filter applies inside the cutoff');
 assert.equal(browsePage(rows,new URLSearchParams({cut:'below97'})).total,6,'invalid cutoffs are ignored');
});
test('average sentence score: breaks flagged-share ties, has its own sorts, and matches between the list and the compact index',async()=>{
 // body0.1 shares: 0,1,2 -> 100% (means .95/.99/.97), 3 -> 50% (mean .6), 4 -> too short, 5 -> 100% without a mean (older snapshot); papers without a usable value go last in title order
 const rows=papers.slice(0,6).map((p,i)=>({...p,...[
  {flags:{'body0.1':[100,100]},mean_score:{body:.95,nobib:.5}},{flags:{'body0.1':[100,100]},mean_score:{body:.99}},{flags:{'body0.1':[100,100]},mean_score:{body:.97}},
  {flags:{'body0.1':[50,100]},mean_score:{body:.6}},{flags:{'body0.1':[9,10]},mean_score:{body:.99}},{flags:{'body0.1':[100,100]}}][i]}));
 const ids=o=>o.map(i=>rows[i].id),{buildIndex,BrowseIndex}=await import('../lib/browse-index.ts');
 const loaded=new BrowseIndex(JSON.parse(JSON.stringify(buildIndex(rows).index)));
 const base={q:'',collection:rows[0].collection,area:'',tag:'',saved:false,ai:'',mode:'body0.1',cut:''};
 const expected={flagged:[1,2,0,5,3,4],'flagged-asc':[3,5,0,2,1,4],score:[1,2,0,3,4,5],'score-asc':[3,0,2,1,4,5]};
 for(const [sort,order] of Object.entries(expected)){
  assert.deepEqual(browsePage(rows,new URLSearchParams({sort,mode:'body0.1'})).items.map(x=>x.id),ids(order),`list ${sort}`);
  assert.deepEqual(loaded.query({...base,sort},new Set(),null,0).items.map(i=>loaded.x.ids[i]),ids(order),`index ${sort}`);
 }
});
