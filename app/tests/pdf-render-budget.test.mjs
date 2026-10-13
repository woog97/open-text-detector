import {test} from 'node:test';
import assert from 'node:assert/strict';
import {rasterSize,createRenderQueue} from '../lib/pdf-render-budget.ts';
test('raster budget caps pixels and dimensions even at high zoom and extreme aspect ratios',()=>{
 for(const [w,h] of [[900,1200],[2200,3300],[500,100000],[100000,500]]){
  const size=rasterSize(w,h,3);
  assert.ok(size.width<=4096&&size.height<=4096);
  assert.ok(size.width*size.height<=4_000_000);
 }
 assert.throws(()=>rasterSize(Infinity,100,2));
 assert.throws(()=>rasterSize(0,100,2));
});
test('render queue limits work, cancels stale queued requests, and releases only once',async()=>{
 const queue=createRenderQueue(2),a=new AbortController(),b=new AbortController(),c=new AbortController(),d=new AbortController();
 const releaseA=await queue(a.signal),releaseB=await queue(b.signal);
 let started=false;
 const cancelled=queue(c.signal);const rejected=assert.rejects(cancelled,/cancelled/);
 const next=queue(d.signal).then(release=>{started=true;return release;});
 await Promise.resolve();assert.equal(started,false);
 c.abort();await rejected;
 releaseA();const releaseD=await next;assert.equal(started,true);
 releaseA();releaseB();releaseD();
 const aborted=new AbortController();aborted.abort();await assert.rejects(queue(aborted.signal),/cancelled/);
 const finalRelease=await queue(new AbortController().signal);finalRelease();
});

test('cancelled hung operations free slots and late rejections remain handled',async()=>{
 const {abortable}=await import('../lib/pdf-render-budget.ts');
 const queue=createRenderQueue(1),controller=new AbortController();
 const release=await queue(controller.signal);
 let fail;
 const hung=new Promise((resolve,reject)=>{fail=reject;});
 const job=abortable(hung,controller.signal).finally(release);
 const rejected=assert.rejects(job,/interrupted/);
 const next=queue(new AbortController().signal);
 controller.abort();await rejected;
 (await next)();fail(new Error('late network error'));await Promise.resolve();
});
test('1,000 scheduled/cancelled render jobs never exceed the concurrency budget',async()=>{
 const queue=createRenderQueue(2);let active=0,peak=0,completed=0;
 const jobs=Array.from({length:1000},(_,i)=>{
  const controller=new AbortController();
  const job=queue(controller.signal).then(async release=>{active++;peak=Math.max(peak,active);await Promise.resolve();active--;completed++;release();},()=>{});
  if(i%3===0)controller.abort();return job;
 });
 await Promise.all(jobs);assert.equal(active,0);assert.ok(peak<=2);assert.ok(completed>600);
});
