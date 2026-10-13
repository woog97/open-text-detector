import {test} from 'node:test';
import assert from 'node:assert/strict';
import {loadCatalogue} from '../lib/atlas-api.ts';
test('304 refresh preserves parsed catalogue; a failed refresh does not replace it',async t=>{
 const calls=[];let step=0;
 t.mock.method(globalThis,'fetch',async(url,options)=>{
  calls.push({url,options});step++;
  if(step===1)return Response.json({items:[]},{headers:{etag:'"version1"'}});
  if(step===3)return new Response('server failure',{status:503});
  return new Response(null,{status:304});
 });
 const signal=new AbortController().signal;
 const a=await loadCatalogue(signal),b=await loadCatalogue(signal);assert.equal(a,b);assert.equal(calls[1].options.headers['If-None-Match'],'"version1"');assert.equal(calls[0].options.signal,signal);
 await assert.rejects(loadCatalogue(signal),/unavailable/);assert.equal(await loadCatalogue(signal),a);
});
