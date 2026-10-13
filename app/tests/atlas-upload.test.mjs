import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../atlas-upload/worker.ts';

test('publication uses conditional writes and returns the catalogue validator',async()=>{
 const token='test-only';
 const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token));
 const env={TOKEN_SHA256:Buffer.from(hash).toString('hex'),EXPIRES_AT:String(Date.now()/1000+60),BUCKET:{
  async put(key,body,options){assert.equal(key,'atlas-public/catalogue.json');assert.equal(options.onlyIf.get('if-match'),'"stale"');return null;},
  async get(){return {body:'{}',httpEtag:'"current"'};}
 }};
 const url='https://test/atlas-public/catalogue.json';
 const response=await worker.fetch(new Request(url,{method:'PUT',headers:{Authorization:`Bearer ${token}`,'If-Match':'"stale"'},body:'{}'}),env);
 assert.equal(response.status,412);
 const get=await worker.fetch(new Request(url,{headers:{Authorization:`Bearer ${token}`}}),env);
 assert.equal(get.headers.get('etag'),'"current"');
 assert.equal((await worker.fetch(new Request(url),env)).status,401);
 assert.equal((await worker.fetch(new Request('https://test/private/file',{headers:{Authorization:`Bearer ${token}`}}),env)).status,404);
});
