import {createAtlasWorker as productionWorker} from '../atlas-cloud/worker.ts';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash,createHmac} from 'node:crypto';
import {createLocalWorker as createAtlasWorker} from '../atlas-local/worker.ts';
const origin='http://127.0.0.1:3010',id='a'.repeat(24),password='test-only-password',sessionKey='b'.repeat(64);
function fixture(){
 const reads=[];
 const env={PRIVATE_AUTH:JSON.stringify({username:'alice',passwordHash:createHash('sha256').update(password).digest('hex'),sessionKey}),PRIVATE_BUCKET:{
  get:async(key,options)=>{reads.push([key,options]);return key==='catalogue.json'?{json:async()=>({items:[{id,title:'Private paper',models:['v5','v8']}]})}:{body:'private bytes'};},
  head:async()=>({size:100})
 },ASSETS:{fetch:async()=>new Response('shell')},BUCKET:{list:async()=>({objects:[],truncated:false}),get:async(key)=>key.startsWith('indexes/')||key.startsWith('atlas-public/public/')?null:({body:'catalogue',httpEtag:'"public"',json:async()=>({items:[],models:[]})})}};
 const worker=createAtlasWorker();
 const fetch=(path,init={})=>worker.fetch(new Request(origin+path,init),env);
 const login=(body={username:'alice',password},headers={})=>fetch('/backend/private/login',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json',...headers},body:JSON.stringify(body)});
 return {env,reads,fetch,login};
}
test('anonymous and forged sessions cannot read any private metadata, text or PDF',async()=>{
 const f=fixture();for(const route of ['/session','/v1/pdf-reader','/v1/pdf-reader/'+id,'/v1/pdf-reader/'+id+'/file']){
  for(const cookie of ['', 'otd_private_local=fake']){const r=await f.fetch('/backend/private'+route,{headers:{Cookie:cookie,Range:'bytes=0-3'}});assert.equal(r.status,401);assert.match(r.headers.get('cache-control'),/no-store/);assert.doesNotMatch(await r.text(),/Private paper|private bytes/);}
 }assert.equal(f.reads.length,0);
});
test('login validates credentials and origin; issues secure session; authenticated reads work',async()=>{
 const f=fixture();assert.equal((await f.login({username:'alice',password:'wrong'})).status,401);
 assert.equal((await f.login(undefined,{Origin:'https://evil.test'})).status,403);
 assert.equal((await f.login({username:'alice',password:'x'.repeat(5000)})).status,413);
 const r=await f.login();assert.equal(r.status,200);const cookie=r.headers.get('set-cookie');for(const flag of ['HttpOnly','SameSite=Strict','Path=/'])assert.ok(cookie.includes(flag));
 const headers={Cookie:cookie.split(';')[0]};
 const list=await f.fetch('/backend/private/v1/pdf-reader',{headers});assert.equal((await list.json()).items[0].id,id);assert.match(list.headers.get('cache-control'),/no-store/);
 const detail=await f.fetch('/backend/private/v1/pdf-reader/'+id,{headers});assert.equal(detail.status,200);assert.equal(detail.headers.get('content-encoding'),'gzip');
 const pdf=await f.fetch('/backend/private/v1/pdf-reader/'+id+'/file',{headers:{...headers,Range:'bytes=10-19'}});assert.equal(pdf.status,206);assert.equal(pdf.headers.get('content-range'),'bytes 10-19/100');assert.match(pdf.headers.get('cache-control'),/no-store/);
 const unknown=await f.fetch('/backend/private/v1/pdf-reader/'+'c'.repeat(24),{headers});assert.equal(unknown.status,404);
 const logout=await f.fetch('/backend/private/logout',{method:'POST',headers:{...headers,Origin:origin}});assert.ok(logout.headers.get('set-cookie').includes('Max-Age=0'));
});
test('expired, tampered, far-future and old-key sessions fail closed',async()=>{
 const f=fixture();const now=Math.floor(Date.now()/1000);
 for(const [expiration,key] of [[now-1,sessionKey],[now+86401,sessionKey],[now+60,'old key']]){
  const payload=expiration+'.'+'c'.repeat(32),signature=createHmac('sha256',key).update(payload).digest('hex');
  assert.equal((await f.fetch('/backend/private/session',{headers:{Cookie:'otd_private_local='+payload+'.'+signature}})).status,401);
 }
 delete f.env.PRIVATE_AUTH;assert.equal((await f.fetch('/backend/private/v1/pdf-reader')).status,503);assert.equal(f.reads.length,0);
});
test('private IDs never resolve through public endpoints',async()=>{
 const f=fixture();assert.equal((await productionWorker().fetch(new Request(origin+'/backend/v1/pdf-reader/'+id),f.env)).status,404);assert.equal(f.reads.length,0);
});
