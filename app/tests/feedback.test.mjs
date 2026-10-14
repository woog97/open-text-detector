import {test} from 'node:test';
import assert from 'node:assert/strict';
import {handleFeedback,MAX_COMMENT,MIN_COMMENT} from '../atlas-cloud/feedback.ts';
const json=(v,status=200)=>Response.json(v,{status});
const paper='a'.repeat(24);
function setup({exists=true,allow=true}={}){
 const puts=[],limits=[];
 const deps={bucket:{put:async(k,v)=>{puts.push([k,JSON.parse(v)]);}},paperExists:async id=>exists&&id===paper,now:()=>new Date('2026-10-13T20:00:00Z'),
  perSender:{limit:async o=>{limits.push(o.key);return {success:allow};}},sitewide:{limit:async()=>({success:true})}};
 const send=(body,headers={},method='POST')=>handleFeedback(new Request('https://opentextdetector.com/backend/v1/feedback',{method,body:method==='GET'?undefined:typeof body==='string'?body:JSON.stringify(body),
  headers:{'Content-Type':'application/json',Origin:'https://opentextdetector.com','CF-Connecting-IP':'198.51.100.7',...headers}}),deps,json);
 return {send,puts,limits};
}
test('a message is stored privately with the paper and the settings shown, and nothing that identifies the sender',async()=>{
 const {send,puts,limits}=setup();
 const r=await send({paper,comment:'  The intro is mine; only the abstract was polished. '+'Every other section was written by hand over several weeks.  ',context:{report:'qwen36-35b-a3b-oct10-x',mode:'body0.1',shown:'87%',extra:'dropped'}});
 assert.equal(r.status,201);assert.equal(puts.length,1);
 const [key,stored]=puts[0];
 assert.match(key,/^feedback\/2026-10-13\/\d+-[0-9a-f-]{36}\.json$/);
 assert.deepEqual(stored,{received_at:'2026-10-13T20:00:00.000Z',paper,comment:'The intro is mine; only the abstract was polished. Every other section was written by hand over several weeks.',context:{report:'qwen36-35b-a3b-oct10-x',mode:'body0.1',shown:'87%'}});
 assert.doesNotMatch(JSON.stringify(stored),/198\.51/);assert.deepEqual(limits,['198.51.100.7']);
});
test('rejects other sites, other methods, bad input, unknown papers and senders over the limit; the hidden field stores nothing',async()=>{
 const {send,puts}=setup();
 assert.equal((await send({paper,comment:'x'},{Origin:'https://evil.example'})).status,403);
 assert.equal((await send({paper,comment:'x'},{},'GET')).status,405);
 assert.equal((await send({paper,comment:'x'},{'Content-Type':'text/plain'})).status,415);
 assert.equal((await send('{nope')).status,400);
 assert.equal((await send({paper,comment:'   '})).status,400);
 assert.equal((await send({paper,comment:'x'.repeat(MAX_COMMENT+1)})).status,400);
 assert.equal((await send({paper,comment:'x'.repeat(9000)})).status,413);
 assert.equal((await send({paper:'../secret',comment:'x'.repeat(MIN_COMMENT)})).status,400);
 assert.equal((await send({paper:'b'.repeat(24),comment:'x'.repeat(MIN_COMMENT)})).status,400);
 assert.equal((await send({paper,comment:'buy now',website:'http://spam.example'})).status,201);
 assert.equal((await setup({allow:false}).send({paper,comment:'x'.repeat(MIN_COMMENT)})).status,429);
 assert.equal(puts.length,0);
});
test('marked passages are stored as sorted unique indices; messages need 100 characters even with marks; bad marks are rejected',async()=>{
 const {send,puts}=setup();
 const long='x'.repeat(MIN_COMMENT);
 assert.equal((await send({paper,comment:'',passages:[12,3,3,40]})).status,400,'marks alone are not enough');
 assert.equal((await send({paper,comment:'x'.repeat(MIN_COMMENT-1)})).status,400,'99 characters is too short');
 assert.equal((await send({paper,comment:long,passages:[12,3,3,40],context:{report:'r'}})).status,201);
 assert.deepEqual(puts[0][1].passages,[3,12,40]);
 assert.equal((await send({paper,comment:long})).status,201);assert.equal('passages' in puts[1][1],false);
 for(const passages of [[-1],[1.5],['3'],'3',Array.from({length:301},(_,i)=>i)])assert.equal((await send({paper,comment:'x'.repeat(MIN_COMMENT),passages})).status,400,JSON.stringify(passages).slice(0,20));
 assert.equal((await send({paper,comment:'',passages:[]})).status,400);
 assert.equal(puts.length,2);
});
