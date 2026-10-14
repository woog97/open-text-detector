import {test} from 'node:test';
import assert from 'node:assert/strict';
import {handleFeedback,MAX_COMMENT,MIN_COMMENT} from '../atlas-cloud/feedback.ts';
const json=(v,status=200)=>Response.json(v,{status});
const paper='a'.repeat(24);
function setup({exists=true,allow=true}={}){
 const puts=[],limits=[];
 const store=new Map();
 // puts: stored messages in order (the empty duplicate markers under feedback/_seen/ are kept in store only).
 const bucket={put:async(k,v)=>{store.set(k,v);if(!k.startsWith('feedback/_seen/'))puts.push([k,JSON.parse(v)]);},
  get:async k=>store.has(k)?{json:async()=>JSON.parse(store.get(k))}:null,head:async k=>store.has(k)?{key:k}:null};
 const deps={bucket,paperExists:async id=>exists&&id===paper,now:()=>new Date('2026-10-13T20:00:00Z'),
  perSender:{limit:async o=>{limits.push(o.key);return {success:allow};}},sitewide:{limit:async()=>({success:true})}};
 const send=(body,headers={},method='POST')=>handleFeedback(new Request('https://opentextdetector.com/backend/v1/feedback',{method,body:method==='GET'?undefined:typeof body==='string'?body:JSON.stringify(body),
  headers:{'Content-Type':'application/json',Origin:'https://opentextdetector.com','CF-Connecting-IP':'198.51.100.7',...headers}}),deps,json);
 return {send,puts,limits,store};
}
test('a message is stored privately with the paper and the settings shown, and nothing that identifies the sender',async()=>{
 const {send,puts,limits}=setup();
 const r=await send({paper,comment:'  The intro is mine; only the abstract was polished. '+'Every other section was written by hand over several weeks.  ',context:{report:'qwen36-35b-a3b-oct10-x',mode:'body0.1',shown:'87%',extra:'dropped'}});
 assert.equal(r.status,201);assert.equal(puts.length,1);
 const [key,stored]=puts[0];
 assert.match(key,/^feedback\/2026-10-13\/\d+-[0-9a-f-]{36}\.json$/);
 const {edit_key_sha256,...rest}=stored;assert.match(edit_key_sha256,/^[0-9a-f]{64}$/);
 assert.deepEqual(rest,{received_at:'2026-10-13T20:00:00.000Z',paper,comment:'The intro is mine; only the abstract was polished. Every other section was written by hand over several weeks.',context:{report:'qwen36-35b-a3b-oct10-x',mode:'body0.1',shown:'87%'}});
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
 assert.equal((await send({paper,comment:'x'.repeat(40000)})).status,413);
 assert.equal((await send({paper,comment:'中'.repeat(4000)})).status,201,'4,000 characters in a non-Latin script fit');
 assert.equal((await send({paper,comment:'x'.repeat(MAX_COMMENT),passages:Array.from({length:2000},(_,i)=>100000+i)})).status,201,'the longest message with the most marks fits');
 puts.length=0;
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
 assert.equal((await send({paper,comment:long+'y'})).status,201);assert.equal('passages' in puts[1][1],false);
 for(const passages of [[-1],[1.5],['3'],'3',Array.from({length:2001},(_,i)=>i)])assert.equal((await send({paper,comment:'x'.repeat(MIN_COMMENT),passages})).status,400,JSON.stringify(passages).slice(0,20));
 assert.equal((await send({paper,comment:'',passages:[]})).status,400);
 assert.equal(puts.length,2);
});
test('site feedback needs no paper, keeps only the page as context, ignores marks and is stored under feedback/site/',async()=>{
 const {send,puts}=setup({exists:false});
 const msg='The calibration page is hard to read on a phone; the table runs off the screen and the chart labels overlap a lot.';
 assert.equal((await send({kind:'site',comment:msg,context:{page:'calibration',report:'x'},passages:[1,2]})).status,201);
 assert.match(puts[0][0],/^feedback\/site\/2026-10-13\/\d+-[0-9a-f-]{36}\.json$/);
 assert.deepEqual(puts[0][1],{received_at:'2026-10-13T20:00:00.000Z',kind:'site',comment:msg,context:{page:'calibration'}});
 assert.equal((await send({kind:'site',comment:'too short'})).status,400);
 assert.equal((await send({comment:msg})).status,400,'paper feedback still needs a paper');
});
test('both kinds keep optional contact fields, trimmed and capped',async()=>{
 const {send,puts}=setup();
 const msg='x'.repeat(120);
 assert.equal((await send({kind:'site',comment:msg,contact:{name:'  Ada ',reach:'ada@example.org',extra:'dropped'}})).status,201);
 assert.deepEqual(puts[0][1].contact,{name:'Ada',reach:'ada@example.org'});
 assert.equal((await send({kind:'site',comment:msg,contact:{name:'',reach:'   '}})).status,201);assert.equal('contact' in puts[1][1],false);
 assert.equal((await send({kind:'site',comment:msg,contact:{reach:'y'.repeat(500)}})).status,201);assert.equal(puts[2][1].contact.reach.length,200);
 assert.equal((await send({paper,comment:msg+'p',contact:{name:'Ada'}})).status,201);assert.deepEqual(puts[3][1].contact,{name:'Ada'});
});
test('paper feedback keeps the sender\'s answer to "Are you an author?"; unknown answers and site feedback drop it',async()=>{
 const {send,puts}=setup();
 const msg='x'.repeat(120);
 assert.equal((await send({paper,comment:msg,author:'first'})).status,201);assert.equal(puts[0][1].author,'first');
 assert.equal((await send({paper,comment:msg+'2',author:'chair'})).status,201);assert.equal('author' in puts[1][1],false);
 assert.equal((await send({paper,comment:msg+'3',author:''})).status,201);assert.equal('author' in puts[2][1],false);
 assert.equal((await send({kind:'site',comment:msg,author:'first'})).status,201);assert.equal('author' in puts[3][1],false);
});
test('paper feedback keeps the sender\'s answer to the hard negative mining consent question; unknown answers and site feedback drop it',async()=>{
 const {send,puts}=setup();
 const msg='x'.repeat(120);
 assert.equal((await send({paper,comment:msg,consent:'yes'})).status,201);assert.equal(puts[0][1].consent,'yes');
 assert.equal((await send({paper,comment:msg+'2',consent:'maybe'})).status,201);assert.equal('consent' in puts[1][1],false);
 assert.equal((await send({paper,comment:msg+'3',consent:''})).status,201);assert.equal('consent' in puts[2][1],false);
 assert.equal((await send({kind:'site',comment:msg,consent:'yes'})).status,201);assert.equal('consent' in puts[3][1],false);
});

test('the sender gets an id and edit key; an edit is stored as a new version next to the first, which is never overwritten',async()=>{
 const {send,puts,store}=setup();
 const first='The methods section was written by me without AI; only the abstract was polished with a tool before submission.';
 const r=await send({paper,comment:first,author:'first'});
 assert.equal(r.status,201);const {id,key}=await r.json();
 assert.match(id,/^2026-10-13\/\d{13}-[0-9a-f-]{36}$/);assert.match(key,/^[0-9a-f]{48}$/);
 assert.equal(puts[0][0],`feedback/${id}.json`);assert.equal(JSON.stringify(puts[0][1]).includes(key),false,'the edit key itself is not stored');
 const second=first+' I also wrote the related work section myself.';
 const e=await send({paper,comment:second,author:'first',contact:{reach:'@someone'},edit:{id,key}});
 assert.equal(e.status,201);assert.deepEqual(await e.json(),{ok:true,id});
 assert.match(puts[1][0],new RegExp(`^feedback/${id}/edit-\\d+\\.json$`));
 assert.equal(puts[1][1].comment,second);assert.equal(puts[1][1].edit_of,id);assert.deepEqual(puts[1][1].contact,{reach:'@someone'});
 assert.equal(JSON.parse(store.get(`feedback/${id}.json`)).comment,first,'first version unchanged');
 const refused=async(body,status=403)=>{const x=await send(body);assert.equal(x.status,status);};
 await refused({paper,comment:second+' again',edit:{id,key:'0'.repeat(48)}});
 await refused({paper:'b'.repeat(24),comment:second+' again',edit:{id,key}},400); // not a known paper
 await refused({paper,comment:second+' again',edit:{id:'2026-10-13/../../x',key}});
 await refused({paper,comment:second+' again',edit:{id:id.replace(/.$/,'0'),key}});
 assert.equal(puts.length,2,'refused edits store nothing');
});
test('an exact repeat (same paper and text, same day) is stored once and gets no edit key',async()=>{
 const {send,puts}=setup();
 const text='This paper was written by hand by all three authors; the flagged sentences in section 3 are standard phrasing.';
 assert.equal((await send({paper,comment:text})).status,201);
 const again=await send({paper,comment:text});assert.equal(again.status,201);assert.deepEqual(await again.json(),{ok:true,duplicate:true});
 assert.equal(puts.length,1);
 assert.equal((await send({paper,comment:text+' Thanks.'})).status,201);assert.equal(puts.length,2);
});
