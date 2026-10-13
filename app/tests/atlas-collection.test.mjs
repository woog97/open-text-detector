import {test} from 'node:test';
import assert from 'node:assert/strict';
// Loader resolves the extensionless internal import used by the frontend.
import {sortPapers,validateCatalogue} from '../lib/atlas-collection.ts';
const paper=(n,scores)=>({id:n.toString(16).padStart(24,'0'),title:`Paper ${n}`,filename:'paper.pdf',collection:'iclr/2025',bytes:n,classified:true,models:['v8'],score_summaries:scores});
const summary=(index)=>({total:1,histogram:Array.from({length:21},(_,i)=>i===index?1:0)});
test('sorting responds to threshold/model, keeps unknowns last and does not mutate input',()=>{
 const papers=[paper(1,{v8:summary(14),v5:summary(20)}),paper(2,{v8:summary(19),v5:summary(1)}),paper(3,{})];
 assert.deepEqual(sortPapers(papers,'share',true,'v8',.8).map(p=>p.id),[papers[1].id,papers[0].id,papers[2].id]);
 assert.equal(sortPapers(papers,'share',true,'v5',.8)[0].id,papers[0].id);
 assert.equal(sortPapers(papers,'share',false,'v8',.8)[0].id,papers[0].id);
 assert.equal(sortPapers(papers,'share',true,'v8',1)[0].id,papers[0].id);
 assert.equal(sortPapers(papers,'size',true,'all',.8)[0].bytes,3);
 assert.equal(papers[0].bytes,1);
});
test('catalogue validation rejects incomplete data, duplicate IDs and corrupt histograms',()=>{
 const good={items:[paper(1,{v8:summary(1)})]};assert.equal(validateCatalogue(good),good);
 for(const bad of [null,{}, {items:[null]}, {items:[good.items[0],good.items[0]]}, {items:[paper(1,{v8:{total:2,histogram:[1]}})]}, {items:[{...good.items[0],models:'v8'}]}, {...good,models:[{id:'v8'}]}])assert.throws(()=>validateCatalogue(bad),/invalid/);
});
