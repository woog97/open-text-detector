import {test} from 'node:test';
import assert from 'node:assert/strict';
import {validateReaderDetail} from '../lib/pdf-reader-data.ts';
const valid=()=>({version:'v1',reports:[{id:'r1',text:'😀 text',model:{name:'v8'},result:{segments:[{start:0,end:6,score:.5,label:'human'}]}}]});
test('accepts codepoint ranges and empty reports',()=>{
 assert.equal(validateReaderDetail(valid()).reports.length,1);
 assert.equal(validateReaderDetail({version:'v1',reports:[]}).reports.length,0);
});
test('rejects missing, overlapping, invalid and duplicate report data before rendering',()=>{
 for(const input of [null,{}, {version:'v',reports:{}}, {version:'v',reports:[null]}])assert.throws(()=>validateReaderDetail(input));
 for(const patch of [{start:-1},{end:7},{score:NaN},{score:1.1},{start:.5},{label:null}]){
  const input=valid();Object.assign(input.reports[0].result.segments[0],patch);assert.throws(()=>validateReaderDetail(input));
 }
 const overlap=valid();overlap.reports[0].result.segments.push({start:2,end:4,score:.5,label:'human'});assert.throws(()=>validateReaderDetail(overlap));
 const duplicate=valid();duplicate.reports.push(duplicate.reports[0]);assert.throws(()=>validateReaderDetail(duplicate));
});
