import {test} from 'node:test';
import assert from 'node:assert/strict';
import {collectionBreakdown,summarizePassages} from '../lib/paper-scores.ts';
const stats=scores=>summarizePassages(scores.map(score=>({score})));
test('groups collections, weights passage totals, and changes with threshold/model',()=>{
 const papers=[{collection:'iclr/2025',score_summaries:{v5:stats([.2,.8,.9]),v8:stats([.9])}},{collection:'iclr/2025',score_summaries:{v5:stats([.1])}},{collection:'colm/2024'}];
 const groups=collectionBreakdown(papers,'v5',.8);
 assert.deepEqual(groups[0],{collection:'colm/2024',papers:1,scored:0,above:0,below:0,bins:[0,0,0,1]});
 assert.deepEqual(groups[1],{collection:'iclr/2025',papers:2,scored:2,above:1,below:3,bins:[1,1,0,0]});
 assert.equal(collectionBreakdown(papers,'v5',.65)[1].above,2);
 assert.deepEqual(collectionBreakdown(papers,'v8',.8)[1].bins,[0,0,1,1]);
 assert.equal(collectionBreakdown(papers,'all',.8)[1].below,1);
 assert.deepEqual(collectionBreakdown([],'all',.8),[]);
});
