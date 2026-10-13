import test from 'node:test';
import assert from 'node:assert/strict';
import { summarizePassages, aboveCount, paperRate, summaryFor, scoreBand } from '../lib/paper-scores.ts';
test('strict thresholds preserve exact boundary scores and exclude invalid scores',()=>{
 const summary=summarizePassages([0,.2,.20001,.5,.8,.80001,1,NaN,-1,2].map(score=>({score})));
 assert.equal(summary.total,7); assert.equal(aboveCount(summary,.8),2); assert.equal(aboveCount(summary,.2),5);
 assert.equal(aboveCount(summary,0),6);assert.equal(aboveCount(summary,1),0);assert.equal(paperRate(summary,.8),2/7);
});
test('missing data is unknown, not a zero score; model selection does not mix results',()=>{
 const v5=summarizePassages([{score:.1}]),v8=summarizePassages([{score:.9}]);const paper={score_summaries:{v5,v8}};
 assert.equal(summaryFor(paper,'all'),v8);assert.equal(summaryFor(paper,'v5'),v5);
 assert.equal(paperRate(undefined,.8),null);assert.equal(paperRate(summarizePassages([]),.8),null);
 assert.deepEqual([.2,.5,.8,.81].map(scoreBand),['low','medium','medium','high']);
});
test('punctuation artifacts do not count as text passages; offsets use Unicode code points',()=>{
 const text='😀 . . 文 42';
 const summary=summarizePassages([{start:0,end:6,score:.99},{start:6,end:7,score:.1},{start:8,end:10,score:.9}],text);
 assert.equal(summary.total,2);assert.equal(summary.excluded,1);assert.equal(summary.policy,2);
 assert.equal(paperRate(summary,.65),.5);
 const empty=summarizePassages([{start:0,end:5,score:1}],'. . .');
 assert.equal(paperRate(empty,.65),null);
});

test('experimental Laya selection stays separate and does not replace default MELD scores',async()=>{
 const {modelLabel}=await import('../lib/paper-scores.ts');
 const laya=summarizePassages([{score:.3}]),v8=summarizePassages([{score:.9}]);
 const p={score_summaries:{laya,v8}};
 assert.equal(summaryFor(p,'laya'),laya);
 assert.equal(summaryFor(p,'all'),v8);
 assert.equal(modelLabel('laya'),'Laya · Experimental');
 assert.equal(modelLabel('v8'),'MELD v8');
});
