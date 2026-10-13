import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mergeHighlightBoxes} from '../lib/pdf-highlight-boxes.ts';
const box=(x,y=10,id=1,w=20,h=10)=>({x,y,id,w,h,band:'low'});
test('joins shuffled adjacent words with small vertical offsets without mutating source',()=>{
 const input=[box(34,10.5),box(10),box(58,9.5)];
 const before=structuredClone(input);
 assert.deepEqual(mergeHighlightBoxes(input),[{id:1,x:10,y:9.5,w:68,h:11,band:'low'}]);
 assert.deepEqual(input,before);
});
test('keeps different passages, lines, and column gaps separate',()=>{
 assert.equal(mergeHighlightBoxes([box(10),box(34,10,2),box(10,23),box(200)]).length,4);
});
test('merging is zoom invariant and unions overlapping rectangles',()=>{
 const input=[box(10),box(25),box(49)];
 assert.equal(mergeHighlightBoxes(input).length,1);
 const scaled=input.map(b=>({...b,x:b.x*3,y:b.y*3,w:b.w*3,h:b.h*3}));
 assert.equal(mergeHighlightBoxes(scaled).length,1);
 assert.equal(mergeHighlightBoxes(scaled)[0].w,mergeHighlightBoxes(input)[0].w*3);
});
test('large separated-line input remains distinct and ignores unusable geometry',()=>{
 const input=Array.from({length:20000},(_,i)=>box(10,i*20));
 const result=mergeHighlightBoxes([...input,box(NaN),box(1,1,1,-1)]);
 assert.equal(result.length,20000);assert.equal(result.at(-1).y,399980);
});
