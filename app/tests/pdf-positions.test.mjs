import {test} from 'node:test';
import assert from 'node:assert/strict';
import {positionPassages,supportsSavedPositions} from '../lib/pdf-positions.ts';
const page={page:1,width:100,height:200,rotation:0,media_box:[0,0,100,200],crop_box:[0,0,100,200]};
test('maps repeated short text by offsets, with Unicode code points and page boundaries',()=>{
 const map={pages:[page,{...page,page:2}],rectangles:[{start:0,end:1,page:1,x0:10,y0:20,x1:20,y1:30},{start:2,end:4,page:1,x0:30,y0:20,x1:50,y1:30},{start:5,end:7,page:2,x0:10,y0:20,x1:30,y1:30}]};
 const result=positionPassages('😀 hi hi',[{start:2,end:7,label:'ai_evidence',score:.8}],map)[0];
 assert.equal(result.text,'hi hi');assert.equal(result.mapped,true);assert.deepEqual(result.rectangles.map(r=>r.page),[1,2]);assert.equal(result.rectangles[0].x,.3);
});
test('accounts for cropping and rejects non-quarter-turn saved geometry',()=>{
 const map={pages:[{...page,crop_box:[10,20,90,180]}],rectangles:[{start:0,end:4,page:1,x0:10,y0:20,x1:30,y1:40}]};
 assert.equal(supportsSavedPositions(map),true);
 assert.deepEqual(positionPassages('word',[{start:0,end:4,label:'human',score:0}],map)[0].rectangles,[{page:1,x:0,y:0,w:.25,h:.125}]);
 assert.equal(supportsSavedPositions({...map,pages:[{...page,rotation:45}]}),false);
});
test('whitespace-only passages receive no invented rectangles',()=>{
 const p=positionPassages('a  b',[{start:1,end:3,label:'human',score:0}],{pages:[page],rectangles:[{start:0,end:1,page:1,x0:1,y0:1,x1:2,y1:2},{start:3,end:4,page:1,x0:3,y0:1,x1:4,y1:2}]})[0];
 assert.equal(p.mapped,false);
});
test('nested word ranges do not break the position search; invalid coordinates are discarded',()=>{
 const map={pages:[page],rectangles:[{start:0,end:10,page:1,x0:0,y0:0,x1:10,y1:10},{start:1,end:2,page:1,x0:10,y0:0,x1:20,y1:10},{start:4,end:5,page:1,x0:NaN,y0:0,x1:20,y1:10}]};
 assert.equal(positionPassages('abcdefghij',[{start:4,end:5,label:'human',score:0}],map)[0].rectangles.length,1);
 assert.equal(supportsSavedPositions({pages:[{...page,crop_box:[0,0,Infinity,200]}],rectangles:[]}),false);
 assert.equal(supportsSavedPositions({pages:null,rectangles:[]}),false);
});

test('rendered word positions use the rotated crop origin for 90, 180 and 270 degrees',()=>{
 const base={...page,media_box:[10,20,110,220],crop_box:[20,50,100,200]};
 // Unrotated crop: left 10, top 20, width 80, height 150.
 for(const [rotation,x0,y0,w,h] of [[0,10,20,80,150],[90,30,10,150,80],[180,10,30,80,150],[270,20,10,150,80]]){
  const map={pages:[{...base,rotation}],rectangles:[{start:0,end:4,page:1,x0,y0,x1:x0+10,y1:y0+10}]};
  assert.equal(supportsSavedPositions(map),true);
  assert.deepEqual(positionPassages('word',[{start:0,end:4,label:'human',score:0}],map)[0].rectangles,[{page:1,x:0,y:0,w:10/w,h:10/h}]);
 }
});
