import type { Passage, AlignedPassage } from './pdf-alignment';
export type SavedMap = {
  pages: {page:number;width:number;height:number;rotation:number;media_box:number[];crop_box:number[]}[];
  rectangles: {start:number;end:number;page:number;x0:number;y0:number;x1:number;y1:number}[];
};
export type PositionedPassage = AlignedPassage & {rectangles?:{page:number;x:number;y:number;w:number;h:number}[]};
export function supportsSavedPositions(map:SavedMap|undefined):map is SavedMap {
  return !!map && Array.isArray(map.pages) && Array.isArray(map.rectangles) && map.pages.length>0 && map.pages.every(p=>p && Number.isInteger(p.page) && p.page>0 && [0,90,180,270].includes(p.rotation) && Array.isArray(p.crop_box) && p.crop_box.length===4 && Array.isArray(p.media_box) && p.media_box.length===4 && [...p.crop_box,...p.media_box].every(Number.isFinite) && p.crop_box[2]>p.crop_box[0] && p.crop_box[3]>p.crop_box[1]);
}
export function positionPassages(text:string,segments:Passage[],map:SavedMap):PositionedPassage[] {
  const chars=Array.from(text), pages=new Map(map.pages.map(p=>[p.page,p]));
  const words=map.rectangles.filter(r=>r && [r.start,r.end,r.x0,r.y0,r.x1,r.y1].every(Number.isFinite) && Number.isInteger(r.start) && Number.isInteger(r.end) && r.start>=0 && r.end>r.start && r.end<=chars.length).sort((a,b)=>a.start-b.start);
  let maxEnd=0;const prefixEnd=words.map(word=>maxEnd=Math.max(maxEnd,word.end));
  return segments.map((segment,id)=>{
    let lo=0,hi=words.length;
    while(lo<hi){const mid=(lo+hi)>>>1;if(prefixEnd[mid]<=segment.start)lo=mid+1;else hi=mid;}
    const rectangles=[];
    for(let n=lo;n<words.length && words[n].start<segment.end;n++){
      const word=words[n],p=pages.get(word.page);if(!p||word.end<=segment.start)continue;
      const [cx0,cy0,cx1,cy1]=p.crop_box;
      // Poppler rectangles already follow the rendered page rotation. Rotate
      // the crop origin/dimensions into that same space; do not rotate words again.
      let leftCrop=cx0-p.media_box[0],topCrop=p.media_box[3]-cy1;
      let w=cx1-cx0,h=cy1-cy0;
      const mw=p.media_box[2]-p.media_box[0],mh=p.media_box[3]-p.media_box[1];
      if(p.rotation===90){[leftCrop,topCrop,w,h]=[mh-topCrop-h,leftCrop,h,w];}
      else if(p.rotation===180){[leftCrop,topCrop]=[mw-leftCrop-w,mh-topCrop-h];}
      else if(p.rotation===270){[leftCrop,topCrop,w,h]=[topCrop,mw-leftCrop-w,h,w];}
      const x0=word.x0-leftCrop,y0=word.y0-topCrop,x1=word.x1-leftCrop,y1=word.y1-topCrop;
      const left=Math.max(0,x0),top=Math.max(0,y0),right=Math.min(w,x1),bottom=Math.min(h,y1);
      if(right>left && bottom>top)rectangles.push({page:word.page,x:left/w,y:top/h,w:(right-left)/w,h:(bottom-top)/h});
    }
    return {...segment,id,text:chars.slice(segment.start,segment.end).join(''),mapped:rectangles.length>0,page:rectangles[0]?.page,pieces:[],rectangles};
  });
}
