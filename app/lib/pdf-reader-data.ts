import type {Passage} from './pdf-alignment';
import type {SavedMap} from './pdf-positions';
export type ReaderReport={id:string;text:string;model:{name:string};text_sha256?:string;result:{segments:Passage[];notice?:string}};
export type ReaderDetail={version:string;version_verified?:boolean;reports:ReaderReport[];position_maps?:Record<string,SavedMap>};
/** Reject incomplete/corrupt reports before they enter React state. */
export function validateReaderDetail(value:unknown):ReaderDetail {
  const d=value as ReaderDetail;
  const fail=()=>{throw new Error('This paper’s saved results are incomplete or invalid. Please try again later.');};
  if(!d||typeof d.version!=='string'||!Array.isArray(d.reports))fail();
  const ids=new Set<string>();
  for(const r of d.reports){
    if(!r||typeof r.id!=='string'||ids.has(r.id)||typeof r.text!=='string'||typeof r.model?.name!=='string'||!Array.isArray(r.result?.segments))fail();
    if(r.result.notice!==undefined && typeof r.result.notice!=="string")fail();
    ids.add(r.id);
    const length=Array.from(r.text).length;
    let end=0;
    for(const p of r.result.segments){
      if(!p||!Number.isInteger(p.start)||!Number.isInteger(p.end)||p.start<end||p.end<p.start||p.end>length||!Number.isFinite(p.score)||p.score<0||p.score>1||typeof p.label!=='string')fail();
      end=p.end;
    }
  }
  if(d.position_maps!==undefined && (!d.position_maps||typeof d.position_maps!=='object'||Array.isArray(d.position_maps)))fail();
  return d;
}
