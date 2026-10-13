/** Only the dedicated public PDF prefix is discoverable; other bucket objects stay private. */
export const publicPdfKey = /^papers\/([a-f0-9]{64})\.pdf$/;
export type DiscoveryMetadata = {forum_id?:string;number?:number;keywords?:string[];primary_area?:string;tldr?:string;abstract_preview?:string;published?:number;updated?:number};
export type PublishedPaper = DiscoveryMetadata & {id:string;title:string;filename:string;collection:string;bytes:number;classified:boolean;models:string[];score_summaries?:Record<string,unknown>;pdf_key:string;detail_key?:string};
type ArchiveRow = DiscoveryMetadata & {key?:string;title?:string;filename?:string;conference?:string;year?:number;collection?:string};
type PdfObject = {key:string;size:number;customMetadata?:Record<string,string>};
export function mergeArchive(published:PublishedPaper[], objects:PdfObject[], rows:ArchiveRow[]):PublishedPaper[] {
  const items=published.map(p=>({...p})), keys=new Set(items.map(p=>p.pdf_key)), ids=new Set(items.map(p=>p.id));
  const metadata=new Map(rows.filter(r=>r&&typeof r.key==='string').map(r=>[r.key,r]));
  const discovery=(row:ArchiveRow|undefined):DiscoveryMetadata=>row?{forum_id:row.forum_id,number:row.number,keywords:row.keywords,primary_area:row.primary_area,tldr:row.tldr,abstract_preview:row.abstract_preview,published:row.published,updated:row.updated}:{};
  for(const item of items)Object.assign(item,discovery(metadata.get(item.pdf_key)));
  for(const object of objects){
    const match=publicPdfKey.exec(object.key);
    if(!match||keys.has(object.key))continue;
    const hash=match[1],id=hash.slice(0,24);
    if(ids.has(id))throw new Error('PDF identifier collision');
    const row=metadata.get(object.key),meta=object.customMetadata;
    const filename=row?.filename||meta?.filename||`${hash}.pdf`;
    const collection=row?.collection||(row?.conference&&row.year?`${row.conference}/${row.year}`:null)||meta?.collection||'Other uploads';
    items.push({...discovery(row),id,title:row?.title||meta?.title||filename.replace(/\.pdf$/i,''),filename,collection,bytes:object.size,classified:false,models:[],pdf_key:object.key});
    keys.add(object.key);ids.add(id);
  }
  return items;
}
type SourceCatalogue = {items:PublishedPaper[];models:{total:number;available:number;complete:boolean}[];published_at:number};
export type ReaderShards = Record<string,Record<string,[string,string]>>;
/** Fields the public catalogue exposes; storage keys stay server-side. */
export function publicPaper(p:PublishedPaper){return {id:p.id,title:p.title,filename:p.filename,collection:p.collection,bytes:p.bytes,classified:p.classified,models:p.models,score_summaries:p.score_summaries,forum_id:p.forum_id,number:p.number,keywords:p.keywords,primary_area:p.primary_area,tldr:p.tldr,abstract_preview:p.abstract_preview,published:p.published,updated:p.updated};}
/**
 * The public catalogue and the reader's id -> (PDF hash, detail hash) shards, built offline by
 * scripts/publish_public_catalogue.py so the Worker never parses the full source catalogue or lists the archive.
 */
export function buildPublicCatalogue(source:SourceCatalogue, objects:PdfObject[], rows:ArchiveRow[]):{catalogue:{items:ReturnType<typeof publicPaper>[];models:unknown[];published_at:number};shards:ReaderShards} {
  if(!Array.isArray(source.items))throw new Error('Invalid catalogue');
  const items=mergeArchive(source.items,[...objects].sort((a,b)=>a.key.localeCompare(b.key)),rows);
  const ids=new Set<string>(),shards:ReaderShards={};
  for(const p of items){
    if(!/^[a-f0-9]{24}$/.test(p.id)||ids.has(p.id)||!/^papers\/[a-f0-9]{64}\.pdf$/.test(p.pdf_key)||(p.detail_key!==undefined&&!/^atlas-public\/objects\/[a-f0-9]{64}\.json\.gz$/.test(p.detail_key))||(p.classified&&!p.detail_key))throw new Error('Invalid published object');
    ids.add(p.id);
    (shards[p.id.slice(0,2)]??={})[p.id]=[p.pdf_key.slice(7,-4),p.detail_key?p.detail_key.slice(21,-8):''];
  }
  const models=source.models.map(m=>({...m,total:items.length,complete:m.available===items.length}));
  return {catalogue:{items:items.map(publicPaper),models,published_at:source.published_at},shards};
}
