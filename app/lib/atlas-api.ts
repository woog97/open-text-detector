import {validateCatalogue,type AtlasCatalogue} from './atlas-collection.ts';
let cached:{etag:string|null;data:AtlasCatalogue}|undefined;
/** Explicit 304 handling preserves parsed data and React references during refresh. */
export async function loadCatalogue(signal:AbortSignal):Promise<AtlasCatalogue>{
  const response=await fetch('/backend/v1/pdf-reader',{signal,cache:'no-cache',headers:cached?.etag?{'If-None-Match':cached.etag}:{}});
  if(response.status===304&&cached)return cached.data;
  if(!response.ok)throw new Error('The paper collection is unavailable. Please try again shortly.');
  const data=validateCatalogue(await response.json());
  cached={etag:response.headers.get('etag'),data};return data;
}
