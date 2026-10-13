// The public site moved from pangram-paper-atlas.woog09.workers.dev to open-text-detector.woog09.workers.dev on
// October 11, 2026, when the project was renamed. This Worker runs under the old name (wrangler.atlas-redirect.json)
// and sends every request to the same path and query on the new address, so old links, bookmarks and installed
// browser extensions keep working. 301/308 are permanent; the short cache lets a rollback take effect within an hour.
export const NEW_ORIGIN='https://open-text-detector.woog09.workers.dev';
export function redirect(request:Request):Response{
 const url=new URL(request.url);
 const status=['GET','HEAD'].includes(request.method)?301:308; // 308 keeps the method and body of uploads
 return new Response(null,{status,headers:{Location:NEW_ORIGIN+url.pathname+url.search,'Cache-Control':'public, max-age=3600'}});
}
export default {fetch:redirect};
