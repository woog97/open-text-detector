import {privateResults,type PrivateEnv} from './private-results.ts';
import {byteRange} from '../atlas-cloud/worker.ts';
interface Env extends PrivateEnv {ASSETS:Fetcher}
// This entry point is for loopback-only local development. Never deploy it.
export function createLocalWorker(){return {async fetch(request:Request,env:Env):Promise<Response>{
 const url=new URL(request.url);
 if(!['localhost','127.0.0.1','[::1]'].includes(url.hostname))return new Response('Local use only',{status:403});
 try{
  if(url.pathname.startsWith('/backend/private/'))return privateResults(request,env,byteRange);
  // Public collection remains readable without copying thousands of public PDFs locally.
  if(url.pathname.startsWith('/backend/v1/pdf-reader')&&['GET','HEAD'].includes(request.method)){
   const headers=new Headers();for(const name of ['range','if-range','if-none-match']){const value=request.headers.get(name);if(value)headers.set(name,value);}
   return fetch('https://open-text-detector.woog09.workers.dev'+url.pathname+url.search,{method:request.method,headers});
  }
  if(url.pathname.startsWith('/backend/'))return new Response('Not found',{status:404});
  return env.ASSETS.fetch(request);
 }catch{return Response.json({detail:'Private results are unavailable.'},{status:503,headers:{'Cache-Control':'no-store'}});}
}};}
export default createLocalWorker();
