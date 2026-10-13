/** Bound raster allocation and concurrent PDF work across mounted viewers. */
export function rasterSize(width:number,height:number,deviceRatio:number) {
  if(!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0)throw new Error('Invalid PDF page dimensions');
  const ratio=Math.min(Math.max(1,deviceRatio||1),2,4096/width,4096/height,Math.sqrt(4_000_000/(width*height)));
  return {ratio,width:Math.max(1,Math.floor(width*ratio)),height:Math.max(1,Math.floor(height*ratio))};
}
export function createRenderQueue(limit=2) {
  let running=0;
  const waiting: {start:()=>void}[]=[];
  return (signal:AbortSignal):Promise<()=>void>=>new Promise((resolve,reject)=>{
    const cancel=()=>{const i=waiting.indexOf(job);if(i>=0)waiting.splice(i,1);reject(new Error('Render cancelled'));};
    const job={start:()=>{
      signal.removeEventListener('abort',cancel);
      if(signal.aborted){reject(new Error('Render cancelled'));return;}
      running++;let released=false;
      resolve(()=>{if(released)return;released=true;running--;while(running<limit&&waiting.length)waiting.shift()!.start();});
    }};
    if(signal.aborted){reject(new Error('Render cancelled'));return;}
    signal.addEventListener('abort',cancel,{once:true});
    if(running<limit)job.start();else waiting.push(job);
  });
}
export const acquirePdfRender=createRenderQueue();

/** Detach cancelled consumers from slow PDF/network work without leaking a queue slot. */
export function abortable<T>(promise:Promise<T>,signal:AbortSignal):Promise<T> {
  return new Promise((resolve,reject)=>{
    const abort=()=>reject(new Error('Render interrupted. Try this page again.'));
    // Always attach rejection handling, including when already aborted.
    promise.then(value=>{signal.removeEventListener('abort',abort);if(!signal.aborted)resolve(value);},error=>{signal.removeEventListener('abort',abort);reject(error);});
    if(signal.aborted)abort();else signal.addEventListener('abort',abort,{once:true});
  });
}
