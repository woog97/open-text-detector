// Private paper data lives in a separate bucket with no public domain.
export interface PrivateEnv { PRIVATE_BUCKET?:R2Bucket; PRIVATE_AUTH?:string }
const cookieName='otd_private_local', ttl=86400;
const encoder=new TextEncoder();
const privateHeaders={'Cache-Control':'private, no-store','Vary':'Cookie','X-Robots-Tag':'noindex, nofollow, noarchive','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer','X-Frame-Options':'DENY'};
function json(value:unknown,status=200,extra:Record<string,string>={}){return Response.json(value,{status,headers:{...privateHeaders,...extra}});}
const hex=(buffer:ArrayBuffer)=>Array.from(new Uint8Array(buffer),n=>n.toString(16).padStart(2,'0')).join('');
async function digest(value:string){return hex(await crypto.subtle.digest('SHA-256',encoder.encode(value)));}
async function key(secret:string){return crypto.subtle.importKey('raw',encoder.encode(secret),{name:'HMAC',hash:'SHA-256'},false,['sign','verify']);}
async function signed(payload:string,secret:string){return hex(await crypto.subtle.sign('HMAC',await key(secret),encoder.encode(payload)));}
function config(env:PrivateEnv){
  if(!env.PRIVATE_AUTH||!env.PRIVATE_BUCKET)return null;
  const c=JSON.parse(env.PRIVATE_AUTH);
  return typeof c.username==='string'&&/^[a-f0-9]{64}$/.test(c.passwordHash)&&typeof c.sessionKey==='string'&&c.sessionKey.length>=32?c:null;
}
async function authenticated(request:Request,c:{sessionKey:string}){
  const token=request.headers.get('cookie')?.split(';').map(s=>s.trim()).find(s=>s.startsWith(cookieName+'='))?.slice(cookieName.length+1);
  const match=token?.match(/^(\d+)\.([a-f0-9]{32})\.([a-f0-9]{64})$/);
  if(!match)return false;
  const expires=Number(match[1]),now=Math.floor(Date.now()/1000);
  if(expires<=now||expires>now+ttl)return false;
  const signature=Uint8Array.from(match[3].match(/../g)!,s=>parseInt(s,16));
  return crypto.subtle.verify('HMAC',await key(c.sessionKey),signature,encoder.encode(match[1]+'.'+match[2]));
}
const cookie=(value:string,age:number)=>`${cookieName}=${value}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${age}`;
/** Called before the public router. No object read happens before authentication. */
export async function privateResults(request:Request,env:PrivateEnv,parseRange:(value:string,size:number)=>{offset:number;length:number}|null):Promise<Response>{
  const url=new URL(request.url),path=url.pathname;
  const c=config(env);
  if(!c)return json({detail:'Private results are not configured yet.'},503);
  if(path==='/backend/private/login'||path==='/backend/private/logout'){
    if(request.method!=='POST')return json({detail:'Method not allowed'},405,{Allow:'POST'});
    if(request.headers.get('origin')!==url.origin)return json({detail:'Invalid origin'},403);
    if(path.endsWith('/logout'))return json({ok:true},200,{'Set-Cookie':cookie('',0)});
    if(!request.headers.get('content-type')?.startsWith('application/json')||Number(request.headers.get('content-length'))>4096)return json({detail:'Invalid request'},400);
    // Bound the body even when Content-Length is absent.
    const reader=request.body?.getReader();if(!reader)return json({detail:'Invalid request'},400);
    const chunks:Uint8Array[]=[];let size=0;
    while(true){const part=await reader.read();if(part.done)break;size+=part.value.length;if(size>4096){await reader.cancel();return json({detail:'Invalid request'},413);}chunks.push(part.value);}
    const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
    let input;try{input=JSON.parse(new TextDecoder().decode(bytes));}catch{return json({detail:'Invalid request'},400);}
    if(typeof input?.username!=='string'||typeof input?.password!=='string')return json({detail:'Username or password is incorrect.'},401);
    const hash=await digest(input.password);
    if(input.username!==c.username||hash!==c.passwordHash)return json({detail:'Username or password is incorrect.'},401);
    const payload=(Math.floor(Date.now()/1000)+ttl)+'.'+hex(crypto.getRandomValues(new Uint8Array(16)).buffer);
    return json({ok:true},200,{'Set-Cookie':cookie(payload+'.'+await signed(payload,c.sessionKey),ttl)});
  }
  if(!['GET','HEAD'].includes(request.method))return json({detail:'Method not allowed'},405,{Allow:'GET, HEAD'});
  if(!await authenticated(request,c))return json({detail:'Please sign in.'},401);
  if(path==='/backend/private/session')return json({username:c.username});
  const list=path==='/backend/private/v1/pdf-reader';
  const match=path.match(/^\/backend\/private\/v1\/pdf-reader\/([a-f0-9]{24})(\/file)?$/);
  if(!list&&!match)return json({detail:'Not found'},404);
  const catalogue=await env.PRIVATE_BUCKET!.get('catalogue.json');
  if(!catalogue)return json({detail:'Private results are unavailable.'},503);
  const data=await catalogue.json<{items:{id:string;title:string;filename:string;collection:string;bytes:number;classified:boolean;models:string[]}[]}>();
  if(list)return json(data);
  const paper=data.items.find(p=>p.id===match![1]);if(!paper)return json({detail:'Not found'},404);
  const file=!!match![2],objectKey=`papers/${paper.id}/${file?'paper.pdf':'detail.json.gz'}`;
  const meta=await env.PRIVATE_BUCKET!.head(objectKey);if(!meta)return json({detail:'Not found'},404);
  const headers=new Headers({...privateHeaders,'Content-Type':file?'application/pdf':'application/json','Content-Length':String(meta.size)});
  if(file){headers.set('Accept-Ranges','bytes');headers.set('Content-Disposition',`inline; filename="${paper.id}.pdf"`);}else headers.set('Content-Encoding','gzip');
  let range={offset:0,length:meta.size},status=200;
  if(file&&request.method!=='HEAD'&&request.headers.has('range')){
    const parsed=parseRange(request.headers.get('range')!,meta.size);
    if(!parsed)return new Response(null,{status:416,headers:{...privateHeaders,'Content-Range':`bytes */${meta.size}`}});
    range=parsed;status=206;headers.set('Content-Range',`bytes ${range.offset}-${range.offset+range.length-1}/${meta.size}`);headers.set('Content-Length',String(range.length));
  }
  if(request.method==='HEAD')return new Response(null,{headers});
  const object=await env.PRIVATE_BUCKET!.get(objectKey,file?{range}:undefined);
  if(!object)return json({detail:'Not found'},404);
  return new Response(object.body,{status,headers,...(!file?{encodeBody:'manual' as const}:{})});
}
