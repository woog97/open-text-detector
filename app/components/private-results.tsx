import {useEffect,useState,lazy,Suspense,type FormEvent} from 'react';
import ReaderErrorBoundary from './reader-error-boundary';
import './private-results.css';
const PDFReader=lazy(()=>import('./pdf-reader'));
async function api(path:string,init?:RequestInit){
  const response=await fetch('/backend/private'+path,{...init,cache:'no-store'});
  if(response.status===401){window.dispatchEvent(new Event('private-session-expired'));throw new Error('Your session ended. Please sign in again.');}
  if(!response.ok)throw new Error('Private results are unavailable. Please retry.');
  return response.json();
}
export default function PrivateResults(){
  const [state,setState]=useState<'loading'|'login'|'ready'|'error'>('loading');
  const [error,setError]=useState(''),[busy,setBusy]=useState(false);
  useEffect(()=>{
    document.title='Private results · Open Text Detector';
    const meta=document.createElement('meta');meta.name='robots';meta.content='noindex, nofollow, noarchive';document.head.append(meta);
    const expired=()=>setState('login');window.addEventListener('private-session-expired',expired);
    let active=true;
    api('/session').then(()=>{if(active)setState('ready');}).catch(e=>{if(active){setState(s=>s==='login'?s:'error');setError(e.message);}});
    return()=>{active=false;meta.remove();window.removeEventListener('private-session-expired',expired);};
  },[]);
  async function login(event:FormEvent<HTMLFormElement>){
    event.preventDefault();setBusy(true);setError('');const data=new FormData(event.currentTarget);
    try{const r=await fetch('/backend/private/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:data.get('username'),password:data.get('password')})});
      if(!r.ok)throw new Error(r.status===401?'Username or password is incorrect.':'Sign-in is unavailable. Please retry.');setState('ready');
    }catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  async function logout(){setBusy(true);try{const r=await fetch('/backend/private/logout',{method:'POST'});if(!r.ok)throw new Error('Could not sign out. Please retry.');setState('login');setError('');}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  return <main className="private-shell"><header className="private-header"><a href="/">Open Text Detector <span>/ Private results</span></a>{state==='ready'&&<button onClick={logout} disabled={busy}>Sign out</button>}</header>
    {state==='ready'?<><div className="private-heading"><span>PRIVATE COLLECTION</span><h1>Your papers</h1><p>Read the original pages alongside their classification results.</p>{error&&<p role="alert">{error}</p>}</div><ReaderErrorBoundary><Suspense fallback={<p className="private-loading">Opening reader…</p>}><PDFReader api={api} hosted fileBasePath="/backend/private/v1/pdf-reader" preferredModel="v8"/></Suspense></ReaderErrorBoundary></>:
    <section className="private-card"><span className="private-eyebrow">OPEN TEXT DETECTOR</span><h1>Sign in</h1><p>Access your private paper results.</p>{state==='loading'?<p>Checking session…</p>:state==='error'?<><p role="alert">{error}</p><button onClick={()=>location.reload()}>Try again</button></>:
    <form onSubmit={login}><label>Username<input name="username" autoComplete="username" required autoFocus maxLength={100}/></label><label>Password<input name="password" type="password" autoComplete="current-password" required maxLength={256}/></label>{error&&!error.includes('session ended')&&<p role="alert" className="private-error">{error}</p>}<button disabled={busy} type="submit">{busy?'Signing in…':'Sign in'}</button></form>}<a className="private-back" href="/">← Back to the atlas</a></section>}
  </main>;
}
