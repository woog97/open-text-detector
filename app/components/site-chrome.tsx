"use client";
/* eslint-disable @next/next/no-html-link-for-pages */
import {useEffect,useRef,useState,type ReactNode} from 'react';
import {ArrowRight,ChevronDown} from 'lucide-react';
import './site.css';
import './beta-theme.css';
import {BETA} from '../lib/ui-mode';

export type SitePage='home'|'discover'|'saved'|'classifications'|'calibration'|'transparency'|'architecture'|'data'|'results'|'limitations'|'manifesto'|'about';
type NavTarget={href:string;onClick?:()=>void};

/** Number of papers in the browser-local reading list (shared with the discover page). */
export function useSavedCount(){
 const [count,setCount]=useState(0);
 useEffect(()=>{const read=()=>{try{const s=JSON.parse(localStorage.getItem('atlas-reading-list')||'[]');setCount(Array.isArray(s)?s.length:0);}catch{setCount(0);}};read();window.addEventListener('storage',read);return()=>window.removeEventListener('storage',read);},[]);
 return count;
}

function NavItem({target,current,children}:{target:NavTarget;current:boolean;children:ReactNode}){
 return <a href={target.href} aria-current={current?'page':undefined} onClick={target.onClick?e=>{e.preventDefault();target.onClick!();}:undefined}>{children}</a>;
}

/** A nav link with a dropdown of sub-pages. Hovering the block or clicking the arrow opens it; the arrow flips when open. */
function NavMenu({href,label,current,items}:{href:string;label:string;current:SitePage;items:[SitePage,string,string][]}){
 const [open,setOpen]=useState(false);const ref=useRef<HTMLDivElement>(null);
 useEffect(()=>{if(!open)return;
  const away=(e:PointerEvent)=>{if(!ref.current?.contains(e.target as Node))setOpen(false);};
  const esc=(e:KeyboardEvent)=>{if(e.key==='Escape')setOpen(false);};
  document.addEventListener('pointerdown',away);document.addEventListener('keydown',esc);
  return()=>{document.removeEventListener('pointerdown',away);document.removeEventListener('keydown',esc);};},[open]);
 const inside=items.some(([k])=>k===current);
 return <div ref={ref} className={`site-menu${open?' is-open':''}${inside?' is-inside':''}`}
  onMouseEnter={()=>{if(matchMedia('(hover:hover)').matches)setOpen(true);}} onMouseLeave={()=>{if(matchMedia('(hover:hover)').matches)setOpen(false);}}
  onBlur={e=>{if(!e.currentTarget.contains(e.relatedTarget as Node))setOpen(false);}}>
  <a href={href} aria-current={current===label.toLowerCase()?'page':undefined}>{label}</a>
  <button type="button" className="site-menu-toggle" aria-expanded={open} aria-label={`${open?'Hide':'Show'} ${label} pages`} onClick={()=>setOpen(o=>!o)}><ChevronDown size={14} aria-hidden/></button>
  {open&&<div className="site-menu-list" role="menu">
   {items.map(([k,name,href])=><a key={k} role="menuitem" href={href} aria-current={current===k?'page':undefined}>{name}</a>)}
  </div>}
 </div>;
}

/** Where the submissions list lives now that `/` is the home page. */
export const SUBMISSIONS_HREF=BETA?'/?collection=iclr%2F2027':'/';


/** One header for every public page. */
export function SiteHeader({page,savedCount,discover={href:SUBMISSIONS_HREF},saved={href:'/?saved=1'},classifications,browse=true,extra}:{page:SitePage;savedCount:number;discover?:NavTarget;saved?:NavTarget;classifications?:NavTarget;browse?:boolean;extra?:ReactNode}){
 if(!BETA)return <><header className="site-nav"><a href="/" className="site-brand" onClick={discover.onClick&&browse?e=>{e.preventDefault();discover.onClick!();}:undefined}><span className="site-mark" aria-hidden>OTD</span>Open Text Detector</a>
  <nav aria-label="Primary">
   {browse&&<NavItem target={discover} current={page==='discover'}>Discover</NavItem>}
   {browse&&<NavItem target={saved} current={page==='saved'}><span className="site-long">Reading list</span><span className="site-short">Saved</span> <span className="site-count">{savedCount}</span></NavItem>}
   {classifications&&<NavItem target={classifications} current={page==='classifications'}>Classifications{page!=='classifications'&&<ArrowRight size={14}/>}</NavItem>}
   <NavItem target={{href:'/?view=calibration'}} current={page==='calibration'}>Calibration</NavItem>
   {extra}
  </nav></header></>;
 return <><header className="site-nav"><a href="/" className="site-brand"><span className="site-mark" aria-hidden>OTD</span>Open Text Detector</a>
  <nav aria-label="Primary">
   {browse&&<NavItem target={discover} current={page==='discover'}>Submissions</NavItem>}
   {browse&&<NavItem target={saved} current={page==='saved'}><span className="site-long">Reading list</span><span className="site-short">Saved</span> <span className="site-count">{savedCount}</span></NavItem>}
   {classifications&&<NavItem target={classifications} current={page==='classifications'}>Classifications{page!=='classifications'&&<ArrowRight size={14}/>}</NavItem>}
   <NavMenu href="/?view=transparency" label="Transparency" current={page} items={[['architecture','Architecture','/?view=architecture'],['data','Data','/?view=data'],['calibration','Calibration','/?view=calibration'],['results','Evaluation','/?view=results'],['limitations','Limitations','/?view=limitations']]}/>
   <span className="site-wide-only"><NavItem target={{href:'/?view=manifesto'}} current={page==='manifesto'}>Manifesto</NavItem></span>
   <span className="site-wide-only"><NavItem target={{href:'/?view=about'}} current={page==='about'}>About</NavItem></span>
   {extra}
  </nav></header></>;
}

export function SiteFooter({children}:{children?:ReactNode}){
 if(!BETA)return <footer className="site-footer"><span>Open Text Detector</span><span>Metadata comes from OpenReview; inclusion does not imply acceptance. Classifications are model outputs, not conclusions about authorship.{children}</span></footer>;
 return <footer className="site-footer"><span>Open Text Detector</span><span>Open source. Not affiliated with any conference or company. Metadata comes from OpenReview; inclusion does not imply acceptance. Classifications are model outputs, not conclusions about authorship.{children}<a href="/?view=transparency">Transparency</a><a href="/?view=manifesto">Manifesto</a><a href="/?view=about">About</a></span></footer>;
}
