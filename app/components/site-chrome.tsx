"use client";
/* eslint-disable @next/next/no-html-link-for-pages */
import {useEffect,useRef,useState,type ReactNode} from 'react';
import {ArrowRight,ArrowUpRight,ChevronDown} from 'lucide-react';
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
/** GitHub's mark (Octicons "mark-github", MIT licence), for the Code link in the Transparency menu. */
function GitHubMark(){return <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="currentColor" style={{verticalAlign:'-2px'}}><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>;}
function NavMenu({href,label,current,items}:{href:string;label:string;current:SitePage;items:[SitePage|'code',ReactNode,string,(boolean|'narrow')?][]}){
 const [open,setOpen]=useState(false);const ref=useRef<HTMLDivElement>(null);
 useEffect(()=>{if(!open)return;
  const away=(e:PointerEvent)=>{if(!ref.current?.contains(e.target as Node))setOpen(false);};
  const esc=(e:KeyboardEvent)=>{if(e.key==='Escape')setOpen(false);};
  document.addEventListener('pointerdown',away);document.addEventListener('keydown',esc);
  return()=>{document.removeEventListener('pointerdown',away);document.removeEventListener('keydown',esc);};},[open]);
 const inside=items.some(([k,,,ext])=>k===current&&ext!=='narrow');
 return <div ref={ref} className={`site-menu${open?' is-open':''}${inside?' is-inside':''}`}
  onMouseEnter={()=>{if(matchMedia('(hover:hover)').matches)setOpen(true);}} onMouseLeave={()=>{if(matchMedia('(hover:hover)').matches)setOpen(false);}}
  onBlur={e=>{if(!e.currentTarget.contains(e.relatedTarget as Node))setOpen(false);}}>
  <a href={href} aria-current={current===label.toLowerCase()?'page':undefined}>{label}</a>
  <button type="button" className="site-menu-toggle" aria-expanded={open} aria-label={`${open?'Hide':'Show'} ${label} pages`} onClick={()=>setOpen(o=>!o)}><ChevronDown size={14} aria-hidden/></button>
  {open&&<div className="site-menu-list" role="menu">
   {items.map(([k,name,href,external])=><a key={k} role="menuitem" href={href} aria-current={current===k?'page':undefined}
    {...(external==='narrow'?{className:'site-narrow-only'}:external?{target:'_blank',rel:'noopener noreferrer',className:'site-menu-ext'}:{})}>{name}</a>)}
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
   <NavMenu href="/?view=transparency" label="Transparency" current={page} items={[['architecture','Architecture','/?view=architecture'],['data','Data','/?view=data'],['calibration','Calibration','/?view=calibration'],['results','Evaluation','/?view=results'],['limitations','Limitations','/?view=limitations'],
    ['manifesto','Manifesto','/?view=manifesto','narrow'],['about','About','/?view=about','narrow'],
    ['code',<><GitHubMark/> Code <ArrowUpRight size={12} aria-hidden/></>,'https://github.com/woog97/open-text-detector',true]]}/>
   <span className="site-wide-only"><NavItem target={{href:'/?view=manifesto'}} current={page==='manifesto'}>Manifesto</NavItem></span>
   <span className="site-wide-only"><NavItem target={{href:'/?view=about'}} current={page==='about'}>About</NavItem></span>
   {extra}
  </nav></header></>;
}

export function SiteFooter({children}:{children?:ReactNode}){
 if(!BETA)return <footer className="site-footer"><span>Open Text Detector</span><span>Metadata comes from OpenReview; inclusion does not imply acceptance. Classifications are model outputs, not conclusions about authorship.{children}</span></footer>;
 return <footer className="site-footer"><span>Open Text Detector</span><span>Open source. Not affiliated with any conference or company. Metadata comes from OpenReview; inclusion does not imply acceptance. Classifications are model outputs, not conclusions about authorship.{children}<a href="/?view=transparency">Transparency</a><a href="/?view=manifesto">Manifesto</a><a href="/?view=about">About</a></span></footer>;
}
