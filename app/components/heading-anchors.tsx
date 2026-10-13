"use client";
/* Section links: every h2/h3 under a page root gets an id from its text and a "#" link shown on hover, so a section can
 * be linked as /?view=data&phase=...#known-limitations (Alice, Oct 13 2026). The link is relative ("#id"), so a copied link
 * keeps the page's current tab. Headings whose text changes with an interactive control (inside .ar-detail), headings
 * that are themselves links, and anything marked data-no-anchor are skipped. Pages render after a lazy load and switch
 * tabs without reloading, so headings are picked up as they appear; the address's #id is scrolled to once it exists. */
import {useEffect,type RefObject} from "react";

const SKIP='.ar-detail, [data-no-anchor]';
export function slug(text:string){return text.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g,'').replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'').slice(0,60).replace(/-+$/,'')||'section';}

export function useHeadingAnchors(root:RefObject<HTMLElement|null>){
 useEffect(()=>{
  const el=root.current;if(!el)return;
  const target=decodeURIComponent(location.hash.slice(1));let settled=false,until=0;
  const stop=()=>{settled=true;};const events=['wheel','touchstart','keydown','mousedown'] as const;
  for(const e of events)addEventListener(e,stop,{passive:true,once:true});
  const add=()=>{
   for(const h of el.querySelectorAll<HTMLElement>('h2, h3')){
    if(h.closest(SKIP))continue;
    let a=h.querySelector<HTMLAnchorElement>(':scope > .hd-anchor');
    if(!a&&h.querySelector('a'))continue;
    const text=(h.textContent||'').replace(/#$/,'').trim();
    if(a&&h.dataset.hdText===text)continue;
    // React can reuse a heading element for another tab's heading, so ids that were made here follow the text.
    if(!h.id||h.dataset.hdAuto){h.removeAttribute('id');let id=slug(text),n=2;const base=id;while(document.getElementById(id))id=`${base}-${n++}`;h.id=id;h.dataset.hdAuto='1';}
    h.dataset.hdText=text;h.classList.add('hd-target');
    if(!a){a=document.createElement('a');a.className='hd-anchor';a.textContent='#';h.appendChild(a);}
    a.href=`#${h.id}`;a.setAttribute('aria-label',`Link to this section: ${text}`);
   }
   // Content that loads later (data fetched by the page) can push the heading down, so it is scrolled to again after
   // each change for a few seconds, until the reader scrolls or types themselves.
   if(target&&!settled){const t=document.getElementById(target);if(t&&el.contains(t)){if(!until)until=performance.now()+4000;setTimeout(()=>{if(!settled&&performance.now()<until)t.scrollIntoView();});}}
  };
  add();
  const watch=new MutationObserver(add);watch.observe(el,{childList:true,subtree:true,characterData:true});
  return ()=>{watch.disconnect();for(const e of events)removeEventListener(e,stop);};
 },[root]);
}
