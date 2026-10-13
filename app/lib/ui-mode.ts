/* Site design. The legacy (pre-redesign) view was removed on Oct 13, 2026: every visitor gets the current design, and old
   `?ui=legacy` links or a stored "legacy" preference are ignored (the stored value is cleared). BETA stays exported
   because components still branch on it; it is always true. */
export type UiMode='beta';
export const UI_MODE:UiMode='beta';
export const BETA=true;
if(typeof window!=='undefined'){try{localStorage.removeItem('atlas-ui');}catch{/* storage unavailable */}
 const p=new URLSearchParams(location.search);if(p.has('ui')){p.delete('ui');history.replaceState(null,'',p.size?`/?${p}`:'/');}}
if(typeof document!=='undefined')document.documentElement.classList.add('ui-beta');
