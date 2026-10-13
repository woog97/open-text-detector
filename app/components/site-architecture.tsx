"use client";
/* How the detector works: text extraction, input format, model architecture and training loss (draft, Oct 11 2026).
   Describes final-moe: the Oct 10 loss-ablation winner (abl-noseg-nomix). The Oct 7 model it replaces is final-moe-oct7. */
import {useEffect,useState,type ReactNode} from 'react';
import './site-architecture.css';

type Phase='extraction'|'input'|'model'|'loss';
const PHASES:[Phase,string,string][]=[
 ['extraction','Text extraction','PDF → clean text'],
 ['input','Input format','text → 510-token windows'],
 ['model','Architecture','window → score per token'],
 ['loss','Training loss','what the model is trained to do'],
];

export function Architecture(){
 const [phase,setPhase]=useState<Phase>(()=>{const p=typeof location!=='undefined'?new URLSearchParams(location.search).get('phase'):null;return PHASES.some(([k])=>k===p)?p as Phase:'model';});
 useEffect(()=>{const p=new URLSearchParams(location.search);p.set('phase',phase);history.replaceState(null,'',`/?${p}`);},[phase]);
 const i=PHASES.findIndex(([k])=>k===phase);
 return <div className="sp-col sp-col-wide ar">
  <section className="sp-hero">
   <h1>How the detector works</h1>
   <p>Four steps turn a PDF into a score for every sentence.</p>
   <p className="sp-fine">This page describes the model that scores every paper on this site, Qwen3.6 35B-A3B MoE · Oct 10.</p>
  </section>
  <nav className="ar-steps" aria-label="Steps">
   {PHASES.map(([k,label,sub],n)=><button key={k} type="button" aria-current={k===phase?'step':undefined} onClick={()=>setPhase(k)}>
    <span className="ar-step-n">{n+1}</span><b>{label}</b><span>{sub}</span>
   </button>)}
  </nav>
  <section className="ar-panel" aria-label={PHASES[i][1]}>
   {phase==='extraction'&&<Extraction/>}
   {phase==='input'&&<InputFormat/>}
   {phase==='model'&&<Model/>}
   {phase==='loss'&&<Loss/>}
  </section>
  <div className="ar-nav">
   {i>0?<button type="button" onClick={()=>setPhase(PHASES[i-1][0])}>← {PHASES[i-1][1]}</button>:<span/>}
   {i<PHASES.length-1&&<button type="button" onClick={()=>setPhase(PHASES[i+1][0])}>{PHASES[i+1][1]} →</button>}
  </div>
 </div>;
}

function Draft({children}:{children:ReactNode}){return <div className="ar-draft"><b>Sketch</b><span>{children}</span></div>;}

/* ---------- 3. Architecture ---------- */

type Part='input'|'backbone'|'lora'|'readout'|'token'|'document';
const PARTS:Record<Part,{title:string;short:string;body:ReactNode}>={
 input:{title:'Input',short:'1,020 tokens: the window, twice',body:<>
  <p>A 510-token window of the paper, fed in two copies back to back (1,020 tokens). No prompt, separator or instruction is added.</p>
  <p>The model reads left to right, so a token in the first copy only sees what came before it. Every token in the second copy has already seen the whole window. Only the second copy is scored. Try it in the demo below.</p>
 </>},
 backbone:{title:'Backbone: Qwen3.6-35B-A3B',short:'40 layers · 35B parameters, ~3B used per token · frozen',body:<>
  <dl className="ar-facts">
   <dt>Layers</dt><dd>40, in 10 blocks of 3 linear-attention layers + 1 full-attention layer</dd>
   <dt>Feed-forward</dt><dd>Mixture of experts: 256 experts per layer; each token uses 8 of them plus 1 shared expert</dd>
   <dt>Size</dt><dd>35B parameters in total, about 3B used for any one token</dd>
   <dt>d_model</dt><dd>2,048</dd>
   <dt>Weights</dt><dd>Unchanged from the public release. Loaded in BF16 and never updated</dd>
  </dl>
 </>},
 lora:{title:'LoRA adapters (trained)',short:'low-rank updates on every linear layer',body:<>
  <p>Each linear layer W gets a low-rank update: it computes Wx + (α/r)·BAx, where A is r × d_in and B is d_out × r. Only A and B are trained.</p>
  <dl className="ar-facts">
   <dt>Rank</dt><dd>128 on every linear layer; 16 on each expert's projections</dd>
   <dt>Scale</dt><dd>alpha 32</dd>
  </dl>
  <p>The base weights are unchanged, so the trained detector is one adapter file loaded on top of the public weights.</p>
 </>},
 readout:{title:'Read-out',short:'keep the 510 second-copy tokens',body:<>
  <p>The final hidden states (d_model = 2,048) of the first copy are discarded; the 510 second-copy hidden states go to the heads.</p>
 </>},
 token:{title:'Token head',short:'p(AI) for every token · used for all site scores',body:<>
  <p>One linear layer (d_model → 2) applied to each token’s final hidden state, then softmax: the probability that the token belongs to AI-written text.</p>
  <p>A sentence's score is the mean of its tokens' probabilities (step 2). Every highlight and flagged share on this site comes from this head.</p>
 </>},
 document:{title:'Document head',short:'p(window contains AI) · training signal',body:<>
  <p>One linear layer (d_model → 2) on the mean of the window’s 510 final hidden states: does this text contain any AI writing?</p>
  <p>It adds a whole-passage training signal. Sentence scores and flags don't use it; the Calibration page shows its curves.</p>
 </>},
};

function Model(){
 const [part,setPart]=useState<Part>('backbone');
 const box=(k:Part,cls='')=><button type="button" className={`ar-box ${cls}`} aria-pressed={k===part} onClick={()=>setPart(k)}><b>{PARTS[k].title}</b><span>{PARTS[k].short}</span></button>;
 return <>
  <p className="sp-lede">A general language model, frozen, with small trained adapters and a one-layer classifier on top. It reads 510 tokens at a time and gives every token a probability of being AI-written.</p>
  <div className="ar-model">
   <div className="ar-diagram" role="group" aria-label="Model parts; select one for details">
    {box('input','is-io')}
    <i className="ar-arrow" aria-hidden/>
    <div className="ar-stack">
     {box('backbone','is-frozen')}
     {box('lora','is-trained')}
    </div>
    <i className="ar-arrow" aria-hidden/>
    {box('readout','is-io')}
    <i className="ar-arrow" aria-hidden/>
    <div className="ar-heads">
     {box('token','is-trained is-main')}
     {box('document','is-trained')}
    </div>
    <p className="ar-key"><span className="is-frozen"/>frozen <span className="is-trained"/>trained</p>
   </div>
   <div className="ar-detail" aria-live="polite"><h3>{PARTS[part].title}</h3>{PARTS[part].body}</div>
  </div>
  <Repeat2Demo/>
  <div className="ar-detail dt-static">
   <h3>How the backbone was chosen</h3>
   <p>The same classifier was trained on six different language models on Oct 3, each on 10% of the training data (ModernBERT-large at full length): two encoders, ModernBERT-large and Ettin 1B, and four decoders, Qwen3.5 4B, Qwen3.5 9B, Gemma 4 12B and Qwen3.6 35B-A3B. The decoders were ahead of the encoders by about 0.16 AUROC on public text and level with them on papers. Qwen3.6 35B-A3B caught the most AI edits inside papers.</p>
   <p>Three finalists were then trained at full length on the same 72,000 windows on Oct 7: Qwen3.6 35B-A3B, Qwen3.5 4B, and Google’s newly released EmbeddingGemma 2, added as a small, fast encoder to test whether a much cheaper model would do. Qwen3.6 35B-A3B led on every held-out set: it caught 70% of AI edits in unseen papers against 61% for Qwen3.5 4B and 33% for EmbeddingGemma 2, and 93% of other models’ text against 53% and 28%. It is a mixture of experts, so only about 3 billion of its 35 billion parameters run for each token, which keeps scoring affordable. Each backbone was trained once. The full comparison tables are on the <a href="/?view=results&tab=runs">Evaluation page under Runs compared</a>.</p>
  </div>
 </>;
}

/** Which tokens a selected token can see, with one copy or two. */
const DEMO=['The','results','were','later','confirmed','by','a','second','team','.'];
function Repeat2Demo(){
 const [repeat,setRepeat]=useState(true),[sel,setSel]=useState(2);
 const n=DEMO.length,seq=repeat?[...DEMO,...DEMO]:DEMO,scored=(j:number)=>repeat?j>=n:true;
 const pick=scored(sel+(repeat?n:0))?sel:2,target=pick+(repeat?n:0);
 const seenWords=new Set(seq.slice(0,target+1).map((_,j)=>j%n));
 return <div className="ar-demo">
  <div className="ar-demo-head">
   <h3>Why the window is fed twice</h3>
   <div role="group" aria-label="Layout" className="sp-chips">
    <button type="button" aria-pressed={!repeat} onClick={()=>setRepeat(false)}>One copy</button>
    <button type="button" aria-pressed={repeat} onClick={()=>setRepeat(true)}>Two copies (used)</button>
   </div>
  </div>
  <p className="sp-fine">Select a word. Shaded words are what the model has read when it scores that word.</p>
  <div className="ar-tokens">
   {seq.map((w,j)=>{const copy=repeat&&j<n?'is-copy1':'';const seen=j<=target;const me=j===target;
    return <button key={j} type="button" className={`ar-tok ${copy} ${seen?'is-seen':''} ${me?'is-me':''}`} disabled={!scored(j)} onClick={()=>setSel(j%n)} aria-label={`${w}${scored(j)?'':' (not scored)'}`}>{w}</button>;})}
  </div>
  <p className="ar-demo-out">{repeat
   ?<>Scoring <b>“{DEMO[pick]}”</b>: the model has read all {n} words of the window ({seenWords.size} of {n}), including the ones after it.</>
   :<>Scoring <b>“{DEMO[pick]}”</b>: the model has read only {pick+1} of {n} words. Attention can’t reach the tokens after it, so a token at the start of a window is scored with almost no context.</>}</p>
 </div>;
}

/* ---------- 1, 2, 4: first sketches ---------- */

type XStep='words'|'boiler'|'rebuild'|'split';
const XSTEPS:Record<XStep,{title:string;short:string;body:ReactNode}>={
 words:{title:'Words and positions',short:'the PDF’s own text layer, no OCR',body:<>
  <p><code>pdftotext</code>, from the open-source PDF library <a href="https://poppler.freedesktop.org/" target="_blank" rel="noreferrer">Poppler</a>, reads every word the PDF already contains, with its page and position.</p>
  <p>The positions are used twice. The clean-up rules below use them to find headers, margin line numbers and page numbers, and to decide which lines to join. The reader uses them to draw each sentence’s highlight on the PDF page. For a paper without saved positions, it matches the text against the PDF’s text layer instead and highlights only passages that match in one place.</p>
  <p>If a PDF has no text layer at all, every page is read with OCR instead (Tesseract, 300 dpi). Characters a font can’t map are dropped and logged.</p>
 </>},
 boiler:{title:'Remove headers and numbering',short:'only when strict conditions hold; numbers in the body are never removed',body:<>
  <p>Each is removed only when all of its conditions hold:</p>
  <dl className="ar-facts">
   <dt>Review header</dt><dd>The exact ICLR, COLM or TMLR template line, in the top 8.5% of the page, on 2 or more pages</dd>
   <dt>Line numbers</dt><dd>A run of 12+ consecutive, evenly spaced numbers in the page margin, on 2 or more pages</dd>
   <dt>Page numbers</dt><dd>A lone number equal to the page index, at the bottom centre, on 2 or more pages</dd>
   <dt>Placeholders</dt><dd>“Anonymous authors / Paper under double-blind review” on page 1; the NeurIPS checklist template text (the answers stay)</dd>
  </dl>
 </>},
 rebuild:{title:'Rebuild the text',short:'join wrapped lines, fix ligatures and hyphens',body:<>
  <dl className="ar-facts">
   <dt>Lines</dt><dd>Wrapped lines of a paragraph are joined with a space only if they are aligned prose lines at normal spacing. Math, tables, code and captions keep their line breaks</dd>
   <dt>Hyphens</dt><dd>A line-end hyphen is removed only if the joined word appears at least twice elsewhere in the paper and isn’t a prefix like non- or self-</dd>
   <dt>Characters</dt><dd>Ligatures (single characters some fonts use for fi, fl, ff, ffi, ffl) split into separate letters: U+FB01 “ﬁ”, one character, becomes “f” + “i”, two characters; invisible characters dropped; letter-spaced small caps rejoined (D IST → DIST)</dd>
   <dt>Order</dt><dd>Left as the PDF gives it, except a prose column interrupted by a side box, or a displaced title</dd>
   <dt>Math</dt><dd>Kept as extracted</dd>
  </dl>
  <p>Every word is either kept or logged as removed with a reason. A paper fails the check if a word goes missing.</p>
 </>},
 split:{title:'Sentences and sections',short:'what gets a score, and what can be left out of counts',body:<>
  <p><b>Sentences.</b> The text is split after . ! or ? followed by a space or the end of the text. Each sentence gets its own score (step 2).</p>
  <p><b>Sections.</b> After scoring, the bibliography is found: a References or Bibliography heading followed by at least 3 citation-like lines in the next 20. Everything after it is the appendix. These sections are still scored; the counting options only decide whether they count toward a paper’s flagged share. The default leaves the bibliography out.</p>
 </>},
};

function Extraction(){
 const [step,setStep]=useState<XStep>('boiler');
 return <>
  <p className="sp-lede">The model reads the PDF’s own text layer after a clean-up step. Each clean-up rule changes the text only when all of its conditions match.</p>
  <div className="ar-model">
   <div className="ar-diagram" role="group" aria-label="Extraction steps; select one for details">
    {(Object.keys(XSTEPS) as XStep[]).map((k,n)=><div key={k} className="ar-xrow">
     {n>0&&<i className="ar-arrow" aria-hidden/>}
     <button type="button" className={`ar-box ${k==='split'?'is-io':''}`} aria-pressed={k===step} onClick={()=>setStep(k)}><b>{XSTEPS[k].title}</b><span>{XSTEPS[k].short}</span></button>
    </div>)}
   </div>
   <div className="ar-detail" aria-live="polite"><h3>{XSTEPS[step].title}</h3>{XSTEPS[step].body}</div>
  </div>
  <CleanupDemo/>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Tables.</b> Tables without ruling lines are often not recognized. Their cells come through as short lines of words and numbers, and long runs of numbers (results tables, hyperparameter lists) are scored like sentences.</li>
    <li><b>Math and code.</b> Display equations, algorithms and pseudocode are kept as extracted. When a font has no Unicode mapping for its symbols, the math comes out garbled.</li>
    <li><b>Reading order.</b> Where a side box or column overlaps body text, the order is left as the PDF gives it and only flagged.</li>
    <li><b>Sections.</b> A bibliography without a References or Bibliography heading, or with unusual formatting, may not be found; its sentences then count toward the paper’s flagged share.</li>
   </ul>
   <p>Earlier ablations suggested these errors have little effect on evaluation results.</p>
  </div>
  <p className="sp-fine">Training text uses the same clean-up for new human paper text. Some older human text (PMC, peS2o, arXiv) comes from other pipelines and is mixed in on purpose, so text format can’t predict the label.</p>
 </>;
}

/** A made-up review-copy page, segment by segment. Each segment is plain text or belongs to one rule. */
type Rule='header'|'anon'|'lineno'|'pagenum'|'join'|'hyphen'|'lig';
const RULES:[Rule,string][]=[['header','Review header'],['anon','Author placeholder'],['lineno','Line numbers'],['pagenum','Page number'],['join','Join lines'],['hyphen','Hyphens'],['lig','Ligatures']];
type Seg=[string,Rule?,string?];  // text, rule it belongs to, replacement when the rule is on
const n=(k:number):Seg=>[String(k).padStart(3,'0')+' ','lineno',''];
/** Page 1 of a made-up ICLR review copy, as the text layer gives it (line by line). */
const PAGE:Seg[]=[
 ['Under review as a conference paper at ICLR 2027','header'],['\n','header',''],
 ['\n'],
 n(0),['SENTENCE-LEVEL DETECTION OF AI EDITS IN PAPERS'],['\n'],
 ['\n'],
 ['Anonymous authors','anon'],['\n','anon',''],
 ['Paper under double-blind review','anon'],['\n','anon',''],
 ['\n'],
 n(1),['ABSTRACT'],['\n'],
 n(2),['Detectors of AI-written text usually score a whole document. We'],['\n','join',' '],
 n(3),['study detectors that score each sentence of a research pa'],['-','hyphen',''],['\n','hyphen',''],
 n(4),['per instead. Existing classi'],['ﬁ','lig','fi'],['ers work well on full documents'],['\n','join',' '],
 n(5),['but miss short edits, and their scores drift with document length.'],['\n','join',' '],
 n(6),['We propose a token-level objective that improves recall at a 1% false-'],['\n','join',''],
 n(7),['positive rate by 9 points on held-out papers.'],['\n'],
 ['\n'],
 n(8),['1 INTRODUCTION'],['\n'],
 n(9),['Language models now help write many submissions. A single rewritten'],['\n','join',' '],
 n(10),['paragraph can change a paper’s claims, so detection has to work at'],['\n','join',' '],
 n(11),['the level of sentences. We train on 72,000 windows and evaluate on'],['\n','join',' '],
 n(12),['three held-out sets (Table 1). The loss we minimize is'],['\n'],
 n(13),['L = Σt ℓ(yt, pt)                                          (1)'],['\n'],
 n(14),['where yt is the label of token t and pt the predicted probability.'],['\n'],
 n(15),['Table 1: Recall at 1% false-positive rate on three held-out sets.'],['\n'],
 ['\n'],
 ['1','pagenum',''],
];

function CleanupDemo(){
 const [on,setOn]=useState<Record<Rule,boolean>>({header:true,anon:true,lineno:true,pagenum:true,join:true,hyphen:true,lig:true});
 const [marks,setMarks]=useState(true);
 const all=RULES.every(([r])=>on[r]);
 return <div className="ar-demo">
  <div className="ar-demo-head">
   <h3>Clean-up on a sample page</h3>
   <label className="ar-check"><input type="checkbox" checked={marks} onChange={e=>setMarks(e.target.checked)}/> Show changes</label>
  </div>
  <p className="sp-fine">Page 1 of a made-up ICLR 2027 review copy, line by line as the PDF’s text layer gives it. Each chip turns one rule on or off. Red text is removed, green text replaces what was there, and ⏎ / ␣ mark a line break that is removed or replaced by a space.</p>
  <div role="group" aria-label="Rules" className="sp-chips">
   <button type="button" aria-pressed={all} onClick={()=>setOn(Object.fromEntries(RULES.map(([r])=>[r,!all])) as Record<Rule,boolean>)}>{all?'All rules':'No rules'}</button>
   {RULES.map(([r,l])=><button key={r} type="button" aria-pressed={on[r]} onClick={()=>setOn({...on,[r]:!on[r]})}>{l}</button>)}
  </div>
  <pre className="ar-page">{PAGE.map(([t,rule,rep],i)=>{
   if(rule==='lig'&&!on[rule])return marks?<mark key={i} className="ar-lig" title="U+FB01: one character that looks like f + i">{t}</mark>:<span key={i}>{t}</span>;
   if(!rule||!on[rule])return <span key={i}>{t}</span>;
   const out=rep??'';
   if(!marks)return <span key={i}>{out}</span>;
   if(out==='')return t==='\n'?<span key={i} className="ar-joined" title="line break removed">⏎</span>:<del key={i} title={RULES.find(x=>x[0]===rule)![1]}>{t}</del>;
   if(t==='\n')return <span key={i} className="ar-joined" title="line break replaced by a space">␣</span>;
   return <ins key={i} title={`${t} → ${out}`}>{out}</ins>;
  })}</pre>
  <ul className="ar-notes">
   <li>“pa-per” loses its hyphen because “paper” appears at least twice elsewhere in the paper.</li>
   <li>“false-positive” keeps its hyphen, since “false-positive” appears unbroken elsewhere in the paper; only the line break is removed.</li>
   <li>The title, the headings, the equation and the table caption keep their own lines: they are not full-width prose lines.</li>
   <li>“classiﬁers” contains the ligature “ﬁ”, one character (U+FB01) that looks the same as “fi”. Turn Ligatures off to see it boxed.</li>
  </ul>
 </div>;
}

type IStep='tok'|'win'|'prob'|'sent'|'cut';
const ISTEPS:Record<IStep,{title:string;short:string;body:ReactNode}>={
 tok:{title:'Tokenize',short:'Qwen tokenizer, no special tokens added',body:<>
  <p>The cleaned text is split into tokens by the backbone’s own tokenizer (vocabulary 248,320). No BOS, EOS, prompt or chat template is added. Each token keeps its character offsets, so scores can be mapped back to sentences and to the PDF.</p>
 </>},
 win:{title:'Cut into windows',short:'510 tokens each, no overlap on the site',body:<>
  <p>The token sequence is cut into windows of 510 tokens, back to back. The last window is moved back so it ends exactly at the paper’s last token, so it overlaps the window before it. A paper shorter than 510 tokens is one window.</p>
  <p>Each window is fed twice, 1,020 tokens in total (step 3), and is scored independently: no token sees text outside its own window.</p>
  <p>The research evaluations use windows every 256 tokens, so most tokens there are covered by two windows. See the demo below.</p>
 </>},
 prob:{title:'Score every token',short:'p(AI) from the token head; overlaps averaged',body:<>
  <p>The token head gives each token a probability of being AI-written (softmax over its two outputs). Where windows overlap, a token’s probabilities from each window are averaged.</p>
 </>},
 sent:{title:'Score every sentence',short:'mean p(AI) of its tokens',body:<>
  <p>A sentence’s score is the mean p(AI) of the tokens that overlap it. Whitespace-only tokens are left out. Sentences with no letters or digits get no score.</p>
  <p>A document-level probability is also computed, from the document head averaged over windows. The site does not show it.</p>
 </>},
 cut:{title:'Apply the cutoff',short:'flag at 0.0211 (0.1% FPR) or 0.00099 (1%)',body:<>
  <dl className="ar-facts">
   <dt>0.1% (default)</dt><dd>Flagged at 0.0211 or above. This cutoff flags 0.1% of sentences in held-out human papers</dd>
   <dt>1%</dt><dd>Flagged at 0.00099 or above</dd>
   <dt>Red</dt><dd>0.38 or above, about 0.015% of human sentences</dd>
   <dt>Calibration</dt><dd>2,400 human papers from 2019–2022, main text only (bibliography and appendices left out, the default); each counting mode has its own cutoffs</dd>
  </dl>
 </>},
};

function InputFormat(){
 const [step,setStep]=useState<IStep>('win');
 return <>
  <p className="sp-lede">A paper is longer than one 510-token window, so it is cut into windows. Every token gets a probability, a sentence’s score is the mean over its tokens, and a calibrated cutoff decides which sentences are flagged.</p>
  <div className="ar-model">
   <div className="ar-diagram" role="group" aria-label="Scoring steps; select one for details">
    {(Object.keys(ISTEPS) as IStep[]).map((k,n)=><div key={k} className="ar-xrow">
     {n>0&&<i className="ar-arrow" aria-hidden/>}
     <button type="button" className="ar-box" aria-pressed={k===step} onClick={()=>setStep(k)}><b>{ISTEPS[k].title}</b><span>{ISTEPS[k].short}</span></button>
    </div>)}
   </div>
   <div className="ar-detail" aria-live="polite"><h3>{ISTEPS[step].title}</h3>{ISTEPS[step].body}</div>
  </div>
  <WindowDemo/>
  <SentenceDemo/>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Context.</b> Each token sees only its own 510-token window. A sentence that crosses a window boundary has its two halves scored with different context.</li>
    <li><b>Document length.</b> On the held-out sets, the 1%-FPR cutoff for human sentences is 0.75–0.8 in documents under 256 tokens and 0.35–0.5 in documents over 510 tokens. The site’s cutoffs are calibrated on full papers, so they would flag more human sentences in short pasted texts.</li>
    <li><b>Control-token strings.</b> Text such as <code>&lt;think&gt;</code> or <code>&lt;|im_start|&gt;</code> written in a paper is read as the model’s control token. About 4.7% of Atlas papers contain one; training text contains none.</li>
    <li><b>No sliding windows on the site.</b> The site cuts papers into windows that don’t overlap, so a token near a window edge is scored with little context on one side. The research evaluations slide the window every 256 tokens and average the overlapping scores. Using non-overlapping windows on the site was an oversight.</li>
    <li><b>No special tokens.</b> Windows are fed with no BOS, EOS or separator tokens, in training and on the site. Tokens marking where a document starts and ends, and where the second copy begins, would tell the model in context whether a window is a whole document or part of one and how long it is. That could improve evaluation performance and reduce the score shift with document length. Not using them was an oversight.</li>
   </ul>
  </div>
 </>;
}

/** Windows over a paper of n tokens, as score_moe.windows: every `stride` tokens, plus one ending at the last token. */
function windowStarts(n:number,stride:number,w=510){
 if(n<=w)return [0];
 const s=new Set<number>();for(let a=0;a<=n-w;a+=stride)s.add(a);s.add(n-w);
 return [...s].sort((a,b)=>a-b);
}
function WindowDemo(){
 const [n,setN]=useState(1700),[stride,setStride]=useState(510);
 const starts=windowStarts(n,stride),cover=new Array(n).fill(0);
 for(const a of starts)for(let i=a;i<Math.min(a+510,n);i++)cover[i]++;
 const twice=cover.filter(c=>c>1).length;
 // Coverage runs for the bar under the windows.
 const runs:[number,number,number][]=[];for(let i=0;i<n;){let j=i;while(j<n&&cover[j]===cover[i])j++;runs.push([i,j,cover[i]]);i=j;}
 const pct=(x:number)=>`${(x/n)*100}%`;
 return <div className="ar-demo">
  <div className="ar-demo-head">
   <h3>Windows over a paper</h3>
   <div role="group" aria-label="Window spacing" className="sp-chips">
    <button type="button" aria-pressed={stride===510} onClick={()=>setStride(510)}>Site: every 510</button>
    <button type="button" aria-pressed={stride===256} onClick={()=>setStride(256)}>Evaluations: every 256</button>
   </div>
  </div>
  <label className="ar-slider">Paper length <input type="range" min={300} max={4000} step={10} value={n} onChange={e=>setN(Number(e.target.value))}/> <b>{n.toLocaleString()} tokens</b></label>
  <div className="ar-wins">
   {starts.map((a,i)=><div key={a} className="ar-win" style={{left:pct(a),width:pct(Math.min(510,n-a))}} title={`Window ${i+1}: tokens ${a+1}–${Math.min(a+510,n)}`}>{i+1}</div>)}
  </div>
  <div className="ar-cover" aria-hidden>{runs.map(([a,b,c])=><i key={a} className={c>1?'is-twice':''} style={{left:pct(a),width:pct(b-a)}}/>)}</div>
  <p className="ar-demo-out">{starts.length} window{starts.length>1?'s':''}; {twice.toLocaleString()} of {n.toLocaleString()} tokens ({Math.round(twice/n*100)}%) are covered by two windows, and their scores are averaged.</p>
 </div>;
}

/** Illustrative token probabilities for three sentences (not real model output). */
const SENTS:{label:string;toks:[string,number][]}[]=[
 {label:'Low',toks:[['We',.0004],[' train',.0006],[' for',.0003],[' 40',.0002],['k',.0004],[' steps',.0005],[' on',.0003],[' 8',.0002],[' GPUs',.0004],['.',.0006]]},
 {label:'Middle',toks:[['These',.06],[' results',.09],[' highlight',.16],[' the',.08],[' importance',.14],[' of',.07],[' careful',.10],[' design',.09],['.',.08]]},
 {label:'High',toks:[['Moreover',.94],[',',.91],[' our',.88],[' framework',.93],[' seamlessly',.97],[' integrates',.95],[' diverse',.92],[' modalities',.90],['.',.89]]},
];
function band(p:number,cut:number){return p>=.38?'is-red':p>=cut?'is-yellow':'is-green';}
function SentenceDemo(){
 const [k,setK]=useState(1),[cut,setCut]=useState(0.021138);
 const toks=SENTS[k].toks,mean=toks.reduce((a,[,p])=>a+p,0)/toks.length;
 return <div className="ar-demo">
  <div className="ar-demo-head">
   <h3>From token scores to a flag</h3>
   <div role="group" aria-label="Cutoff" className="sp-chips">
    <button type="button" aria-pressed={cut>.01} onClick={()=>setCut(0.021138)}>0.1% FPR · 0.0211</button>
    <button type="button" aria-pressed={cut<.01} onClick={()=>setCut(0.000994)}>1% FPR · 0.00099</button>
   </div>
  </div>
  <div role="group" aria-label="Example sentence" className="sp-chips">
   {SENTS.map((x,i)=><button key={x.label} type="button" aria-pressed={i===k} onClick={()=>setK(i)}>{x.label} example</button>)}
  </div>
  <div className="ar-sent">{toks.map(([t,p],i)=><span key={i} className="ar-stok"><b>{t.replace(/^ /,'␣')}</b><i>{p<.01?p.toFixed(4):p.toFixed(3)}</i></span>)}</div>
  <p className="ar-demo-out">Mean p(AI) = <b>{mean<.01?mean.toFixed(4):mean.toFixed(3)}</b> → <span className={`ar-flag ${band(mean,cut)}`}>{mean>=.38?'flagged, red':mean>=cut?'flagged, yellow':'not flagged'}</span></p>
  <p className="sp-fine">Token probabilities here are made up to show the arithmetic. ␣ marks a token that starts with a space.</p>
 </div>;
}

type LStep='token'|'sentence'|'document';
const LSTEPS:Record<LStep,{title:string;short:string;body:ReactNode}>={
 token:{title:'Token term',short:'weight 1 · every token against its label',body:<>
  <p>Cross-entropy between each token’s p(AI) from the token head and its label (AI or human).</p>
  <p>Within a row, the edited passage (the target) and the text around it (the context) each count half: the term is ½ × mean over target tokens + ½ × mean over context tokens. A row with only one of the two uses that one. Rows then count equally in the batch mean.</p>
  <p>Without the split, a one-sentence edit in a 510-token window would be a few percent of its row’s token loss. The Oct 10 ablation found a plain mean over all tokens did as well (within noise).</p>
 </>},
 sentence:{title:'Sentence term',short:'weight 0.2 · each sentence’s mean logit against its label',body:<>
  <p>For every sentence whose tokens all share one label, the token head’s logits are averaged over the sentence and scored with cross-entropy against that label. Sentences that mix AI and human tokens are skipped. Per row, the sentence losses are averaged; then rows are averaged.</p>
  <p>This trains the quantity the site reports: a sentence’s score is the mean of its tokens’ probabilities. Removing the term cost 1.5 points of small-edit recall in the Oct 10 ablation.</p>
 </>},
 document:{title:'Document term',short:'weight 0.1 · does this row contain AI text?',body:<>
  <p>Cross-entropy on the document head (mean of the final hidden states over the row’s non-whitespace tokens). The label is AI if any part of the row is AI-written.</p>
  <p>Document-only rows (5% of the training windows) have no token or sentence labels and train only this term. Every row still counts 1/32 of a step (see the formula).</p>
 </>},
};

function Loss(){
 const [step,setStep]=useState<LStep>('token');
 return <>
  <p className="sp-lede">Training compares the model’s outputs with the true labels at three levels: each token, each sentence and the whole row. Only the LoRA adapters and the heads are updated.</p>
  <div className="ar-formula" aria-label="Loss formula">
   <div><span>L</span><sub>step</sub> = <sup>1</sup>&frasl;<sub>32</sub> &sum;<sub>rows r</sub> [ <span>&#8467;</span><sub>token</sub>(r) + 0.2 <span>&#8467;</span><sub>sentence</sub>(r) + 0.1 <span>&#8467;</span><sub>document</sub>(r) ]</div>
   <small>One optimizer step sees 32 rows. A document-only row has only the document term.</small>
   <details><summary>How the code computes it</summary>
    <p>The loss is computed per micro-batch: the token and sentence terms are averaged over the micro-batch’s rows that have token labels and multiplied by those rows’ share of the micro-batch; the document term is averaged over all its rows. Each micro-batch loss is weighted by its size / 32 before the gradients are summed. Together this equals the sum above. It holds exactly for 71,998 of the 72,000 training rows: 2 paragraph-edit rows have no single-label sentence, so the sentence mean in their micro-batch is over the other rows. (<code>modeling_sweep.py:124–142</code>, <code>train_sweep.py:552–559</code>)</p>
   </details>
  </div>
  <div className="ar-model">
   <div className="ar-diagram" role="group" aria-label="Loss terms; select one for details">
    {(Object.keys(LSTEPS) as LStep[]).map(k=><button key={k} type="button" className={`ar-box`} aria-pressed={k===step} onClick={()=>setStep(k)} style={{marginBottom:8}}><b>{LSTEPS[k].title}</b><span>{LSTEPS[k].short}</span></button>)}
   </div>
   <div className="ar-detail" aria-live="polite"><h3>{LSTEPS[step].title}</h3>{LSTEPS[step].body}</div>
  </div>
  <RowDemo/>
  <div className="ar-demo">
   <h3>Training setup</h3>
   <dl className="sp-dl sp-dl-narrow ar-hp">
    <dt>Data</dt><dd>72,000 distinct windows, one pass, all three terms, two copies per window</dd>
    <dt>Steps</dt><dd>2,250 optimizer steps of 32 rows</dd>
    <dt>Micro-batch</dt><dd>All 32 rows in one forward pass, packed into one sequence; no gradient accumulation; 1 H200</dd>
    <dt>Optimizer</dt><dd>AdamW, 8-bit (bitsandbytes), weight decay 0.01; gradient norm clipped at 1.0</dd>
    <dt>Learning rate</dt><dd>LoRA 1e-4, heads 2e-5; 6% linear warmup, then cosine decay to 0</dd>
    <dt>LoRA</dt><dd>Rank 128 on linear layers, 16 on experts, alpha 32, no dropout</dd>
    <dt>Precision</dt><dd>BF16 weights and compute</dd>
    <dt>Seed</dt><dd>1</dd>
    <dt>Checkpoint</dt><dd>End of training</dd>
    <dt>Divergence guard</dt><dd>On (roll back and skip steps if the loss spikes); no rollbacks in this run</dd>
   </dl>
  </div>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Window length.</b> Training uses 510-token windows only. A model trained on 1,020-token windows with the same rows, extended with their surrounding paper text, matched it on the standard held-out sets (Oct 10, one seed). Whether longer windows help when whole papers are scored, as on this site, has not been measured.</li>
    <li><b>Weight on human text.</b> Across the 72,000 training windows, about 70% of the token, sentence and document loss weight is on human text and 30% on AI text.</li>
    <li><b>Coarse document label.</b> A window with one AI sentence and a fully AI window get the same document label.</li>
    <li><b>Mixed sentences.</b> Sentences that contain both AI and human tokens get no sentence-level loss.</li>
    <li><b>No special tokens.</b> Nothing marks where a document starts or ends, or where the second copy begins (see Input format).</li>
    <li><b>One seed.</b> Each recipe was trained once; differences of about a point between recipes are within noise.</li>
   </ul>
  </div>
  <div className="ar-history">
   <h3>How the recipe changed</h3>
   <p>The first version of this model, trained on Oct 7, 2026, used a longer recipe. It differed from the current one in three ways:</p>
   <ul>
    <li><b>A preliminary training stage.</b> Before the main training, the model spent 375 steps (12,000 windows) learning only one task: estimating what share of the window’s text is AI-written, as one of 15 equal bins (0–7%, 7–13%, … 93–100%). Each window was fed as a single copy.</li>
    <li><b>Two extra outputs.</b> Besides the token and document outputs described above, the model had two more, both read from the last token of the window: the AI-share bin from the preliminary stage (loss weight 0.2), and whether the window is substantially mixed, meaning the smaller of its AI and human parts covers more than 15% of its tokens (loss weight 0.1). Neither output was ever used to score text.</li>
    <li><b>Smaller groups of rows.</b> Each step processed its 32 rows in 4 groups of 8 and combined their gradients. This reduces memory use and does not change what is learned.</li>
   </ul>
   <p>On Oct 10 we trained versions of the model that each left out some of these parts, on the same 72,000 training windows, and compared how many small AI edits each caught in held-out papers at a 1% false-positive rate. The version without the preliminary stage and without the two extra outputs performed best, so it became the current model.</p>
  </div>
 </>;
}

/** Illustrative training rows: sentences of word-tokens with labels (1 AI, 0 human, -1 none) and the target passage. */
type DemoRow={label:string;doc:0|1;docOnly?:boolean;sents:{words:string[];y:-1|0|1;target:boolean}[]};
const ROWS:DemoRow[]=[
 {label:'Small edit',doc:1,sents:[
  {words:['We','evaluate','on','three','benchmarks.'],y:0,target:false},
  {words:['Each','run','uses','one','seed.'],y:0,target:false},
  {words:['This','robust','setup','underscores','the','reliability','of','our','findings.'],y:1,target:true},
  {words:['Results','are','in','Table','2.'],y:0,target:false}]},
 {label:'AI section',doc:1,sents:[
  {words:['Section','4','describes','the','data.'],y:0,target:false},
  {words:['Our','approach','offers','a','comprehensive','framework.'],y:1,target:true},
  {words:['It','seamlessly','integrates','diverse','signals.'],y:1,target:true},
  {words:['This','paves','the','way','for','future','work.'],y:1,target:true}]},
 {label:'Human text',doc:0,sents:[
  {words:['We','train','for','40k','steps.'],y:0,target:true},
  {words:['Learning','rate','is','1e-4.'],y:0,target:true},
  {words:['Batch','size','is','32.'],y:0,target:true}]},
 {label:'Document-only',doc:1,docOnly:true,sents:[
  {words:['Chapter','3','presents','the','survey','results.'],y:-1,target:false},
  {words:['Participants','were','recruited','online.'],y:-1,target:false}]},
];

function RowDemo(){
 const [k,setK]=useState(0),[split,setSplit]=useState(true);
 const row=ROWS[k],toks=row.sents.flatMap((st,si)=>st.words.map(w=>({w,si,y:st.y,target:st.target})));
 const lab=toks.filter(t=>t.y>=0),nT=lab.filter(t=>t.target).length,nC=lab.length-nT,both=nT>0&&nC>0;
 const weight=(t:typeof toks[number])=>t.y<0?0:!split||!both?1/lab.length:t.target?.5/nT:.5/nC;
 const max=Math.max(...toks.map(weight),1e-9);
 const aiShare=lab.length?lab.filter(t=>t.y===1).reduce((a,t)=>a+weight(t),0):0;
 const sentTerms=row.sents.filter(st=>st.y>=0);
 return <div className="ar-demo">
  <div className="ar-demo-head">
   <h3>One training row</h3>
   <div role="group" aria-label="Token averaging" className="sp-chips">
    <button type="button" aria-pressed={split} onClick={()=>setSplit(true)}>½ target + ½ context</button>
    <button type="button" aria-pressed={!split} onClick={()=>setSplit(false)}>Plain mean</button>
   </div>
  </div>
  <div role="group" aria-label="Example row" className="sp-chips">
   {ROWS.map((r,i)=><button key={r.label} type="button" aria-pressed={i===k} onClick={()=>setK(i)}>{r.label}</button>)}
  </div>
  <div className="ar-row">{toks.map((t,i)=><span key={i} className={`ar-rtok ${t.y===1?'is-ai':t.y===0?'is-human':'is-none'} ${t.target?'is-target':''}`}>
   <i style={{height:`${Math.round(weight(t)/max*28)}px`}} title={`token-term weight ${weight(t).toFixed(3)}`}/>
   <b>{t.w}</b>
  </span>)}</div>
  <p className="ar-key"><span className="ar-sw is-ai"/>AI label <span className="ar-sw is-human"/>human label <span className="ar-sw is-none"/>no token label <u>underlined</u> = target passage · bar height = weight in the token term</p>
  <table className="ar-terms"><tbody>
   <tr><th>Token</th><td>{row.docOnly?'Not used: no token labels.':<>{lab.length} labelled tokens{both?<>; target {nT}, context {nC}</>:null}. AI tokens carry <b>{Math.round(aiShare*100)}%</b> of the row’s token-term weight.</>}</td></tr>
   <tr><th>Sentence</th><td>{row.docOnly?'Not used.':<>{sentTerms.length} sentences, each scored on its mean logit: {sentTerms.map((st,i)=><span key={i} className={`ar-flag ${st.y?'is-red':'is-green'}`}>{st.y?'AI':'human'}</span>)}</>}</td></tr>
   <tr><th>Document</th><td>Label <span className={`ar-flag ${row.doc?'is-red':'is-green'}`}>{row.doc?'contains AI':'human'}</span>{row.docOnly?' (the only term this row trains)':''}</td></tr>
  </tbody></table>
  <p className="sp-fine">Rows are made up and shortened; real rows are up to 510 tokens, and real tokens are word pieces.</p>
 </div>;
}
