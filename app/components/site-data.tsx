"use client";
/* Where the training and evaluation text comes from and how the AI text was made (draft, Oct 11 2026).
   Facts and sources: research/site-architecture-20261011/DATA_FACTS.md. Styles shared with site-architecture.css. */
import {useEffect,useState,type ReactNode} from 'react';
import './site-architecture.css';
import './site-data.css';
import {PROMPTS} from './ai-prompts';

type Phase='human'|'ai'|'mix'|'eval'|'cal';
const PHASES:[Phase,string,string][]=[
 ['human','Human text','where the human data comes from'],
 ['ai','AI text','how the AI data is generated'],
 ['mix','Training mix','the 72,000 training windows'],
 ['eval','Evaluation sets','held-out text the scores are measured on'],
 ['cal','Calibration','the human papers that set the cutoffs'],
];

export function DataPage(){
 const [phase,setPhase]=useState<Phase>(()=>{const p=typeof location!=='undefined'?new URLSearchParams(location.search).get('phase'):null;return PHASES.some(([k])=>k===p)?p as Phase:'eval';});
 useEffect(()=>{const p=new URLSearchParams(location.search);p.set('phase',phase);history.replaceState(null,'',`/?${p}`);},[phase]);
 const i=PHASES.findIndex(([k])=>k===phase);
 return <div className="sp-col sp-col-wide ar">
  <section className="sp-hero">
   <h1>Training and evaluation data</h1>
   <p>Where the human and AI text comes from, how it was combined for training, and what the model is tested on.</p>
  </section>
  <nav className="ar-steps dt-steps" aria-label="Sections">
   {PHASES.map(([k,label,sub],n)=><button key={k} type="button" aria-current={k===phase?'step':undefined} onClick={()=>setPhase(k)}>
    <span className="ar-step-n">{n+1}</span><b>{label}</b><span>{sub}</span>
   </button>)}
  </nav>
  <section className="ar-panel" aria-label={PHASES[i][1]}>
   {phase==='human'&&<HumanText/>}
   {phase==='ai'&&<AiText/>}
   {phase==='mix'&&<TrainingMix/>}
   {phase==='eval'&&<EvalSets/>}
   {phase==='cal'&&<Calibration/>}
  </section>
  <div className="ar-nav">
   {i>0?<button type="button" onClick={()=>setPhase(PHASES[i-1][0])}>← {PHASES[i-1][1]}</button>:<span/>}
   {i<PHASES.length-1&&<button type="button" onClick={()=>setPhase(PHASES[i+1][0])}>{PHASES[i+1][1]} →</button>}
  </div>
 </div>;
}


/* ---------- Evaluation sets ---------- */

/** One edited passage and its untouched control, for each kind of edit (made-up text). */
type Pair={label:string;before:string;ai:string;after:string;original:string};
const PAIRS:Pair[]=[
 {label:'Rewrite one sentence',before:'We train on 12 languages. ',ai:'This diverse linguistic coverage underscores the robustness of our approach. ',after:'Results are in Table 3.',original:'Most of them have little web text. '},
 {label:'Rewrite two sentences',before:'We train on 12 languages. ',ai:'This broad coverage enables a comprehensive assessment. It further highlights the generality of the method. ',after:'Results are in Table 3.',original:'Most of them have little web text. We report each one separately. '},
 {label:'Insert one sentence',before:'We train on 12 languages. Most of them have little web text. ',ai:'This setting closely reflects real-world deployment conditions. ',after:'Results are in Table 3.',original:''},
 {label:'Split one sentence',before:'We train on 12 languages. ',ai:'Most of these languages have little web text. As a result, we report each one separately. ',after:'Results are in Table 3.',original:'Most of them have little web text, so we report each one separately. '},
 {label:'Write a paragraph',before:'Section 4 describes the data. ',ai:'Our evaluation spans a diverse set of benchmarks designed to probe both accuracy and robustness. Each benchmark targets a distinct capability, ensuring a holistic view of model behavior. ',after:'Section 5 reports results.',original:'We use three benchmarks: two for accuracy and one for robustness to typos. '},
];
function PairDemo(){
 const [k,setK]=useState(0);const p=PAIRS[k];
 return <div className="ar-demo">
  <div className="ar-demo-head"><h3>Every edit has a control</h3></div>
  <p className="sp-fine">Each AI-edited passage is scored next to the same passage with the original human text put back. Recall is measured on the AI sentences; false positives on the control’s sentences. Text below is made up.</p>
  <div role="group" aria-label="Kind of edit" className="sp-chips">{PAIRS.map((x,i)=><button key={x.label} type="button" aria-pressed={i===k} onClick={()=>setK(i)}>{x.label}</button>)}</div>
  <div className="dt-pair">
   <div><h4>Edited (scored for recall)</h4><p>{p.before}<span className="dt-ai">{p.ai}</span>{p.after}</p></div>
   <div><h4>Control (scored for false positives)</h4><p>{p.before}{p.original?<span className="dt-human">{p.original}</span>:<span className="dt-empty">(nothing inserted)</span>}{p.original?'':' '}{p.after}</p></div>
  </div>
  <p className="ar-key"><span className="ar-sw is-ai"/>AI-written <span className="ar-sw is-human"/>original human text at the same spot</p>
 </div>;
}

type Tag='paper'|'writer'|'format'|'context'|'stress'|'domain'|'screen';
const TAGS:[Tag,string][]=[['paper','Unseen papers'],['writer','Unseen writers'],['format','Site text format'],['context','Full-paper context'],['stress','Stress tests'],['domain','Other domains'],['screen','Screening only']];
type EvalSet={name:string;id:string;tags:Tag[];rows:string;sents:string;writers:string;edits:string;heldout:string;notes:ReactNode};
const SETS:EvalSet[]=[
 {name:'Held-out papers',id:'heldout-writers',tags:['paper'],rows:'2,342 rows · 313 papers',sents:'1,590 AI · 16,591 control',
  writers:'Claude Opus 5.5, Sonnet 5.5, Haiku 4.5; GPT-6 Luna (all also used in training)',edits:'Rewrite one or two sentences, insert one, split one',
  heldout:'Papers and paragraphs never used in training',
  notes:<>Passages are three consecutive paragraphs, mostly from NeurIPS, ICML and ACL papers from 2013–2022; the middle one is edited. Edits that change anything besides the named sentence are rejected. One control per passage, shared by the writers that edited it.</>},
 {name:'Held-out papers and unseen writers',id:'t21-heldout-v2',tags:['paper','writer'],rows:'8,000 rows · 2,017 papers',sents:'4,086 small-edit AI · 76,504 control',
  writers:'Claude Fable 5.1 and GPT-6 Sol (never trained); Claude Opus, Sonnet, Haiku; GPT-6.1 Sol',edits:'Small edits; paragraph written from its neighbours or rewritten; section written or rewritten',
  heldout:'15% of the papers with Claude Opus or Sonnet edits, chosen at random, and every paper with a Fable 5.1 or GPT-6 Sol edit',
  notes:<>Human text from PubMed Central, peS2o, arXiv, ACL and ML conference papers dated 2022 or earlier. Every row on a held-out paper is kept out of training, matched by paper id and by hashed text, so re-hosted copies are caught too.</>},
 {name:'Site text format and unseen writers',id:'wide1-heldout',tags:['paper','writer','format'],rows:'3,142 rows · 1,542 papers',sents:'1,567 small-edit AI · 16,974 control',
  writers:'GLM 5.3 Flash and MiMo V2.6 Flash (never trained); DeepSeek V4.1 Flash (15% of its papers held out)',edits:'Small edits; paragraph written or rewritten',
  heldout:'All GLM and MiMo papers; a fixed 15% of DeepSeek papers',
  notes:<>The only set extracted with the same clean-up as the site (step 1 on the Architecture page): papers from OpenReview and other sources dated 2022 or earlier. Every small edit has its own control.</>},
 {name:'Full-paper context versions',id:'*-lc',tags:['paper','context'],rows:'Same rows as the three sets above',sents:'Same 7,243 small-edit AI sentences',
  writers:'As above',edits:'As above',heldout:'As above',
  notes:<>Each passage is placed back inside up to 1,000 tokens of its real paper; the edit and its control get identical added context. Tests how the model behaves on surrounding paper text: headings, captions, references and math. 75–100% of rows could be extended, depending on the set.</>},
 {name:'Light polish',id:'polish-eval-v1',tags:['writer','stress'],rows:'598 rows · 299 papers',sents:'Token-level labels; 3,959 control',
  writers:'Claude Fable 5.1',edits:'Light polishing of a whole paragraph: about 9% of words changed (median)',heldout:'Writer never trained',
  notes:<>Labels come from a word-level diff, so most sentences are part human, part AI. Measured with a document-level score, since sentence recall is undefined.</>},
 {name:'Humanizer',id:'humanizer-eval-v1',tags:['writer','stress'],rows:'600 rows · 61 papers',sents:'10,583 AI · 12,841 control',
  writers:'Claude Fable 5.1 rewriting Claude-written paragraphs',edits:'An AI paragraph rewritten to sound human, inside its human section',heldout:'Writer never trained; held-out sections',
  notes:<>Each humanized paragraph is scored next to the same paragraph before humanizing and the human original.</>},
 {name:'Co-writing',id:'cowrite-eval-v1',tags:['writer','stress'],rows:'200 rows',sents:'339 AI · 1,824 control',
  writers:'Claude Fable 5.1',edits:'Three rounds of co-writing: human, AI and AI-edited segments mixed',heldout:'Writer never trained',
  notes:<>Human segments must appear word for word in the original; the AI share of each passage is 10–90%.</>},
 {name:'Other domains',id:'cross-model-s500',tags:['domain'],rows:'996 rows',sents:'8,439 AI · 22,477 control',
  writers:'Claude Opus 5.5, Sonnet 5.5, Haiku 4.5; GPT-6 Luna, GPT-6.1 Sol; 100 documents from a model labelled “Opus 3” in a browser interface (not verified)',edits:'Multi-paragraph spans expanded from a short brief and spliced back into the text',heldout:'New kinds of text',
  notes:<>Fiction (Project Gutenberg, Standard Ebooks), Federal Reserve Beige Books, Wikipedia and JMLR papers up to 2014. Controls are other human passages, not the originals of the edited ones.</>},
 {name:'Screening set',id:'sweep-eval-rows',tags:['paper','screen'],rows:'3,511 rows',sents:'339 small-edit AI · 13,139 human',
  writers:'GPT-6 Luna; public detector datasets',edits:'Missing sentence, two sentences or paragraph rewritten from notes',heldout:'Papers never trained',
  notes:<>Used to compare checkpoints during training. Too few small edits for headline numbers.</>},
];

function EvalSets(){
 const [tag,setTag]=useState<Tag|'all'>('all');const [open,setOpen]=useState<string|null>('heldout-writers');
 const shown=SETS.filter(s=>tag==='all'||s.tags.includes(tag));
 return <>
  <p className="sp-lede">The detector has two goals: catch light AI editing in a research paper, and catch papers written mostly or entirely by AI with little human oversight, in both cases without falsely flagging human writing. The evaluation so far is built around the first. Headline numbers are small-edit recall at a 1% false-positive rate, pooled over the first three sets below: 7,243 AI-written sentences in held-out papers.</p>
  <div className="ar-demo">
   <h3>How the evaluation is designed</h3>
   <ol className="dt-principles">
    <li><b>Fix the false-positive rate first, then measure recall.</b> A false flag on a human author is the costly mistake, so the cutoff is set where 1% of human sentences are flagged, and the score is the share of AI sentences caught at that cutoff. A fixed cutoff of 0.5 would mostly measure calibration. AUROC is reported next to it as a measure that needs no cutoff.</li>
    <li><b>Make the hardest realistic case the headline.</b> Whole AI documents are comparatively easy; the open problem is an author who has AI rewrite or add one or two sentences. The headline counts only those small-edit sentences. Paragraphs, sections and whole documents are reported separately.</li>
    <li><b>Compare each edit with its own original.</b> Every edited passage is scored next to the same passage with the human text put back, and each set’s cutoff is set on its own controls. Otherwise a model could score well by noticing topic, venue or formatting instead of the writing; with matched controls, the only difference between the two is the edit.</li>
    <li><b>Hold out one thing per set.</b> Each set changes one condition from training: new papers, new writing models, the site’s text format, full-paper context, other kinds of writing, or writing deliberately disguised. When a score drops, the set says which change caused it. The filters below group the sets this way.</li>
    <li><b>Use enough data to tell models apart.</b> The headline pools 7,243 sentences from three sets, intervals resample whole papers, and two models are compared on the same sentences. The 339-sentence screening set is used only to compare checkpoints during training: on it, one model scored 51% on one half and 63% on the other.</li>
    <li><b>Keep evaluation text out of training</b> by paper, by text and by writing model (details below).</li>
   </ol>
  </div>
  <PairDemo/>
  <div className="ar-demo">
   <div className="ar-demo-head"><h3>The sets</h3></div>
   <div role="group" aria-label="Filter sets" className="sp-chips">
    <button type="button" aria-pressed={tag==='all'} onClick={()=>setTag('all')}>All · {SETS.length}</button>
    {TAGS.map(([t,l])=><button key={t} type="button" aria-pressed={tag===t} onClick={()=>setTag(t)}>{l} · {SETS.filter(s=>s.tags.includes(t)).length}</button>)}
   </div>
   <div className="dt-sets">{shown.map(s=><div key={s.id} className={`dt-set ${open===s.id?'is-open':''}`}>
    <button type="button" aria-expanded={open===s.id} onClick={()=>setOpen(open===s.id?null:s.id)}>
     <b>{s.name}</b><span>{s.rows}</span><span>{s.sents}</span>
    </button>
    {open===s.id&&<div className="dt-set-body">
     <dl className="ar-facts">
      <dt>Held out</dt><dd>{s.heldout}</dd>
      <dt>Writers</dt><dd>{s.writers}</dd>
      <dt>Edits</dt><dd>{s.edits}</dd>
      <dt>Sentences</dt><dd>{s.sents}</dd>
     </dl>
     <p>{s.notes}</p>
     <p className="sp-fine">File: <code>{s.id}</code></p>
    </div>}
   </div>)}</div>
  </div>
  <div className="ar-limits">
   <h3>Keeping evaluation text out of training</h3>
   <ul>
    <li><b>Papers.</b> Every paper in an evaluation set is excluded from training by paper id and by hashes of its paragraphs or of 60-character pieces of its text, which also catches the same text under another id.</li>
    <li><b>Sentences.</b> A training row is dropped if it shares any sentence of 10 or more words with the screening, checkpoint-selection or calibration text.</li>
    <li><b>Writers.</b> Claude Fable 5.1, GPT-6 Sol, GLM 5.3 Flash and MiMo V2.6 Flash never write training text.</li>
   </ul>
  </div>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Whole AI-written papers are not measured.</b> No evaluation set scores complete AI-generated papers against human papers the way the site scores them. Held-out AI papers exist (120 synthetic manuscripts and 44 Claude-written papers set aside during generation) but are not yet part of the evaluation.</li>
    <li><b>Research-paper text only for the headline.</b> The headline sets are all research papers; the other-domains set is small.</li>
    <li><b>Writers overlap with training</b> in the held-out-papers set and the other-domains set, so those measure new text, not new models.</li>
    <li><b>Most edits are in short passages.</b> Without the full-paper versions, passages are a few hundred tokens, shorter than the site’s 510-token windows over whole papers.</li>
    <li><b>Some “Claude Fable 5.1” edits were probably written by Claude Opus 5.</b> When a safety classifier stopped a writer agent’s response, the agent continued on Claude Opus 5. This happened in 6 Fable 5.1 batches, which supply 299 of the 1,366 Fable 5.1 rows in the held-out-papers-and-unseen-writers set. Opus 5 also never wrote training text, so these rows still test an unseen writer, but per-writer results for Fable 5.1 on that set are partly Opus 5.</li>
    <li><b>No ground truth for the papers on this site.</b> The ICLR 2027 submissions shown here have no known labels; only the sets above are measured.</li>
   </ul>
  </div>
 </>;
}

/* ---------- Calibration ---------- */

function Calibration(){
 return <>
  <p className="sp-lede">The cutoffs that decide which sentences are flagged are set on 2,400 papers written before AI writing tools were common, so every flag in them counts as a false positive.</p>
  <div className="ar-model">
   <div className="ar-detail dt-static">
    <h3>The papers</h3>
    <dl className="ar-facts">
     <dt>Conference papers</dt><dd>2,188 ICLR 2019–2022, NeurIPS 2021–2022 and CoRL 2021–2022 papers, sampled by venue and year from the PDFs on this site</dd>
     <dt>Archive papers</dt><dd>212 papers from 2020–2022 from an earlier collection</dd>
     <dt>Removed</dt><dd>111 papers that share text with a scored model’s training data; 3 archive papers labelled 2022 whose text cites work from 2023–2025</dd>
     <dt>Sentences</dt><dd>638,746 in the main text, with the bibliography and appendices left out (the default counting)</dd>
    </dl>
    <p>The project treats papers from 2022 or earlier as human-written.</p>
   </div>
   <div className="ar-detail dt-static">
    <h3>Setting and checking a cutoff</h3>
    <p>Every paper is scored the same way as the papers on this site. For a target rate, the cutoff is the score above which that share of the human sentences falls: 0.0211 for 0.1%, 0.00099 for 1% (main text only, the default). Each counting mode gets its own cutoffs.</p>
    <p>To check a cutoff, it is set on a random half of the papers and the flag rate is measured on the other half. At the 0.1% cutoff the other half’s rate is 0.12% (95% interval 0.09–0.15%, 1,226 papers). Intervals resample whole papers.</p>
   </div>
  </div>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Mostly machine-learning papers.</b> Nearly all calibration papers are ICLR, NeurIPS and CoRL papers. Human writing in other fields may be flagged at a different rate.</li>
    <li><b>Same window setup as the site.</b> Calibration papers are scored with non-overlapping windows, like the site, and not with the sliding windows used in the research evaluations.</li>
    <li><b>Full papers only.</b> The cutoffs fit full papers; on short texts the same cutoff flags more human sentences (see Input format on the Architecture page).</li>
   </ul>
  </div>
 </>;
}

/* ---------- Pie chart (validated categorical palette, fixed order) ---------- */
const SERIES=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948'];
type Slice={label:string;value:number;note?:string};
function Pie({title,items,unit,selected,onSelect}:{title:string;items:Slice[];unit:string;selected?:number;onSelect?:(i:number)=>void}){
 const [hover,setHover]=useState<number|null>(null);
 const sum=items.reduce((a,x)=>a+x.value,0),R=80,C=90;
 let a0=-Math.PI/2;
 const arcs=items.map(x=>{const a1=a0+x.value/sum*2*Math.PI,large=a1-a0>Math.PI?1:0;
  const d=`M${C},${C} L${C+R*Math.cos(a0)},${C+R*Math.sin(a0)} A${R},${R} 0 ${large} 1 ${C+R*Math.cos(a1)},${C+R*Math.sin(a1)} Z`;a0=a1;return d;});
 const focus=hover??selected??null;
 const pct=(v:number)=>`${(v/sum*100).toFixed(1)}%`;
 return <figure className="dt-pie">
  <figcaption>{title}</figcaption>
  <div className="dt-pie-body">
   <svg viewBox="0 0 180 180" role="img" aria-label={title}>
    {arcs.map((d,i)=><path key={i} d={d} fill={SERIES[i%SERIES.length]} stroke="#fff" strokeWidth={2} strokeLinejoin="round"
      opacity={focus===null||focus===i?1:.35} onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)} onClick={()=>onSelect?.(i)} style={{cursor:onSelect?'pointer':'default'}}>
     <title>{`${items[i].label}: ${items[i].value.toLocaleString()} ${unit} (${pct(items[i].value)})${items[i].note?` · ${items[i].note}`:''}`}</title></path>)}
   </svg>
   <table className="dt-legend"><tbody>{items.map((x,i)=><tr key={x.label} className={focus===i?'is-focus':''} onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)} onClick={()=>onSelect?.(i)} style={{cursor:onSelect?'pointer':'default'}}>
    <td><span className="dt-sw" style={{background:SERIES[i%SERIES.length]}}/></td><td>{x.label}{x.note&&<small className="dt-note">{x.note}</small>}</td><td className="dt-num">{x.value.toLocaleString()}</td><td className="dt-num">{pct(x.value)}</td>
   </tr>)}</tbody></table>
  </div>
 </figure>;
}

/* ---------- Human text ---------- */

const OTHER_HUMAN:[string,number,string][]=[
 ['Creative writing',1104,'Gutenberg, WritingPrompts, SCP, Wikisource'],
 ['Scientific',946,'arXiv, PubMed Central, peS2o, ACL'],
 ['Reference',744,'Wikipedia, Wikibooks, Pressbooks, LibreTexts'],
 ['Social',684,'Stack Exchange, Ubuntu IRC, Wikipedia talk'],
 ['General web',549,'CCCC, Foodista, Wikivoyage, EFF'],
 ['Reviews',461,'Amazon, IMDb, OpinRank'],
 ['News',428,'VOA, Global Voices, Slate, Wikinews, SciDev'],
 ['Professional',184,'GovReport, World Bank, Federal Reserve, OANC'],
];
function HumanText(){
 return <>
  <p className="sp-lede">All human text is dated 2022 or earlier, before AI writing tools were in wide use. The project treats such text as human-written. The one exception is GRADTEX, a public dataset whose documents come with their own human or AI labels (see below).</p>
  <div className="dt-sources">
   <div className="ar-detail dt-static">
    <h3>Research papers</h3>
    <dl className="ar-facts">
     <dt>Site format</dt><dd>ICLR and NeurIPS 2021–2022 papers, extracted with the same clean-up as the site: a pool of 10,413 papers</dd>
     <dt>Older format</dt><dd>ACL, ICML and NeurIPS papers from 2013–2021, extracted from PDF with a simpler method</dd>
     <dt>Passages</dt><dd>Scientific paragraphs from PubMed Central, peS2o, arXiv and ACL, dated 2022 or earlier</dd>
     <dt>Archive</dt><dd>Papers dated 2022 or earlier from the project’s paper archive, used as hosts for some small edits and sections (132 human-only windows)</dd>
    </dl>
    <p>Most AI edits are written into these papers, so the same papers also provide the human context around each edit.</p>
   </div>
   <div className="ar-detail dt-static">
    <h3>Other human writing</h3>
    <p>32 sources across fiction, reference works, news, forums, reviews, textbooks and government reports, for example Project Gutenberg, Wikipedia, Wikinews, Stack Exchange, IMDb and Amazon reviews, and Federal Reserve and World Bank reports. At most 2 windows are taken per document.</p>
    <p>These give the model human examples outside research papers, so an unfamiliar style is not on its own a sign of AI.</p>
   </div>
  </div>
  <div className="ar-demo">
   <h3>Human-only training windows</h3>
   <p className="dt-p">24,904 of the 72,000 training windows contain no AI text. 1,804 of them are human documents from GRADTEX, a public dataset derived from MAGE, labelled human by that dataset; like GRADTEX’s AI documents, they carry only a document label and train only the document output. Other human writing is 5,100 windows: 20.5% of the human-only windows and 7.1% of all training windows.</p>
   <div className="dt-pies">
    <Pie title="Human-only windows by source (24,904)" unit="windows" items={[
     {label:'Untouched paper text',value:9000},{label:'Human version of an AI section',value:3000},{label:'Human version of an AI paragraph',value:3000},
     {label:'Other paragraphs of edited papers',value:3000},{label:'Other human writing',value:5100},{label:'GRADTEX human documents',value:1804,note:'document label only'}]}/>
    <Pie title="Other human writing by category (5,100)" unit="windows" items={OTHER_HUMAN.map(([l,v,n])=>({label:l,value:v,note:n}))}/>
   </div>
   <p className="sp-fine">The categories of other human writing are modeled after the <a href="https://arxiv.org/abs/2607.27183" target="_blank" rel="noreferrer">Pangram 4 technical report</a>.</p>
  </div>
  <div className="ar-demo">
   <h3>Matching text formats</h3>
   <p className="dt-p">Text extracted in different ways looks different: line breaks, hyphenation, leftover headers. If AI rows came mostly in one format and human rows in another, the model could learn the format instead of the writing. So human windows are drawn from each format roughly in proportion to the AI rows that use it, and the AI share is checked per format.</p>
   <p className="sp-fine">In the final mix the AI share is 0.51 for text in the site’s format and 0.69–0.78 for the older formats. The site’s format has relatively more human text.</p>
  </div>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Formats are not fully balanced.</b> The AI share still differs by format (0.51 versus 0.69–0.78).</li>
    <li><b>Mostly machine learning.</b> Research-paper text comes mainly from machine-learning and computational-linguistics venues.</li>
   </ul>
  </div>
 </>;
}

/* ---------- AI text ---------- */

type Gen='small'|'para'|'section'|'full';
const GENS:Record<Gen,{title:string;short:string;body:ReactNode}>={
 small:{title:'Small edits',short:'one or two sentences inside a human paragraph',body:<>
  <p>A writing model changes one or two sentences of a human paragraph and must copy every other character unchanged. Only the new sentences are labelled AI. Four instructions, in the ratio 30 / 30 / 30 / 10:</p>
  <dl className="ar-facts">
   <dt>Rewrite one</dt><dd>Reword the named sentence noticeably, keeping its meaning</dd>
   <dt>Rewrite two</dt><dd>The same for two adjacent sentences</dd>
   <dt>Insert one</dt><dd>Add one plausible sentence after the named sentence</dd>
   <dt>Split one</dt><dd>Split the named sentence into two</dd>
  </dl>
  <h4 className="dt-h4">How each writer’s edits were made</h4>
  <p className="sp-fine">The differences between writers come from when the data was made: GPT-6 Luna’s text (Sep 30–Oct 5) used earlier pipelines; Claude, GPT-6.1 Sol and DeepSeek (Oct 6–7) used a later one with stricter checks.</p>
  <div className="sp-scroll"><table className="sp-table">
   <thead><tr><th>Writer</th><th>Rows</th><th>Made by</th><th>Sees the paragraphs around it</th><th>Rejected if</th></tr></thead>
   <tbody>
    <tr><td>Claude Opus 5.5, Sonnet 5.5, Haiku 4.5</td><td>13,334</td><td rowSpan={3}>Prompted edit</td><td rowSpan={3}>Yes</td><td rowSpan={3}>Anything besides the named sentences changed; the wrong number of sentences; or a rewrite more than 90% similar to the original</td></tr>
    <tr><td>DeepSeek V4.1 Flash</td><td>3,356</td></tr>
    <tr><td>GPT-6.1 Sol</td><td>378</td></tr>
    <tr><td>GPT-6 Luna</td><td>2,615</td><td>Prompted edit</td><td>No</td><td>More than one changed block, or a block of the wrong size. The change does not have to be on the named sentence.</td></tr>
    <tr><td>GPT-6 Luna</td><td>5,517</td><td>Splice: no model call. One or two sentences from a GPT-6 Luna paragraph draft (see Paragraph edits) replace the matching human sentences.</td><td>Not applicable</td><td>The two sentences are under 25% or over 85% similar, either has fewer than 6 words, or their lengths differ by more than 0.6–1.6×</td></tr>
   </tbody>
  </table></div>
  <h4 className="dt-h4">Rows by kind of edit</h4>
  <div className="sp-scroll"><table className="sp-table dt-num">
   <thead><tr><th>Writer</th><th>Rewrite one</th><th>Rewrite two</th><th>Insert one</th><th>Split one</th><th>Splice one</th><th>Splice two</th><th>Total</th></tr></thead>
   <tbody>
    <tr><td>Claude</td><td>4,058</td><td>3,978</td><td>4,001</td><td>1,297</td><td>–</td><td>–</td><td>13,334</td></tr>
    <tr><td>GPT-6 Luna</td><td>785</td><td>725</td><td>831</td><td>265</td><td>4,019</td><td>1,498</td><td>8,132*</td></tr>
    <tr><td>DeepSeek</td><td>1,032</td><td>998</td><td>1,001</td><td>325</td><td>–</td><td>–</td><td>3,356</td></tr>
    <tr><td>GPT-6.1 Sol</td><td>114</td><td>116</td><td>113</td><td>35</td><td>–</td><td>–</td><td>378</td></tr>
   </tbody>
  </table></div>
  <p className="sp-fine">* Includes 9 rows from an early trial that merged two sentences into one.</p>
 </>},
 para:{title:'Paragraph edits',short:'a whole paragraph written or rewritten',body:<>
  <p>A writing model replaces one whole paragraph of a human text. The paragraph is labelled AI; the text around it stays human. Each one also comes with the original human paragraph, in the same context, as a separate human row.</p>
  <p className="sp-fine">The differences between writers come from when the data was made: GPT-6 Luna’s text (Sep 30–Oct 5) used earlier pipelines; Claude, GPT-6.1 Sol and DeepSeek (Oct 6–7) used a later one with stricter checks.</p>
  <div className="dt-cards">
   <div className="dt-card">
    <h4>Draft or rewrite</h4>
    <p className="sp-fine">Claude Opus 5.5, Sonnet 5.5, Haiku 4.5, GPT-6.1 Sol and DeepSeek V4.1 Flash · 4,315 rows</p>
    <ul>
     <li><b>Draft</b> (half): sees the paragraphs before and after a gap and writes the missing one, without seeing the original.</li>
     <li><b>Rewrite</b> (half): polishes the original paragraph, keeping its content, numbers and citations.</li>
    </ul>
    <p><b>Rejected if</b> it is not one paragraph, adds markdown or commentary, is more than 60% similar to a neighbouring paragraph, is under 0.5× or over 1.8× the original’s length, or is more than 90% similar to the original. A rewrite must also change at least 12%.</p>
   </div>
   <div className="dt-card">
    <h4>Draft from notes</h4>
    <p className="sp-fine">GPT-6 Luna · 4,385 rows</p>
    <ol>
     <li><b>Notes:</b> a call that sees the hidden paragraph writes detailed notes on its claims, qualifications and citations.</li>
     <li><b>Write:</b> a second call writes the paragraph from the notes, the abstract and the neighbouring paragraphs, without seeing the original.</li>
     <li><b>Audit:</b> a third call compares the result with the original.</li>
    </ol>
    <p><b>Rejected if</b> the output is not valid JSON or is under 35 or over 450 words (retried up to 3 times). The audit was recorded but not used to filter:</p>
    <table className="sp-table dt-num">
     <thead><tr><th>Audit result</th><th>Rows</th></tr></thead>
     <tbody>
      <tr><td>No issues</td><td>1,194</td></tr>
      <tr><td>Only broken PDF extraction in the human text around it</td><td>1,976</td></tr>
      <tr><td>An issue with the paragraph itself</td><td>1,215</td></tr>
     </tbody>
    </table>
    <p className="sp-fine">The most common paragraph issues: a changed number or citation (450), a general “needs review” (443), meaning not preserved (195). They concern faithfulness to the original; the paragraph is still AI-written and labelled so.</p>
   </div>
  </div>
 </>},
 section:{title:'AI sections',short:'one section of a paper replaced',body:<>
  <p>Claude writes one top-level section of a human paper, either drafting it from the title, abstract, neighbouring sections, heading and target length, or rewriting the original. Section roles: introduction 25%, related work 25%, discussion 20%, method 15%, experiments 15%.</p>
  <p>Rejected if it contains markdown or LaTeX or its length is off by more than 0.5–1.8×. The human version of each section is a separate human row.</p>
 </>},
 full:{title:'Fully AI documents',short:'mirrors, manuscripts, full papers',body:<>
  <dl className="ar-facts">
   <dt>Mirrors</dt><dd>GPT-6 Luna writes a standalone text on a short topic taken from a human passage in the general human sources, without seeing the passage</dd>
   <dt>Manuscripts</dt><dd>Research papers written from a human paper’s title and abstract, then reviewed and revised; the title and abstract are removed. The mix uses 130 by GPT-6.1 Sol and 123 by GPT-6 Luna. A few mix the two models, for example a GPT-6 Luna draft revised by GPT-6.1 Sol</dd>
   <dt>Claude papers</dt><dd>Full papers by Claude Opus, Sonnet and Haiku from seeds dated 2022 or earlier: drafted, reviewed by Claude Opus, revised once</dd>
  </dl>
 </>},
};

/* Real training rows, a fixed sample built by research/site-architecture-20261011/build_ai_examples.py and served from R2. */
type Example={id:string;key:string;kind:Gen;writer:string;op:string;source:string|null;format:string;text:string;regions:[number,number,number][];group_rows:number;human:{text:string}|null};
const OP_LABEL:Record<string,string>={rewrite_one:'Rewrite one',rewrite_two:'Rewrite two',insert_one:'Insert one',split_one:'Split one',merge_two:'Merge two (early trial)',
 splice_one:'Splice, one sentence',splice_two:'Splice, two sentences',draft:'Draft',rewrite:'Rewrite',draft_from_notes:'Draft from notes',mirror:'Mirror',manuscript:'Manuscript',claude_paper:'Full paper'};
const HUMAN_NOTE:Record<Gen,string>={
 small:'Prompted small edits have no separate human version: the human paragraphs around the edit are in the same row. GPT-6 Luna splices do: for 5,514 of the 5,517, the human original is also a training row.',
 para:'Most paragraph edits also have their human original, in the same context, as a separate training row: 2,990 of the 4,315 Claude, GPT-6.1 Sol and DeepSeek edits, and 4,383 of the 4,385 GPT-6 Luna drafts.',
 section:'2,963 of the 4,200 AI sections also have their human original, in the same context, as a separate training row.',
 full:'Fully AI documents have no human version.'};
const EXAMPLES_URL='/backend/v1/transparency/ai-examples';
function useExamples(){
 const [data,setData]=useState<Example[]|null|'error'>(null);
 useEffect(()=>{let live=true;fetch(EXAMPLES_URL).then(r=>{if(!r.ok)throw new Error(String(r.status));return r.json() as Promise<{examples:Example[]}>;}).then(d=>{if(live)setData(d.examples);}).catch(()=>{if(live)setData('error');});return()=>{live=false;};},[]);
 return data;
}
function Labelled({text,regions}:{text:string;regions:[number,number,number][]}){
 const rs=regions.length?regions:[[0,text.length,0] as [number,number,number]];
 return <div className="dt-ex-text">{rs.map(([a,b,l],i)=>l===1?<span key={i} className="dt-ex-ai">{text.slice(a,b)}</span>:<span key={i}>{text.slice(a,b)}</span>)}</div>;
}
function Chips<T extends string>({label,value,options,onChange}:{label:string;value:T|'all';options:[T,string,number][];onChange:(v:T|'all')=>void}){
 return <div className="dt-ex-filter"><span>{label}</span><div role="group" aria-label={label} className="sp-chips">
  <button type="button" aria-pressed={value==='all'} onClick={()=>onChange('all')}>All</button>
  {options.map(([v,l,n])=><button key={v} type="button" aria-pressed={value===v} onClick={()=>onChange(v)}>{l} · {n}</button>)}
 </div></div>;
}
function AiExamples({kind}:{kind:Gen}){
 const data=useExamples();
 const [writer,setWriter]=useState<string|'all'>('all');const [op,setOp]=useState<string|'all'>('all');const [i,setI]=useState(0);const [human,setHuman]=useState(false);const [onlyHuman,setOnlyHuman]=useState(false);
 useEffect(()=>{setWriter('all');setOp('all');setI(0);setHuman(false);setOnlyHuman(false);},[kind]);
 useEffect(()=>{setI(0);setHuman(false);},[writer,op,onlyHuman]);
 const prompts=PROMPTS.filter(p=>p.kind===kind&&(writer==='all'||p.writers.includes(writer)));
 const inKind=Array.isArray(data)?data.filter(e=>e.kind===kind):[];const hasHuman=inKind.some(e=>e.human);
 const all=inKind.filter(e=>!onlyHuman||e.human);
 const count=(f:(e:Example)=>boolean)=>all.filter(f).length;
 const writers=[...new Set(all.map(e=>e.writer))].sort().map(w=>[w,w,count(e=>e.writer===w&&(op==='all'||e.op===op))] as [string,string,number]);
 const ops=[...new Set(all.map(e=>e.op))].map(o=>[o,OP_LABEL[o]||o,count(e=>e.op===o&&(writer==='all'||e.writer===writer))] as [string,string,number]);
 const shown=all.filter(e=>(writer==='all'||e.writer===writer)&&(op==='all'||e.op===op));const ex=shown[Math.min(i,shown.length-1)];
 return <div className="ar-demo">
  <h3>Examples from the training data</h3>
  <p className="sp-fine">Real training rows, a fixed sample of up to 3 per writer and kind of edit. <span className="dt-ex-key"><span className="dt-ex-ai">Highlighted text</span> is labelled AI; the rest is labelled human.</span> {HUMAN_NOTE[kind]}</p>
  {data===null&&<p className="sp-fine">Loading examples…</p>}
  {data==='error'&&<p className="sp-fine">The examples could not be loaded.</p>}
  {Array.isArray(data)&&<>
   {hasHuman&&<label className="ar-check"><input type="checkbox" checked={onlyHuman} onChange={e=>setOnlyHuman(e.target.checked)}/>Only examples with a human version ({inKind.filter(e=>e.human).length} of {inKind.length})</label>}
   <Chips label="Writer" value={writer} options={writers} onChange={setWriter}/>
   {ops.length>1&&<Chips label="Kind" value={op} options={ops} onChange={setOp}/>}
   {ex?<div className="dt-ex">
    <div className="dt-ex-head">
     <span><b>{ex.writer}</b> · {OP_LABEL[ex.op]||ex.op}{ex.source?<> · {ex.source}</>:null} · <span className="sp-fine">{ex.group_rows.toLocaleString()} training rows of this writer and kind</span></span>
     <span className="dt-ex-nav"><button type="button" onClick={()=>setI((i-1+shown.length)%shown.length)} aria-label="Previous example">‹</button>{Math.min(i,shown.length-1)+1} of {shown.length}<button type="button" onClick={()=>setI((i+1)%shown.length)} aria-label="Next example">›</button></span>
    </div>
    {ex.human&&<label className="ar-check"><input type="checkbox" checked={human} onChange={e=>setHuman(e.target.checked)}/>Show the human version of this passage (a separate human training row)</label>}
    {human&&ex.human?<Labelled text={ex.human.text} regions={[]}/>:<Labelled text={ex.text} regions={ex.regions}/>}
   </div>:<p className="sp-fine">No example for this combination.</p>}
  </>}
  <h3>Prompts</h3>
  {prompts.length?prompts.map(p=><details key={p.title} className="dt-prompt">
   <summary><b>{p.title}</b></summary>
   <p className="sp-fine">{p.note}</p>
   {p.parts.map(x=><details key={x.label} className="dt-prompt-part"><summary>{x.label}</summary><pre>{x.text}</pre></details>)}
  </details>):<p className="sp-fine">No prompt for this writer.</p>}
 </div>;
}

function AiText(){
 const [g,setG]=useState<Gen>('small');
 return <>
  <p className="sp-lede">We generated almost all of the AI training text by prompting writing models with real human text, in the four kinds below: 45,300 of the 47,096 training windows that contain AI. The other 1,796 come from GRADTEX, a public dataset derived from MAGE, and carry only a document label (AI-involved or human) with no character labels. The largest group, small edits inside human writing, is the hardest case to catch. Only the characters a model wrote are labelled AI. Automatic checks reject malformed outputs; what each check rejects differs by writer and is listed with each kind.</p>
  <nav className="ar-steps dt-kinds" aria-label="Kinds of AI text">
   {(Object.keys(GENS) as Gen[]).map((k,n)=><button key={k} type="button" aria-current={k===g?'step':undefined} onClick={()=>setG(k)}>
    <span className="ar-step-n">{n+1}</span><b>{GENS[k].title}</b><span>{GENS[k].short}</span></button>)}
  </nav>
  <div className="ar-demo dt-gen" aria-live="polite">{GENS[g].body}</div>
  <AiExamples kind={g}/>
  <div className="ar-demo">
   <h3>Writing models</h3>
   <p className="sp-fine">Writers marked “evaluation only” never wrote training text, so the evaluation sets can test writers the model has not seen.</p>
   <div className="sp-scroll"><table className="sp-table">
    <thead><tr><th>Model</th><th>Training text</th><th>Evaluation sets</th><th>Run through</th></tr></thead>
    <tbody>
     <tr><td><b>Claude Opus 5.5, Sonnet 5.5, Haiku 4.5</b></td><td>Small edits, paragraphs, sections, full papers</td><td>Held-out papers; Held-out papers and unseen writers; Other domains</td><td>Claude Code</td></tr>
     <tr><td><b>GPT-6.1 Sol</b></td><td>Small edits, paragraphs, manuscripts</td><td>Held-out papers and unseen writers; Other domains</td><td>Codex</td></tr>
     <tr><td><b>GPT-6 Luna</b></td><td>Small edits (prompted and splices), paragraphs, mirrors, manuscripts</td><td>Held-out papers; Other domains; Screening set</td><td>OpenRouter</td></tr>
     <tr><td><b>DeepSeek V4.1 Flash</b></td><td>Small edits, paragraphs</td><td>Site text format and unseen writers (15% of its papers held out)</td><td>OpenRouter</td></tr>
     <tr><td><b>Claude Fable 5.1</b></td><td>Evaluation only</td><td>Held-out papers and unseen writers; Light polish; Humanizer; Co-writing</td><td>Claude Code</td></tr>
     <tr><td><b>GPT-6 Sol</b></td><td>Evaluation only</td><td>Held-out papers and unseen writers</td><td>Codex</td></tr>
     <tr><td><b>GLM 5.3 Flash</b></td><td>Evaluation only</td><td>Site text format and unseen writers</td><td>OpenRouter</td></tr>
     <tr><td><b>MiMo V2.6 Flash</b></td><td>Evaluation only</td><td>Site text format and unseen writers</td><td>OpenRouter</td></tr>
     <tr><td><b>“Opus 3” (unverified)</b></td><td>Evaluation only</td><td>Other domains (100 documents)</td><td>A browser chat interface; the name is the label shown in its model picker, and the actual model was not recorded</td></tr>
    </tbody>
   </table></div>
  </div>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Instructed edits.</b> Edits follow fixed instructions. Real authors may prompt differently, edit the output afterwards, or use AI over several rounds.</li>
    <li><b>Few GPT small edits.</b> GPT-6.1 Sol produced 378 of the 1,800 planned small edits (a second round was planned but never run); the shortfall was filled from the other small-edit writers.</li>
    <li><b>Some Opus 5.5 text may be Claude Opus 5.</b> When a safety classifier stopped a writer agent’s response, the agent continued on Claude Opus 5. 124 training rows (117 paragraph edits, 7 small edits) come from batches where this happened and are labelled Claude Opus 5.5; which parts each model wrote is not recorded. 0.17% of the training windows.</li>
    <li><b>GPT-6 Luna’s text was not regenerated with the later pipeline.</b> Its small edits, splices and paragraph drafts were made before the stricter checks and the neighbouring-paragraph context were introduced, and were kept as they were.</li>
    <li><b>Haiku 4.5 instead of Haiku 5.5.</b> The Claude Haiku text (small edits, paragraphs, sections and full papers) was written by Claude Haiku 4.5. The intended model was Haiku 5.5; the model setting resolved to the older version, which was a mistake.</li>
    <li><b>Most GPT-6 Luna small edits are splices.</b> 5,517 of 8,132 reuse sentences from GPT-6 Luna paragraph drafts rather than an editing prompt, and its 2,615 prompted edits were checked less strictly than the other writers’.</li>
    <li><b>Audit not used as a filter.</b> 3,191 of the 4,385 GPT-6 Luna paragraphs were flagged by their audit (1,215 for the generated paragraph, 1,976 only for broken extraction in the human text around it) and still used for training.</li>
    <li><b>Claude-heavy.</b> Over half of the small edits (13,334 of 25,200) are by Claude models. This was a cost choice: generating text through Claude Code subscriptions gave more data for the money than Codex, and we avoided paying per-call API prices out of pocket where we could.</li>
    <li><b>Untraced “Opus 3” documents.</b> 100 documents in the other-domains set are labelled “Opus 3”, the name shown in a browser chat interface’s model picker. The code and logs that generated them could not be found, so which model wrote them is unknown.</li>
   </ul>
  </div>
 </>;
}

/* ---------- Training mix ---------- */

type Group={name:string;note?:string;rows:number;ai:string;parts:string;about:string};
const GROUPS:Group[]=[
 {name:'Small edits',rows:25200,ai:'about 10% of tokens',parts:'Claude 13,334 · GPT-6 Luna 8,132 (5,517 of them splices) · DeepSeek 3,356 · GPT-6.1 Sol 378',about:'One or two AI sentences inside a human paragraph, with the paragraphs around it.'},
 {name:'Human paper text',rows:18000,ai:'none',parts:'Untouched papers 9,000 · human version of each AI section 3,000 · human version of each AI paragraph 3,000 · other paragraphs from edited papers 3,000',about:'Human research-paper text. A third is the matched human version of AI-edited text, so the same passages appear both ways.'},
 {name:'Paragraph edits',rows:8700,ai:'35–56% of tokens',parts:'GPT-6 Luna 4,385 · Claude and GPT-6.1 Sol 3,505 · DeepSeek 810',about:'One AI paragraph inside human context.'},
 {name:'Fully AI',rows:7200,ai:'all tokens',parts:'Mirrors 4,200 · manuscripts 1,500 · Claude papers 1,500',about:'Documents written entirely by AI, cut into windows.'},
 {name:'Other human writing',rows:5100,ai:'none',parts:'32 sources',about:'Human text outside research papers: fiction, news, forums, reviews, reference works, reports.'},
 {name:'AI sections',rows:4200,ai:'about 92% of tokens',parts:'Claude',about:'Windows from a paper section written by Claude, cropped around the AI text.'},
 {name:'GRADTEX human and AI documents',note:'document label only',rows:3600,ai:'1,796 AI-involved documents; no token labels',parts:'1,804 human · 1,796 AI-involved',about:'A public dataset derived from MAGE with only a document label (AI-involved or human). Trains only the document term.'},
];
function TrainingMix(){
 const [sel,setSel]=useState(0);const max=GROUPS[0].rows,g=GROUPS[sel];
 return <>
  <p className="sp-lede">The model trains on 72,000 windows of up to 510 tokens, each seen once. About a third are small edits; a third are human text with no AI at all.</p>
  <div className="ar-demo">
   <h3>What the 72,000 windows are</h3>
   <Pie title="Training windows by group (72,000)" unit="windows" items={GROUPS.map(x=>({label:x.name,value:x.rows,note:x.note}))} selected={sel} onSelect={setSel}/>
   <div className="dt-bar-detail">
    <b>{g.name}</b> · {g.rows.toLocaleString()} windows · AI: {g.ai}
    <p>{g.about}</p>
    <p className="sp-fine">{g.parts}</p>
   </div>
  </div>
  <div className="ar-model">
   <div className="ar-detail dt-static">
    <h3>Cutting rows into windows</h3>
    <p>A row longer than 510 tokens is cropped to a window of up to 500 tokens that contains the AI text, at a random position, snapped to a word boundary. Long human and document-only texts are cut into back-to-back windows from a random start. Every row must fit under the tokenizer of each model trained on the mix.</p>
   </div>
   <div className="ar-detail dt-static">
    <h3>What is kept out</h3>
    <p>Evaluation, checkpoint-selection and calibration papers, matched by id, paragraph hash and 60-character pieces of text; any row sharing a 10+-word sentence with them; and anything written by the evaluation-only models. Duplicates within a group are removed. See Evaluation sets.</p>
   </div>
  </div>
  <div className="ar-limits">
   <h3>Known limitations</h3>
   <ul>
    <li><b>Human-heavy loss.</b> About 70% of the total loss weight is on human text (see Training loss on the Architecture page).</li>
    <li><b>Planned counts were not all met.</b> GPT-6.1 Sol small edits (378 of 1,800) and DeepSeek paragraph edits (810 of 999) fell short and were filled from other writers in the same group.</li>
    <li><b>One mix.</b> Larger mixes from the same sources (134,000 and 566,000 windows) were trained but are not the current model.</li>
   </ul>
  </div>
 </>;
}
