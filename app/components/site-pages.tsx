"use client";
/* Static public pages: home, transparency, manifesto and about. Ported from the approved design canvas. */
import {useState,type FormEvent,type ReactNode} from 'react';
import {SiteHeader,SiteFooter,SUBMISSIONS_HREF,useSavedCount} from './site-chrome';
import {Transparency} from './site-transparency';
import {Architecture} from './site-architecture';
import {DataPage} from './site-data';
import {ResultsPage} from './site-results';
import {LimitationsPage} from './site-limitations';
import './site-pages.css';

export type StaticPage='home'|'transparency'|'architecture'|'data'|'results'|'limitations'|'manifesto'|'about';

export default function SitePages({page}:{page:StaticPage}){
 const savedCount=useSavedCount();
 return <div className="site sp">
  <SiteHeader page={page} savedCount={savedCount}/>
  <main>{page==='home'?<Home/>:page==='transparency'?<Transparency/>:page==='architecture'?<Architecture/>:page==='data'?<DataPage/>:page==='results'?<ResultsPage/>:page==='limitations'?<LimitationsPage/>:page==='manifesto'?<Manifesto/>:<About/>}</main>
  <SiteFooter/>
 </div>;
}

function Section({title,children}:{title:string;children:ReactNode}){
 return <section className="sp-section"><h2>{title}</h2>{children}</section>;
}

/* ---------------- Home ---------------- */
function Home(){
 const [q,setQ]=useState('');
 function submit(e:FormEvent){e.preventDefault();const p=new URLSearchParams({collection:'iclr/2027'});if(q.trim())p.set('q',q.trim());location.href=`/?${p}`;}
 return <div className="sp-col">
  <section className="sp-hero">
   <h1>Check an ICLR 2027 submission for AI-written text</h1>
   <p>This site runs an independently built, open-source detector over every public ICLR 2027 submission and shows, passage by passage, where the writing looks generated, edited or paraphrased by a language model. It is free and independent, with no company or commercial funding behind it. <a href="/?view=manifesto">Why I built it</a>.</p>
   <form className="sp-lookup" onSubmit={submit} role="search">
    <label htmlFor="sp-q" className="sp-sr">Submission number or title</label>
    <input id="sp-q" type="search" value={q} onChange={e=>setQ(e.target.value)} placeholder="Submission number or title, e.g. 24819" autoComplete="off"/>
    <button type="submit">Look up</button>
   </form>
   <div className="sp-meta">ICLR 2027 submissions · scores are provisional until launch · <a href={SUBMISSIONS_HREF}>Browse all submissions</a></div>
  </section>

  <Section title="Why this exists">
   <p>ICLR runs its review in the open. Every submission, review and rebuttal is public while the process is under way. That openness makes it possible to look at how papers are written, not only what they claim.</p>
   <p>Undisclosed use of language models in writing is hard to see and slow to address. Conferences ask for disclosure but have few ways to check it while review is still happening. This site puts the same evidence in front of everyone at the same time, so that authors, reviewers and chairs can discuss it with the paper in hand.</p>
   <p>A score is a model estimate, not proof of misconduct, and the detector makes errors. Read the highlighted passages before drawing any conclusion.</p>
  </Section>

  <Section title="Who made this">
   <p>Built by woog, self-funded and unpaid, with help from a collaborator, sky.moo, and donated compute from a Hugging Face community grant (through zmuwztd) and nev. I curated and produced the training data, trained the detectors and evaluated them. Commercial detectors keep their models and data closed. This one is meant to be fully open: the results are on this site, and the training recipe, a full dataset disclosure and the LoRA adapters will be published.</p>
   <p className="sp-links"><a href="/?view=manifesto">Why I built this</a><a href="/?view=transparency">Transparency</a><a href="/?view=about">Credits</a></p>
  </Section>

  <Section title="How to use it">
   <dl className="sp-dl">
    <dt>Reviewers and area chairs</dt><dd>Look up the papers in your batch. Open a flagged paper to see which passages scored high and decide whether it matters for your assessment.</dd>
    <dt>Authors</dt><dd>Check your own submission and see which passages were flagged.</dd>
    <dt>Everyone else</dt><dd>Browse submissions by research area, or read how the detector was built and tested.</dd>
   </dl>
  </Section>

  <Section title="What the detector looks for">
   <div className="sp-grid2">
    <div><b>Fully generated text</b><span>Passages written by a language model from a prompt.</span></div>
    <div><b>Partial generation</b><span>Some sections generated, others written by the authors.</span></div>
    <div><b>Light edits</b><span>Human writing polished or corrected by a model.</span></div>
    <div><b>Rewriting and paraphrasing</b><span>Existing text reworded by a model.</span></div>
   </div>
   <p className="sp-links"><a href="/?view=transparency">How well it works, and where it fails</a></p>
  </Section>

  <Section title="Build on it">
   <p>This is meant to be a starting point. Once the releases are out, run the detectors yourself, check the numbers, train a better one from the released data, or fork the whole site for another venue.</p>
  </Section>
 </div>;
}

/* ---------------- Manifesto ---------------- */
const MANIFESTO=[
 'ICLR is unusual. Every submission, review and rebuttal is public while the review is still happening, not after it ends. Anyone can read along. I think that is one of the best things about this field, and it is also what made this project possible.',
 'Over the last few years, more and more of the writing in machine learning papers has come from language models. Sometimes it’s a whole paper. More often it’s a section here and a rewritten paragraph there. Most of it is never disclosed. Conferences have started writing policies about it, but a policy only means something if someone can check it, and review moves much faster than the institutions around it.',
 'I don’t think peer review and official conference proceedings, as we know them, will survive what is coming. Given where AI is heading, I think they are doomed, and I don’t see a way for them to recover.',
 'But we aren’t there yet. We are in the transition, taking part in a review process whose limits are being tested harder than ever. If this conference is going to have any chance of working, it needs radical change, and that change can’t wait for the official process. Official processes are extremely hard to change once a status quo has been in place for years, and the logistics move far too slowly. By the time anything official happens, the decisions will already have been made.',
 'There is already a company that sells this kind of detection. Its models are closed, its training data is closed, and you are asked to trust the result. That’s a reasonable way to run a business. I don’t think it’s a good foundation for making claims about other people’s work, especially in a community that is supposed to run on evidence anyone can check.',
 'So I built my own and paid for most of it out of pocket. I collected and generated the training data, trained the models, and evaluated them. I didn’t do it entirely alone: sky.moo worked on it with me and put real effort in, and zmuwztd, through a Hugging Face community grant, and nev gave me GPU time I couldn’t have afforded otherwise. Beating them was never the point. The point was to show it could be done in the open, and then give the whole thing away.',
 'The results are already on this site, along with a full account of how they were reached. The code, the training data, the held-out evaluation sets and the LoRA adapter are public too (see Releases on the About page). If you think my models are wrong, I want you to be able to prove it with my own data, and then build something better.',
 'I’m not writing a paper about this. I care about whether it’s useful.',
 'I also want to be clear about what it’s for. It isn’t meant to catch or shame anyone. Using a model to help you write is a choice people are entitled to make. Hiding it from the people reviewing your work is the problem. A score is a model estimate, not proof of misconduct, and the detector makes errors. That’s why every flag points to the exact passage, so anyone can check it for themselves. Evidence you can see and argue with is fairer to authors than suspicion you can’t.',
 'I don’t know whether this will change much. But I keep coming back to the fact that none of it needed permission. The papers were public, the methods were known and the compute could be rented. What was missing was someone deciding to do the work. I think a lot of things that seem stuck are stuck in exactly that way, and that we have more power to fix them ourselves than we usually act like we do.',
 'If this is useful to you, take it. Run it on another conference, train a stronger detector, fork the site or replace it entirely.',
];
function Manifesto(){
 return <article className="sp-col sp-essay">
  <header><h1>Why I built this</h1><div className="sp-byline">woog</div></header>
  {MANIFESTO.map((p,i)=><p key={i}>{p}</p>)}
  <footer><div>woog</div><p className="sp-links"><a href="/?view=transparency">Transparency</a><a href="/?view=about">Releases and contact</a></p></footer>
 </article>;
}

/* ---------------- About ---------------- */
/** Release status as of Oct 13, 2026 (everything released; the GitHub links work once the repository is public): [item, what it is, status, status kind, link].
 * Per-paper scores are released through the reader only; no bulk download (Alice, Oct 12). */
const RELEASES:[string,string,string,'out'|'part'|'soon',string?][]=[
 ['Evaluation results','Accuracy and error rates on held-out data, for the live model and the runs it was chosen from.','Done: Evaluation page','out','/?view=results'],
 ['Per-paper scores','Sentence-level scores for every classified paper.','Done: in the reader for each paper','out','/?collection=iclr%2F2027'],
 ['Training data','The 72,000 training windows behind the live model, with sources and construction.','Done: on Hugging Face','out','https://huggingface.co/datasets/woog/open-text-detector-mix-oct7'],
 ['LoRA adapter','The trained weights of the live model.','Done: on Hugging Face','out','https://huggingface.co/woog/open-text-detector-qwen36-35b-a3b-oct10'],
 ['Evaluation data','The held-out sets behind the Evaluation page.','Done: on Hugging Face','out','https://huggingface.co/datasets/woog/open-text-detector-heldout'],
 ['Training recipe','Configurations and scripts to reproduce each model.','Done: on GitHub, with the exact code of the live model','out','https://github.com/woog97/open-text-detector/tree/main/benchmarks/otd/training/final-moe-oct10-snapshot'],
 ['Site and inference code','Everything that runs this website and scores papers.','Done: on GitHub','out','https://github.com/woog97/open-text-detector'],
];
function About(){
 return <div className="sp-col">
  <section className="sp-hero">
   <h1>About this project</h1>
   <p>Open Text Detector is a free, open-source tool for finding AI-written text in machine learning papers. It is built by woog, with help from a collaborator and donated compute, and paid for mostly out of pocket. There is no company or commercial funding behind it, and it will never become a product. <a href="/?view=manifesto">Why I built it</a>.</p>
  </section>
  <Section title="How it was built">
   <p>Commercial detectors exist, but their models, training data and evaluations are closed. You have to trust their numbers. I wanted a detector anyone could inspect, so I built one from scratch.</p>
   <p>I curated and produced the training data, trained the classifiers, and evaluated them, on my own time and at my own expense. The model served on this site is the result. <a href="/?view=transparency">The full story is on the Transparency page</a>.</p>
  </Section>
  <Section title="Credits">
   <dl className="sp-dl sp-dl-narrow">
    <dt>woog</dt><dd>Project lead. Training data, model training, evaluation and this site.</dd>
    <dt>sky.moo</dt><dd>Collaborator.</dd>
    <dt>zmuwztd</dt><dd>Supported the project with a Hugging Face community grant, which grew from 1 to 8 A100s.</dd>
    <dt>nev</dt><dd>Shared access to their 4× H200 node.</dd>
   </dl>
  </Section>
  <Section title="Releases">
   <p>Everything behind the detector is public: the code, the evaluations, the training recipe, the data and the weights.</p>
   <table className="sp-table"><thead><tr><th>Item</th><th>Status</th></tr></thead><tbody>
    {RELEASES.map(([n,d,st,k,href])=><tr key={n}><td><b>{n}</b><span>{d}</span></td><td className={k==='soon'?'sp-soon':k==='part'?'sp-part':'sp-out'}>{href?<a href={href}>{st}</a>:st}</td></tr>)}
   </tbody></table>
  </Section>
  <Section title="Choose your classifier">
   <p>No single detector is right for every case, and I don’t want this site to decide that for you. The site currently serves one model, Qwen3.6 35B-A3B MoE · Oct 10; how it compares with the other candidates is under Runs compared on the <a href="/?view=results&tab=runs">Evaluation page</a>. Switching between models on result pages is planned.</p>
  </Section>
  <Section title="Build something better">
   <p>The point of releasing everything is that someone else can take it further. Train a stronger model on the released data, run it on another venue, or replace this site entirely. If you do, I would like to hear about it.</p>
  </Section>
  <Section title="Limits">
   <p>A score is a model estimate, not proof of misconduct, and the detector makes errors. The highlighted passages are there so you can judge for yourself.</p>
   <p className="sp-links"><a href="/?view=limitations">Known limitations</a><a href="/?view=calibration">Calibration</a></p>
  </Section>
  <Section title="Contact">
   <p>I’m <b>woog</b> on Discord, which is the best place to reach me. I’m active on EleutherAI #offtopic.</p>
   <p className="sp-links"><a href="https://x.com/woog09" rel="noopener">@woog09 on X</a><a href="mailto:rigg.alice0@gmail.com">rigg.alice0@gmail.com</a></p>
  </Section>
 </div>;
}
