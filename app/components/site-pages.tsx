"use client";
/* Static public pages: home, transparency, manifesto and about. Ported from the approved design canvas. */
import {useRef,useState,type FormEvent,type ReactNode} from 'react';
import {useHeadingAnchors} from './heading-anchors';
import {SiteHeader,SiteFooter,SUBMISSIONS_HREF,useSavedCount} from './site-chrome';
import {Transparency} from './site-transparency';
import {Architecture} from './site-architecture';
import {DataPage} from './site-data';
import {ResultsPage} from './site-results';
import {LimitationsPage} from './site-limitations';
import './site-pages.css';
import ad8eAvatar from './assets/ad8e-avatar.png';
/** Vite gives an image import as a URL string; the Next-era type declarations describe an object with `src`. */
const imgSrc=(i:unknown)=>typeof i==='string'?i:(i as {src:string}).src;

export type StaticPage='home'|'transparency'|'architecture'|'data'|'results'|'limitations'|'manifesto'|'about';

const NO_ROOT={current:null};
/** Pages whose section headings get a "#" link (heading-anchors.tsx). */
const LINKED:StaticPage[]=['transparency','architecture','data','results','limitations'];

export default function SitePages({page}:{page:StaticPage}){
 const savedCount=useSavedCount();
 const main=useRef<HTMLElement>(null);useHeadingAnchors(LINKED.includes(page)?main:NO_ROOT);
 return <div className="site sp">
  <SiteHeader page={page} savedCount={savedCount}/>
  <main ref={main}>{page==='home'?<Home/>:page==='transparency'?<Transparency/>:page==='architecture'?<Architecture/>:page==='data'?<DataPage/>:page==='results'?<ResultsPage/>:page==='limitations'?<LimitationsPage/>:page==='manifesto'?<Manifesto/>:<About/>}</main>
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
   <h1>Check an ICLR 2027 submission for AI-assisted writing</h1>
   <p>This site runs an open-source classifier I trained to detect LLM writing in ML papers. I ran it on all 42,000~ish ICLR 2027 submissions and host the sentence-level classification results here with an integrated pdf viewer. It is free and independent, with no company or commercial funding behind it. <a href="/?view=manifesto">Why I built it</a>.</p>
   <form className="sp-lookup" onSubmit={submit} role="search">
    <label htmlFor="sp-q" className="sp-sr">Submission number or title</label>
    <input id="sp-q" type="search" value={q} onChange={e=>setQ(e.target.value)} placeholder="Submission number or title, e.g. 24819" autoComplete="off"/>
    <button type="submit">Look up</button>
   </form>
   <div className="sp-meta">ICLR 2027 submissions · <a href={SUBMISSIONS_HREF}>Browse all submissions</a></div>
  </section>

  <Section title="This is not a reliable slop detector">
   <p>The classifier measures whether writing resembles text generated or edited by a language model. It does not measure a paper’s quality, whether its ideas came from a human or model, or whether its reasoning holds up. A paper with many flagged sentences is not, on that evidence alone, a worse paper.</p>
   <p>When making this project originally, I was hoping that the proportion of AI aided writing in a paper would be highly correlated to its likelihood of being slop. The jury is still out on whether this is true, but I don't have enough evidence to make this claim yet. What I do have evidence of, is that the classifier assigns substantially higher confidence scores on held-out eval sets I designed to imitate slop (I just don't trust the evals that much). See out the ROC curves in the <a href="/?view=calibration">calibration section</a> for details.</p>
   <p>The site offers a few heuristics along these lines, such as the share of a paper’s sentences above a threshold, and sorting submissions by that share. You can choose which human false positive rate to use and control the sensitivity of the classifier. I set the default human FPR to 0.1%, but you can change it to 0.01% to try catching more egregious forms of slop (at the cost of worse recall. Again, check the <a href="/?view=calibration#which-cutoff-to-use">ROC curves</a>). Mandatory reminder: a score is a model estimate, not proof of misconduct, and the detector makes errors. Read the highlighted passages before drawing any conclusion.</p>
  </Section>

  <Section title="Why this exists">
   <p>I wrote a <a href="/?view=manifesto">manifesto</a> explaining this. In short: AI makes it easier than ever to produce research at every stage, which is good for the frontier. But it also lets low-effort work be produced at a scale that saturates the submission pool and overburdens double-blind review. Official processes change far too slowly to keep up, and ICLR 2027’s open review makes it possible to act now instead of waiting.</p>
   <p>One of the things we need is forced transparency: everyone should be able to see where, when and how AI was used in a paper, without needing to trust authors to report it faithfully. More broadly, being able to tell whether writing online is human- or AI-authored should be a free public service, not a paid product. This site is a first step toward that: an open detector anyone can inspect.</p>
  </Section>

  <Section title="Who made this">
   <p>Built by woog, self-funded and unpaid, with help from a collaborator, sky.moo, and donated compute from a Hugging Face community grant (through zmuwztd) and nev. I curated and produced the training data, trained the detectors and evaluated them. Commercial detectors keep their models and data closed. This one is fully open: the results are on this site, and the code, training recipe, training data and LoRA adapters are published (links on the <a href="/?view=about">About page</a>).</p>
   <p className="sp-links"><a href="/?view=manifesto">Why I built this</a><a href="/?view=transparency">Transparency</a><a href="/?view=about">Credits</a></p>
  </Section>

  <Section title="What the detector looks for">
   <div className="sp-grid2">
    <div><b>Drafted paragraphs and sections</b><span>Parts of a human paper written by a model.</span></div>
    <div><b>Inserted sentences</b><span>One or two sentences a model added to a human paragraph.</span></div>
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
/** Alice's text, verbatim (Oct 13, 2026). *…* marks emphasis. */
const MANIFESTO=[
 "AI is getting good, increasing everyone's ability to produce higher quality work at all parts of the research process. This means the frontier is moving faster than ever before. This is good. However, it also means the tail of low effort work is able to bootstrap more than ever before, enabling certain people to pump out many times more low effort / low quality papers than the frontier without consequence, saturating the submissions pool, and overburdening the double blind review process.",
 "For the record, I think the peer review process and these official conference proceedings is completely and utterly doomed, with no chance of recovery. Nonetheless, we are in this transition period where we are partaking in one of these review processes where its limits are really being stress tested. For this conference to have any chance at success, radical change is necessary, and we cannot wait for changes to be made in the official processes, as changing official processes is extremely difficult when the status quo has been in place for years and logistics move very slowly, far too slowly. we're lucky that ICLR 2027 has a radically open review process, allowing this project to take place.",
 "I originally asked Pangram for credits to do this project. They replied within an hour saying it's a fantastic idea but unfortunately they couldn't give me the ~$100k in api credits i had asked for, because they planned to collab directly with ICLR organizers. I was like \"great, how can i get involved?\" then they ghosted me. Cool. So I did it myself. I strongly believe we have the power to take matters in our own hands, and make things better for everyone.",
 "I'm somewhat of an extremist. I have real, strong opinions on how things ought to be to make things better. I'm skeptical as to how effective their official collaboration will be, because of the constraints that everything needs to fit under the conference official rules and guidelines, as well as Pangram's corporate incentive to minimize reputational risk. No matter what they end up deploying in their collaboration, I don't think it will be enough. Radical, polarizing change is needed now. One of the things we need is forced transparency, such that disclosing AI usage in a section isn't optional or even necessary, so that everyone has the ability to see exactly where/when/how it is used.",
 "The original goal of this project was to accurately detect AI slop so we could spend less time on those papers. My motivation on this project was driven by this hypothesis that higher proportions of AI assisted writing in papers is strongly negatively correlated with paper quality. I do not have enough evidence to make this conclusion however. A major limitation of this project in its current form is that the classifier detects *writing that has been assisted/edit by AI*, and makes no strong claims about the quality of work, about whether the ideas behind the writing are by humans vs LLMs, logical consistency and flow within papers, and other clues we associate with the term *AI slop*.",
 "In short, I believe I failed at accomplishing this original goal of AI slop detection. Nonetheless, I hope the project still provides some amount of value. I prioritized releasing a less perfect result as early into the ICLR 2027 process as possible, to allow other people involved to use it as much as possible. I hope this serves the research community diffusely. I hope it also fosters an initiative to improve open source AI writing detection, to give people an actually good alternative to for-profit products.",
 "There's a reasonable chance this project damages my academic reputation beyond any chance of recovery, as a consequence of people interpreting this project as attacking every researcher in the whole field. I can accept this risk largely because I think academia is doomed either way. Nevertheless, I hope to avoid misinterpretations wherever possible. I hope I've done my due diligence in avoiding *unnecessarily* pissing off tens of thousands of people. All *necessary* pissing off will of course remain.",
];
const MANIFESTO_SIGNOFF=["With metta,", "woog"];
/** Renders *emphasis* as italics; the rest is plain text. */
const withEmphasis=(t:string)=>t.split(/\*([^*]+)\*/).map((x,i)=>i%2?<em key={i}>{x}</em>:x);
function Manifesto(){
 return <article className="sp-col sp-essay">
  <header><h1>Why I built this</h1></header>
  {MANIFESTO.map((p,i)=><p key={i}>{withEmphasis(p)}</p>)}
  <footer><div>{MANIFESTO_SIGNOFF.map((l,i)=><span key={i}>{i>0&&<br/>}{l}</span>)}</div><p className="sp-links"><a href="/?view=transparency">Transparency</a><a href="/?view=about">Releases and contact</a></p></footer>
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
 ['Training recipe','Configurations and scripts to reproduce the model.','Done: on GitHub, with the exact code of the live model','out','https://github.com/woog97/open-text-detector/tree/main/benchmarks/otd/training/final-moe-oct10-snapshot'],
 ['Site and inference code','Everything that runs this website and scores papers.','Done: on GitHub','out','https://github.com/woog97/open-text-detector'],
];
function About(){
 return <div className="sp-col">
  <section className="sp-hero">
   <h1>About this project</h1>
   <p>Open Text Detector is a free, open-source tool for finding AI-assisted and AI-edited writing in machine learning papers. It is built by woog, with help from a collaborator and donated compute, and paid for mostly out of pocket. There is no company or commercial funding behind it, and it will never become a product. <a href="/?view=manifesto">Why I built it</a>.</p>
  </section>
  <Section title="How it was built">
   <p>Commercial detectors exist, but their models, training data and evaluations are closed. You have to trust their numbers. I wanted a detector anyone could inspect, so I built one from scratch.</p>
   <p>I curated and produced the training data, trained the classifiers, and evaluated them, on my own time and at my own expense. The model served on this site is the result. All code including the website have been vibed up initially with Codex, and later Claude. The only prose I can assure is in my original unedited voice on this website is <a href="/?view=manifesto">the manifesto</a>.</p>
   <p>The full details can be found in the <a href="/?view=transparency">Transparency section</a>.</p>
  </Section>
  <Section title="Credits">
   <dl className="sp-dl sp-dl-narrow">
    <dt>woog</dt><dd>Project lead. Training data, model training, evaluation and this site.</dd>
    <dt>sky.moo</dt><dd>Collaborator. Assisted with data generation, ablations, and early training runs.</dd>
    <dt>zmuwztd</dt><dd>Supported the project with a Hugging Face community grant, which grew from 1 to 8 A100s. HF compute support lasted from Sept 30th to Oct 8th, 2026.</dd>
    <dt>nev</dt><dd>Shared access to their 4× H200 node. Got access from Oct 1st, 2026.</dd>
   </dl>
  </Section>
  <Section title="Releases">
   <p>Everything behind the detector is public: the code, the evaluations, the training recipe, the data and the weights.</p>
   <table className="sp-table"><thead><tr><th>Item</th><th>Status</th></tr></thead><tbody>
    {RELEASES.map(([n,d,st,k,href])=><tr key={n}><td><b>{n}</b><span>{d}</span></td><td className={k==='soon'?'sp-soon':k==='part'?'sp-part':'sp-out'}>{href?<a href={href}>{st}</a>:st}</td></tr>)}
   </tbody></table>
  </Section>
  <Section title="Questions">
   <h3 className="sp-q">How good is the model?</h3>
   <p>It’s okay but we can do a lot better. The biggest limitation is there’s not enough training data: I spent about $80 total generating AI labeled data. Ideally we’d have around 100x more at least, to better match the proportion of human data available. There’s certainly a lot of other small optimizations that can be made too. I have comments on some of these in limitations sections at the bottom of the individual transparency pages.</p>
   <h3 className="sp-q">Aren’t you worried people will train against this to humanize text?</h3>
   <p>I’m not worried. As a wise person on a random discord server once said:</p>
   {/* Their public Discord message, redrawn in the page's colours (not a screenshot). */}
   <figure className="sp-chat" aria-label="Discord message from ad8e, September 30, 2026">
    <img className="sp-chat-avatar" src={imgSrc(ad8eAvatar)} alt="" width={40} height={40}/>
    <div><div className="sp-chat-head"><b>ad8e</b><time dateTime="2026-09-30T20:23">9/30/26, 8:23 PM</time></div>
    <p>this domain's bottleneck is researcher skill, not the underlying perfect-play equilibrium. it is easy to train a model better than pangram; next year's pangram will accomplish this. it is also easy to train a model that defeats pangram, but notice all the attempts at this are sad and scammy. that's because the better ML researchers have better things to do</p></div>
   </figure>
  </Section>
  <Section title="Privacy">
   <p>This site keeps usage statistics so I can see how it is used: which pages are visited, which papers are opened, and which sorting and filtering options are used, along with each visit’s country and city, the network or institution it comes from, the device type, browser, language and referring site. There are no cookies or tracking scripts, and IP addresses are not stored.</p>
   <p>If you send a message about a paper, your browser keeps a copy of it, with a private key, so you can edit it later. That copy stays in your browser. Every version you send is kept.</p>
  </Section>
  <Section title="Contact">
   <p>I’m <b>woog</b> on Discord, which is the best place to reach me. I’m active on EleutherAI #offtopic.</p>
   <p className="sp-links"><a href="https://x.com/woog09" rel="noopener">@woog09 on X</a></p>
  </Section>
 </div>;
}
