"use client";
/* Limitations page (Transparency), first version Oct 12 2026. Sources: the parked-limitations doc (research notes,
   Oct 11-12), the Known limitations lists on the Data and Evaluation pages, and recalibrate.py's Oct 10 numbers on the
   2,400 human calibration papers (main text only, the default counting): at the 0.1% cutoff 10.6% of papers have a flagged
   sentence, 99th-percentile paper 2.2% flagged; at 1%, 66% and 7.5%. */
import './site-architecture.css';
import './site-data.css';

type Item=[title:string,body:React.ReactNode];

const READING:Item[]=[
 ['A flagged share is not the share of a paper that AI wrote.','It counts sentences that score above a cutoff. The detector misses many edited sentences at that cutoff (about half of small edits at 0.1%, see Evaluation), and some human sentences score above it.'],
 ['Human papers get flags too.','At the default 0.1% cutoff, 11% of the 2,400 human calibration papers have at least one flagged sentence; the most-flagged 1% have about 2% of their sentences flagged. At the 1% option, two thirds have at least one, and the most-flagged 1% have about 8%. One or two flagged sentences in a paper say little on their own.'],
 ['No ground truth for the papers on this site.','The ICLR 2027 submissions have no known labels. Their flag rates describe what the model does on them; they are not accuracy measurements.'],
 ['Flag rates on recent papers are not explained yet.','Recent submissions are flagged far more often than papers from 2022 or earlier. Some of that may come from how recent papers differ from the calibration papers (newer topics, citation styles, PDF layouts) rather than from AI use. We have not separated the two.'],
 ['Mostly machine-learning papers.','Nearly all calibration papers are ICLR, NeurIPS and CoRL papers. Human writing in other fields may be flagged at a different rate.'],
];

const WEAK:Item[]=[
 ['Co-writing over several rounds.','When human and AI text are mixed sentence by sentence over three rounds, 26% of AI sentences are caught at 1% and 12% at 0.1%.'],
 ['Light polishing.','When AI changes about 9% of a paragraph’s words, few sentences are mostly AI, so sentence flags rarely appear. Scored per document the polished texts separate from human ones only partly (AUROC 0.81).'],
 ['Rewording a single sentence.','The hardest small edit: 64% caught at 1% and 35% at 0.1%. Added sentences are easiest (93% and 81%).'],
 ['Paragraph rewrites in the site’s own text format.','69% of sentences caught at 1% and 45% at 0.1%, against 97% and 87% for the same kind of edit in the cleaner research format.'],
];

const SETUP:Item[]=[
 ['Edits inside whole papers are not measured.','Every evaluation set scores the edited passage on its own. The site scores whole papers, with headings, captions, references and math around each passage.'],
 ['Whole AI-written papers are not measured.','No evaluation set scores complete AI-generated papers against human papers the way the site does.'],
 ['The site does not use overlapping windows.','The site cuts each paper into back-to-back 510-token windows; the evaluations slide the window every 256 tokens and average the overlapping scores. A sentence near a window edge gets less context on the site. The site’s cutoffs are set the same way as its scores, so the two are consistent, but the size of the difference from the evaluation setup is not measured.'],
 ['Reference lists are left out.','The model gives reference-list sentences high scores, so the site always leaves them out; appendices are left out too by default, so a score counts only the main text. Reference lists are found by a rule (a “References” heading followed by citation-like lines), which can miss unusual layouts; any missed reference lines are scored like body text.'],
 ['Scores depend on document length.','In our evaluation runs the cutoff that flags 1% of human sentences was about 0.75–0.8 for texts under 256 tokens and 0.35–0.5 for texts over 510 tokens. The site’s cutoffs are set on full papers and fit full papers; the same cutoff would flag more human sentences in short texts.'],
 ['Some paper text becomes control tokens.','Strings such as “<think>” or “<|endoftext|>” in a paper are read by the model as single special tokens it never saw in training. About 4.7% of papers on this site contain one, mostly papers about language models and agents. The effect on their scores is not measured.'],
 ['No markers for where a document starts or ends.','Windows are fed to the model without start or end tokens, so it cannot tell whether a window is a whole document or part of one. This may contribute to the length effect above; not measured.'],
];

const MISTAKES:Item[]=[
 ['Two scoring oversights.','The site scores papers without overlapping windows, and nothing marks where a document starts or ends (both above). Neither was a deliberate choice.'],
 ['The wrong Claude Haiku.','The Claude Haiku text was meant to come from Haiku 5.5; a model setting resolved to Haiku 4.5, and I did not catch it before the text was used.'],
 ['A writer switch I did not catch.','When a safety filter stopped a writing agent, it carried on with Claude Opus 5 while its output kept the original label. 124 training rows and 299 evaluation rows are affected.'],
 ['Early data kept as it was.','GPT-6 Luna’s text was made before the stricter checks existed and was not regenerated, and paragraphs its audit flagged were still used for training.'],
];

const BUDGET:Item[]=[
 ['One training run per recipe.','Each recipe was trained once, so differences of about a point between recipes are within noise. More runs per recipe would cost proportionally more compute.'],
 ['No labelled real-world papers.','There is no set of real submissions with known AI use; building one would need paid expert annotation. Every accuracy number comes from edits made for the purpose.'],
 ['Text written through subscriptions.','Most AI text was generated through Claude Code subscriptions rather than per-call APIs, which gave more text for the money but made the data Claude-heavy.'],
 ['Planned data not finished.','A second round of GPT-6.1 Sol edits was never run (378 of 1,800 planned small edits), and some other writers fell short of their planned counts.'],
 ['Longer windows not tested on whole papers.','A model trained on 1,020-token windows matched this one on the standard held-out sets. Whether it does better on whole papers, as the site scores them, would need a test built from real full papers, which has not been run.'],
 ['No outside review.','This is a one-person, self-funded project on donated compute. The work has not been peer-reviewed or independently audited.'],
];

function List({items}:{items:Item[]}){return <ul>{items.map(([t,b])=><li key={t}><b>{t}</b> {b}</li>)}</ul>;}

export function LimitationsPage(){
 return <div className="sp-col sp-col-wide ar">
  <section className="sp-hero">
   <h1>Limitations</h1>
   <p>What the scores on this site can and cannot tell you, where the detector is weak, and what has not been measured. This list is incomplete and will grow.</p>
  </section>
  <div className="ar-limits"><h3>Reading a score</h3><List items={READING}/></div>
  <div className="ar-limits"><h3>Where the detector is weak</h3><List items={WEAK}/>
   <p>Numbers are recall on held-out sets at cutoffs that flag 1% or 0.1% of each set’s human sentences; details on the <a href="/?view=results">Evaluation page</a>.</p></div>
  <div className="ar-limits"><h3>How the site differs from the evaluations</h3><List items={SETUP}/></div>
  <div className="ar-limits"><h3>My mistakes</h3><List items={MISTAKES}/></div>
  <div className="ar-limits"><h3>What time and money did not cover</h3><List items={BUDGET}/></div>
  <div className="ar-limits"><h3>Training and evaluation data</h3>
   <p>Limits of the data itself (which models wrote which text, counts that fell short of plan, writer labels that may be wrong) are listed under Known limitations on the <a href="/?view=data">Data page</a>.</p></div>
 </div>;
}
