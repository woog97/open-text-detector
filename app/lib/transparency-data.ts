/* Transparency page content: abridged from the project logs and final evaluation reports (Oct 2026). */
export const TABS:[string,string,string][]=[['timeline','Timeline','9'],['hyp','Hypotheses','21'],['data','Data','5 mixes'],['dead','Dead ends','10'],['over','Against the AI’s advice','10'],['effort','Effort and compute',''],['open','Open problems','7']];
export const OUTCOME:Record<string,string>={c:'Confirmed',r:'Refuted',i:'Inconclusive',o:'Open'};

export const HYPOTHESES:[q:string,how:string,k:string,res:string,when:string][]=[
 ['Decoder models beat encoders','Six backbones at 10% length','c','Decoders ahead by about 0.16 AUROC on public text; tied on papers.','Oct 3'],
 ['High constant learning rates train LoRA faster at no cost','Overnight sweep, about 40 runs','r','Collapsed near step 1,000 without warmup. A 2e-4 cosine schedule became the default.','Oct 4–5'],
 ['With warmup and cosine decay, learning rate barely matters','Same sweep, 1e-4 to 5e-4','c','All landed at 0.77–0.78 recall on edits.','Oct 5'],
 ['Training longer improves small-edit detection','4B at 20% vs full length','r','Paper rewrites rose 0.77 → 0.89; small edits stayed at 0.21–0.24.','Oct 5'],
 ['Oversampling short AI spans helps','Sweep arm L, 3 runs','r','0.72 on edits against 0.79 for the default; small edits unchanged.','Oct 5'],
 ['Small-edit recall is limited by data, not recipe','Audit of 24,000 training windows, then splices','c','Only 2 windows held an AI span under 200 characters. Splices lifted small edits 0.25 → 0.58–0.63.','Oct 5'],
 ['The detector learns PDF-extraction artifacts as a shortcut','Cleaned human text; added artifacts to AI text','r','No change in false positives; 0 of 1,074 math sentences flagged.','Oct 5'],
 ['Cutoffs from calibration windows transfer to test text','Fit on calibration windows, applied to test','r','Realised false positives of 2–23%. Cutoffs from a separate half of the papers landed at 0.6–2.1%.','Oct 5'],
 ['Thresholds from ordinary human papers hold on harder human text','16 checkpoints','r','Flagged 4.5–7% of human sentences next to edits.','Oct 5'],
 ['The MoE improves with full-length training','MoE at 2e-4, then 1e-4 with a guard','r','2e-4 diverged; 1e-4 overfit. The short run stayed best.','Oct 5'],
 ['A vision-language PDF parser gives better text','Full run of MinerU on an A100','r','About 10 h lost to memory errors; too slow; cut to a 300-paper subset.','Oct 5'],
 ['ICLR 2027’s high flag rate reflects real AI use','Flag rates by year on 7,102 papers','o','Rates rise sharply from 2025. Changes in format or topic are not ruled out yet.','Oct 5'],
 ['Splice gains carry over to other writers','Held-out Claude and Luna edits','r','Caught Luna edits well but Sonnet and Opus only 37–41%: the gain was specific to one writer.','Oct 5–6'],
 ['GPT writers paraphrase more than Claude','Similarity to the original','c','GPT rewrites 44–52% similar to the original, Claude 60–77%.','Oct 6'],
 ['A balanced multi-writer mix fixes the writer gap','4B, 3 runs each, T2 vs splices','c','Unseen writers 0.47 → 0.60; other models’ text 0.41 → 0.83. Cost: in-domain small edits 0.63 → 0.48.','Oct 6'],
 ['Holding out writers and papers costs nothing','T2.1 vs T2, 3 runs each','c','Matched T2 on every AUROC.','Oct 6'],
 ['Humanizer tools defeat the detector','200 humanized AI sections','r','T2 went 0.91 → 0.90. Older splice-only models caught just 0.18.','Oct 6'],
 ['Clause-level labels help with edits inside sentences','Synthetic edits with exact truth, then a detector A/B','o','+0.10 macro-F1 on synthetic edits. Detector A/B not finished.','Oct 6'],
 ['The MoE generalises better than the 4B','Final full-length runs on the same data, one run each','c','Ahead on every held-out set: unseen papers 0.70 vs 0.61, other models’ text 0.93 vs 0.53 at the selected checkpoints. A first comparison on Oct 6 had been within noise.','Oct 6–7'],
 ['A small encoder (EmbeddingGemma 2) is good enough','Final full-length run on the same data','r','Unseen papers 0.33 against 0.70 for the MoE; other models’ text 0.28.','Oct 7'],
 ['The detector catches real co-writing','100 co-written documents, 3 rounds','o','Final MoE about 0.25–0.27; earlier models 0.14–0.17. Still the largest known blind spot.','Oct 6–7'],
];

export const CATEGORIES:Record<string,[string,string]>={small:['Small AI edits (1–2 sentences)','#c2417a'],para:['Paragraph edits','#4a3aa7'],sec:['AI sections','#1baf7a'],ai:['Fully AI text','#eb6834'],hpap:['Human paper text','#2a78d6'],hgen:['Other human text','#86b6ef'],grad:['GRADTEX (public)','#b9bdb6']};
export const MIXES:[name:string,when:string,why:string,parts:Record<string,number>][]=[
 ['BASE','Oct 3','Mostly whole human vs fully AI documents. Almost no short AI spans.',{hgen:6000,ai:7200,hpap:3011,para:2989,grad:4800}],
 ['SPG','Oct 5','Splices replace GRADTEX: small edits jump, but only for one writer.',{hgen:6000,ai:7200,hpap:3011,para:2989,small:4800}],
 ['T2','Oct 6','Small edits from Claude, Luna and GPT; more human paper text; AI sections.',{small:8400,hpap:6000,para:2900,sec:1400,ai:2400,hgen:1700,grad:1200}],
 ['T2.1','Oct 6','T2 with a held-out set of writers and papers, so tests are clean.',{small:8400,hpap:6000,para:2900,sec:1400,ai:2400,hgen:1700,grad:1200}],
 ['Final','Oct 7 · live model trains on it','T2.1 scaled to 72,000 distinct windows seen once, with DeepSeek edits added. Human paper text is matched so its format can’t give the answer away.',{small:25200,hpap:18000,para:8700,sec:4200,ai:7200,hgen:5100,grad:3600}],
];
export const COLLECTIONS:[string,string,string][]=[
 ['Human source passages','100,000 · 56.4M tokens','Broad human writing. Origin and permissions only partly verified.*'],
 ['Fully AI mirrors (Luna)','30,000 · 15.7M tokens','28,120 passed automated checks.'],
 ['Paper paragraph pairs','10,000 + 10,000','Human originals paired with AI rewrites in context.'],
 ['Synthetic research papers','600 · 4.4M tokens','Written end to end by three GPT models.'],
 ['Claude data','~120M tokens generated','4,905 sentence edits, 2,078 sections, 254 papers, 1,228 paragraph edits.'],
 ['GPT edits','1,418','1,000 small edits and 418 paragraph edits.'],
 ['Luna sentence edits','5,000','Cost $0.46 to generate.'],
 ['DeepSeek edits','4,359 in the final mix','3,360 small edits and 999 paragraph edits.'],
];
export const DEAD_ENDS:[string,string,string][]=[
 ['Cloud restarts wiped temporary storage (4 times)','a sweep’s checkpoints and a whole wave of runs','storage mirrored every 5 minutes; a restart watcher'],
 ['PDF parser settings changed after testing','about 10 GPU-hours of failed work','settings frozen between test and full run'],
 ['Training blow-ups noticed late (5 runs)','about 3 hours of H200 time','a divergence guard that rolls back automatically'],
 ['Shared disk quota ran out','two runs killed mid-training','a RAM disk for files that can be re-downloaded'],
 ['Thousands of single-file transfers','a 30-minute rate-limit lockout','bundled transfers, 8 → 88 MB/s'],
 ['AI edits landed on the wrong sentence','93 training and 27 test examples','one strict check shared by every pipeline'],
 ['First T2 build dropped long rows','all AI sections, caught before training','crop to a window around the AI span'],
 ['Best checkpoints pruned (7 runs)','the selected epochs of those runs','never prune the selected checkpoint'],
 ['Parallel AI sessions interfered','duplicate GPU runs and stale notes','handoff notes and one task per session'],
 ['Wrong claims later corrected (9)','misleading numbers for hours','corrected; every claim checked against logs'],
];
export const OVERRIDES:[string,string,string][]=[
 ['Keep ICLR 2027 flags private until drift is understood','Publish them as placeholders to get feedback','Still provisional; final scores before launch'],
 ['Rescale scores to a false-positive scale','Keep raw scores','Raw scores plus calibration became the standard'],
 ['Keep a baseline recipe alongside LoRA Without Regret','Use LoRA Without Regret only','Its high learning rate collapsed; switched to cosine with warmup'],
 ['Don’t use a vision-language PDF parser','Try it anyway','Too slow; cut to 300 papers'],
 ['Run exact 10%-length comparisons','Run a 20% MoE instead','Length vs recipe still not separated'],
 ['Use 3 seeds for splice wave 2','Use 1 seed, for speed','Lost in a restart before scoring'],
 ['Pause the GPUs to save money','Keep them running','GPUs are a grant; only storage costs money'],
 ['Set an API budget for data generation','Use subagents and a Codex handoff','~120M Claude tokens and 1,418 GPT edits'],
 ['Strip omission markers from text','Mark omissions on both human and AI text','Open; deprioritised'],
 ['Score all ICLR papers straight away','Benchmark small batches first','Throughput went from 14 to 23 windows a second'],
];
export const EFFORT:[string,string][]=[['90+','training runs'],['42,419','ICLR 2027 papers scored'],['~120M','tokens of AI data generated'],['41,000+','papers extracted and cleaned'],['2,400','human papers for calibration'],['~$30','paid compute; the rest donated']];
export const COMPUTE:[string,string,string][]=[
 ['Before Sep 30','Short GPU rentals, ~$30','vast.ai, paid by woog'],
 ['Sep 30','1× A100','Hugging Face community grant, through zmuwztd'],
 ['Oct 1','Grant raised to 4× A100','Hugging Face'],
 ['Oct 1','Shared 4× H200 node','nev. Shared with about 6 other users; 2–3 GPUs usable at a time'],
 ['Oct 5','Grant raised to 8× A100, the highest tier','Hugging Face'],
];
export const OPEN_PROBLEMS:[string,string][]=[
 ['Co-writing','The final MoE catches about a quarter of text co-written over 3 rounds.'],
 ['Calibration years','The cutoff was set on human papers from 2019–2022. Recent human papers may be flagged more often.'],
 ['False positives next to AI text','Human sentences beside AI text are flagged 5–7 times more often than in fully human papers.'],
 ['ICLR 2027’s flag rate','Not yet checked by reading flagged sentences or comparing with ICLR 2026.'],
 ['Light polishing','Can’t be measured sentence by sentence; needs a word-level metric.'],
 ['Never measured','Non-native and translated writing; fields outside machine learning. Only one training run and one seed.'],
 ['No real AI-assisted papers','Co-written documents stand in for them.'],
];
export const TIMELINE:[string,string,string][]=[
 ['Sep 25 – Oct 2','Collect and build data','14,561 older papers classified with MELD. 100,000 human passages, 30,000 AI mirrors, 600 synthetic papers.'],
 ['Oct 3','Six-backbone comparison','Decoders beat encoders. Becomes atlas-fast10, the first placeholder model.'],
 ['Oct 4 – 5','Overnight recipe sweep','About 40 runs. High learning rates collapse. Small edits stuck at 0.21–0.25 whatever the recipe.'],
 ['Oct 5','Find the data gap','Only 2 of 24,000 training windows hold a short AI span. ICLR 2027 scored and published as experimental.'],
 ['Oct 5 – 6','Splices','AI sentences swapped into human text: small edits 0.25 → 0.63, but only for one writer.'],
 ['Oct 6','Multi-writer data: T2, then T2.1','Claude and GPT edits, writers held out. Other models’ text 0.41 → 0.83.'],
 ['Oct 7','Final full-length runs','One pass over 72,000 windows with DeepSeek added, on three backbones. The MoE leads on every held-out set.'],
 ['Oct 8','Calibrate on human papers','Cutoff set on 2,403 human papers from 2019–2022: 1.0% of held-out human sentences flagged.'],
 ['Oct 11','Stricter default cutoff','Three archive papers that cite 2023–2025 work removed from calibration (2,400 papers). Default cutoff now flags 0.1% of human sentences.'],
 ['Oct 12','Oct 10 model live','Every paper rescored with the Oct 10 MoE (no stage 1, no segment or mixed loss). Its own cutoffs from the same 2,400 human papers; red marks scores of 0.38 or more. Appendices are now left out by default, as the bibliography already was.'],
];
