"""Collapsible takeaway panels for each results tab. Each claim cites the numbers it rests on."""


def panel(title, clear, likely, unclear, nxt):
    li = lambda items: ''.join(f'<li>{x}</li>' for x in items)
    return (f'<details class="takeaways"><summary>{title}</summary><div class="tk">'
            f'<div class="tkg clear"><h4>Clear signals</h4><ul>{li(clear)}</ul></div>'
            f'<div class="tkg likely"><h4>Likely, with some uncertainty</h4><ul>{li(likely)}</ul></div>'
            f'<div class="tkg unclear"><h4>Inconclusive</h4><ul>{li(unclear)}</ul></div>'
            f'<div class="tkg next"><h4>Highest-leverage next steps (few runs left)</h4><ol>{li(nxt)}</ol></div>'
            '</div></details>')


COMPARE = panel(
    'Takeaways: old vs new',
    clear=[
        '<strong>The same Qwen3.5 4B got much better.</strong> All-edit recall at the held-out 1% cutoff went from 0.21 (Oct 3) to 0.78–0.79 (overnight), consistent across 5 overnight runs (3 at 20%, 2 at full length).',
        '<strong>At the same full schedule, Qwen beats ModernBERT on paper_v3 and standalone rewrites</strong> (0.84–0.86 vs 0.69; 0.77 vs 0.70), with equal all-edit recall (0.78–0.82). The gaps are well beyond the seed spread seen on these metrics.',
        '<strong>Cutoffs need calibration data that looks like the test text.</strong> Cutoffs from the calibration windows produced 2–23% realized FPR; cutoffs from the dev half produced 0.6–2.1%.',
    ],
    likely=[
        '<strong>Most of the 4B jump comes from fixing the Oct 3 recipe, not from length alone.</strong> The Oct 3 run used micro-batch 1 and a loss-picked checkpoint (the earliest stage-2 epoch). Uncertainty: there is no overnight 10% run, so length (10% to 20%) and recipe changes cannot be separated.',
        '<strong>The MoE is the most efficient backbone per unit of training:</strong> at 10% it matches the full-length 9B on all edits (0.80) and leads on small edits (0.35). Uncertainty: one run, one seed, trained with the older recipe.',
        '<strong>The MoE retrained with the overnight recipe (20%) is the best edit detector:</strong> 0.84 all edits and 0.98 paragraph edits at 1.5% held-out FPR. Uncertainty: one seed; small edits (0.30) and standalone rewrites (0.55) are no better than the Oct 3 MoE.',
        '<strong>The full-length 9B is the best on standalone rewrites</strong> (0.77) and paper_v3 (0.86). Uncertainty: one seed; its public AUROC fell to 0.93.',
    ],
    unclear=[
        '<strong>Small-edit ranking between models.</strong> MoE 0.35, 9B 0.30–0.33, 4B 0.22–0.28 and ModernBERT 0.25 differ by less than the noise. The test half has about 150 small-edit sentences (48 one-sentence), so each estimate is roughly ±0.07, and across seeds the same 4B recipe varies by ±0.02–0.07.',
        '<strong>Public-text comparisons.</strong> Only about 200 public human documents per half, so 1% cutoffs realize at 2.5–6% FPR and public recall is noisy.',
        '<strong>Ettin</strong> is missing from these tables (incomplete saved scores).',
    ],
    nxt=[
        '<strong>Create sentence-scale training data:</strong> one- and two-sentence AI edits inside human paragraphs. Nothing else moved small edits, and only 2 of 24,000 training windows contain one.',
        '<strong>A full-length MoE run with the overnight recipe</strong> (about 5 hours on one H200, and it now resumes after a crash). At 20% it already leads on all edits and paragraph edits; length is what lifted paper_v3 and standalone rewrites for the Qwens.',
        '<strong>Build a calibration set from hard human text</strong> (sentences next to edits, paired originals, untouched passages from edited papers), so a deployable 1% cutoff can be fixed once and reused.',
        'If you want the length vs recipe question settled: a 10% overnight-recipe run of the 4B and 9B takes about 20 minutes each on an H200.',
    ])

SWEEP = panel(
    'Takeaways: overnight sweep',
    clear=[
        '<strong>A constant learning rate with no warmup is unstable at 5e-4 and above.</strong> Collapses appear on the 4B and 9B, at 20% and full length, across seeds; 1e-3 flags every human sentence.',
        '<strong>With cosine decay and warmup, learning rate barely matters at 20%:</strong> 1e-4, 2e-4, 3e-4 and 5e-4 all land at 0.77–0.78 all-edit recall (seed sd about 0.01).',
        '<strong>Training to full length improves paper_v3 sentences and standalone rewrites</strong> (paper_v3 0.77 to 0.84–0.89; standalone 0.51 to 0.70–0.80), on 2 seeds of the 4B and 1 of the 9B.',
        '<strong>H200s train about twice as fast as A100s</strong> here (20% 4B: 39 vs 72–88 minutes).',
    ],
    likely=[
        '<strong>Small edits are limited by training data, not the recipe.</strong> No change to learning rate, schedule, loss weight, model size or length moved small-edit recall off 0.21–0.26. Uncertainty: this is an inference; new data has not been tested yet.',
        '<strong>Raising the sentence-loss weight or the head learning rate doesn\'t help.</strong> K (3 seeds) matched the baseline. C (head LR) was worse, but it ran on the unstable schedule, so that result is confounded.',
        '<strong>5e-4 with cosine decay is fine at 20% but worse at full length</strong> on paper_v3 (0.66) and public AUROC (0.83). Uncertainty: one full-length seed.',
        '<strong>Standard AdamW does not rescue the constant schedule</strong> (G, 2 seeds). Uncertainty: it was not tested on the stable schedule.',
    ],
    unclear=[
        '<strong>9B vs 4B.</strong> Within noise at 20% (3 seeds each). At full length the 9B is better on small edits (0.26 vs 0.21–0.24) but worse on public AUROC (0.93 vs 0.97), from one 9B seed.',
        '<strong>Choosing checkpoints by dev recall instead of the final epoch.</strong> Every epoch was scored but not analyzed. The learning curves look flat after about 20%, so the gain is probably small.',
        '<strong>Short-span oversampling (L).</strong> It hurt paragraph edits, and small edits stayed flat. But it only reweights 1–3 sentence spans that already exist; true single-sentence spans are almost absent.',
        '<strong>Standalone-rewrite recall is very noisy</strong> across seeds (sd 0.16–0.22). Don\'t rank arms on it.',
    ],
    nxt=[
        '<strong>New sentence-scale edit data</strong> is the main lever left for small edits.',
        '<strong>Spend scarce runs at full length on an H200</strong> (about 3.5–4 hours for the 9B), with the stable recipe: 2e-4, cosine, warmup, micro-batch 8. Short runs plateau.',
        '<strong>Use dev-half cutoffs (or a hard-human calibration set) for every report,</strong> not a 0.5 cutoff or the calibration windows.',
    ])

OCT3 = panel(
    'Takeaways: Oct 3 backbone comparison',
    clear=[
        '<strong>Decoders generalize better to public text than encoders:</strong> AUROC 0.90–0.92 (Qwen) vs 0.73–0.74 (ModernBERT, Ettin). Ettin and the Qwens had the same 10% budget.',
        '<strong>The uncalibrated 0.5 cutoff is very conservative</strong> (0.1–0.5% human FPR), so recall at 0.5 understates what the models can do.',
        '<strong>Edits are caught mainly in context:</strong> the same rewritten paragraph is caught 92% of the time next to human text, but 13% standalone (MoE).',
    ],
    likely=[
        '<strong>The MoE was the best small-edit detector</strong> on the sentence-level ROC. Uncertainty: one seed per model and different training budgets.',
        '<strong>The weak Qwen3.5 4B result was a checkpoint-selection artifact.</strong> The overnight runs confirmed it: the same backbone reaches 0.78.',
    ],
    unclear=[
        '<strong>The backbone ranking beyond the MoE:</strong> single seeds, ModernBERT at full length against 10% for the rest, and loss-picked checkpoints.',
        '<strong>Ettin\'s sentence-level results</strong> are missing because its saved scores are incomplete.',
    ],
    nxt=[
        'These questions continue in the other two tabs. The main open item from this round is retraining the MoE with the improved recipe.',
    ])
