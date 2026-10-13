# Prepared design: paired ranking on known paper targets

Status: implementation and existing-TRAIN pair pool audited. User authorized this independent existing-paper hypothesis without waiting for GRADTEX. Registration remains gated on shared budget checks and pair-aware BF16 preflight; verify dispatcher state for actual launch status.

Question: does a small relative-ordering loss help distinguish a generated replacement from the matched original, beyond the existing absolute token classification objective? Ranking alone cannot control human false positives, so absolute classification stays intact.

The primary comparison requires TWO isolated arms: identical pair-batched data with coefficient0, versus coefficient0.1. No additional model head. Historical control serves only as context for the data/batching change. Stage1 unchanged. Stage2 retains original token/sentence/segment/mixed losses and adds ranking on verified matched target spans. Checkpoint selection remains original combined paper validation loss, and calibration/test suites remain frozen.

Score is mean tokenhead AI-minus-human logit over nonwhitespace verified target tokens. Pair loss is softplus(score_human-score_AI). Multiply mean pair loss by coefficient0.1 and fraction of microbatch rows participating in pairs. The derivative is bounded by this coefficient; the loss itself is intentionally not hard-clipped. Coefficient0 is exactlyzero. Both members must occur in the same microbatch; incomplete or repeated memberships are rejected. Model forwards must remain BF16; synthetic-logit tests use BF16 without model inference.

Dataset preparation audited original TRAIN pair_id, same paper, exact preserved prefix/suffix, non-identical generated target, and pure0human/pure1AI target token provenance. It produced1572pairs across790paper groups; one pair was excluded for ambiguous target token provenance. Full target crops fit510rawtokens+2specials. This is observed generation provenance, not inferred document labels. Existing source exclusions and paper-split limits remain.

Required budget manifest before launch:
- Keep6000stage1 and12000rows perstage2epoch:42kdraws total. A pair consumes TWO rows, never one effective-batch slot.
- Replace about10%stage2 processed input tokens with paired targets; record actual share, absolute token totals, class exposure, source repetition, and discarded candidates.
- Deterministically place complete pairs in adjacent slots within eight-row microbatches. Both arms use byte-identical manifests/order and effectivebatch32.
- Match each substituted row's processed token length within3 where possible, require overall epoch token totals within0.5%, cap each source paper at3pair draws perepoch, and freeze the manifest before either arm trains.
- Paired-target ordering and source exposure differ from historical control. Neither arm should be compared to historical control as a pure objective ablation.

Remaining GPU readiness checks: build/freeze identical paired budgets; validate all adapter rows and pair membership at every microbatch boundary; run the newly implemented real BF16 preflight on a complete pair, alongside longest-row memory checks; test frozen-weight gradients and state roundtrip; verify code/config/data hashes. The registration script publishes dispatcher readiness only after algebra and data checks; worker preflight must pass before training.

HIP extension: use the data agent's strengthened source-grouped v2manifest only. HIP has whole-document paraphrase provenance, not token spans. A HIP pair must retain absolute document-level classification plus ranking, without assigning AI labels to every token. The paper-only integration patch MUST NOT be reused to turn HIP labels into token gold. It needs a separate document-supervision integration and matched coefficient0control.
