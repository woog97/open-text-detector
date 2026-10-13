# No-generation diversity pilot

Authorized October 1, 2026. Remote root: `/data/workspace/paper-diversity-v1`.

| Arm | GPU | Change |
|---|---:|---|
| control | 0 | Fresh paper-only ModernBERT-large LoRA control |
| raid | 1 | Replace ~20% of processed tokens with length-matched clean RAID abstracts |
| mage | 3 | Replace ~20% of processed tokens with length-matched MAGE human or topical/specified responses |
| boundary | 0, after control and its evaluations | Up to every fifth draw becomes a contiguous short-AI boundary crop from training papers |

GPU2's preexisting batch sweep is undisturbed. Each worker proceeds directly to calibration and both frozen evaluations, independently of other workers. The fourth arm uses the control's GPU only after it is released.

## Contracts

- Same pinned ModernBERT-large initialization, seed42, LoRA rank128/alpha32, attention+MLP coverage, effective batch32, microbatch8, optimizer, learning rates, stages and objectives as the existing paper recipe. All forwards BF16. All models/checkpoints remain on the Space.
- 6,000 stage1 draws; 12,000 draws in each of three stage2 epochs. Try every fifth position first, then seeded fallback positions until external token exposure reaches20%. Skip positions with no sufficiently long donor. Alternate external human/AI labels and sample source buckets uniformly among sufficiently long candidates. Actual counts in each `exposure.json`.
- External windows match replaced windows within3 source tokens; total processed real-token exposure must stay within0.5% of control per epoch. Padding compute and wall time may differ. No claim of identical human-token fractions or supervised-token exposure.
- Original selection and calibration windows stay fixed. Checkpoint selection retains minimum stage-specific validation loss to isolate this data comparison. Every arm receives the same calibration and workflow/comparison evaluations; evaluate all predeclared arms. Do not tune thresholds on test or retroactively pick test metrics. Promotion requires paper-localization benefit with acceptable observed human FPR and a follow-up seed.
- No OpenRouter or other generation API calls.

## Dataset provenance and splitting

RAID clean publisher file: `https://dataset.raid-bench.xyz/train_none.csv`; downloaded bytes pinned by SHA256. Card: https://huggingface.co/datasets/liamdugan/raid. Only clean abstracts used. Stable hash of source_id partitions80/10/10; only train groups used. A match from any generation excludes its entire source group. Keep human and generated variants together.

MAGE official revision and file digests are frozen in `download-complete.json`. Card: https://huggingface.co/datasets/yaful/MAGE. Only official train used; all official valid/test texts enter the exclusion index. Human label1 is remapped to internal0. Generated topical/specified responses become internal1. Exclude continuation setups because human-prefix provenance requires separate handling. Retain only domains containing both classes. Cap reservoir at700 eligible rows per source bucket before balanced sampling.

All current suite profiles (including full), original selection/calibration prepared files, and MAGE official valid/test enter a normalized exact-text and13-word shared-span exclusion index. Held-out shingles sampled every8 positions; candidates scanned at every position. This conservative lexical exclusion catches duplicates and substantial copied passages; it is **not proof of semantic or topic independence**, especially when MAGE lacks parent source IDs. Audit/pool counts, RAID split assignments, external draw IDs and source labels are persisted. External datasets are not used as new final benchmarks in this pilot.

Token labels describe text provenance: human source text or outputs from fully generated topical/specified/title prompts. Do not reuse this conversion for edited or mixed document labels. No original human prompt is prepended to generated training text.

## Short-AI boundary arm

Existing stage2 windows contain18–25 mixed examples per12,000draws with fewer than200AI characters. The boundary arm crops existing human→AI or AI→human transitions to20–200AI characters and2–25% AI by characters. It keeps a contiguous substring of an existing training example, with original regions projected to new offsets. No sentences from unrelated texts are spliced together. It replaces eligible fifth-position draws at matched length; actual replacement and class-token fractions may differ from control. This tests a data package, not a perfectly isolated AI-fraction intervention.

## Operations

`prepare.py` performs the external-data audit and prepares three trials. `prepare_boundary.py` prepares the fourth. `worker.py` preflights, trains, calibrates and evaluates one trial. Remote `boundary_waiter.py` handles GPU0 handoff. Per-arm process/status records and append-only phase logs remain on the Space. Never overwrite partial or completed runs when recovering.

The existing30-minute heartbeat supervises these experiments and reports progress; it must stay active until these and earlier authorized work finish.
