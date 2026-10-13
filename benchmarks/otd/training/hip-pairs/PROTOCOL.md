# HIP readiness decision

HIP provides a distinct potential *paired* document-supervision question: for the same source passage, can the detector rank the released AI paraphrase above its human original? This controls source/topic more directly than an unpaired mixture. It does not provide localized edit spans and is not a new source-domain collection: the underlying passages derive from RAID and MAGE.

Do not queue a generic HIP mixture merely to occupy an idle GPU. First inspect the ongoing GRADTEX auxiliary-document experiment. If a paired objective is then justified, use only source groups retained by this audit, make a deterministic grouped train/selection/calibration partition, and compare against a matched auxiliary-objective control. The released human-prefix evaluation overlaps the first 256 training rows; it is not an independent blind test and must not be represented as such. The current audit excludes matching source groups.

A future preparation needs whole-pair token-length eligibility, both texts retained without inventing token labels, a token/exposure budget, capped source repetition, and a source-matched pairwise-loss sanity check. Retain frozen paper evaluation and its calibration discipline. The dataset supplies neither the paper's generated continuations nor proof that these particular paraphrases evade the current detector. No new paid generation is required for this bounded candidate.

Official source: https://huggingface.co/datasets/YixuanEvenXu/HIP-training-and-evaluation-data
Pinned revision: b574d982f6fabda9dec95dcb62dec3723ab685f5
Remote audit: /data/workspace/paper-diversity-v1/public-source-audit/hip-audit/
