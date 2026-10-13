"""MoE Atlas sentence thresholds without the three archive papers that are not from 2022 or earlier (Oct 11 2026).

The 215-paper archive (human-pre2023-batch001-v1) labels these TMLR 2022, but their text cites 2023-2025 work
(UMP-Net builds on LLaMA, LLaMA-Adapter and LLaMA-Excitor). UMP-Net alone held 160 of the 825 human sentences above the
nobib 0.1% threshold. Same code as research/atlas-bib-exclusion-20261008/recompute.py calibrate with fpr10.py's
targets (0.1%, 1%, 5%, 10%); the only change is that EXCLUDE is dropped from every mode.
Usage: recalibrate.py WORK OUT.json   (WORK as for recompute.py: out-cal/, cal-text/, old-text/, overlap.json, selection.json)
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008')); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import cal_analyze as ca  # noqa: E402
import recompute as rc  # noqa: E402

EXCLUDE = {
    'old:a46aaf09ea4a50245ec81d5b6727af529416788f303f8cf1155fafe13f047010': 'UMP-Net (forum EehtvgNXAl): LLaMA, LLaMA-Adapter, LLaMA-Excitor',
    'old:857075500638db9ce6edcf32d990aa77a686bb4fbcd77126a8ed231cf52b78e4': 'Scalable Generative Modeling of Weighted Graphs (forum yWKkBOcD18): cites 2024-2025 work',
    'old:06ab694121b8f93ca2ef6aecaee2139d0ad417c4255a12a89930e0b1fb42cdbc': 'Adaptive Self-Distillation for Federated Learning (forum K58n87DE4s): cites 2024-2025 work',
}


def main(work, out_json):
    ca.TARGETS = [0.001, 0.01, 0.05, 0.1]
    by_mode, found, shares = rc.human_scores_by_mode(work)
    missing = set(EXCLUDE) - set(by_mode['all']); assert not missing, missing
    res = {'excluded': EXCLUDE, 'modes': {}}
    for m, cal in by_mode.items():
        for mode_name, c in ((m + '/before', cal), (m, {k: v for k, v in cal.items() if k not in EXCLUDE})):
            ca.rng = np.random.default_rng(20261007)  # same seed for every block, so before/after differ only by the papers
            block, share, _ = ca.sentence_block(c)
            res['modes'][mode_name] = dict(block, human_paper_flag_share_p50=float(np.median(list(share.values()))),
                                           human_paper_flag_share_p99=float(np.quantile(list(share.values()), .99)))
            print(mode_name, block['n_papers'], block['n_sentences'], [(t['target_fpr'], round(t['threshold'], 6), [round(x, 5) for x in t['threshold_ci95']], round(t['heldout_fpr'], 6)) for t in block['thresholds']], flush=True)
    Path(out_json).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
