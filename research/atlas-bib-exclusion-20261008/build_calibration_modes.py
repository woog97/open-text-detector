"""Calibration page data for the final MoE under each counting mode (bibliography and appendices in or out).

Runs research/atlas-rescore-moe-20261007/build_moe_calibration.py unchanged, once per mode, with two substitutions:
sentence scores keep only the sentences the mode counts (bib.py sections), and the thresholds are that mode's
(recompute.py calibrate). The ICLR 2027 year row uses the per-paper counts from recompute.py atlas. Document-level
curves do not depend on the mode (they use whole-document scores). Mode 'all' must reproduce the published page data.
Usage: build_calibration_modes.py WORK OUT_DIR
WORK: the recompute.py work directory plus suite-sentences-moe.json.gz, out-suite/, out-yd/, out-space/data,
out-h200/data, yd-text/ (year-diagnostic-v1 texts), modes-*.json and calibration-modes.json.
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import build_moe_calibration as bm  # noqa: E402
import cal_analyze as ca  # noqa: E402
from bib import MODES, included, section_of, sections  # noqa: E402
from recompute import sentence_scores_spans  # noqa: E402

FILES = {'all': 'qwen36-35b-a3b-final.json', 'nobib': 'qwen36-35b-a3b-final-nobib.json',
         'noapp': 'qwen36-35b-a3b-final-noapp.json', 'body': 'qwen36-35b-a3b-final-body.json'}
LABEL = {'all': 'All sentences', 'nobib': 'Bibliography excluded', 'noapp': 'Appendices excluded', 'body': 'Main text only (bibliography and appendices excluded)'}


def main(work, out_dir):
    work = Path(work); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = work / 'calroot'; (root / 'staging').mkdir(parents=True, exist_ok=True)
    for name in ('suite-sentences-moe.json.gz', 'out-cal', 'out-suite', 'out-yd', 'out-space', 'out-h200'):
        if not (root / name).exists(): (root / name).symlink_to(work / name)
    if not (root / 'staging' / 'atlas-details-index-moe.parquet').exists():
        (root / 'staging' / 'atlas-details-index-moe.parquet').symlink_to(work / 'atlas-details-index-moe.parquet')
    bm.D = root; bm.HC = work; bm.CAL_TEXT = work / 'cal-text'; bm.OLD_TEXT = work / 'old-text'; bm.YD_TEXT = work / 'yd-text'
    cal = json.loads((work / 'calibration-modes.json').read_text())['modes']
    meta = {p['id']: p for p in json.loads((work / 'selection.json').read_text())['papers']}
    groups = sorted({(m['venue'], m['year']) for m in meta.values()} | {('archive', 2022)})
    counts = {}
    for f in sorted(glob.glob(str(work / 'modes-[0-9].json'))): counts.update(json.load(open(f))['papers'])
    original_score, original_row = ca.sentence_scores, bm.row
    for mode in MODES:
        def scores(text, off, prob, mode=mode):
            s, starts = sentence_scores_spans(text, off, prob); spans, tail = sections(text)
            keep = np.array([included(section_of(a, spans, tail), mode) for a in starts], bool)
            return s[keep] if len(s) else s
        def row(label, kind, year, ps, h99, mode=mode):
            if label == 'ICLR 2027 submissions':  # per-paper counts for this mode; document scores are unchanged
                docs = [p['doc'] for p in ps]
                ps = [{'share': c['modes'][mode][0] / max(c['modes'][mode][1], 1), 'doc': None} for c in counts.values()]
                ps[:len(docs)] = [{**p, 'doc': d} for p, d in zip(ps, docs)]
            return original_row(label, kind, year, ps, h99)
        ca.sentence_scores = scores if mode != 'all' else original_score
        bm.row = row
        block = cal[mode]; block['groups'] = [{'venue': v, 'year': y} for v, y in groups]
        (root / 'calibration-moe.json').write_text(json.dumps({'variants': {'kept+old (as Atlas calibration)': block}}))
        out = out_dir / FILES[mode]
        bm.main(out)
        data = json.loads(out.read_text()); t1 = next(t['threshold'] for t in data['thresholds'] if t['target_fpr'] == 0.01)
        data['counting'] = {'mode': mode, 'include_bibliography': MODES[mode][0], 'include_appendices': MODES[mode][1], 'label': LABEL[mode],
                            'note': ('Sentences in bibliographies and appendices are found from the extracted text (research/atlas-bib-exclusion-20261008/bib.py); '
                                     'the threshold is recalibrated on the human papers with the same sentences left out. Document-level curves use whole documents in every mode.')}
        data['caveats'] = [c.replace('about 0.006', f'about {t1:.4f}') for c in data['caveats']]
        out.write_text(json.dumps(data, separators=(',', ':')))
    ca.sentence_scores, bm.row = original_score, original_row


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
