"""Create ID-only assignments for the user-requested paper-editing trials."""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research/data/paper-eval-workflows-luna-20260930'
OUT = Path(__file__).resolve().parent
SEED = 'stage3-paper-edit-half-split-20261002-v1'
CONDITIONS = {'proofread', 'light_polish', 'substantial_rewrite'}

def read(name):
    return [json.loads(line) for line in (SOURCE / name).read_text().splitlines()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    papers = {p['paper_id']: p for p in read('papers.jsonl')}
    rows = read('dataset.jsonl')
    before = {name: digest(SOURCE / name) for name in ['papers.jsonl', 'dataset.jsonl']}
    parent = {pid: pid for pid in papers}

    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    owners = {}
    for pid, p in papers.items():
        for name in p['authors']:
            key = re.sub(r'\W+', '', unicodedata.normalize('NFKC', name).casefold())
            if not key:
                continue
            if key in owners:
                a, b = find(pid), find(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = pid
    groups = defaultdict(set)
    for pid in papers:
        groups[find(pid)].add(pid)

    def subset(available, count, salt):
        candidates = [g for g in groups.values() if g <= available]
        candidates.sort(key=lambda g: hashlib.sha256((SEED + salt + '|'.join(sorted(g))).encode()).hexdigest())
        choices = {0: set()}
        for group in candidates:
            for size, selected in list(choices.items())[::-1]:
                total = size + len(group)
                if total <= count and total not in choices:
                    choices[total] = selected | group
        assert count in choices, 'Requested count cannot preserve author groups'
        return choices[count]

    # Keep the fixed evaluation half inside the previous test split, avoiding
    # pilot/calibration papers that were used while designing the generation.
    old_test = {pid for pid, p in papers.items() if p['split'] == 'test'}
    evaluation = subset(old_test, 95, 'evaluation')
    train_pool = set(papers) - evaluation
    half_train = subset(train_pool, 47, 'train50')
    arms = {'0': set(), '50': half_train, '100': train_pool}
    assert len(papers) == 189 and len(train_pool) == 94
    assert not evaluation & train_pool and half_train <= train_pool
    assert all(g <= evaluation or g <= train_pool for g in groups.values())
    assert all(g <= half_train or not g & half_train for g in groups.values())
    edits = [r for r in rows if r['condition'] in CONDITIONS]
    assert len({r['id'] for r in rows}) == len(rows)
    assert all(Counter((r['condition'], r['view']) for r in edits if r['paper_id'] == pid).values() for pid in papers)
    assert len({(r['paper_id'], r['condition']) for r in edits}) == 567
    for ids in [evaluation, train_pool, half_train]:
        selected = [r for r in edits if r['paper_id'] in ids]
        assert len(selected) == len(ids) * 6
        assert len({(r['paper_id'], r['condition']) for r in selected}) == len(ids) * 3

    assignments = [{
        'paper_id': pid, 'original_split': papers[pid]['split'],
        'role': 'fixed_evaluation' if pid in evaluation else 'training_trial_pool',
        'author_group_id': find(pid),
        'training_arms_percent': [int(k) for k, v in arms.items() if pid in v],
        'conference': papers[pid]['conference'], 'year': papers[pid]['year'],
    } for pid in sorted(papers)]
    row_assignments = [{
        'id': r['id'], 'paper_id': r['paper_id'], 'condition': r['condition'], 'view': r['view'],
        'role': 'fixed_evaluation' if r['paper_id'] in evaluation else 'training_trial_pool',
        'selected_for_stage3': r['condition'] in CONDITIONS,
        'training_arms_percent': [int(k) for k, v in arms.items() if r['paper_id'] in v and r['condition'] in CONDITIONS],
    } for r in rows]
    manifest = {
        'version': 1, 'date': '2026-10-02', 'seed': SEED,
        'authorization': 'User accepts GRADTEX as-is without another deep review; set HIP aside; split paper edits approximately in half, vary use of one half while always reserving the other for evaluation.',
        'sources': {'GRADTEX': 'accepted existing released training data; no additional deep quality audit requested', 'HIP': 'excluded from current stage-3 plan, files preserved'},
        'scope': 'Three editing conditions only; IDs for all other views/conditions carry source-level exclusion roles, not new training authorization.',
        'source_directory': str(SOURCE.relative_to(ROOT)), 'source_sha256': before,
        'conditions': sorted(CONDITIONS),
        'grouping': 'All variants and views of a paper stay together; connected normalized author-name groups stay together.',
        'evaluation_selection': 'Deterministic 95-paper subset of the former 108-paper test split; no detector scores used.',
        'fixed_evaluation': {'papers': 95, 'distinct_edits': 285, 'rows_including_both_views': 570, 'paper_ids': sorted(evaluation)},
        'training_trial_pool': {'papers': 94, 'distinct_edits': 282, 'rows_including_both_views': 564, 'paper_ids': sorted(train_pool)},
        'trial_arms': {k: {'training_papers': len(v), 'distinct_training_edits': len(v)*3, 'rows_including_both_views': len(v)*6, 'paper_ids': sorted(v)} for k, v in arms.items()},
        'future_evaluation_rules': [
            'Always exclude the fixed evaluation papers, their other paragraphs, variants and author groups from training.',
            'At 0%, no paper-editing examples from the trial pool are added; it does not mean zero GRADTEX or zero overall training.',
            'Do not compare these trials on the old full evaluation suite: remove every trial-pool paper from evaluation in every arm, even at 0%.',
            'Prior benchmark exposure remains historical exposure; the fixed half is not claimed as a newly unseen test.',
            'Use separate development data for thresholds and model selection; never fit them on the fixed evaluation half.',
            'Keep other training data and exposure policy matched across arms; decide the token budget at training time.',
        ],
        'materialization': 'ID-only overlay; original datasets and historical splits remain unchanged. Future loaders must use this overlay.',
        'validation': {'passed': True, 'paper_overlap': 0, 'normalized_author_group_overlap': 0, 'fifty_percent_arm_nested': True, 'all_edit_views_accounted_for': True},
    }
    assert before == {name: digest(SOURCE / name) for name in before}
    for name, data in [('paper-assignments.jsonl', assignments), ('row-assignments.jsonl', row_assignments)]:
        (OUT / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in data))
    (OUT / 'split-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'evaluation': 95, 'training_pool': 94, 'training_edits_by_arm': {k: len(v)*3 for k, v in arms.items()}, 'validation': manifest['validation']}))

if __name__ == '__main__':
    raise SystemExit('Cancelled by user; do not recreate or apply this split.')
