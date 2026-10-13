import json, re
from pathlib import Path
NEVER_TRAIN_FILES = [Path('/tmp/pangram-t2-20261006/inputs') / n for n in
 ['heldout-eval-never-train.json', 'fullpapers-heldout-never-train.json', 'sections-heldout-never-train.json', 'hosted-heldout-never-train.json', 't21-heldout-never-train.json']]

def norm_hash(t):
    import hashlib
    return hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()

def never_train_hit(row, _cache={}):
    """For training builders: True if a row must not be used for training (held-out eval paper or paragraph).
    Loads every file in NEVER_TRAIN_FILES (sentence-edit, full-paper and section held-out lists). Also checks row['id'] parts.

    Checks the row's paper_id/group (with or without a 'paper:' prefix) against the never-train paper list, and every
    blank-line-separated paragraph of row['text'] (normalized: lowercase, non-word characters removed) against the
    paragraph hashes. Rows whose paragraphs were re-split differently should additionally be screened with the
    hashed 60-character normalized shingles ('paragraph_shingle_sha256_12', stored at every 10th offset; the row is queried
    at every offset, so any copied held-out stretch of 70+ normalized characters is found)."""
    if not _cache:
        _cache.update(papers=set(), hashes=set(), shingles=set())
        for f in NEVER_TRAIN_FILES:
            if f.exists():
                d = json.loads(f.read_text())
                _cache['papers'] |= set(d['paper_ids']); _cache['hashes'] |= set(d['paragraph_sha256'])
                _cache['shingles'] |= set(d['paragraph_shingle_sha256_12'])
    ids = {str(row.get(k, '')).replace('paper:', '').split('/')[0] for k in ('paper_id', 'group', 'group_id', 'seed_id', 'item_id')}
    ids |= {x for x in re.split(r'[/]', str(row.get('id', ''))) if x.startswith(('claude-fp-', 'claude-sec-'))} | {str(row.get('id', '')).replace('fullpaper-', '')}
    if ids & _cache['papers']:
        return True
    text = row.get('text', '')
    # Skip short blocks (headings like '## 1 Introduction', boilerplate lines): they recur across unrelated documents.
    if any(norm_hash(p) in _cache['hashes'] for p in text.split('\n\n') if len(re.sub(r'\W+', '', p)) >= 100):
        return True
    import hashlib
    n = re.sub(r'\W+', '', text.lower())
    return any(hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] in _cache['shingles'] for i in range(max(0, len(n) - 59)))
