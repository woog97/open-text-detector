"""Versioned character-span labels; original text and earlier labels are immutable."""
import hashlib
import re
from collections import Counter

LABELS = {'human': 0, 'ai_assisted': 1, 'ai_generated': 2}
SCHEMA = 'pangram-stage4-character-labels-v1'


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def summarize(text, spans):
    """Require exact code-point coverage and compute the report's weighted fraction."""
    cursor = 0
    counts = Counter({name: 0 for name in LABELS})
    for span in spans:
        assert span['label'] in LABELS
        assert type(span['start']) is int and type(span['end']) is int
        assert span['start'] == cursor < span['end'] <= len(text)
        counts[span['label']] += span['end'] - span['start']
        cursor = span['end']
    assert cursor == len(text)
    assert len(text) > 0
    return {
        'character_counts': dict(counts),
        'weighted_ai_fraction': (0.5 * counts['ai_assisted'] + counts['ai_generated']) / len(text),
        'offset_unit': 'unicode_code_points',
        'offset_convention': 'start_inclusive_end_exclusive',
        'whitespace_policy': 'all characters counted; intervening whitespace belongs to its assigned span',
    }


def record(identifier, dataset, text, spans, method, **metadata):
    assert identifier and text
    return {
        'schema': SCHEMA, 'id': identifier, 'dataset': dataset,
        'text_sha256': text_hash(text), 'text_length': len(text),
        'labels': LABELS, 'spans': spans, 'label_method': method,
        **summarize(text, spans), **metadata,
    }


def uniform(identifier, dataset, text, label, **metadata):
    assert label in LABELS
    return record(identifier, dataset, text, [{'start': 0, 'end': len(text), 'label': label}],
                  'known_writing_process', **metadata)


def project_offsets(text, spans, offsets):
    """Unambiguous token projections; tokens crossing unlike labels are masked."""
    summarize(text, spans)
    labels, mask = [], []
    for start, end in offsets:
        assert 0 <= start <= end <= len(text)
        hits = {s['label'] for s in spans if s['start'] < end and s['end'] > start}
        if end > start and len(hits) == 1:
            labels.append(LABELS[hits.pop()]); mask.append(True)
        else:
            labels.append(-100); mask.append(False)
    return labels, mask


def word_offsets(text):
    return [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
