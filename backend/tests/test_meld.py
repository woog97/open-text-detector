import importlib.util
import math
import pytest
from open_text_detector_backend.providers.localization import localize, window_starts
from open_text_detector_backend.providers.checkpoints import MELD_ID
from open_text_detector_backend.schemas import ModelConfig


def test_localization_preserves_unicode_and_sentence_partition():
    text = 'One 🧪.\n\nNext sentence! '
    # Two byte-level tokens both refer to the emoji. Their display must not duplicate it.
    offsets = [(0, 3), (4, 5), (4, 5), (5, 6), (8, 12), (13, 21), (21, 22)]
    tokens, segments = localize(text, offsets, [0, 2, 4, 0, 4, 4, 4], 1.9)
    assert tokens[1]['raw_score'] == 3
    assert tokens[1]['token_count'] == 2
    assert all(a['end'] <= b['start'] for a, b in zip(tokens, tokens[1:]))
    assert ''.join(text[s['start']:s['end']] for s in segments) == text
    assert segments[0]['label'] == 'low_evidence'
    assert segments[-1]['label'] == 'ai_evidence'


@pytest.mark.parametrize('length', [1, 2046, 2047, 4000, 10001])
def test_windows_cover_entire_document(length):
    covered = set()
    starts = window_starts(length, 2046, 256)
    for start in starts:
        covered.update(range(start, min(start + 2046, length)))
    assert covered == set(range(length))
    assert len(starts) == len(set(starts))
    assert starts[-1] == max(0, length - 2046)


def test_meld_model_validation_and_registry(workspace):
    c, admin, _, _ = workspace
    payload = dict(name='MELD v5', provider='meld', model_id=MELD_ID, enabled=True)
    response = c.post('/v1/models', headers=admin, json=payload)
    assert response.status_code == 201
    assert response.json()['provider'] == 'meld'
    with pytest.raises(ValueError):
        ModelConfig(**(payload | {'model_id': 'unreviewed/repo'}))
    with pytest.raises(ValueError):
        ModelConfig(**(payload | {'task': 'image'}))


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='optional ML dependencies')
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_predict_excludes_special_tokens_and_stitches_without_duplication(batch_size):
    import threading
    from open_text_detector_backend.providers.meld import Meld

    class Tokenizer:
        cls_token_id = sep_token_id = 999
        def __call__(self, text, **kwargs):
            return {'input_ids': list(range(1, len(text) + 1)),
                    'offset_mapping': [(i, i+1) for i in range(len(text))]}

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def prepare_for_model(self, ids, **kwargs):
            return {'input_ids': [999] + ids + [999], 'attention_mask': [1] * (len(ids)+2),
                    'special_tokens_mask': [1] + [0] * len(ids) + [1]}

    class Model:
        cfg = {'max_length': 300, 'rho': .25, 'score_offsets': {'overall': {'fpr_0.01': 1.9}}}

        def token_scores(self, input_ids, attention_mask):
            return input_ids.float() / 100

    provider = Meld.__new__(Meld)
    provider.batch_size = batch_size
    provider.lock = threading.Lock()
    provider.model, provider.tokenizer, provider.device = Model(), Tokenizer(), 'cpu'
    text = 'word. ' * 100
    result = provider.predict({}, text)
    expected = sum(range(451, 601)) / 150 / 100
    assert math.isclose(result['raw_score'], expected, abs_tol=1e-6)
    assert len(result['tokens']) == len(text)
    assert result['tokens'][-1]['end'] == len(text)
    assert result['inference']['windows'] > 1
    assert result['score_type'] == 'ai_evidence'
    assert result['label'] == 'ai_evidence'
    assert result['localization']['span_threshold_calibrated'] is False
    assert provider.predict({}, 'short')['label'] == 'uncertain'
