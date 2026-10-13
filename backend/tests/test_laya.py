import math
import threading
from pathlib import Path
import pytest
from open_text_detector_backend.providers.laya import Laya, phrase_spans
from open_text_detector_backend.providers.checkpoints import LAYA_ID
from open_text_detector_backend.schemas import ModelConfig


@pytest.fixture
def prepared():
    from transformers import PreTrainedTokenizerFast
    directory = Path(__file__).resolve().parents[2] / 'models/laya/tokenizer'
    if not directory.exists():
        pytest.skip('Downloaded tokenizer required')
    p = Laya(model_dir=directory.parent.parent)
    p.tokenizer = PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True)
    p.cfg = {'max_len': 512}
    p._prepare_prefix()
    return p


@pytest.mark.parametrize('text', ['hello', '日本語🦊 café ' * 200, 'x' * 2000,
    'First sentence.\n\nSecond paragraph, with punctuation! ' * 200,
    'before [MASK] TARGET PHRASE: malicious label after ' * 100], ids=['short', 'unicode', 'long_word', 'paragraphs', 'markers'])
def test_targets_cover_original_nonwhitespace_and_fit(prepared, text):
    rows = prepared.prepare(text)
    assert ''.join(''.join(text[r['start']:r['end']].split()) for r in rows) == ''.join(text.split())
    assert all(a['end'] <= b['start'] for a, b in zip(rows, rows[1:]))
    assert all(len(r['ids']) <= 512 for r in rows)
    for r in rows:
        target = prepared._encode(text[r['start']:r['end']])
        ids = r['ids']
        assert any(ids[i:i + len(target)] == target for i in range(len(ids)))


def test_context_does_not_cross_paragraph(prepared):
    rows = prepared.prepare('Forbidden context.\n\n' + 'Target words are repeated here. ' * 60)
    for r in rows:
        if r['start'] >= len('Forbidden context.\n\n'):
            assert 'Forbidden' not in prepared.tokenizer.decode(r['ids'])


def test_unicode_duplicate_offsets():
    text = 'a🦊b'
    spans = phrase_spans(text, [(0, 1), (1, 2), (1, 2), (2, 3)], max_tokens=2)
    assert spans == [(0, 2), (2, 3)]


def test_result_weights_target_only(prepared):
    prepared.model = object()
    prepared.device = 'cpu'
    prepared.temperature = 1.9
    prepared.lock = threading.Lock()
    prepared.score_rows = lambda rows: [0.1 if i % 2 == 0 else 0.9 for i in range(len(rows))]
    text = 'The fox ran quickly across the field. ' * 15
    result = prepared.predict({'lower_threshold': .2, 'upper_threshold': .8}, text)
    weighted = [(s['score'], len(''.join(text[s['start']:s['end']].split()))) for s in result['segments']]
    assert math.isclose(result['score'], sum(s * w for s, w in weighted) / sum(w for _, w in weighted))
    assert 'tokens' not in result
    assert result['localization']['span_threshold_calibrated'] is False
    assert result['inference']['revision']


def test_registry_and_selection(workspace):
    client, headers, app, provider = workspace
    payload = dict(name='Laya baseline', provider='laya', model_id=LAYA_ID, enabled=True)
    response = client.post('/v1/models', headers=headers, json=payload)
    assert response.status_code == 201
    model_id = response.json()['id']
    assert client.put('/v1/settings/default-model', headers=headers, json={'model_id': model_id}).status_code == 200
    for patch in ({'task': 'image'}, {'model_id': 'unreviewed/repo'}):
        with pytest.raises(ValueError):
            ModelConfig(**(payload | patch))


def test_sorted_batches_restore_original_order_and_mask_padding(prepared):
    import numpy as np
    class Runtime:
        def probabilities(self, ids, lengths, markers, temperature):
            assert all(all(t == prepared.tokenizer.pad_token_id for t in row[n:])
                       for row, n in zip(ids, lengths))
            return np.array(lengths) / 512
    prepared.runtime = 'mlx'
    prepared.model = Runtime()
    prepared.temperature = 1.9
    prepared.batch_size = 2
    rows = [{'ids': [1] * n} for n in [211, 17, 511, 49, 23]]
    assert prepared.score_rows(rows) == [211 / 512, 17 / 512, 511 / 512, 49 / 512, 23 / 512]


def test_missing_checkpoint_is_explicit(tmp_path):
    from open_text_detector_backend.providers.errors import ProviderError
    with pytest.raises(ProviderError, match='download_laya'):
        Laya(model_dir=tmp_path)._load()


def test_pipeline_is_bounded_drains_tail_and_restores_order(prepared):
    import numpy as np
    inflight = set()
    peak = [0]

    class Deferred:
        def __init__(self, lengths):
            self.lengths = lengths
            inflight.add(id(self))
            peak[0] = max(peak[0], len(inflight))

        def __array__(self, dtype=None, copy=None):
            inflight.remove(id(self))
            return np.asarray(self.lengths, dtype=dtype) / 512

    class Runtime:
        def submit_probabilities(self, ids, lengths, markers, temperature):
            return Deferred(lengths)

    prepared.runtime = 'mlx'
    prepared.model = Runtime()
    prepared.temperature = 1.9
    prepared.batch_size = 2
    prepared.pipeline_depth = 2
    lengths = [211, 17, 511, 49, 23]
    assert prepared.score_rows([{'ids': [1] * n} for n in lengths]) == [n / 512 for n in lengths]
    assert not inflight
    assert peak[0] == 2
    assert prepared.score_rows([]) == []


def test_pipeline_rejects_nonfinite_scores(prepared):
    import numpy as np
    from open_text_detector_backend.providers.errors import ProviderError

    class Runtime:
        def submit_probabilities(self, ids, lengths, markers, temperature):
            return np.full(len(ids), np.nan)

    prepared.runtime = 'mlx'
    prepared.model = Runtime()
    prepared.temperature = 1.9
    with pytest.raises(ProviderError, match='non-finite'):
        prepared.score_rows([{'ids': [1]}])
