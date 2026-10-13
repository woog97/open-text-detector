"""Verify custom classifier weights survive loading; never test with real weights."""
import importlib.util
import pytest
from open_text_detector_backend.schemas import ModelConfig
from open_text_detector_backend.providers.checkpoints import QWEN_ID


def test_merged_model_config():
    value = ModelConfig(name='Qwen EditLens', provider='editlens', model_id=QWEN_ID)
    assert value.base_model_id == QWEN_ID
    with pytest.raises(ValueError):
        ModelConfig(name='Wrong base', provider='editlens', model_id=QWEN_ID, base_model_id='Qwen/Qwen3-4B')


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='optional ML dependencies')
def test_normed_head_roundtrip_and_reject_missing_head(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForSequenceClassification
    from open_text_detector_backend.providers.qwen import EditLensQwen, load_merged_qwen
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        head_dim=8, num_labels=4, pad_token_id=0)
    torch.manual_seed(7)
    model = EditLensQwen(config).eval()
    with torch.no_grad():
        model.score.norm.weight.fill_(1.7)
        model.score.norm.bias.fill_(.3)
        model.score.linear.weight.normal_()
    model.save_pretrained(tmp_path / 'valid')
    loaded = load_merged_qwen(tmp_path / 'valid', torch.float32).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        assert torch.allclose(model(inputs).logits, loaded(inputs).logits)
    assert torch.equal(model.score.linear.weight, loaded.score.linear.weight)
    Qwen3ForSequenceClassification(config).save_pretrained(tmp_path / 'invalid')
    with pytest.raises(ValueError, match='Checkpoint tensors'):
        load_merged_qwen(tmp_path / 'invalid', torch.float32)
