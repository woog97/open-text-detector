import pytest
import torch
from open_text_detector_backend.providers.meld_v8_attention import tiled


@pytest.mark.parametrize("length", [1, 17, 64, 65, 127, 128, 129, 257])
def test_tiled_attention_preserves_window_edges(length):
    torch.manual_seed(19)
    q, k, v = [torch.randn(2, 3, length, 64) for _ in range(3)]
    positions = torch.arange(length)
    mask = (positions[:, None] - positions[None, :]).abs() <= 64
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(tiled(q, k, v), expected, atol=1e-6, rtol=1e-5)


def test_tiled_attention_does_not_leak_across_windows():
    torch.manual_seed(20)
    q, k, v = [torch.randn(1, 2, 257, 64) for _ in range(3)]
    original = tiled(q, k, v)
    changed = v.clone()
    changed[:, :, 200:] += 100
    actual = tiled(q, k, changed)
    torch.testing.assert_close(actual[:, :, :136], original[:, :, :136], atol=1e-6, rtol=1e-5)
