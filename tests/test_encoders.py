import pytest
import torch

from stt.encoders import Conv2dSubsampling, MultiHeadAttention, build_encoder, lengths_to_mask
from tests.conftest import micro_model_cfg


def test_lengths_to_mask():
    m = lengths_to_mask(torch.tensor([3, 1]), 4)
    assert m.tolist() == [[True, True, True, False], [True, False, False, False]]


@pytest.mark.parametrize("factor", [2, 4])
def test_subsampling_lengths_match_output(factor):
    sub = Conv2dSubsampling(40, 16, 8, factor)
    x = torch.randn(2, 37, 40)
    y, lengths = sub(x, torch.tensor([37, 20]))
    assert y.shape[:2] == (2, lengths.max())
    assert int(lengths[0]) == y.size(1)


@pytest.mark.parametrize("encoder", ["bilstm", "transformer", "conformer"])
def test_encoders_contract_and_padding_invariance(encoder):
    torch.manual_seed(0)
    enc = build_encoder(micro_model_cfg("ctc", encoder), 40).eval()
    x = torch.randn(1, 60, 40)
    out1, len1 = enc(x, torch.tensor([60]))
    padded = torch.cat([x, torch.randn(1, 30, 40)], dim=1)
    out2, len2 = enc(padded, torch.tensor([60]))
    assert int(len1) == int(len2)
    n = int(len1)
    assert out1.shape[-1] == 32
    # the valid frames must not depend on what is in the padding
    assert torch.allclose(out1[0, :n], out2[0, :n], atol=1e-4)


def test_attention_causal_mask_blocks_future():
    attn = MultiHeadAttention(8, 2, 0.0).eval()
    x = torch.randn(1, 5, 8)
    y1 = attn(x, x, x, causal=True)
    x2 = x.clone()
    x2[0, 4] += 10.0  # change the last position only
    y2 = attn(x2, x2, x2, causal=True)
    assert torch.allclose(y1[0, :4], y2[0, :4], atol=1e-5)
    assert not torch.allclose(y1[0, 4], y2[0, 4])


def test_relative_attention_bias_shape():
    attn = MultiHeadAttention(8, 2, 0.0, relative=True, max_rel=4)
    x = torch.randn(2, 12, 8)
    assert attn(x, x, x).shape == (2, 12, 8)
