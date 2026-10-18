import numpy as np
import torch

from stt.config import DataConfig
from stt.features import LogMel, cmvn, hz_to_mel, mel_filterbank, mel_to_hz, resample, spec_augment


def test_mel_scale_inverts():
    for f in (0.0, 100.0, 1000.0, 3999.0):
        assert abs(mel_to_hz(hz_to_mel(f)) - f) < 1e-6


def test_filterbank_shape_and_coverage():
    fb = mel_filterbank(8000, 256, 40, 20.0, None)
    assert fb.shape == (129, 40)
    assert (fb >= 0).all()
    assert (fb.sum(0) > 0).all()  # every filter catches some bin


def test_logmel_frames_and_tone_position():
    cfg = DataConfig(sample_rate=8000, n_mels=40)
    lm = LogMel(cfg)
    t = torch.arange(8000) / 8000
    low = torch.sin(2 * torch.pi * 300 * t)
    high = torch.sin(2 * torch.pi * 3000 * t)
    f_low, f_high = lm(low), lm(high)
    assert f_low.shape == (lm.n_frames(8000), 40)
    assert f_low.mean(0).argmax() < f_high.mean(0).argmax()
    batched = lm(torch.stack([low, high]))
    assert batched.shape == (2, lm.n_frames(8000), 40)
    assert torch.allclose(batched[0], f_low, atol=1e-4)


def test_cmvn_normalises():
    x = torch.randn(50, 8) * 3 + 2
    y = cmvn(x)
    assert torch.allclose(y.mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(y.std(0), torch.ones(8), atol=1e-3)


def test_spec_augment_masks_and_keeps_shape():
    x = torch.ones(100, 40)
    g = torch.Generator().manual_seed(0)
    y = spec_augment(x, freq_masks=1, freq_mask_width=10, time_masks=1, time_mask_width=20, generator=g)
    assert y.shape == x.shape
    assert (y == 0).any()
    assert (x == 1).all()  # input untouched


def test_resample_keeps_duration():
    wav = np.random.RandomState(0).randn(16000).astype(np.float32)
    out = resample(wav, 16000, 8000)
    assert out.shape == (8000,)
    assert resample(wav, 16000, 16000) is wav
