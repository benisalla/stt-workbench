"""From a waveform to the log-mel spectrogram every model in this repo listens to.

The pipeline (the classic one, from Kaldi to Whisper):

1. load the file (`soundfile`) and resample to the configured rate (`scipy.signal.resample_poly`);
2. short-time Fourier transform with 25 ms Hann windows every 10 ms (`torch.stft`);
3. power spectrum |X|^2, then a bank of `n_mels` triangular filters spaced on the mel scale,
   mel(f) = 2595 log10(1 + f / 700), so that low frequencies (where vowels live) get more bins;
4. log(. + 1e-6), then per-utterance mean/variance normalisation (CMVN) so the model does
   not care about recording level;
5. SpecAugment (Park et al. 2019) during training: a few random frequency bands and time
   spans are zeroed, which makes the model rely on context instead of single cues.

Shapes: waveform (T,) -> features (frames, n_mels) with frames = T // hop + 1.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from stt.config import DataConfig


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Mono float32 waveform at `sample_rate`, in [-1, 1]."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        wav = resample(wav, sr, sample_rate)
    return np.ascontiguousarray(wav, dtype=np.float32)


def resample(wav: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    """Polyphase resampling (anti-aliased), exact for integer ratios."""
    if orig_sr == new_sr:
        return wav
    g = math.gcd(orig_sr, new_sr)
    return resample_poly(wav, new_sr // g, orig_sr // g).astype(np.float32)


def hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float | None) -> torch.Tensor:
    """(n_fft // 2 + 1, n_mels) triangular filters, Slaney-style area normalised."""
    f_max = f_max or sample_rate / 2
    n_bins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2, n_bins)
    mel_points = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m) for m in mel_points.tolist()])
    lower, center, upper = hz_points[:-2], hz_points[1:-1], hz_points[2:]
    f = fft_freqs[:, None]
    rising = (f - lower) / (center - lower).clamp(min=1e-6)
    falling = (upper - f) / (upper - center).clamp(min=1e-6)
    fb = torch.clamp(torch.minimum(rising, falling), min=0.0)
    fb = fb * (2.0 / (upper - lower))[None, :]  # equal-area filters
    return fb


class LogMel(torch.nn.Module):
    """Log-mel spectrogram front end. Works on (T,) or (B, T) waveforms."""

    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.n_fft = cfg.n_fft
        self.hop = cfg.hop_length
        self.win = cfg.win_length
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)
        self.register_buffer(
            "fb", mel_filterbank(cfg.sample_rate, self.n_fft, cfg.n_mels, cfg.f_min, cfg.f_max), persistent=False
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(T,) -> (frames, n_mels) or (B, T) -> (B, frames, n_mels)."""
        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav[None]
        spec = torch.stft(
            wav, self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
            center=True, pad_mode="reflect", return_complex=True,
        )
        power = spec.abs().pow(2)  # (B, bins, frames)
        mel = torch.matmul(power.transpose(1, 2), self.fb)  # (B, frames, n_mels)
        logmel = torch.log(mel + 1e-6)
        return logmel[0] if squeeze else logmel

    def n_frames(self, n_samples: int) -> int:
        return n_samples // self.hop + 1


def cmvn(feats: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-utterance mean/variance normalisation over time. (frames, n_mels)."""
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True)
    return (feats - mean) / (std + eps)


def spec_augment(
    feats: torch.Tensor,
    freq_masks: int = 2,
    freq_mask_width: int = 15,
    time_masks: int = 2,
    time_mask_width: int = 40,
    time_mask_max_ratio: float = 0.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SpecAugment masking on one (frames, n_mels) utterance (time warping is left out: it
    costs a lot and helps little, see the SpecAugment paper's ablation)."""
    feats = feats.clone()
    n_frames, n_mels = feats.shape
    fill = 0.0  # features are CMVN'd: zero = the mean

    def rand_int(high: int) -> int:
        if high <= 0:
            return 0
        return int(torch.randint(0, high + 1, (1,), generator=generator))

    for _ in range(freq_masks):
        w = rand_int(min(freq_mask_width, n_mels))
        f0 = rand_int(n_mels - w)
        feats[:, f0 : f0 + w] = fill
    max_t = min(time_mask_width, int(n_frames * time_mask_max_ratio))
    for _ in range(time_masks):
        w = rand_int(max_t)
        t0 = rand_int(n_frames - w)
        feats[t0 : t0 + w, :] = fill
    return feats


__all__ = ["LogMel", "cmvn", "hz_to_mel", "load_audio", "mel_filterbank", "mel_to_hz", "resample", "spec_augment"]
