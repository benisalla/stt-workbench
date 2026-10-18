"""The front end: waveform -> log-mel -> CMVN -> SpecAugment, drawn step by step.

    python examples/04_features_and_specaugment.py        # writes examples/output/features.png

Every model in the repo sees exactly these features; SpecAugment is applied only in training.
Needs data/examples/synthetic (example 01) or any wav file passed as the first argument.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from stt.config import DataConfig
from stt.features import LogMel, cmvn, load_audio, spec_augment

wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(Path("data/examples/synthetic/val").glob("*.wav"))
cfg = DataConfig(sample_rate=8000, n_mels=40)  # 25 ms windows every 10 ms, 40 mel bins up to 4 kHz

wav = torch.from_numpy(load_audio(wav_path, cfg.sample_rate))  # resampled to cfg.sample_rate if needed
logmel = LogMel(cfg)(wav)  # (frames, n_mels)
normed = cmvn(logmel)  # per-utterance mean 0 / std 1 per mel bin: removes the recording level
masked = spec_augment(normed, freq_masks=2, freq_mask_width=6, time_masks=2, time_mask_width=15,
                      generator=torch.Generator().manual_seed(0))

print(f"{wav_path.name}: {wav.numel()} samples = {wav.numel() / cfg.sample_rate:.2f} s -> {logmel.shape[0]} frames x "
      f"{logmel.shape[1]} mels (hop {cfg.hop_length} samples, window {cfg.win_length}, n_fft {cfg.n_fft})")

fig, axes = plt.subplots(4, 1, figsize=(10, 9))
axes[0].plot(wav.numpy(), lw=0.5)
axes[0].set_title("waveform")
for ax, feats, title in zip(axes[1:], (logmel, normed, masked), ("log-mel", "after CMVN", "after SpecAugment (training only)")):
    ax.imshow(feats.T.numpy(), origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_ylabel("mel bin")
axes[-1].set_xlabel("frame (10 ms)")
fig.tight_layout()
out = Path("examples/output/features.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print("wrote", out)
