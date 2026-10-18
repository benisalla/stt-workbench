"""Train a small CTC recogniser on the synthetic toy language, on a CPU, in about a minute.

    python examples/01_train_on_toy_data.py

What happens: `write_synthetic` makes 200 + 30 short sentences of a 12-letter toy language
(each letter is a fixed tone pattern, 8 kHz, with noise), `train` runs the normal training
loop with a 2-layer Conformer and a CTC head, and the checkpoint lands in
checkpoints/examples-ctc/ (best.pt, last.pt, log.csv, results.json).
Examples 02 and 03 load that checkpoint.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import torch

from stt.config import DataConfig, ModelConfig, TrainConfig
from stt.datasets import write_synthetic
from stt.train import train

torch.set_num_threads(2)

DATA = Path("data/examples/synthetic")
OUT = Path("checkpoints/examples-ctc")

if not (DATA / "train.jsonl").exists():
    write_synthetic(DATA, n_train=200, n_val=30, seed=0, max_words=3)
    print(f"wrote {DATA}")

# Three frozen dataclasses hold every setting; the CLI flags of stt-train map onto them 1:1.
data_cfg = DataConfig(data_dir=DATA, sample_rate=8000, n_mels=40)  # 8 kHz audio -> 40 mel bins are plenty
model_cfg = ModelConfig(model="ctc", encoder="conformer", d_model=64, n_layers=2, n_heads=4, subsampling_channels=16)
train_cfg = TrainConfig(output_dir=OUT, batch_size=16, max_steps=1200, warmup_steps=100, eval_every=300, log_every=25,
                        device="cpu", resume=False)

result = train(data_cfg, model_cfg, train_cfg)
print(f"\nfinal: WER {100 * result['wer']:.1f}%  CER {100 * result['cer']:.1f}%  "
      f"({result['parameters'] / 1e6:.2f} M parameters, {result['train_seconds'] / 60:.1f} min)")
print(f"checkpoint: {OUT / 'best.pt'}")
