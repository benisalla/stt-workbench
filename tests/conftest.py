import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import pytest
import torch

from stt.config import DataConfig, ModelConfig
from stt.datasets import write_synthetic

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    """A 24 + 8 sentence synthetic dataset, written once per test session."""
    out = tmp_path_factory.mktemp("synthetic")
    return write_synthetic(out, n_train=24, n_val=8, seed=1, max_words=2)


@pytest.fixture
def data_cfg(synthetic_dir) -> DataConfig:
    return DataConfig(data_dir=synthetic_dir, sample_rate=8000, n_mels=40)


def micro_model_cfg(model: str = "ctc", encoder: str = "conformer") -> ModelConfig:
    return ModelConfig(model=model, encoder=encoder, d_model=32, n_heads=2, n_layers=2, ffn_mult=2, conv_kernel=7,
                       subsampling_channels=8, dec_dim=32, dec_layers=1, pred_dim=16, joint_dim=16, dropout=0.0)
