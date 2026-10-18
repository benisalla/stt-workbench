from pathlib import Path

import pytest

from stt.config import DataConfig, ModelConfig, TrainConfig, config_from_dict, config_to_dict


def test_feature_geometry_follows_sample_rate():
    cfg = DataConfig(sample_rate=8000)
    assert (cfg.win_length, cfg.hop_length, cfg.n_fft) == (200, 80, 256)
    cfg = DataConfig(sample_rate=16000)
    assert (cfg.win_length, cfg.hop_length, cfg.n_fft) == (400, 160, 512)


def test_invalid_configs_raise():
    with pytest.raises(ValueError):
        ModelConfig(model="nope")
    with pytest.raises(ValueError):
        ModelConfig(encoder="cnn")
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(conv_kernel=8)
    with pytest.raises(ValueError):
        DataConfig(tokenizer="bpe")
    with pytest.raises(ValueError):
        TrainConfig(precision="fp8")


def test_round_trip_through_dict():
    cfg = TrainConfig(output_dir=Path("x/y"), lr=3e-4)
    d = config_to_dict(cfg)
    assert d["output_dir"] == "x/y"
    back = config_from_dict(TrainConfig, {**d, "unknown_key": 1})
    assert back == cfg and isinstance(back.output_dir, Path)


def test_model_flags():
    assert ModelConfig(model="hybrid").has_ctc and ModelConfig(model="hybrid").has_attention_decoder
    assert not ModelConfig(model="rnnt").has_ctc
