"""All hyperparameters live here, grouped by what they control.

Configs are frozen dataclasses: they cannot be changed by accident after creation, and
they are saved inside every checkpoint, so a trained model can be rebuilt without
remembering any flag (`load_checkpoint` in train.py).

The defaults are deliberately SMALL (a few million parameters at most): this repo exists
to learn how each ASR approach works, not to chase leaderboard numbers. Every size can be
changed from the command line (`stt-train --help`) or in the notebook.

Model types (`ModelConfig.model`):
    ctc     encoder + CTC head                               (models.CTCModel)
    aed     encoder + Transformer decoder with cross-attention (models.AttentionModel)
    las     encoder + LSTM decoder with location-aware attention (models.AttentionModel)
    hybrid  CTC head + Transformer decoder, joint loss and joint decoding (models.AttentionModel)
    rnnt    encoder + prediction network + joiner, transducer loss (models.TransducerModel)
Encoders (`ModelConfig.encoder`): bilstm | transformer | conformer   (encoders.py)
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

MODEL_TYPES = ("ctc", "aed", "las", "hybrid", "rnnt")
ENCODER_TYPES = ("bilstm", "transformer", "conformer")
DECODER_METHODS = ("greedy", "beam")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the speech is, how it is turned into features, and how it is batched.

    A dataset is a folder with `train.jsonl` and `val.jsonl` manifests (see docs/data.md):
    one JSON object per line, `{"audio": "<path>", "text": "<transcript>", "duration": <s>}`.
    Paths are absolute or relative to the manifest's folder.
    """

    data_dir: Path = Path("data/synthetic")
    train_manifest: str = "train.jsonl"
    val_manifest: str = "val.jsonl"
    sample_rate: int = 16_000  # audio is resampled to this on load
    # Log-mel front end (features.py): 25 ms windows every 10 ms is the ASR standard.
    n_mels: int = 80
    win_ms: float = 25.0
    hop_ms: float = 10.0
    f_min: float = 20.0
    f_max: float | None = None  # None = sample_rate / 2
    # Text: "char" (one token per character) or "bpe" (SentencePiece model at `bpe_model`).
    tokenizer: str = "char"
    bpe_model: Path | None = None
    lowercase: bool = True
    # SpecAugment (training only). Masks are in feature frames / mel bins.
    specaugment: bool = True
    freq_masks: int = 2
    freq_mask_width: int = 15
    time_masks: int = 2
    time_mask_width: int = 40
    time_mask_max_ratio: float = 0.2  # a time mask never hides more than this fraction of the clip
    # Batching: clips are sorted by duration into buckets so that padding is small.
    max_duration_s: float = 20.0  # longer training clips are skipped
    bucket_size: int = 64  # clips per duration bucket (batches are drawn inside a bucket)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.n_mels <= 0:
            raise ValueError("sample_rate and n_mels must be positive")
        if self.tokenizer not in ("char", "bpe"):
            raise ValueError("tokenizer must be 'char' or 'bpe'")
        if self.tokenizer == "bpe" and self.bpe_model is None:
            raise ValueError("tokenizer='bpe' needs bpe_model")

    @property
    def n_fft(self) -> int:
        """Smallest power of two holding the analysis window."""
        win = self.win_length
        n = 1
        while n < win:
            n *= 2
        return n

    @property
    def win_length(self) -> int:
        return round(self.sample_rate * self.win_ms / 1000)

    @property
    def hop_length(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1000)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the recogniser: which approach, which encoder, and how big."""

    model: str = "ctc"  # one of MODEL_TYPES
    encoder: str = "conformer"  # one of ENCODER_TYPES
    # Conv2d subsampling in front of every encoder: 2 layers of stride 2 = 4x fewer frames
    # (40 ms per encoder frame). 4 ms-per-frame features are too many for attention to look at.
    subsampling_channels: int = 64
    subsampling_factor: int = 4  # 2 or 4
    d_model: int = 144  # encoder width
    n_heads: int = 4
    n_layers: int = 6
    ffn_mult: int = 4  # feed-forward width = ffn_mult * d_model
    conv_kernel: int = 15  # Conformer convolution module kernel (odd)
    dropout: float = 0.1
    # Attention decoders (aed / hybrid: Transformer decoder; las: LSTM decoder).
    dec_layers: int = 2
    dec_dim: int = 144
    label_smoothing: float = 0.1
    # Hybrid: loss = ctc_weight * CTC + (1 - ctc_weight) * attention; also used in joint decoding.
    ctc_weight: float = 0.3
    # RNN-T: prediction network (LSTM over previous tokens) and joiner.
    pred_dim: int = 128
    pred_layers: int = 1
    joint_dim: int = 128

    def __post_init__(self) -> None:
        if self.model not in MODEL_TYPES:
            raise ValueError(f"model must be one of {MODEL_TYPES}, got {self.model!r}")
        if self.encoder not in ENCODER_TYPES:
            raise ValueError(f"encoder must be one of {ENCODER_TYPES}, got {self.encoder!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.subsampling_factor not in (2, 4):
            raise ValueError("subsampling_factor must be 2 or 4")
        if self.conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be odd")
        if not 0.0 <= self.ctc_weight <= 1.0:
            raise ValueError("ctc_weight must be in [0, 1]")

    @property
    def has_ctc(self) -> bool:
        return self.model in ("ctc", "hybrid")

    @property
    def has_attention_decoder(self) -> bool:
        return self.model in ("aed", "las", "hybrid")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """How long, how fast, where to save, how to evaluate."""

    output_dir: Path = Path("checkpoints/run")
    batch_size: int = 16
    max_steps: int = 20_000
    max_epochs: int | None = None  # stop after this many epochs if set (max_steps still applies)
    lr: float = 1e-3
    warmup_steps: int = 1_000  # linear warm-up, then cosine decay to `min_lr`
    min_lr: float = 1e-5
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    precision: str = "fp32"  # fp32 | bf16 | fp16 (GPU autocast; CTC/RNN-T losses stay in fp32)
    num_workers: int = 0
    device: str = "auto"  # auto | cpu | cuda
    seed: int = 0
    log_every: int = 50
    eval_every: int = 500  # steps; also at the end
    eval_batches: int | None = None  # None = whole validation manifest
    decode: str = "greedy"  # validation decoding: greedy | beam
    beam_size: int = 4
    select_by: str = "wer"  # best.pt keeps the lowest of: wer | cer | loss
    time_budget_hours: float | None = None  # stop cleanly (after saving) before a session limit
    resume: bool = True  # continue from output_dir/last.pt if it exists

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32, bf16 or fp16")
        if self.decode not in DECODER_METHODS:
            raise ValueError(f"decode must be one of {DECODER_METHODS}")
        if self.select_by not in ("wer", "cer", "loss"):
            raise ValueError("select_by must be wer, cer or loss")


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """A JSON-friendly dict (Paths as strings)."""
    out = {}
    for k, v in asdict(cfg).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def config_from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Inverse of config_to_dict; unknown keys (older/newer checkpoints) are ignored."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if v is not None and ("Path" in str(f.type)) and isinstance(v, str):
            v = Path(v)
        kwargs[f.name] = v
    return cls(**kwargs)


__all__ = [
    "DECODER_METHODS",
    "ENCODER_TYPES",
    "MODEL_TYPES",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "config_from_dict",
    "config_to_dict",
]
