"""The acoustic encoders: log-mel frames in, one context vector per (subsampled) frame out.

All three encoders share the same front end and the same output contract, so every
approach (CTC, attention, RNN-T) can use any of them:

    Conv2dSubsampling   two 3x3 stride-2 convolutions over the (time, mel) image. Each halves
                        the frame rate: 10 ms frames become 40 ms, 4x fewer positions for the
                        encoder (and for attention: quadratic cost) to deal with.
    BiLSTMEncoder       stacked bidirectional LSTMs (Graves et al. 2013, Deep Speech 2). The
                        forward pass sees the past, the backward pass the future.
    TransformerEncoder  pre-norm self-attention blocks with sinusoidal positions
                        (Vaswani et al. 2017; Speech-Transformer, Dong et al. 2018).
    ConformerEncoder    Conformer blocks (Gulati et al. 2020): macaron feed-forward halves,
                        relative-position multi-head self-attention, and a depthwise-conv
                        module for local patterns. The strongest of the three per parameter.

Every encoder's forward is `(feats, lengths) -> (out, out_lengths)` with
feats (B, T, n_mels), out (B, T', d_model); `out_lengths` are needed because batches are
padded and the losses must know where each utterance really ends.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from stt.config import ModelConfig


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """(B,) lengths -> (B, T) boolean mask, True on valid positions."""
    max_len = max_len or int(lengths.max())
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class Conv2dSubsampling(nn.Module):
    """(B, T, n_mels) -> (B, T // factor, d_model), factor 2 or 4."""

    def __init__(self, n_mels: int, d_model: int, channels: int, factor: int):
        super().__init__()
        n_convs = {2: 1, 4: 2}[factor]
        layers: list[nn.Module] = []
        c_in = 1
        freq = n_mels
        for _ in range(n_convs):
            layers += [nn.Conv2d(c_in, channels, 3, stride=2, padding=1), nn.ReLU()]
            c_in = channels
            freq = (freq + 2 - 3) // 2 + 1
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(channels * freq, d_model)
        self.factor = factor
        self.n_convs = n_convs

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x[:, None])  # (B, C, T', F')
        b, c, t, f = x.shape
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        for _ in range(self.n_convs):
            lengths = (lengths + 2 - 3) // 2 + 1
        return x, lengths


class SinusoidalPositions(nn.Module):
    """Adds the fixed sin/cos position code of "Attention is all you need"."""

    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len)[:, None].float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)][None]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float, activation: nn.Module | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, mult * d_model), activation or nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mult * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product attention; `relative=True` adds Transformer-XL style
    relative position bias (Dai et al. 2019), as used by the Conformer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, relative: bool = False, max_rel: int = 256):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.relative = relative
        self.max_rel = max_rel
        if relative:
            # one learned bias per (head, relative distance), clipped to [-max_rel, max_rel]
            self.rel_bias = nn.Embedding(2 * max_rel + 1, n_heads)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        key_mask: torch.Tensor | None = None, causal: bool = False,
    ) -> torch.Tensor:
        b, tq, _ = query.shape
        tk = key.size(1)
        q = self.q(query).view(b, tq, self.h, self.dk).transpose(1, 2)  # (B, h, Tq, dk)
        k = self.k(key).view(b, tk, self.h, self.dk).transpose(1, 2)
        v = self.v(value).view(b, tk, self.h, self.dk).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dk)  # (B, h, Tq, Tk)
        if self.relative:
            rel = torch.arange(tk, device=query.device)[None, :] - torch.arange(tq, device=query.device)[:, None]
            rel = rel.clamp(-self.max_rel, self.max_rel) + self.max_rel
            scores = scores + self.rel_bias(rel).permute(2, 0, 1)[None]
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if causal:
            future = torch.triu(torch.ones(tq, tk, dtype=torch.bool, device=query.device), diagonal=1)
            scores = scores.masked_fill(future[None, None], float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, tq, self.h * self.dk)
        return self.o(out)


class TransformerBlock(nn.Module):
    """Pre-norm: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_mult, cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        x = x + self.drop(self.attn(h, h, h, key_mask=mask))
        return x + self.ffn(x)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise GLU -> depthwise conv -> BN -> SiLU -> pointwise."""

    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).masked_fill(~mask[..., None], 0.0).transpose(1, 2)  # (B, D, T)
        h = F.glu(self.pw1(h), dim=1).masked_fill(~mask[:, None, :], 0.0)  # the bias would leak into padding
        h = F.silu(self.bn(self.dw(h)))
        h = self.pw2(h).transpose(1, 2)
        return self.drop(h)


class ConformerBlock(nn.Module):
    """x + 1/2 FFN; x + RelMHSA; x + Conv; x + 1/2 FFN; LayerNorm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ffn1 = FeedForward(cfg.d_model, cfg.ffn_mult, cfg.dropout, nn.SiLU())
        self.norm_attn = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, relative=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.conv = ConvModule(cfg.d_model, cfg.conv_kernel, cfg.dropout)
        self.ffn2 = FeedForward(cfg.d_model, cfg.ffn_mult, cfg.dropout, nn.SiLU())
        self.norm_out = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        h = self.norm_attn(x)
        x = x + self.drop(self.attn(h, h, h, key_mask=mask))
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ffn2(x)
        return self.norm_out(x)


class BiLSTMEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_mels: int):
        super().__init__()
        self.sub = Conv2dSubsampling(n_mels, cfg.d_model, cfg.subsampling_channels, cfg.subsampling_factor)
        self.lstm = nn.LSTM(
            cfg.d_model, cfg.d_model // 2, num_layers=cfg.n_layers, batch_first=True,
            bidirectional=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.sub(feats, lengths)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.drop(out), lengths


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_mels: int):
        super().__init__()
        self.sub = Conv2dSubsampling(n_mels, cfg.d_model, cfg.subsampling_channels, cfg.subsampling_factor)
        self.pos = SinusoidalPositions(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.scale = math.sqrt(cfg.d_model)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.sub(feats, lengths)
        mask = lengths_to_mask(lengths, x.size(1))
        x = self.drop(self.pos(x * self.scale))
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x), lengths


class ConformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_mels: int):
        super().__init__()
        self.sub = Conv2dSubsampling(n_mels, cfg.d_model, cfg.subsampling_channels, cfg.subsampling_factor)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.sub(feats, lengths)
        mask = lengths_to_mask(lengths, x.size(1))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, mask)
        return x, lengths


def build_encoder(cfg: ModelConfig, n_mels: int) -> nn.Module:
    return {"bilstm": BiLSTMEncoder, "transformer": TransformerEncoder, "conformer": ConformerEncoder}[cfg.encoder](
        cfg, n_mels
    )


__all__ = [
    "BiLSTMEncoder",
    "ConformerBlock",
    "ConformerEncoder",
    "Conv2dSubsampling",
    "ConvModule",
    "FeedForward",
    "MultiHeadAttention",
    "SinusoidalPositions",
    "TransformerBlock",
    "TransformerEncoder",
    "build_encoder",
    "lengths_to_mask",
]
