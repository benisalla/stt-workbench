"""The recognisers. Each one answers the same question, "which text was said?", with a
different way of aligning audio frames (many) to tokens (few):

CTCModel  (Graves et al. 2006)
    The encoder emits one distribution per frame over vocabulary + blank. A frame path such
    as `h h <b> e l l <b> l o` collapses to "hello" (merge repeats, drop blanks). The CTC loss
    sums the probability of EVERY path that collapses to the transcript (dynamic programming,
    `nn.CTCLoss`). Frames are assumed independent given the audio, so the model has no
    language model of its own: cheap, fast, parallel, good with a beam search + LM.

AttentionModel, model="aed"  (Chan et al. 2016 "LAS"; Speech-Transformer)
    An autoregressive decoder emits tokens one at a time, each time looking at the encoder
    output through cross-attention: p(y_i | y_<i, x). Learns an implicit language model, and
    the alignment is soft and free. Teacher forcing at training, beam search at test. Can
    loop or skip words since nothing forces the attention to move monotonically.
AttentionModel, model="las"
    Same idea with an LSTM decoder and location-aware attention (Chorowski et al. 2015):
    the attention also sees where it looked at the previous step (a convolution over the
    previous weights), which pushes it to move forward through the audio.
AttentionModel, model="hybrid"  (Watanabe et al. 2017)
    The CTC head and the attention decoder share the encoder. The CTC loss forces a
    monotonic alignment during training, and at decoding the attention beam is rescored
    by the CTC prefix probability. Best of both; the default in ESPnet.

TransducerModel  (Graves 2012, RNN-T)
    Three parts: the encoder (audio), a prediction network (an LSTM over the previous
    tokens = a small language model) and a joiner that combines encoder frame t with
    prediction state u into a distribution over vocabulary + blank. The transducer loss sums
    over all monotonic paths through the (T x U) lattice. Streaming-friendly (no attention
    over the future), the standard in on-device ASR.

All models take `(feats, feat_lengths, targets, target_lengths)` in `forward` and return
a dict with "loss" (and parts), and expose `encode(feats, lengths)` for decoding.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stt.config import ModelConfig
from stt.encoders import MultiHeadAttention, SinusoidalPositions, build_encoder, lengths_to_mask
from stt.tokenizer import BLANK, EOS, SOS


def add_sos_eos(targets: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder input `<sos> y` and decoder target `y <eos>` (pad = BLANK, ignored in the loss)."""
    b, u = targets.shape
    inp = torch.full((b, u + 1), BLANK, dtype=torch.long, device=targets.device)
    out = torch.full((b, u + 1), BLANK, dtype=torch.long, device=targets.device)
    inp[:, 0] = SOS
    inp[:, 1:] = targets
    out[:, :u] = targets
    out[torch.arange(b), lengths] = EOS
    return inp, out


class CTCModel(nn.Module):
    def __init__(self, cfg: ModelConfig, n_mels: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.encoder = build_encoder(cfg, n_mels)
        self.ctc_head = nn.Linear(cfg.d_model, vocab_size)

    def encode(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(feats, lengths)

    def ctc_log_probs(self, enc: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.ctc_head(enc), dim=-1)

    def forward(self, feats, feat_lengths, targets, target_lengths) -> dict[str, torch.Tensor]:
        enc, enc_lengths = self.encode(feats, feat_lengths)
        loss = ctc_loss(self.ctc_log_probs(enc), enc_lengths, targets, target_lengths)
        return {"loss": loss, "ctc": loss.detach()}


def ctc_loss(log_probs: torch.Tensor, enc_lengths, targets, target_lengths) -> torch.Tensor:
    """`log_probs` (B, T, V) -> mean CTC loss per target token (fp32, as the kernel needs)."""
    lp = log_probs.float().transpose(0, 1)  # (T, B, V)
    loss = F.ctc_loss(lp, targets, enc_lengths, target_lengths, blank=BLANK, reduction="sum", zero_infinity=True)
    return loss / target_lengths.sum().clamp(min=1)


class TransformerDecoder(nn.Module):
    """Pre-norm Transformer decoder: causal self-attention over previous tokens, cross-attention
    over the encoder frames, feed-forward; `forward(tokens, enc, enc_mask) -> logits`."""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        d = cfg.dec_dim
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = SinusoidalPositions(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.enc_proj = nn.Linear(cfg.d_model, d) if cfg.d_model != d else nn.Identity()
        self.layers = nn.ModuleList()
        for _ in range(cfg.dec_layers):
            self.layers.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "self": MultiHeadAttention(d, cfg.n_heads, cfg.dropout),
                "n2": nn.LayerNorm(d), "cross": MultiHeadAttention(d, cfg.n_heads, cfg.dropout),
                "n3": nn.LayerNorm(d),
                "ffn": nn.Sequential(nn.Linear(d, cfg.ffn_mult * d), nn.ReLU(), nn.Dropout(cfg.dropout),
                                     nn.Linear(cfg.ffn_mult * d, d)),
            }))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size)
        self.scale = d**0.5

    def forward(self, tokens: torch.Tensor, enc: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.pos(self.embed(tokens) * self.scale))
        mem = self.enc_proj(enc)
        for layer in self.layers:
            h = layer["n1"](x)
            x = x + self.drop(layer["self"](h, h, h, causal=True))
            h = layer["n2"](x)
            x = x + self.drop(layer["cross"](h, mem, mem, key_mask=enc_mask))
            x = x + self.drop(layer["ffn"](layer["n3"](x)))
        return self.out(self.norm(x))


class LocationAwareAttention(nn.Module):
    """Additive attention whose score also sees a conv over the previous attention weights
    (Chorowski et al. 2015): e_t = w^T tanh(W s + V h_t + U (f * a_prev)_t + b)."""

    def __init__(self, dec_dim: int, enc_dim: int, attn_dim: int = 128, conv_channels: int = 10, kernel: int = 31):
        super().__init__()
        self.query = nn.Linear(dec_dim, attn_dim, bias=False)
        self.key = nn.Linear(enc_dim, attn_dim)
        self.loc_conv = nn.Conv1d(1, conv_channels, kernel, padding=kernel // 2, bias=False)
        self.loc = nn.Linear(conv_channels, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, state: torch.Tensor, keys: torch.Tensor, enc: torch.Tensor, prev_attn: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc(self.loc_conv(prev_attn[:, None]).transpose(1, 2))  # (B, T, A)
        e = self.v(torch.tanh(self.query(state)[:, None] + keys + loc)).squeeze(-1)  # (B, T)
        e = e.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(e, dim=-1)
        context = torch.bmm(attn[:, None], enc).squeeze(1)  # (B, D)
        return context, attn


class LASDecoder(nn.Module):
    """LSTM decoder with location-aware attention: at each step, input = [embed(y_{i-1}), context],
    then attend, then predict y_i from [state, context]. Stateful: see `step`."""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        d = cfg.dec_dim
        self.embed = nn.Embedding(vocab_size, d)
        self.rnn = nn.LSTMCell(d + cfg.d_model, d)
        self.attn = LocationAwareAttention(d, cfg.d_model)
        self.out = nn.Linear(d + cfg.d_model, vocab_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.enc_dim = cfg.d_model
        self.dec_dim = d

    def init_state(self, enc: torch.Tensor) -> dict:
        b, t, _ = enc.shape
        z = enc.new_zeros(b, self.dec_dim)
        return {"h": z, "c": z.clone(), "context": enc.new_zeros(b, self.enc_dim),
                "attn": enc.new_zeros(b, t), "keys": self.attn.key(enc)}

    def step(self, token: torch.Tensor, state: dict, enc: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = torch.cat([self.drop(self.embed(token)), state["context"]], dim=-1)
        h, c = self.rnn(x, (state["h"], state["c"]))
        context, attn = self.attn(h, state["keys"], enc, state["attn"], mask)
        logits = self.out(self.drop(torch.cat([h, context], dim=-1)))
        return logits, {"h": h, "c": c, "context": context, "attn": attn, "keys": state["keys"]}

    def forward(self, tokens: torch.Tensor, enc: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        state = self.init_state(enc)
        logits = []
        for i in range(tokens.size(1)):
            step_logits, state = self.step(tokens[:, i], state, enc, enc_mask)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)


class AttentionModel(nn.Module):
    """Encoder + attention decoder (Transformer for aed/hybrid, LSTM for las); hybrid adds a CTC head."""

    def __init__(self, cfg: ModelConfig, n_mels: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.encoder = build_encoder(cfg, n_mels)
        self.decoder = LASDecoder(cfg, vocab_size) if cfg.model == "las" else TransformerDecoder(cfg, vocab_size)
        self.ctc_head = nn.Linear(cfg.d_model, vocab_size) if cfg.model == "hybrid" else None
        self.ctc_weight = cfg.ctc_weight if cfg.model == "hybrid" else 0.0

    def encode(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(feats, lengths)

    def ctc_log_probs(self, enc: torch.Tensor) -> torch.Tensor:
        assert self.ctc_head is not None
        return F.log_softmax(self.ctc_head(enc), dim=-1)

    def forward(self, feats, feat_lengths, targets, target_lengths) -> dict[str, torch.Tensor]:
        enc, enc_lengths = self.encode(feats, feat_lengths)
        mask = lengths_to_mask(enc_lengths, enc.size(1))
        dec_in, dec_out = add_sos_eos(targets, target_lengths)
        logits = self.decoder(dec_in, enc, mask)
        att = F.cross_entropy(
            logits.reshape(-1, self.vocab_size).float(), dec_out.reshape(-1), ignore_index=BLANK,
            label_smoothing=self.cfg.label_smoothing,
        )
        out = {"att": att.detach()}
        with torch.no_grad():
            pred = logits.argmax(-1)
            valid = dec_out != BLANK
            out["token_acc"] = ((pred == dec_out) & valid).sum() / valid.sum().clamp(min=1)
        if self.ctc_head is not None:
            ctc = ctc_loss(self.ctc_log_probs(enc), enc_lengths, targets, target_lengths)
            out["ctc"] = ctc.detach()
            out["loss"] = self.ctc_weight * ctc + (1 - self.ctc_weight) * att
        else:
            out["loss"] = att
        return out


class PredictionNetwork(nn.Module):
    """RNN-T prediction network: an LSTM over the previous non-blank tokens (blank = start)."""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.pred_dim)
        self.lstm = nn.LSTM(cfg.pred_dim, cfg.pred_dim, num_layers=cfg.pred_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.pred_layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple]:
        out, state = self.lstm(self.drop(self.embed(tokens)), state)
        return out, state


class Joiner(nn.Module):
    """joint(t, u) = Linear(tanh(W_enc h_t + W_pred g_u)). Broadcasts to (B, T, U, V)."""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.enc = nn.Linear(cfg.d_model, cfg.joint_dim)
        self.pred = nn.Linear(cfg.pred_dim, cfg.joint_dim)
        self.out = nn.Linear(cfg.joint_dim, vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.enc(enc) + self.pred(pred)))


class TransducerModel(nn.Module):
    def __init__(self, cfg: ModelConfig, n_mels: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.encoder = build_encoder(cfg, n_mels)
        self.predictor = PredictionNetwork(cfg, vocab_size)
        self.joiner = Joiner(cfg, vocab_size)

    def encode(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(feats, lengths)

    def forward(self, feats, feat_lengths, targets, target_lengths) -> dict[str, torch.Tensor]:
        try:
            from torchaudio.functional import rnnt_loss
        except ImportError as e:  # pragma: no cover
            raise ImportError("RNN-T training needs torchaudio: pip install 'stt-workbench[rnnt]'") from e
        enc, enc_lengths = self.encode(feats, feat_lengths)
        b, _u = targets.shape
        pred_in = torch.cat([torch.full((b, 1), BLANK, dtype=torch.long, device=targets.device), targets], dim=1)
        pred, _ = self.predictor(pred_in)  # (B, U+1, P)
        logits = self.joiner(enc[:, :, None, :], pred[:, None, :, :])  # (B, T, U+1, V)
        loss = rnnt_loss(
            logits.float(), targets.int(), enc_lengths.int(), target_lengths.int(), blank=BLANK, reduction="sum",
        ) / target_lengths.sum().clamp(min=1)
        return {"loss": loss, "rnnt": loss.detach()}


def build_model(cfg: ModelConfig, n_mels: int, vocab_size: int) -> nn.Module:
    if cfg.model == "ctc":
        return CTCModel(cfg, n_mels, vocab_size)
    if cfg.model in ("aed", "las", "hybrid"):
        return AttentionModel(cfg, n_mels, vocab_size)
    if cfg.model == "rnnt":
        return TransducerModel(cfg, n_mels, vocab_size)
    raise ValueError(cfg.model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "AttentionModel",
    "CTCModel",
    "Joiner",
    "LASDecoder",
    "LocationAwareAttention",
    "PredictionNetwork",
    "TransducerModel",
    "TransformerDecoder",
    "add_sos_eos",
    "build_model",
    "count_parameters",
    "ctc_loss",
]
