"""Fine-tuning pretrained speech models from Hugging Face (`pip install 'stt-workbench[pretrained]'`).

The models in models.py learn everything from the training set. With little data, a model
that already knows what speech sounds like does far better:

wav2vec2 + CTC  (Baevski et al. 2020)
    A convolutional waveform encoder + Transformer, pretrained by contrastive prediction of
    masked latent frames on 1000s of hours of unlabelled audio. Fine-tuning adds a linear
    CTC head over characters on top and trains with the CTC loss: the same objective as
    `CTCModel`, with a much better encoder. `Wav2Vec2Fine` wraps `Wav2Vec2ForCTC` to the
    repo's (feats, lengths, targets, target_lengths) interface, except that it eats raw
    16 kHz waveforms, not log-mels.

Whisper  (Radford et al. 2022)
    An attention encoder-decoder (like `aed`) trained on 680 000 h of weakly labelled audio,
    with its own tokenizer and log-mel front end. Fine-tuning = continuing its cross-entropy
    training on your transcripts. `WhisperFine` wraps `WhisperForConditionalGeneration`.

Both wrappers are thin: `forward(...)` -> {"loss": ...}, `transcribe(waveforms)` -> texts.
Training them is a notebook job (GPU, model downloads); this module only holds the code
so that the notebook and the package share it. `freeze_feature_encoder` keeps the
convolutional front end fixed, which is standard for small datasets.
"""

from __future__ import annotations

import torch
from torch import nn

from stt.tokenizer import SPECIALS


class Wav2Vec2Fine(nn.Module):
    """wav2vec2 with a fresh CTC head over the repo's character vocabulary (blank = 0)."""

    def __init__(self, tokenizer, name: str = "facebook/wav2vec2-base", freeze_feature_encoder: bool = True,
                 config_overrides: dict | None = None, model=None):
        super().__init__()
        self.tokenizer = tokenizer
        if model is None:
            from transformers import Wav2Vec2ForCTC

            model = Wav2Vec2ForCTC.from_pretrained(
                name, vocab_size=tokenizer.vocab_size, pad_token_id=0, ctc_loss_reduction="sum",
                ignore_mismatched_sizes=True, **(config_overrides or {}),
            )
        self.model = model
        if freeze_feature_encoder and hasattr(self.model, "freeze_feature_encoder"):
            self.model.freeze_feature_encoder()

    def forward(self, waves: torch.Tensor, wave_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor):
        mask = torch.arange(waves.size(1), device=waves.device)[None] < wave_lengths[:, None]
        labels = targets.clone()
        labels[torch.arange(labels.size(1), device=labels.device)[None] >= target_lengths[:, None]] = -100
        out = self.model(input_values=waves, attention_mask=mask.long(), labels=labels)
        return {"loss": out.loss / target_lengths.sum().clamp(min=1)}

    @torch.no_grad()
    def transcribe(self, waves: torch.Tensor, wave_lengths: torch.Tensor) -> list[str]:
        from stt.decoding import ctc_greedy

        mask = torch.arange(waves.size(1), device=waves.device)[None] < wave_lengths[:, None]
        logits = self.model(input_values=waves, attention_mask=mask.long()).logits
        out_lengths = self.model._get_feat_extract_output_lengths(wave_lengths)
        return [self.tokenizer.decode(seq) for seq in ctc_greedy(torch.log_softmax(logits, -1), out_lengths)]


class WhisperFine(nn.Module):
    """Whisper encoder-decoder fine-tuned on (log-mel, text) pairs with its own tokenizer."""

    def __init__(self, name: str = "openai/whisper-tiny", language: str = "en", model=None, processor=None):
        super().__init__()
        if model is None or processor is None:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            processor = processor or WhisperProcessor.from_pretrained(name)
            model = model or WhisperForConditionalGeneration.from_pretrained(name)
        self.model = model
        self.processor = processor
        self.language = language
        self.model.generation_config.language = language
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None

    def features(self, waves: list, sample_rate: int = 16_000) -> torch.Tensor:
        return self.processor.feature_extractor(waves, sampling_rate=sample_rate, return_tensors="pt").input_features

    def labels(self, texts: list[str]) -> torch.Tensor:
        batch = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)
        return labels

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor):
        out = self.model(input_features=input_features, labels=labels)
        return {"loss": out.loss}

    @torch.no_grad()
    def transcribe(self, input_features: torch.Tensor, max_new_tokens: int = 128) -> list[str]:
        ids = self.model.generate(input_features=input_features, max_new_tokens=max_new_tokens)
        return [t.strip().lower() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


def char_vocab_for_hf(tokenizer) -> dict[str, int]:
    """A vocab.json-style mapping for Wav2Vec2CTCTokenizer from the repo's CharTokenizer
    ("|" stands for the space, as Hugging Face expects)."""
    vocab = {}
    for i, ch in enumerate(tokenizer.itos):
        if i < len(SPECIALS):
            vocab[["<pad>", "<s>", "</s>", "<unk>"][i]] = i
        else:
            vocab["|" if ch == " " else ch] = i
    return vocab


__all__ = ["Wav2Vec2Fine", "WhisperFine", "char_vocab_for_hf"]
