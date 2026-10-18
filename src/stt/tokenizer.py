"""Turning text into integer tokens and back.

Every model needs a few special ids:
    blank (0)  CTC / RNN-T "emit nothing" symbol. Also the padding value of label tensors.
    sos   (1)  start of sentence, fed to attention decoders as the first input
    eos   (2)  end of sentence, the attention decoders learn to emit it to stop
    unk   (3)  anything not in the vocabulary
Keeping blank at 0 matches `torch.nn.CTCLoss` and `torchaudio.functional.rnnt_loss` defaults.

Two tokenizers share this interface:
    CharTokenizer      one token per character (space included). Simple, no training, the
                       right choice for small data: the model only has to learn ~30 classes.
    SentencePieceTokenizer  sub-word pieces (BPE/unigram) from a `.model` file; fewer, longer
                       tokens shorten the output sequence, which helps attention decoders and
                       RNN-T on big datasets. Needs the `bpe` extra (sentencepiece).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BLANK, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<blank>", "<sos>", "<eos>", "<unk>"]


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Collapse whitespace, optionally lowercase. Keeps apostrophes and letters of any script."""
    text = text.strip()
    if lowercase:
        text = text.lower()
    return re.sub(r"\s+", " ", text)


class CharTokenizer:
    """Character vocabulary built from the training transcripts."""

    def __init__(self, chars: list[str]):
        self.itos = SPECIALS + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> CharTokenizer:
        chars = sorted({c for t in texts for c in t})
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i >= len(SPECIALS))

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.itos[len(SPECIALS) :]}

    @classmethod
    def from_dict(cls, d: dict) -> CharTokenizer:
        return cls(d["chars"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))


class SentencePieceTokenizer:
    """Sub-word tokenizer around a trained SentencePiece model (`pip install sentencepiece`).

    Ids are shifted by len(SPECIALS) so the four special ids above keep their meaning.
    Train a model with `train_sentencepiece(texts, prefix, vocab_size)`.
    """

    def __init__(self, model_path: str | Path):
        import sentencepiece as spm  # lazy: optional dependency

        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.offset = len(SPECIALS)

    @property
    def vocab_size(self) -> int:
        return self.offset + self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return [i + self.offset for i in self.sp.encode(text, out_type=int)]

    def decode(self, ids: list[int]) -> str:
        pieces = [i - self.offset for i in ids if i >= self.offset]
        return self.sp.decode(pieces)

    def to_dict(self) -> dict:
        return {"type": "bpe", "model_path": self.model_path}

    @classmethod
    def from_dict(cls, d: dict) -> SentencePieceTokenizer:
        return cls(d["model_path"])


def train_sentencepiece(texts: list[str], prefix: str | Path, vocab_size: int = 256, model_type: str = "bpe") -> Path:
    """Train a SentencePiece model on `texts`; returns the `.model` path."""
    import tempfile

    import sentencepiece as spm

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(texts))
        corpus = f.name
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=str(prefix), vocab_size=vocab_size, model_type=model_type,
        character_coverage=1.0, bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1,
    )
    return prefix.with_suffix(".model")


def tokenizer_from_dict(d: dict):
    """Rebuild whichever tokenizer was saved in a checkpoint."""
    if d["type"] == "char":
        return CharTokenizer.from_dict(d)
    if d["type"] == "bpe":
        return SentencePieceTokenizer.from_dict(d)
    raise ValueError(f"unknown tokenizer type {d['type']!r}")


__all__ = [
    "BLANK",
    "EOS",
    "SOS",
    "SPECIALS",
    "UNK",
    "CharTokenizer",
    "SentencePieceTokenizer",
    "normalize_text",
    "tokenizer_from_dict",
    "train_sentencepiece",
]
