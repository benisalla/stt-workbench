"""A count-based n-gram language model over token ids, to bias beam search towards likely text.

p(w | h) for history h of the last n-1 tokens, with interpolated absolute discounting
(the simple cousin of Kneser-Ney): mass D is taken from every seen continuation and given
to the lower-order model, recursively down to a uniform distribution.

    p_n(w | h) = max(c(h, w) - D, 0) / c(h)  +  D * N1+(h) / c(h) * p_{n-1}(w | h')

Here N1+(h) is the number of distinct words seen after h. Unseen histories fall back
entirely to the shorter one. The LM scores tokens of whatever tokenizer the model uses
(characters by default), and `<eos>` is a normal token so the LM knows how sentences end.

A character 5-gram trained on the training transcripts is a cheap, honest baseline;
decoding.py adds `lm_weight * log p_lm` to the acoustic score of every extension.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from stt.tokenizer import EOS, SOS


class NGramLM:
    def __init__(self, order: int = 5, vocab_size: int = 32, discount: float = 0.75):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.vocab_size = vocab_size
        self.discount = discount
        # counts[k][history tuple][word] for history length k
        self.counts: list[dict[tuple, dict[int, int]]] = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]

    def train(self, sequences: list[list[int]]) -> NGramLM:
        for seq in sequences:
            tokens = [SOS] * (self.order - 1) + list(seq) + [EOS]
            for i in range(self.order - 1, len(tokens)):
                w = tokens[i]
                for k in range(self.order):
                    h = tuple(tokens[i - k : i])
                    self.counts[k][h][w] += 1
        return self

    def score(self, history: list[int], token: int) -> float:
        """log p(token | history)."""
        return math.log(max(self.prob(history, token), 1e-12))

    def prob(self, history: list[int], token: int) -> float:
        hist = ([SOS] * (self.order - 1) + list(history))[-(self.order - 1) :] if self.order > 1 else []
        return self._prob(tuple(hist), token, len(hist))

    def _prob(self, h: tuple, w: int, k: int) -> float:
        if k < 0:
            return 1.0 / self.vocab_size
        table = self.counts[k].get(h)
        if not table:
            return self._prob(h[1:] if k > 0 else h, w, k - 1)
        total = sum(table.values())
        n1 = len(table)
        high = max(table.get(w, 0) - self.discount, 0.0) / total
        low = self.discount * n1 / total * self._prob(h[1:] if k > 0 else h, w, k - 1)
        return high + low

    def perplexity(self, sequences: list[list[int]]) -> float:
        logp, n = 0.0, 0
        for seq in sequences:
            hist: list[int] = []
            for w in list(seq) + [EOS]:
                logp += self.score(hist, w)
                hist.append(w)
                n += 1
        return math.exp(-logp / max(n, 1))

    def save(self, path: str | Path) -> None:
        data = {
            "order": self.order, "vocab_size": self.vocab_size, "discount": self.discount,
            "counts": [{" ".join(map(str, h)): dict(t) for h, t in table.items()} for table in self.counts],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> NGramLM:
        data = json.loads(Path(path).read_text())
        lm = cls(data["order"], data["vocab_size"], data["discount"])
        for k, table in enumerate(data["counts"]):
            for h, t in table.items():
                key = tuple(int(x) for x in h.split()) if h else ()
                for w, c in t.items():
                    lm.counts[k][key][int(w)] = c
        return lm


__all__ = ["NGramLM"]
