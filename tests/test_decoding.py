import math

import pytest
import torch

from stt.decoding import (
    CTCPrefixScorer,
    attention_beam_search,
    attention_greedy,
    ctc_greedy,
    ctc_prefix_beam_search,
    decode_batch,
    rnnt_beam_search,
    rnnt_greedy,
)
from stt.lm import NGramLM
from stt.models import build_model
from stt.tokenizer import SPECIALS
from tests.conftest import micro_model_cfg

V = 10


def peaked(path, v=V, sharp=8.0):
    lp = torch.full((len(path), v), -sharp)
    for t, c in enumerate(path):
        lp[t, c] = 0.0
    return torch.log_softmax(lp, -1)


def test_ctc_greedy_collapses():
    lp = peaked([0, 4, 4, 0, 4, 5, 5, 0])
    assert ctc_greedy(lp[None], torch.tensor([8])) == [[4, 4, 5]]
    assert ctc_greedy(lp[None], torch.tensor([3])) == [[4]]


def test_prefix_beam_matches_greedy_when_peaked():
    path = [0, 4, 4, 0, 4, 5, 0, 6, 6]
    lp = peaked(path)
    assert ctc_prefix_beam_search(lp, beam=4) == ctc_greedy(lp[None], torch.tensor([9]))[0]


def test_prefix_beam_merges_paths_greedy_misses():
    """Classic case: greedy picks the single best path, beam search sums paths of the same prefix."""
    # frame 0: blank .4, a .3, b .3 ; frame 1: blank .4, a .3, b .3 -> greedy "" ; beam: p(a)=.3*.4+.4*.3+.3*.3=.33 > p("")=.16
    lp = torch.log(torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.3, 0.3], [0.4, 0.0, 0.0, 0.0, 0.3, 0.3]]) + 1e-9)
    assert ctc_greedy(lp[None], torch.tensor([2])) == [[]]
    assert ctc_prefix_beam_search(lp, beam=4) in ([4], [5])


def test_prefix_scorer_sums_to_sequence_probability():
    """p(prefix 'ab' then end) from the scorer must equal the brute-force CTC probability."""
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(4, 6), -1)
    scorer = CTCPrefixScorer(lp)
    state = scorer.initial()
    state, _ = scorer.extend(state, 4)
    state, _ = scorer.extend(state, 5)
    total = scorer.final(state)
    # brute force over all 6^4 paths collapsing to [4, 5]
    import itertools

    brute = 0.0
    for path in itertools.product(range(6), repeat=4):
        seq, prev = [], 0
        for c in path:
            if c != 0 and c != prev:
                seq.append(c)
            prev = c
        if seq == [4, 5]:
            brute += math.exp(sum(float(lp[t, c]) for t, c in enumerate(path)))
    assert abs(math.exp(total) - brute) < 1e-6


def test_lm_biases_beam_search():
    lp = torch.log(torch.tensor([[0.2, 0, 0, 0, 0.4, 0.4], [0.2, 0, 0, 0, 0.4, 0.4]]) + 1e-9)
    lm = NGramLM(order=2, vocab_size=6).train([[5, 5]] * 20)
    assert ctc_prefix_beam_search(lp, beam=4, lm=lm, lm_weight=1.0) == [5]
    lm2 = NGramLM(order=2, vocab_size=6).train([[4, 4]] * 20)
    assert ctc_prefix_beam_search(lp, beam=4, lm=lm2, lm_weight=1.0) == [4]


@pytest.mark.parametrize("model", ["aed", "las"])
def test_attention_decoders_return_tokens_without_specials(model):
    torch.manual_seed(0)
    m = build_model(micro_model_cfg(model), 40, V).eval()
    enc, _ = m.encode(torch.randn(1, 40, 40), torch.tensor([40]))
    mask = torch.ones(1, enc.size(1), dtype=torch.bool)
    for seq in (attention_greedy(m, enc, mask, 8), attention_beam_search(m, enc, mask, 8, beam=3)):
        assert all(t >= len(SPECIALS) for t in seq) and len(seq) <= 8


def test_rnnt_decoders():
    pytest.importorskip("torchaudio")
    torch.manual_seed(0)
    m = build_model(micro_model_cfg("rnnt"), 40, V).eval()
    enc, _ = m.encode(torch.randn(1, 40, 40), torch.tensor([40]))
    for seq in (rnnt_greedy(m, enc), rnnt_beam_search(m, enc, beam=3)):
        assert all(t >= len(SPECIALS) for t in seq)


@pytest.mark.parametrize("model", ["ctc", "aed", "las", "hybrid", "rnnt"])
@pytest.mark.parametrize("method", ["greedy", "beam"])
def test_decode_batch_every_model(model, method):
    if model == "rnnt":
        pytest.importorskip("torchaudio")
    torch.manual_seed(0)
    m = build_model(micro_model_cfg(model), 40, V).eval()
    out = decode_batch(m, torch.randn(2, 50, 40), torch.tensor([50, 30]), method, beam=2)
    assert len(out) == 2 and all(isinstance(s, list) for s in out)
