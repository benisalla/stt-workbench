"""From model outputs to token sequences: the search algorithms.

CTC
    greedy       argmax per frame, merge repeats, drop blanks. One pass, no search.
    prefix beam  (Hannun et al. 2014) keeps the best `beam` PREFIXES (not frame paths), each
                 with two probabilities: ending in blank (p_b) and ending in a non-blank
                 (p_nb), so that all paths collapsing to the same prefix are merged. An
                 n-gram LM (lm.py) can add `lm_weight * log p_lm(c | prefix)` at each
                 extension, with `word_bonus` per emitted token to counter the LM's bias
                 towards short outputs.

Attention decoders (aed / las / hybrid)
    greedy       feed <sos>, take the argmax token, feed it back, stop at <eos>.
    beam         keep the best `beam` partial sentences by summed log-prob; finished ones
                 (ended with <eos>) are set aside; scores are length-normalised at the end
                 so long sentences are not punished. For hybrid models, the CTC prefix
                 score (Watanabe et al. 2017) of each extension is mixed in:
                 score = (1 - w) * log p_att + w * log p_ctc_prefix, which rejects extensions
                 the CTC alignment finds impossible (skips, loops).

RNN-T
    greedy       for each encoder frame, emit symbols until the joiner says blank
                 (at most `max_symbols` per frame), updating the prediction network.
    beam         "modified" beam search (one symbol per frame at most): at each frame every
                 hypothesis either emits one symbol or blank; hypotheses with the same token
                 sequence are merged by log-sum-exp.

Every function returns token id lists (specials removed), one per utterance; the
tokenizer's `decode` turns them into text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from stt.lm import NGramLM
from stt.tokenizer import BLANK, EOS, SOS, SPECIALS

NEG_INF = float("-inf")


def logsumexp(*xs: float) -> float:
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ---------------------------------------------------------------------------------------- CTC
def ctc_greedy(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """(B, T, V) log-probs -> token lists. Collapse repeats, remove blanks."""
    best = log_probs.argmax(-1).cpu()
    out = []
    for b in range(best.size(0)):
        seq, prev = [], BLANK
        for t in range(int(lengths[b])):
            tok = int(best[b, t])
            if tok != BLANK and tok != prev:
                seq.append(tok)
            prev = tok
        out.append(seq)
    return out


def ctc_prefix_beam_search(
    log_probs: torch.Tensor, beam: int = 8, lm: NGramLM | None = None, lm_weight: float = 0.5,
    word_bonus: float = 0.0, prune: float = -10.0,
) -> list[int]:
    """One utterance, (T, V) log-probs -> best token list."""
    lp = log_probs.cpu()
    T, V = lp.shape
    # prefix -> (log p_blank, log p_nonblank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}
    lm_cache: dict[tuple[int, ...], float] = {(): 0.0}

    def lm_score(prefix: tuple[int, ...]) -> float:
        if prefix not in lm_cache:
            lm_cache[prefix] = lm_cache[prefix[:-1]] + lm_weight * lm.score(list(prefix[:-1]), prefix[-1]) + word_bonus
        return lm_cache[prefix]

    for t in range(T):
        frame = lp[t]
        candidates = [c for c in range(V) if frame[c] > prune or c == BLANK]
        new: dict[tuple[int, ...], list[float]] = {}

        def add(prefix, pb, pnb, new=new):
            cur = new.setdefault(prefix, [NEG_INF, NEG_INF])
            cur[0] = logsumexp(cur[0], pb)
            cur[1] = logsumexp(cur[1], pnb)

        for prefix, (pb, pnb) in beams.items():
            total = logsumexp(pb, pnb)
            for c in candidates:
                p = float(frame[c])
                if c == BLANK:
                    add(prefix, total + p, NEG_INF)
                elif prefix and c == prefix[-1]:
                    add(prefix, NEG_INF, pnb + p)  # repeat without blank: same prefix
                    add(prefix + (c,), NEG_INF, pb + p)  # repeat after blank: new token
                else:
                    add(prefix + (c,), NEG_INF, total + p)

        def rank(item):
            prefix, (pb, pnb) = item
            s = logsumexp(pb, pnb)
            return s + (lm_score(prefix) if lm is not None else word_bonus * len(prefix))

        beams = dict(sorted(((k, tuple(v)) for k, v in new.items()), key=rank, reverse=True)[:beam])
    best = max(beams.items(), key=lambda kv: logsumexp(*kv[1]) + (lm_score(kv[0]) if lm else 0.0))
    return list(best[0])


class CTCPrefixScorer:
    """CTC prefix probability p(g, ...) for joint CTC/attention decoding (Watanabe et al. 2017).

    For a prefix g, keeps r_n[t] / r_b[t] = log-prob of g at frame t ending in non-blank/blank.
    `extend(state, c)` returns the new state and log p_ctc(g + c), the probability that the
    output STARTS with g + c (any continuation), which attention beam search uses as the
    CTC score of the extension.
    """

    def __init__(self, log_probs: torch.Tensor):
        self.lp = log_probs.cpu()
        self.T = self.lp.size(0)

    def initial(self) -> tuple[torch.Tensor, torch.Tensor, int, float]:
        r_b = torch.cumsum(self.lp[:, BLANK], dim=0)  # all-blank paths
        r_n = torch.full((self.T,), NEG_INF)
        return r_n, r_b, BLANK, 0.0

    def extend(self, state, c: int) -> tuple[tuple, float]:
        r_n, r_b, last, _ = state
        new_n = torch.full((self.T,), NEG_INF)
        new_b = torch.full((self.T,), NEG_INF)
        lp_c = self.lp[:, c]
        lp_b = self.lp[:, BLANK]
        # phi[t-1]: probability of g at t-1 that may be followed by c (no merge with a repeat)
        prev = torch.logaddexp(r_n, r_b) if c != last else r_b
        if last == BLANK and r_n[0] == NEG_INF and r_b[0] != NEG_INF:
            new_n[0] = lp_c[0]  # empty prefix: c may start at frame 0
        for t in range(1, self.T):
            new_n[t] = torch.logaddexp(new_n[t - 1], prev[t - 1]) + lp_c[t]
            new_b[t] = torch.logaddexp(new_n[t - 1], new_b[t - 1]) + lp_b[t]
        # p(g + c, ...) = sum over t of phi[t-1] * p(c | t), plus starting at 0
        psi = torch.logsumexp(torch.cat([prev[:-1] + lp_c[1:], new_n[:1]]), dim=0)
        return (new_n, new_b, c, float(psi)), float(psi)

    def final(self, state) -> float:
        r_n, r_b, _, _ = state
        return float(torch.logaddexp(r_n[-1], r_b[-1]))


# ------------------------------------------------------------------------------ attention
class DecoderInterface:
    """A uniform step API over the two attention decoders (models.py).

    `init(enc, mask)` -> state; `step(state, prefix_tokens)` -> (log_probs (V,), new state).
    The Transformer decoder re-runs on the whole prefix (simple; fine for short outputs);
    the LAS decoder carries its LSTM/attention state along.
    """

    def __init__(self, model, enc: torch.Tensor, mask: torch.Tensor):
        self.model = model
        self.enc = enc  # (1, T, D)
        self.mask = mask
        self.is_las = hasattr(model.decoder, "step")

    def init(self):
        return self.model.decoder.init_state(self.enc) if self.is_las else None

    def step(self, state, prefix: list[int]):
        if self.is_las:
            token = torch.tensor([prefix[-1]], device=self.enc.device)
            logits, state = self.model.decoder.step(token, state, self.enc, self.mask)
            return torch.log_softmax(logits[0].float(), -1), state
        tokens = torch.tensor([prefix], device=self.enc.device)
        logits = self.model.decoder(tokens, self.enc, self.mask)
        return torch.log_softmax(logits[0, -1].float(), -1), None


@dataclass(order=True)
class Hypothesis:
    score: float
    tokens: list[int] = field(compare=False)
    state: object = field(compare=False, default=None)
    ctc_state: object = field(compare=False, default=None)
    att_score: float = field(compare=False, default=0.0)


@torch.no_grad()
def attention_greedy(model, enc: torch.Tensor, mask: torch.Tensor, max_len: int) -> list[int]:
    dec = DecoderInterface(model, enc, mask)
    state, prefix = dec.init(), [SOS]
    for _ in range(max_len):
        lp, state = dec.step(state, prefix)
        tok = int(lp.argmax())
        if tok == EOS:
            break
        prefix.append(tok)
    return [t for t in prefix[1:] if t >= len(SPECIALS)]


@torch.no_grad()
def attention_beam_search(
    model, enc: torch.Tensor, mask: torch.Tensor, max_len: int, beam: int = 4, ctc_weight: float = 0.0,
    ctc_log_probs: torch.Tensor | None = None, lm: NGramLM | None = None, lm_weight: float = 0.3,
    length_penalty: float = 1.0,
) -> list[int]:
    """One utterance (enc (1, T, D)). Joint CTC/attention when ctc_weight > 0 and ctc_log_probs given."""
    dec = DecoderInterface(model, enc, mask)
    scorer = CTCPrefixScorer(ctc_log_probs) if (ctc_weight > 0 and ctc_log_probs is not None) else None
    start = Hypothesis(0.0, [SOS], dec.init(), scorer.initial() if scorer else None)
    live, done = [start], []
    for _ in range(max_len):
        candidates = []
        for hyp in live:
            lp, state = dec.step(hyp.state, hyp.tokens)
            topk = torch.topk(lp, min(beam * 2, lp.numel()))
            for score, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                if tok in (SOS, BLANK):
                    continue
                if tok == EOS:
                    total = hyp.score + (1 - ctc_weight) * score
                    if scorer is not None:
                        total += ctc_weight * (scorer.final(hyp.ctc_state) - hyp.ctc_state[3])
                    if lm is not None:
                        total += lm_weight * lm.score(hyp.tokens[1:], EOS)
                    done.append(Hypothesis(total, hyp.tokens[1:], None))
                    continue
                total = hyp.score + (1 - ctc_weight) * score
                ctc_state = None
                if scorer is not None:
                    ctc_state, psi = scorer.extend(hyp.ctc_state, tok)
                    total += ctc_weight * (psi - hyp.ctc_state[3])  # prefix score increment
                if lm is not None:
                    total += lm_weight * lm.score(hyp.tokens[1:], tok)
                candidates.append(Hypothesis(total, hyp.tokens + [tok], state, ctc_state))
        if not candidates:
            break
        live = sorted(candidates, reverse=True)[:beam]
        if done and len(done) >= beam:
            best_done = max(h.score / (len(h.tokens) + 1) ** length_penalty for h in done)
            if all(h.score / (len(h.tokens)) ** length_penalty < best_done for h in live):
                break
    if not done:
        done = [Hypothesis(h.score, h.tokens[1:]) for h in live]
    best = max(done, key=lambda h: h.score / (len(h.tokens) + 1) ** length_penalty)
    return [t for t in best.tokens if t >= len(SPECIALS)]


# ------------------------------------------------------------------------------------ RNN-T
@torch.no_grad()
def rnnt_greedy(model, enc: torch.Tensor, max_symbols: int = 4) -> list[int]:
    """One utterance, enc (1, T, D)."""
    tokens: list[int] = []
    pred_out, state = model.predictor(torch.tensor([[BLANK]], device=enc.device))
    for t in range(enc.size(1)):
        for _ in range(max_symbols):
            logits = model.joiner(enc[:, t], pred_out[:, -1])
            tok = int(logits.argmax(-1))
            if tok == BLANK:
                break
            tokens.append(tok)
            pred_out, state = model.predictor(torch.tensor([[tok]], device=enc.device), state)
    return [t for t in tokens if t >= len(SPECIALS)]


@torch.no_grad()
def rnnt_beam_search(model, enc: torch.Tensor, beam: int = 4, lm: NGramLM | None = None, lm_weight: float = 0.3) -> list[int]:
    """Modified beam search (at most one symbol per frame), merging identical token sequences."""
    pred_out, state = model.predictor(torch.tensor([[BLANK]], device=enc.device))
    hyps: dict[tuple[int, ...], Hypothesis] = {(): Hypothesis(0.0, [], (pred_out, state))}
    for t in range(enc.size(1)):
        new: dict[tuple[int, ...], Hypothesis] = {}
        for key, hyp in hyps.items():
            pred_out, state = hyp.state
            lp = torch.log_softmax(model.joiner(enc[:, t], pred_out[:, -1])[0].float(), -1)
            # blank: same tokens, same predictor state
            blank_score = hyp.score + float(lp[BLANK])
            if key in new:
                new[key].score = logsumexp(new[key].score, blank_score)
            else:
                new[key] = Hypothesis(blank_score, hyp.tokens, hyp.state)
            topk = torch.topk(lp, min(beam, lp.numel()))
            for score, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                if tok == BLANK or tok < len(SPECIALS):
                    continue
                total = hyp.score + score + (lm_weight * lm.score(hyp.tokens, tok) if lm else 0.0)
                new_key = key + (tok,)
                if new_key in new:
                    new[new_key].score = logsumexp(new[new_key].score, total)
                else:
                    new[new_key] = Hypothesis(total, hyp.tokens + [tok], None)
        # run the predictor once per new token sequence (lazy state)
        kept = sorted(new.values(), reverse=True)[:beam]
        for h in kept:
            if h.state is None:
                parent = hyps.get(tuple(h.tokens[:-1]))
                p_out, p_state = model.predictor(torch.tensor([[h.tokens[-1]]], device=enc.device),
                                                 parent.state[1] if parent else None)
                h.state = (p_out, p_state)
        hyps = {tuple(h.tokens): h for h in kept}
    best = max(hyps.values())
    return best.tokens


# ---------------------------------------------------------------------------------- entry
@torch.no_grad()
def decode_batch(
    model, feats: torch.Tensor, lengths: torch.Tensor, method: str = "greedy", beam: int = 4,
    lm: NGramLM | None = None, lm_weight: float = 0.3, max_len_ratio: float = 1.0,
) -> list[list[int]]:
    """Decode a padded batch with whatever the model is. Returns token id lists."""
    cfg = model.cfg
    enc, enc_lengths = model.encode(feats, lengths)
    if cfg.model == "ctc":
        lp = model.ctc_log_probs(enc)
        if method == "greedy":
            return ctc_greedy(lp, enc_lengths)
        return [ctc_prefix_beam_search(lp[b, : int(enc_lengths[b])], beam, lm, lm_weight) for b in range(lp.size(0))]
    out = []
    for b in range(enc.size(0)):
        n = int(enc_lengths[b])
        e = enc[b : b + 1, :n]
        if cfg.model == "rnnt":
            out.append(rnnt_greedy(model, e) if method == "greedy" else rnnt_beam_search(model, e, beam, lm, lm_weight))
            continue
        mask = torch.ones(1, n, dtype=torch.bool, device=enc.device)
        max_len = max(4, int(n * max_len_ratio))
        if method == "greedy":
            out.append(attention_greedy(model, e, mask, max_len))
        else:
            ctc_lp = model.ctc_log_probs(e)[0] if cfg.model == "hybrid" else None
            out.append(attention_beam_search(
                model, e, mask, max_len, beam, ctc_weight=cfg.ctc_weight if cfg.model == "hybrid" else 0.0,
                ctc_log_probs=ctc_lp, lm=lm, lm_weight=lm_weight,
            ))
    return out


__all__ = [
    "CTCPrefixScorer",
    "DecoderInterface",
    "Hypothesis",
    "attention_beam_search",
    "attention_greedy",
    "ctc_greedy",
    "ctc_prefix_beam_search",
    "decode_batch",
    "rnnt_beam_search",
    "rnnt_greedy",
]
