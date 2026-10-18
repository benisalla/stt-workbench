"""Beam search with a character n-gram language model, trained on the training transcripts.

    python examples/03_beam_search_with_lm.py

The acoustic model alone scores every letter sequence; an n-gram LM adds `lm_weight *
log p(next char | previous chars)` to each prefix during the beam search, which removes
impossible spellings. On the toy language the gain is small (random words), on real
transcripts it is large (see the FSDD numbers in the README).
"""

from pathlib import Path

from stt.data import read_manifest
from stt.lm import NGramLM
from stt.metrics import cer, wer
from stt.train import load_checkpoint
from stt.transcribe import transcribe_files

ckpt = load_checkpoint("checkpoints/examples-ctc/best.pt")  # run examples/01_train_on_toy_data.py first
tok = ckpt["tokenizer"]
train = read_manifest(Path("data/examples/synthetic/train.jsonl"))
val = read_manifest(Path("data/examples/synthetic/val.jsonl"))

# 1. An LM over token ids: Kneser-Ney smoothed counts, nothing to optimise.
lm = NGramLM(order=5, vocab_size=tok.vocab_size).train([tok.encode(u.text) for u in train])

# 2. What it knows: the most likely next characters after a prefix.
prefix = train[0].text[:3]
history = tok.encode(prefix)
ranked = sorted(range(tok.vocab_size), key=lambda t: -lm.score(history, t))[:5]
print(f"after {prefix!r} the LM expects: " + ", ".join(f"{tok.decode([t]) or '<special>'!r}" for t in ranked))

# 3. Decode the validation set three ways.
files = [Path(u.audio) for u in val]
refs = [u.text for u in val]
for name, kw in (("greedy", {"method": "greedy"}),
                 ("beam 8", {"method": "beam", "beam": 8}),
                 ("beam 8 + 5-gram LM", {"method": "beam", "beam": 8, "lm": lm, "lm_weight": 0.3})):
    hyps = transcribe_files(ckpt, files, **kw)
    print(f"{name:20s} WER {100 * wer(refs, hyps):5.1f}%   CER {100 * cer(refs, hyps):5.1f}%")

lm.save("checkpoints/examples-ctc/lm.json")  # stt-transcribe ... --decode beam --lm checkpoints/examples-ctc/lm.json
print("saved checkpoints/examples-ctc/lm.json")
