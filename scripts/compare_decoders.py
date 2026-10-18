"""Greedy vs beam vs beam + n-gram LM for one checkpoint, and a word confusion matrix.

    python scripts/compare_decoders.py checkpoints/fsdd-ctc/best.pt data/fsdd docs/images/fsdd_confusion.png

Trains a character 5-gram LM on the training transcripts, decodes the validation manifest
three ways and prints a markdown table with WER / CER and time, then draws the confusion
matrix of the substituted words (greedy decoding) when the vocabulary is small.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stt.data import read_manifest
from stt.lm import NGramLM
from stt.metrics import cer, edit_alignment, wer
from stt.transcribe import load_checkpoint, transcribe_files

ckpt_path, data_dir, out_png = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
ckpt = load_checkpoint(ckpt_path, "cpu")
tokenizer = ckpt["tokenizer"]
train, val = read_manifest(data_dir / "train.jsonl"), read_manifest(data_dir / "val.jsonl")
lm = NGramLM(order=5, vocab_size=tokenizer.vocab_size).train([tokenizer.encode(u.text) for u in train])
refs = [u.text for u in val]
files = [Path(u.audio) for u in val]

print("| decoding | val WER | val CER | time |")
print("|---|---|---|---|")
hyps_greedy = None
for name, method, beam, use_lm in (("greedy", "greedy", 1, False), ("beam 8", "beam", 8, False),
                                   ("beam 8 + char 5-gram LM (0.3)", "beam", 8, True)):
    t0 = time.time()
    hyps = transcribe_files(ckpt, files, method, beam, lm if use_lm else None, 0.3)
    hyps_greedy = hyps if hyps_greedy is None else hyps_greedy
    print(f"| {name} | {100 * wer(refs, hyps):.1f}% | {100 * cer(refs, hyps):.1f}% | {time.time() - t0:.0f} s |")

vocab = sorted({w for r in refs for w in r.split()})
if len(vocab) <= 20:
    labels = vocab + ["<del>"]
    counts = Counter()
    for ref, hyp in zip(refs, hyps_greedy):
        for op, r, h in edit_alignment(ref.split(), hyp.split()):
            if op == "D":
                counts[(r, "<del>")] += 1
            elif op in ("S", "=") and r is not None and h is not None:
                counts[(r, h if h in vocab else "<del>")] += 1
    mat = np.array([[counts[(r, h)] for h in labels] for r in vocab], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(mat / np.maximum(mat.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(vocab)), vocab)
    ax.set_xlabel("hypothesis (greedy)")
    ax.set_ylabel("reference")
    ax.set_title(f"word confusion, {Path(ckpt_path).parent.name}")
    for i in range(len(vocab)):
        for j in range(len(labels)):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)
