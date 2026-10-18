"""Word error rate (WER) and character error rate (CER), with the alignment behind them.

WER = (S + D + I) / N over the reference words: substitutions, deletions and insertions
found by the Levenshtein alignment that needs the fewest edits, divided by the number of
reference words. CER is the same thing over characters (spaces included). Both can exceed
100% (many insertions). Error rates are computed over the whole test set (sum of edits
over sum of reference lengths), not averaged per utterance.

`stt-eval` compares two files (reference and hypothesis manifests / text files) and prints
WER, CER and the worst utterances with their alignments, e.g.

    REF: the cat  sat on the mat
    HYP: the cat  sat in *** mat
                      S   D
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self


@dataclass
class EditStats:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.ref_length if self.ref_length else (1.0 if self.errors else 0.0)

    def __iadd__(self, other: EditStats) -> Self:
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions
        self.ref_length += other.ref_length
        return self


def edit_alignment(ref: Sequence, hyp: Sequence) -> list[tuple[str, object, object]]:
    """Levenshtein alignment: list of (op, ref_item, hyp_item) with op in {"=", "S", "D", "I"}."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i, "D"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            sub = cost[i - 1][j - 1] + (0 if same else 1)
            dele = cost[i - 1][j] + 1
            ins = cost[i][j - 1] + 1
            best = min(sub, dele, ins)
            cost[i][j] = best
            back[i][j] = ("=" if same else "S") if best == sub else ("D" if best == dele else "I")
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("=", "S"):
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif op == "D":
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def edit_stats(ref: Sequence, hyp: Sequence) -> EditStats:
    ops = edit_alignment(ref, hyp)
    return EditStats(
        substitutions=sum(op == "S" for op, _, _ in ops),
        deletions=sum(op == "D" for op, _, _ in ops),
        insertions=sum(op == "I" for op, _, _ in ops),
        ref_length=len(ref),
    )


def wer(refs: list[str], hyps: list[str]) -> float:
    total = EditStats()
    for r, h in zip(refs, hyps, strict=True):
        total += edit_stats(r.split(), h.split())
    return total.rate


def cer(refs: list[str], hyps: list[str]) -> float:
    total = EditStats()
    for r, h in zip(refs, hyps, strict=True):
        total += edit_stats(list(r), list(h))
    return total.rate


def error_report(refs: list[str], hyps: list[str]) -> dict:
    """WER/CER plus the counts behind them and per-utterance WER."""
    words, chars, per_utt = EditStats(), EditStats(), []
    for r, h in zip(refs, hyps, strict=True):
        w = edit_stats(r.split(), h.split())
        words += w
        chars += edit_stats(list(r), list(h))
        per_utt.append(w.rate)
    return {
        "wer": words.rate, "cer": chars.rate, "n_utterances": len(refs),
        "word_substitutions": words.substitutions, "word_deletions": words.deletions,
        "word_insertions": words.insertions, "ref_words": words.ref_length, "per_utterance_wer": per_utt,
    }


def format_alignment(ref: str, hyp: str) -> str:
    """Three aligned lines (REF / HYP / ops) for one utterance, word level."""
    ops = edit_alignment(ref.split(), hyp.split())
    cells = []
    for op, r, h in ops:
        r = r or "***"
        h = h or "***"
        width = max(len(r), len(h))
        cells.append((r.ljust(width), h.ljust(width), (op if op != "=" else " ").ljust(width)))
    return "\n".join(f"{label}: " + " ".join(c[i] for c in cells) for i, label in enumerate(("REF", "HYP", "   ")))


def read_texts(path: Path) -> list[str]:
    """A .jsonl manifest (field "text") or a plain text file (one utterance per line)."""
    lines = path.read_text().splitlines()
    if path.suffix == ".jsonl":
        return [json.loads(line)["text"] for line in lines if line.strip()]
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="WER / CER between a reference and a hypothesis file")
    p.add_argument("reference", type=Path, help=".jsonl manifest or text file")
    p.add_argument("hypothesis", type=Path, help=".jsonl (field 'text') or text file, same order")
    p.add_argument("--worst", type=int, default=5, help="print the N worst utterances")
    p.add_argument("--json", type=Path, default=None, help="write the report here")
    args = p.parse_args(argv)
    refs, hyps = read_texts(args.reference), read_texts(args.hypothesis)
    report = error_report(refs, hyps)
    print(f"WER {100 * report['wer']:.2f}%  CER {100 * report['cer']:.2f}%  ({report['n_utterances']} utterances, "
          f"S={report['word_substitutions']} D={report['word_deletions']} I={report['word_insertions']} "
          f"N={report['ref_words']})")
    worst = sorted(range(len(refs)), key=lambda i: -report["per_utterance_wer"][i])[: args.worst]
    for i in worst:
        print(f"\n[{i}] WER {100 * report['per_utterance_wer'][i]:.0f}%\n{format_alignment(refs[i], hyps[i])}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


__all__ = ["EditStats", "cer", "edit_alignment", "edit_stats", "error_report", "format_alignment", "main", "wer"]

if __name__ == "__main__":
    main()
