"""Load the checkpoint of example 01, transcribe the validation clips and score them.

    python examples/02_transcribe_and_score.py

Shows the three things an inference pipeline does: rebuild model + tokenizer from a
checkpoint, turn audio files into text (`transcribe_files`, greedy and beam search), and
compare the hypotheses with the references (WER / CER and word-level alignments).
"""

from pathlib import Path

from stt.data import read_manifest
from stt.metrics import error_report, format_alignment
from stt.train import load_checkpoint
from stt.transcribe import transcribe_files

ckpt = load_checkpoint("checkpoints/examples-ctc/best.pt")  # run examples/01_train_on_toy_data.py first
print(f"{ckpt['model_config'].model} / {ckpt['model_config'].encoder}, step {ckpt['step']}, "
      f"vocabulary of {ckpt['tokenizer'].vocab_size} tokens")

val = read_manifest(Path("data/examples/synthetic/val.jsonl"))
files = [Path(u.audio) for u in val]
refs = [u.text for u in val]

hyps = {}
for method, beam in (("greedy", 1), ("beam", 8)):
    hyps[method] = transcribe_files(ckpt, files, method=method, beam=beam)
    r = error_report(refs, hyps[method])
    print(f"{method:6s}: WER {100 * r['wer']:.1f}%  CER {100 * r['cer']:.1f}%  "
          f"(S {r['word_substitutions']}  D {r['word_deletions']}  I {r['word_insertions']} over {r['ref_words']} words)")

print("\nworst utterances (greedy):")
per_utt = error_report(refs, hyps["greedy"])["per_utterance_wer"]
for i in sorted(range(len(refs)), key=lambda i: -per_utt[i])[:3]:
    print(format_alignment(refs[i], hyps["greedy"][i]))
    print()

print("one clip, the way an application would do it:")
text = transcribe_files(ckpt, [files[0]])[0]
print(f"  {files[0].name} -> {text!r}")
