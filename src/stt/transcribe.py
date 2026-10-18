"""`stt-transcribe checkpoint.pt audio.wav [more.wav | a folder]`: text out.

Loads the checkpoint (train.py), computes features exactly as in training (minus SpecAugment),
decodes with greedy or beam search (decoding.py), optionally with an n-gram LM (lm.py),
and prints one line per file (`--jsonl` writes a manifest-like file instead).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stt.data import SpeechDataset, Utterance
from stt.decoding import decode_batch
from stt.features import load_audio
from stt.lm import NGramLM
from stt.train import load_checkpoint, pick_device

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg")


def list_audio(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_SUFFIXES) if p.is_dir() else [p]
    return files


@torch.no_grad()
def transcribe_files(ckpt: dict, files: list[Path], method: str = "greedy", beam: int = 4, lm: NGramLM | None = None,
                     lm_weight: float = 0.3, batch_size: int = 8) -> list[str]:
    model, tokenizer, data_cfg = ckpt["model"], ckpt["tokenizer"], ckpt["data_config"]
    device = next(model.parameters()).device
    ds = SpeechDataset(data_cfg, [Utterance(f, "", 0.0, f.stem) for f in files], tokenizer, train=False)
    texts = []
    for start in range(0, len(files), batch_size):
        chunk = files[start : start + batch_size]
        feats = [ds.features(load_audio(f, data_cfg.sample_rate)) for f in chunk]
        lengths = torch.tensor([f.size(0) for f in feats])
        padded = torch.zeros(len(feats), int(lengths.max()), feats[0].size(1))
        for i, f in enumerate(feats):
            padded[i, : f.size(0)] = f
        ids = decode_batch(model, padded.to(device), lengths.to(device), method, beam, lm, lm_weight)
        texts += [tokenizer.decode(seq) for seq in ids]
    return texts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Transcribe audio files with a trained checkpoint")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("audio", type=Path, nargs="+", help="files and/or folders")
    p.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--lm", type=Path, default=None, help="n-gram LM saved by stt.lm.NGramLM.save")
    p.add_argument("--lm-weight", type=float, default=0.3)
    p.add_argument("--device", default="auto")
    p.add_argument("--jsonl", type=Path, default=None, help="write {audio, text} lines here")
    args = p.parse_args(argv)
    ckpt = load_checkpoint(args.checkpoint, pick_device(args.device))
    lm = NGramLM.load(args.lm) if args.lm else None
    files = list_audio(args.audio)
    texts = transcribe_files(ckpt, files, args.decode, args.beam_size, lm, args.lm_weight)
    if args.jsonl:
        with args.jsonl.open("w") as f:
            for file, text in zip(files, texts):
                f.write(json.dumps({"audio": str(file), "text": text}) + "\n")
    for file, text in zip(files, texts):
        print(f"{file}\t{text}")


__all__ = ["list_audio", "main", "transcribe_files"]

if __name__ == "__main__":
    main()
