"""Manifests, the dataset, and batching.

A dataset is a folder with `train.jsonl` and `val.jsonl`: one JSON object per line,

    {"audio": "wavs/0001.wav", "text": "yes no yes", "duration": 2.1}

Relative audio paths are resolved against the manifest's folder. Any format soundfile
reads (wav, flac, ogg) at any sample rate works; audio is resampled on load.

`SpeechDataset` loads one clip, computes its log-mel features (features.py), applies CMVN
and (training) SpecAugment, and tokenizes the text. `collate` pads a list of items into
tensors + lengths. `BucketSampler` groups clips of similar duration into batches so that
little of the batch is padding, which matters for the sequence losses and for speed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from stt.config import DataConfig
from stt.features import LogMel, cmvn, load_audio, spec_augment
from stt.tokenizer import BLANK, CharTokenizer, SentencePieceTokenizer, normalize_text


@dataclass
class Utterance:
    audio: Path
    text: str
    duration: float
    id: str = ""


def read_manifest(path: str | Path, lowercase: bool = True) -> list[Utterance]:
    path = Path(path)
    utts = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        audio = Path(d["audio"])
        if not audio.is_absolute():
            audio = path.parent / audio
        utts.append(Utterance(audio, normalize_text(d["text"], lowercase), float(d.get("duration", 0.0)),
                              d.get("id", audio.stem)))
    return utts


def write_manifest(path: str | Path, utts: list[Utterance] | list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u in utts:
            d = u if isinstance(u, dict) else {"audio": str(u.audio), "text": u.text, "duration": u.duration, "id": u.id}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def build_tokenizer(cfg: DataConfig, utts: list[Utterance]):
    if cfg.tokenizer == "bpe":
        return SentencePieceTokenizer(cfg.bpe_model)
    return CharTokenizer.from_texts([u.text for u in utts])


class SpeechDataset(Dataset):
    def __init__(self, cfg: DataConfig, utts: list[Utterance], tokenizer, train: bool, seed: int = 0):
        self.cfg = cfg
        self.utts = [u for u in utts if not (train and u.duration > cfg.max_duration_s)]
        self.tokenizer = tokenizer
        self.train = train
        self.logmel = LogMel(cfg)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.utts)

    def features(self, wav) -> torch.Tensor:
        feats = cmvn(self.logmel(torch.as_tensor(wav)))
        if self.train and self.cfg.specaugment:
            feats = spec_augment(feats, self.cfg.freq_masks, self.cfg.freq_mask_width, self.cfg.time_masks,
                                 self.cfg.time_mask_width, self.cfg.time_mask_max_ratio, self.generator)
        return feats

    def __getitem__(self, i: int) -> dict:
        u = self.utts[i]
        wav = load_audio(u.audio, self.cfg.sample_rate)
        return {"feats": self.features(wav), "tokens": torch.tensor(self.tokenizer.encode(u.text), dtype=torch.long),
                "text": u.text, "id": u.id}


def collate(items: list[dict]) -> dict:
    feat_lengths = torch.tensor([it["feats"].size(0) for it in items])
    target_lengths = torch.tensor([it["tokens"].numel() for it in items])
    feats = torch.zeros(len(items), int(feat_lengths.max()), items[0]["feats"].size(1))
    targets = torch.full((len(items), max(int(target_lengths.max()), 1)), BLANK, dtype=torch.long)
    for b, it in enumerate(items):
        feats[b, : it["feats"].size(0)] = it["feats"]
        targets[b, : it["tokens"].numel()] = it["tokens"]
    return {"feats": feats, "feat_lengths": feat_lengths, "targets": targets, "target_lengths": target_lengths,
            "texts": [it["text"] for it in items], "ids": [it["id"] for it in items]}


class BucketSampler(Sampler[list[int]]):
    """Batches of similar-duration clips. Clips are sorted by duration, cut into buckets of
    `bucket_size`, each bucket shuffled and cut into batches; the batch order is shuffled too."""

    def __init__(self, durations: list[float], batch_size: int, bucket_size: int, shuffle: bool, seed: int = 0):
        self.order = sorted(range(len(durations)), key=lambda i: durations[i])
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for start in range(0, len(self.order), self.bucket_size):
            bucket = self.order[start : start + self.bucket_size]
            if self.shuffle:
                rng.shuffle(bucket)
            batches += [bucket[i : i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(-(-min(self.bucket_size, len(self.order) - s) // self.batch_size)
                   for s in range(0, len(self.order), self.bucket_size))


def make_loader(dataset: SpeechDataset, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 0) -> DataLoader:
    sampler = BucketSampler([u.duration for u in dataset.utts], batch_size, dataset.cfg.bucket_size, shuffle, seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers)


__all__ = [
    "BucketSampler",
    "SpeechDataset",
    "Utterance",
    "build_tokenizer",
    "collate",
    "make_loader",
    "read_manifest",
    "write_manifest",
]
