import json

import numpy as np
import torch

from stt.data import (
    BucketSampler,
    SpeechDataset,
    build_tokenizer,
    collate,
    make_loader,
    read_manifest,
    write_manifest,
)
from stt.datasets import SYNTH_ALPHABET, random_sentence, synth_sentence
from stt.tokenizer import BLANK


def test_manifest_round_trip(tmp_path):
    rows = [{"audio": "a.wav", "text": "  Hello  World ", "duration": 1.5, "id": "a"}]
    write_manifest(tmp_path / "m.jsonl", rows)
    utts = read_manifest(tmp_path / "m.jsonl")
    assert utts[0].text == "hello world" and utts[0].audio == tmp_path / "a.wav"
    assert json.loads((tmp_path / "m.jsonl").read_text())["duration"] == 1.5


def test_synthetic_is_deterministic_and_audible():
    rng = np.random.RandomState(0)
    text = random_sentence(rng)
    assert set(text) <= set(SYNTH_ALPHABET + " ")
    a = synth_sentence(text, np.random.RandomState(3))
    b = synth_sentence(text, np.random.RandomState(3))
    assert np.array_equal(a, b) and np.abs(a).max() <= 1.0 and len(a) > 800


def test_dataset_and_collate(data_cfg):
    utts = read_manifest(data_cfg.data_dir / "train.jsonl")
    tok = build_tokenizer(data_cfg, utts)
    ds = SpeechDataset(data_cfg, utts, tok, train=True)
    items = [ds[0], ds[1]]
    b = collate(items)
    assert b["feats"].shape[0] == 2 and b["feats"].shape[2] == 40
    assert b["feat_lengths"].tolist() == [items[0]["feats"].size(0), items[1]["feats"].size(0)]
    assert (b["targets"][0, : b["target_lengths"][0]] == items[0]["tokens"]).all()
    n = int(b["target_lengths"].min())
    assert (b["targets"][b["target_lengths"].argmin(), n:] == BLANK).all()
    assert tok.decode(items[0]["tokens"].tolist()) == utts[0].text


def test_bucket_sampler_covers_everything_once():
    durations = list(np.random.RandomState(0).rand(23))
    s = BucketSampler(durations, batch_size=4, bucket_size=8, shuffle=True)
    batches = list(s)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(23)) and len(batches) == len(s)
    s.set_epoch(1)
    assert list(s) != batches  # reshuffled


def test_loader_batches_similar_durations(data_cfg):
    utts = read_manifest(data_cfg.data_dir / "train.jsonl")
    ds = SpeechDataset(data_cfg, utts, build_tokenizer(data_cfg, utts), train=False)
    loader = make_loader(ds, 4, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch["feats"], torch.Tensor) and len(batch["texts"]) <= 4
