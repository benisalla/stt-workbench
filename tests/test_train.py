import dataclasses

import pytest
import torch

from stt.config import TrainConfig
from stt.train import evaluate, load_checkpoint, lr_at, train
from stt.transcribe import transcribe_files
from tests.conftest import micro_model_cfg


def test_lr_schedule():
    cfg = TrainConfig(lr=1.0, warmup_steps=10, max_steps=110, min_lr=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(60, cfg) == pytest.approx(0.55, abs=1e-6)
    assert lr_at(110, cfg) == pytest.approx(0.1)


@pytest.mark.parametrize("model", ["ctc", "hybrid", "rnnt"])
def test_train_eval_resume_transcribe(tmp_path, data_cfg, model):
    if model == "rnnt":
        pytest.importorskip("torchaudio")
    out = tmp_path / model
    train_cfg = TrainConfig(output_dir=out, batch_size=4, max_steps=6, warmup_steps=2, eval_every=3, log_every=3,
                            device="cpu", eval_batches=1)
    result = train(data_cfg, micro_model_cfg(model, "transformer"), train_cfg, log=lambda *_: None)
    assert result["step"] == 6 and (out / "last.pt").exists() and (out / "best.pt").exists()
    assert (out / "log.csv").read_text().count("\n") >= 3
    # resume for 2 more steps
    more = dataclasses.replace(train_cfg, max_steps=8, eval_every=4, log_every=4)
    result = train(data_cfg, micro_model_cfg(model, "transformer"), more, log=lambda *_: None)
    assert result["step"] == 8
    ckpt = load_checkpoint(out / "best.pt")
    assert ckpt["model_config"].model == model and ckpt["tokenizer"].vocab_size > 4
    wav = next((data_cfg.data_dir / "val").glob("*.wav"))
    texts = transcribe_files(ckpt, [wav], method="greedy")
    assert len(texts) == 1 and isinstance(texts[0], str)
    texts = transcribe_files(ckpt, [wav], method="beam", beam=2)
    assert len(texts) == 1


def test_evaluate_reports_wer(tmp_path, data_cfg):
    from stt.data import SpeechDataset, build_tokenizer, make_loader, read_manifest
    from stt.models import build_model

    utts = read_manifest(data_cfg.data_dir / "val.jsonl")
    tok = build_tokenizer(data_cfg, utts)
    model = build_model(micro_model_cfg("ctc"), 40, tok.vocab_size)
    loader = make_loader(SpeechDataset(data_cfg, utts, tok, train=False), 4, False)
    res = evaluate(model, loader, tok, torch.device("cpu"), max_batches=1)
    assert 0.0 <= res["wer"] and res["n"] == 4 and len(res["examples"]) == 4
