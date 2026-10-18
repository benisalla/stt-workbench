"""Training loop shared by every approach: `stt-train --model ctc --encoder conformer ...`.

What happens each step:
1. a bucketed batch of log-mel features and token targets (data.py);
2. the model computes its own loss (CTC, cross-entropy, joint, or transducer; models.py)
   under autocast if `--precision bf16|fp16` (the sequence losses themselves run in fp32);
3. gradient clipping, AdamW, linear warm-up then cosine decay of the learning rate.

Every `eval_every` steps the validation manifest is decoded with the configured decoder
(greedy by default; `--decode beam`) and WER / CER are computed (metrics.py). `last.pt` is
written every evaluation, `best.pt` when the selected metric improves. Checkpoints hold
the three configs and the tokenizer, so `load_checkpoint(path)` rebuilds everything.
`--time-budget-hours` stops cleanly before a Kaggle/Colab session ends; `--resume` (default)
continues from `last.pt`.

Curves go to `<output_dir>/log.csv` (step, loss, lr, wer, cer, ...) and the final numbers
to `<output_dir>/results.json`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stt.config import (
    DECODER_METHODS,
    ENCODER_TYPES,
    MODEL_TYPES,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_from_dict,
    config_to_dict,
)
from stt.data import SpeechDataset, build_tokenizer, make_loader, read_manifest
from stt.decoding import decode_batch
from stt.metrics import error_report
from stt.models import build_model, count_parameters
from stt.tokenizer import tokenizer_from_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up to `lr`, then cosine decay to `min_lr` at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model, optimizer, step: int, best: float, data_cfg, model_cfg, train_cfg, tokenizer,
                    history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "step": step,
        "best": best, "data_config": config_to_dict(data_cfg), "model_config": config_to_dict(model_cfg),
        "train_config": config_to_dict(train_cfg), "tokenizer": tokenizer.to_dict(), "history": history,
    }, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Rebuild model + tokenizer + configs from a checkpoint. Returns a dict with those keys."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    data_cfg = config_from_dict(DataConfig, ckpt["data_config"])
    model_cfg = config_from_dict(ModelConfig, ckpt["model_config"])
    train_cfg = config_from_dict(TrainConfig, ckpt["train_config"])
    tokenizer = tokenizer_from_dict(ckpt["tokenizer"])
    model = build_model(model_cfg, data_cfg.n_mels, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return {"model": model, "tokenizer": tokenizer, "data_config": data_cfg, "model_config": model_cfg,
            "train_config": train_cfg, "step": ckpt["step"], "best": ckpt["best"], "history": ckpt.get("history", []),
            "optimizer": ckpt.get("optimizer")}


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, method: str = "greedy", beam: int = 4, max_batches: int | None = None,
             lm=None, lm_weight: float = 0.3) -> dict:
    """Loss + WER/CER over the loader; also returns a few example (ref, hyp) pairs."""
    model.eval()
    refs, hyps, losses = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        feats = batch["feats"].to(device)
        out = model(feats, batch["feat_lengths"].to(device), batch["targets"].to(device), batch["target_lengths"].to(device))
        losses.append(float(out["loss"]))
        ids = decode_batch(model, feats, batch["feat_lengths"].to(device), method, beam, lm, lm_weight)
        hyps += [tokenizer.decode(seq) for seq in ids]
        refs += batch["texts"]
    model.train()
    report = error_report(refs, hyps) if refs else {"wer": float("nan"), "cer": float("nan")}
    examples = list(zip(refs[:5], hyps[:5]))
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "wer": report["wer"], "cer": report["cer"],
            "examples": examples, "n": len(refs)}


def print_flush(*args) -> None:
    print(*args, flush=True)


def train(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, log=print_flush) -> dict:
    set_seed(train_cfg.seed)
    device = pick_device(train_cfg.device)
    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_utts = read_manifest(Path(data_cfg.data_dir) / data_cfg.train_manifest, data_cfg.lowercase)
    val_utts = read_manifest(Path(data_cfg.data_dir) / data_cfg.val_manifest, data_cfg.lowercase)
    last = out_dir / "last.pt"
    resume = train_cfg.resume and last.exists()
    if resume:
        state = load_checkpoint(last, device)
        tokenizer, model = state["tokenizer"], state["model"]
        step, best, history = state["step"], state["best"], state["history"]
        log(f"resumed from {last} at step {step}")
    else:
        tokenizer = build_tokenizer(data_cfg, train_utts)
        model = build_model(model_cfg, data_cfg.n_mels, tokenizer.vocab_size).to(device)
        step, best, history = 0, float("inf"), []
    model.train()
    log(f"model {model_cfg.model}/{model_cfg.encoder}: {count_parameters(model) / 1e6:.2f} M parameters, "
        f"vocab {tokenizer.vocab_size}, {len(train_utts)} train / {len(val_utts)} val utterances, device {device}")

    train_ds = SpeechDataset(data_cfg, train_utts, tokenizer, train=True, seed=train_cfg.seed)
    val_ds = SpeechDataset(data_cfg, val_utts, tokenizer, train=False)
    train_loader = make_loader(train_ds, train_cfg.batch_size, True, train_cfg.num_workers, train_cfg.seed)
    val_loader = make_loader(val_ds, train_cfg.batch_size, False, train_cfg.num_workers)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in name or "bias" in name else decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": train_cfg.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=train_cfg.lr, betas=(0.9, 0.98))
    if resume and state["optimizer"]:
        optimizer.load_state_dict(state["optimizer"])

    use_amp = device.type == "cuda" and train_cfg.precision != "fp32"
    amp_dtype = torch.bfloat16 if train_cfg.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and train_cfg.precision == "fp16")
    autocast = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()

    log_path = out_dir / "log.csv"
    fieldnames = ["step", "epoch", "loss", "lr", "grad_norm", "val_loss", "wer", "cer", "elapsed_s"]
    if not resume or not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write_log(row: dict) -> None:
        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({k: row.get(k, "") for k in fieldnames})

    def run_eval() -> dict:
        res = evaluate(model, val_loader, tokenizer, device, train_cfg.decode, train_cfg.beam_size, train_cfg.eval_batches)
        log(f"  val loss {res['loss']:.3f}  WER {100 * res['wer']:.1f}%  CER {100 * res['cer']:.1f}%  ({res['n']} utts)")
        for ref, hyp in res["examples"][:2]:
            log(f"    ref: {ref!r}\n    hyp: {hyp!r}")
        return res

    t0 = time.time()
    epoch = step // max(1, len(train_loader))
    running: dict[str, float] = {}
    n_running = 0
    stop = False
    budget_s = train_cfg.time_budget_hours * 3600 if train_cfg.time_budget_hours else None
    final: dict = {}
    while not stop:
        train_loader.batch_sampler.set_epoch(epoch)
        for batch in train_loader:
            lr = lr_at(step, train_cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr
            with autocast:
                out = model(batch["feats"].to(device), batch["feat_lengths"].to(device),
                            batch["targets"].to(device), batch["target_lengths"].to(device))
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip))
            if math.isfinite(grad_norm):
                scaler.step(optimizer)
            scaler.update()
            step += 1
            for k, v in out.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            n_running += 1

            if step % train_cfg.log_every == 0:
                parts = "  ".join(f"{k} {v / n_running:.3f}" for k, v in running.items())
                log(f"step {step}  {parts}  lr {lr:.2e}  gn {grad_norm:.1f}  {time.time() - t0:.0f}s")
                write_log({"step": step, "epoch": epoch, "loss": running["loss"] / n_running, "lr": lr,
                           "grad_norm": grad_norm, "elapsed_s": time.time() - t0})
                running, n_running = {}, 0

            out_of_time = budget_s is not None and time.time() - t0 > budget_s
            done = step >= train_cfg.max_steps or out_of_time
            if step % train_cfg.eval_every == 0 or done:
                res = run_eval()
                history.append({"step": step, **{k: res[k] for k in ("loss", "wer", "cer")}})
                write_log({"step": step, "epoch": epoch, "val_loss": res["loss"], "wer": res["wer"], "cer": res["cer"],
                           "elapsed_s": time.time() - t0})
                metric = res[train_cfg.select_by]
                if metric < best:
                    best = metric
                    save_checkpoint(out_dir / "best.pt", model, None, step, best, data_cfg, model_cfg, train_cfg,
                                    tokenizer, history)
                    log(f"  new best {train_cfg.select_by} {best:.4f} -> best.pt")
                save_checkpoint(last, model, optimizer, step, best, data_cfg, model_cfg, train_cfg, tokenizer, history)
                final = {"step": step, "val_loss": res["loss"], "wer": res["wer"], "cer": res["cer"],
                         f"best_{train_cfg.select_by}": best, "train_seconds": time.time() - t0,
                         "parameters": count_parameters(model), "model": model_cfg.model, "encoder": model_cfg.encoder}
                (out_dir / "results.json").write_text(json.dumps(final, indent=1))
            if done:
                stop = True
                break
        epoch += 1
        if train_cfg.max_epochs is not None and epoch >= train_cfg.max_epochs:
            stop = True
    if not final:  # max_steps reached before the first eval (should not happen) -> evaluate anyway
        res = run_eval()
        final = {"step": step, "wer": res["wer"], "cer": res["cer"], "val_loss": res["loss"]}
    log(f"done: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()})}")
    return final


def build_parser() -> argparse.ArgumentParser:
    d, m, t = DataConfig(), ModelConfig(), TrainConfig()
    p = argparse.ArgumentParser(description="Train a speech-to-text model",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = p.add_argument_group("data")
    data.add_argument("--data-dir", type=Path, default=d.data_dir, help="folder with train.jsonl and val.jsonl")
    data.add_argument("--train-manifest", default=d.train_manifest)
    data.add_argument("--val-manifest", default=d.val_manifest)
    data.add_argument("--sample-rate", type=int, default=d.sample_rate)
    data.add_argument("--n-mels", type=int, default=d.n_mels)
    data.add_argument("--tokenizer", choices=["char", "bpe"], default=d.tokenizer)
    data.add_argument("--bpe-model", type=Path, default=None)
    data.add_argument("--no-specaugment", action="store_true")
    data.add_argument("--max-duration-s", type=float, default=d.max_duration_s)
    model = p.add_argument_group("model")
    model.add_argument("--model", choices=MODEL_TYPES, default=m.model)
    model.add_argument("--encoder", choices=ENCODER_TYPES, default=m.encoder)
    model.add_argument("--d-model", type=int, default=m.d_model)
    model.add_argument("--n-heads", type=int, default=m.n_heads)
    model.add_argument("--n-layers", type=int, default=m.n_layers)
    model.add_argument("--ffn-mult", type=int, default=m.ffn_mult)
    model.add_argument("--conv-kernel", type=int, default=m.conv_kernel)
    model.add_argument("--subsampling-factor", type=int, choices=[2, 4], default=m.subsampling_factor)
    model.add_argument("--dropout", type=float, default=m.dropout)
    model.add_argument("--dec-layers", type=int, default=m.dec_layers)
    model.add_argument("--dec-dim", type=int, default=m.dec_dim)
    model.add_argument("--label-smoothing", type=float, default=m.label_smoothing)
    model.add_argument("--ctc-weight", type=float, default=m.ctc_weight, help="hybrid: weight of the CTC loss")
    model.add_argument("--pred-dim", type=int, default=m.pred_dim)
    model.add_argument("--joint-dim", type=int, default=m.joint_dim)
    tr = p.add_argument_group("training")
    tr.add_argument("--output-dir", type=Path, default=t.output_dir)
    tr.add_argument("--batch-size", type=int, default=t.batch_size)
    tr.add_argument("--max-steps", type=int, default=t.max_steps)
    tr.add_argument("--max-epochs", type=int, default=None)
    tr.add_argument("--lr", type=float, default=t.lr)
    tr.add_argument("--warmup-steps", type=int, default=t.warmup_steps)
    tr.add_argument("--weight-decay", type=float, default=t.weight_decay)
    tr.add_argument("--grad-clip", type=float, default=t.grad_clip)
    tr.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=t.precision)
    tr.add_argument("--num-workers", type=int, default=t.num_workers)
    tr.add_argument("--device", default=t.device)
    tr.add_argument("--seed", type=int, default=t.seed)
    tr.add_argument("--log-every", type=int, default=t.log_every)
    tr.add_argument("--eval-every", type=int, default=t.eval_every)
    tr.add_argument("--eval-batches", type=int, default=None)
    tr.add_argument("--decode", choices=DECODER_METHODS, default=t.decode, help="validation decoder")
    tr.add_argument("--beam-size", type=int, default=t.beam_size)
    tr.add_argument("--select-by", choices=["wer", "cer", "loss"], default=t.select_by)
    tr.add_argument("--time-budget-hours", type=float, default=None)
    tr.add_argument("--no-resume", action="store_true")
    return p


def configs_from_args(args: argparse.Namespace) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_cfg = DataConfig(
        data_dir=args.data_dir, train_manifest=args.train_manifest, val_manifest=args.val_manifest,
        sample_rate=args.sample_rate, n_mels=args.n_mels, tokenizer=args.tokenizer, bpe_model=args.bpe_model,
        specaugment=not args.no_specaugment, max_duration_s=args.max_duration_s,
    )
    model_cfg = ModelConfig(
        model=args.model, encoder=args.encoder, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ffn_mult=args.ffn_mult, conv_kernel=args.conv_kernel, subsampling_factor=args.subsampling_factor,
        dropout=args.dropout, dec_layers=args.dec_layers, dec_dim=args.dec_dim, label_smoothing=args.label_smoothing,
        ctc_weight=args.ctc_weight, pred_dim=args.pred_dim, joint_dim=args.joint_dim,
    )
    train_cfg = TrainConfig(
        output_dir=args.output_dir, batch_size=args.batch_size, max_steps=args.max_steps, max_epochs=args.max_epochs,
        lr=args.lr, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        precision=args.precision, num_workers=args.num_workers, device=args.device, seed=args.seed,
        log_every=args.log_every, eval_every=args.eval_every, eval_batches=args.eval_batches, decode=args.decode,
        beam_size=args.beam_size, select_by=args.select_by, time_budget_hours=args.time_budget_hours,
        resume=not args.no_resume,
    )
    return data_cfg, model_cfg, train_cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_cfg, model_cfg, train_cfg = configs_from_args(args)
    for cfg in (data_cfg, model_cfg, train_cfg):
        print(f"{type(cfg).__name__}: {dataclasses.asdict(cfg)}")
    train(data_cfg, model_cfg, train_cfg)


__all__ = [
    "build_parser",
    "configs_from_args",
    "evaluate",
    "load_checkpoint",
    "lr_at",
    "main",
    "pick_device",
    "save_checkpoint",
    "set_seed",
    "train",
]

if __name__ == "__main__":
    main()
