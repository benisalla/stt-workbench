"""`stt-export checkpoint.pt model.onnx`: the encoder (+ CTC head when there is one) as ONNX.

Exported graph: log-mel features (1, T, n_mels) + lengths (1,) -> encoder output (1, T', d)
and, for ctc/hybrid, CTC log-probs (1, T', V). The autoregressive decoders (attention, RNN-T
prediction network) are search procedures, not single graphs, and stay in PyTorch; the
exported CTC log-probs can be decoded with `decoding.ctc_greedy` on any numpy array.
Needs the `onnx` extra. The BiLSTM encoder packs sequences and is exported with fixed lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stt.train import load_checkpoint


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        enc, enc_lengths = self.model.encode(feats, lengths)
        if getattr(self.model, "ctc_head", None) is not None:
            return enc, enc_lengths, self.model.ctc_log_probs(enc)
        return enc, enc_lengths


def export_onnx(ckpt: dict, out: Path, n_frames: int = 200, opset: int = 17) -> Path:
    model = ckpt["model"].cpu().eval()
    wrapper = EncoderWrapper(model)
    feats = torch.randn(1, n_frames, ckpt["data_config"].n_mels)
    lengths = torch.tensor([n_frames])
    has_ctc = getattr(model, "ctc_head", None) is not None
    outputs = ["encoder_out", "encoder_lengths"] + (["ctc_log_probs"] if has_ctc else [])
    dynamic = {"feats": {1: "frames"}, "encoder_out": {1: "enc_frames"}}
    if has_ctc:
        dynamic["ctc_log_probs"] = {1: "enc_frames"}
    torch.onnx.export(wrapper, (feats, lengths), str(out), input_names=["feats", "lengths"], output_names=outputs,
                      dynamic_axes=dynamic, opset_version=opset, dynamo=False)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export the encoder (+ CTC head) to ONNX")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args(argv)
    export_onnx(load_checkpoint(args.checkpoint), args.out, opset=args.opset)
    print(f"wrote {args.out}")


__all__ = ["EncoderWrapper", "export_onnx", "main"]

if __name__ == "__main__":
    main()
