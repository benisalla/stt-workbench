"""Build every model and encoder, run one training step and one decoding pass on random data.

    python examples/05_model_zoo.py

No data or checkpoint needed. Shows the shared interface: `model(feats, feat_lengths, targets,
target_lengths)` returns a dict with "loss" (+ the loss terms), `model.encode` gives the encoder
output, and `decode_batch` runs the decoder that fits the model (greedy or beam).
"""

import torch

from stt.config import ModelConfig
from stt.decoding import decode_batch
from stt.models import build_model, count_parameters
from stt.tokenizer import CharTokenizer

torch.manual_seed(0)
tok = CharTokenizer(list("abcdefgh "))  # specials (blank, sos/eos, unk) are added by the tokenizer
N_MELS = 40

# A micro batch: 2 utterances of 120 and 90 frames, transcripts "abc de" and "hgf".
feats = torch.randn(2, 120, N_MELS)
feat_lengths = torch.tensor([120, 90])
targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(tok.encode(t)) for t in ("abc de", "hgf")], batch_first=True)
target_lengths = torch.tensor([6, 3])


def micro(model: str, encoder: str = "conformer") -> ModelConfig:
    return ModelConfig(model=model, encoder=encoder, d_model=32, n_heads=2, n_layers=2, ffn_mult=2, conv_kernel=7,
                       subsampling_channels=8, dec_dim=32, dec_layers=1, pred_dim=16, joint_dim=16, dropout=0.0)


print("model      params   loss terms                         greedy output (random weights)")
for name in ("ctc", "rnnt", "aed", "las", "hybrid"):
    try:
        model = build_model(micro(name), N_MELS, tok.vocab_size)
    except ImportError as e:  # rnnt needs torchaudio for the transducer loss
        print(f"{name:8s}   skipped: {e}")
        continue
    out = model(feats, feat_lengths, targets, target_lengths)
    out["loss"].backward()  # every model trains with the same call
    model.eval()
    with torch.no_grad():
        ids = decode_batch(model, feats, feat_lengths, method="greedy")
    terms = ", ".join(f"{k} {float(v.detach()):.2f}" for k, v in out.items())
    print(f"{name:8s} {count_parameters(model) / 1e3:6.0f} k  {terms:34s} {[tok.decode(seq)[:20] for seq in ids]}")

print("\nencoders (ctc head, 2 layers of width 32): parameters and output frames for 120 input frames")
for encoder in ("bilstm", "transformer", "conformer"):
    model = build_model(micro("ctc", encoder), N_MELS, tok.vocab_size)
    enc, enc_lengths = model.encode(feats, feat_lengths)
    print(f"  {encoder:12s} {count_parameters(model) / 1e3:6.0f} k   {tuple(enc.shape)}  lengths {enc_lengths.tolist()}"
          f"  (subsampling x{micro('ctc').subsampling_factor})")
