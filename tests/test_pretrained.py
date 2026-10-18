import pytest
import torch

from stt.pretrained import Wav2Vec2Fine, char_vocab_for_hf
from stt.tokenizer import CharTokenizer


def test_char_vocab_for_hf():
    tok = CharTokenizer.from_texts(["a b"])
    vocab = char_vocab_for_hf(tok)
    assert vocab["<pad>"] == 0 and vocab["|"] == tok.stoi[" "] and vocab["a"] == tok.stoi["a"]


class StubHF(torch.nn.Module):
    """Stands in for Wav2Vec2ForCTC: waveform -> logits every 320 samples, loss = CTC sum."""

    def __init__(self, vocab):
        super().__init__()
        self.lin = torch.nn.Linear(1, vocab)

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths // 320

    def forward(self, input_values, attention_mask=None, labels=None):
        frames = input_values[:, : input_values.size(1) // 320 * 320].reshape(input_values.size(0), -1, 320).mean(-1, keepdim=True)
        logits = self.lin(frames)
        out = type("Out", (), {})()
        out.logits = logits
        if labels is not None:
            lp = torch.log_softmax(logits, -1).transpose(0, 1)
            tl = (labels != -100).sum(1)
            out.loss = torch.nn.functional.ctc_loss(lp, labels.clamp(min=0), torch.full((logits.size(0),), logits.size(1)), tl, reduction="sum", zero_infinity=True)
        return out


def test_wav2vec2_wrapper_with_stub():
    tok = CharTokenizer.from_texts(["ab"])
    m = Wav2Vec2Fine(tok, model=StubHF(tok.vocab_size), freeze_feature_encoder=False)
    waves = torch.randn(2, 3200)
    out = m(waves, torch.tensor([3200, 1600]), torch.tensor([[4, 5], [4, 0]]), torch.tensor([2, 1]))
    assert torch.isfinite(out["loss"])
    texts = m.transcribe(waves, torch.tensor([3200, 1600]))
    assert len(texts) == 2


def test_real_hf_models_optional():
    pytest.importorskip("transformers")
    pytest.skip("downloads a pretrained model; run in the notebook")
