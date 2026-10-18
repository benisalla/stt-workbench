import pytest
import torch

from stt.models import add_sos_eos, build_model, count_parameters
from stt.tokenizer import BLANK, EOS, SOS
from tests.conftest import micro_model_cfg

VOCAB = 12


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 64, 40, generator=g)
    y = torch.randint(4, VOCAB, (2, 6), generator=g)
    return x, torch.tensor([64, 40]), y, torch.tensor([6, 3])


def test_add_sos_eos():
    y = torch.tensor([[5, 6, 7], [8, 0, 0]])
    inp, out = add_sos_eos(y, torch.tensor([3, 1]))
    assert inp.tolist() == [[SOS, 5, 6, 7], [SOS, 8, 0, 0]]
    assert out.tolist() == [[5, 6, 7, EOS], [8, EOS, BLANK, BLANK]]


@pytest.mark.parametrize("model", ["ctc", "aed", "las", "hybrid", "rnnt"])
@pytest.mark.parametrize("encoder", ["bilstm", "transformer", "conformer"])
def test_forward_backward_every_combination(model, encoder):
    if model == "rnnt":
        pytest.importorskip("torchaudio")
    torch.manual_seed(0)
    m = build_model(micro_model_cfg(model, encoder), 40, VOCAB)
    out = m(*batch())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert count_parameters(m) < 200_000


def test_hybrid_loss_is_weighted_sum():
    torch.manual_seed(0)
    m = build_model(micro_model_cfg("hybrid"), 40, VOCAB).eval()
    out = m(*batch())
    expected = 0.3 * out["ctc"] + 0.7 * out["att"]
    assert torch.allclose(out["loss"], expected, atol=1e-5)


def test_ctc_loss_is_lower_for_matching_frames():
    """A CTC model should prefer a spiky distribution matching the target over a random one."""
    from stt.models import ctc_loss

    lp = torch.full((1, 10, 5), -5.0)
    path = [0, 4, 0, 3, 3, 0, 4, 0, 0, 0]
    for t, c in enumerate(path):
        lp[0, t, c] = 0.0
    lp = torch.log_softmax(lp, -1)
    good = ctc_loss(lp, torch.tensor([10]), torch.tensor([[4, 3, 4]]), torch.tensor([3]))
    bad = ctc_loss(lp, torch.tensor([10]), torch.tensor([[3, 4, 3]]), torch.tensor([3]))
    assert good < bad


def test_overfit_one_batch_ctc():
    """Sanity check of the learning signal: the loss on one batch must fall quickly."""
    torch.manual_seed(0)
    m = build_model(micro_model_cfg("ctc", "transformer"), 40, VOCAB)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    x, xl, y, yl = batch()
    first = None
    for _ in range(40):
        loss = m(x, xl, y, yl)["loss"]
        first = first or float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first
