import pytest

from stt.tokenizer import BLANK, EOS, SOS, SPECIALS, UNK, CharTokenizer, normalize_text, tokenizer_from_dict


def test_specials_are_first():
    assert (BLANK, SOS, EOS, UNK) == (0, 1, 2, 3)
    tok = CharTokenizer.from_texts(["ab ba"])
    assert tok.itos[: len(SPECIALS)] == SPECIALS
    assert tok.vocab_size == len(SPECIALS) + 3


def test_encode_decode_round_trip_and_unknown():
    tok = CharTokenizer.from_texts(["hello world"])
    assert tok.decode(tok.encode("hello world")) == "hello world"
    assert tok.encode("z")[0] == UNK
    assert tok.decode([SOS, EOS, BLANK]) == ""


def test_normalize_text():
    assert normalize_text("  Hello   World ") == "hello world"
    assert normalize_text("A  b", lowercase=False) == "A b"


def test_from_dict():
    tok = CharTokenizer.from_texts(["abc"])
    again = tokenizer_from_dict(tok.to_dict())
    assert again.itos == tok.itos
    with pytest.raises(ValueError):
        tokenizer_from_dict({"type": "other"})


def test_sentencepiece_optional():
    pytest.importorskip("sentencepiece")
    import tempfile
    from pathlib import Path

    from stt.tokenizer import SentencePieceTokenizer, train_sentencepiece

    texts = ["the cat sat on the mat", "the dog ate the cat", "a mat on a cat"] * 10
    with tempfile.TemporaryDirectory() as d:
        model = train_sentencepiece(texts, Path(d) / "bpe", vocab_size=24)
        tok = SentencePieceTokenizer(model)
        assert tok.vocab_size == 24 + len(SPECIALS)
        assert tok.decode(tok.encode("the cat")) == "the cat"
        assert min(tok.encode("the cat")) >= len(SPECIALS)
