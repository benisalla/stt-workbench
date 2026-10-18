from stt.lm import NGramLM
from stt.tokenizer import EOS


def test_ngram_probabilities_sum_to_one_and_prefer_seen():
    lm = NGramLM(order=3, vocab_size=8).train([[4, 5, 6], [4, 5, 7], [4, 5, 6]])
    total = sum(lm.prob([4, 5], w) for w in range(8))
    assert abs(total - 1.0) < 1e-6
    assert lm.prob([4, 5], 6) > lm.prob([4, 5], 7) > lm.prob([4, 5], 3)
    assert lm.prob([6], EOS) > lm.prob([6], 4)


def test_backoff_on_unseen_history():
    lm = NGramLM(order=3, vocab_size=8).train([[4, 5, 6]])
    assert lm.prob([7, 7], 6) > 0
    assert sum(lm.prob([7, 7], w) for w in range(8)) - 1.0 < 1e-6


def test_perplexity_and_save_load(tmp_path):
    lm = NGramLM(order=2, vocab_size=8).train([[4, 5, 6]] * 5)
    ppl = lm.perplexity([[4, 5, 6]])
    assert 1.0 <= ppl < 2.0
    lm.save(tmp_path / "lm.json")
    again = NGramLM.load(tmp_path / "lm.json")
    assert abs(again.perplexity([[4, 5, 6]]) - ppl) < 1e-9
