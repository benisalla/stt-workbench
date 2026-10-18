# Testing

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev,rnnt]"
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q      # ~10 s, CPU only
.venv/bin/ruff check src tests
```

The suite (`tests/`, 80 tests) uses micro models (d_model 32, 2 layers) and a 24-sentence
synthetic dataset written once per session (`conftest.py`), so it runs anywhere.

| file | checks |
|---|---|
| `test_config.py` | feature geometry from the sample rate, validation errors, dict round trip |
| `test_features.py` | mel scale inversion, filterbank coverage, a 300 Hz tone lands below a 3 kHz tone, CMVN, SpecAugment, resampling |
| `test_tokenizer.py` | special ids, round trip, unknown chars, SentencePiece (skipped without the extra) |
| `test_encoders.py` | subsampling lengths, all three encoders, **padding invariance** (valid frames do not depend on padding), causal mask, relative bias |
| `test_models.py` | forward/backward for all 5 models × 3 encoders, hybrid loss weighting, CTC prefers the matching path, **one-batch overfit** |
| `test_decoding.py` | greedy collapse, prefix beam = greedy on peaky input, prefix beam beats greedy on the classic merge case, **CTC prefix scorer vs brute force**, LM steering, every decoder on every model |
| `test_lm.py` | probabilities sum to one, seen > unseen, back-off, perplexity, save/load |
| `test_metrics.py` | S/D/I counts, corpus-level WER, alignment ops, the `stt-eval` CLI |
| `test_data.py` | manifest round trip, deterministic synthetic audio, collate padding, bucket sampler covers every index once |
| `test_train.py` | LR schedule, **train → eval → checkpoint → resume → transcribe** for ctc / hybrid / rnnt |
| `test_pretrained.py` | HF vocab mapping, the wav2vec2 wrapper on a stub model (no download) |

Two tests are skipped on purpose: real Hugging Face downloads, and SentencePiece when
not installed.

Beyond unit tests, a model repo must show that it learns: `docs/evaluation.md` records
short real trainings on the synthetic language and on FSDD with their WER / CER.
