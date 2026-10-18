# Examples

Short scripts that use the `stt` package from Python, in the order one would use it. All run
on a CPU in minutes; run them from the repository root after `pip install -e ".[dev]"`.

| script | what it shows | needs |
|---|---|---|
| [01_train_on_toy_data.py](01_train_on_toy_data.py) | the three config dataclasses and `train()`: a 2-layer Conformer + CTC on 200 synthetic sentences, about a minute | nothing (writes `data/examples/synthetic`) |
| [02_transcribe_and_score.py](02_transcribe_and_score.py) | `load_checkpoint`, `transcribe_files` (greedy / beam), WER / CER and aligned worst utterances | 01 |
| [03_beam_search_with_lm.py](03_beam_search_with_lm.py) | a character 5-gram LM on the training transcripts, what it predicts, and greedy vs beam vs beam + LM | 01 |
| [04_features_and_specaugment.py](04_features_and_specaugment.py) | the front end step by step: log-mel, CMVN, SpecAugment, drawn to `examples/output/features.png` | 01 or any wav |
| [05_model_zoo.py](05_model_zoo.py) | all five models (ctc, rnnt, aed, las, hybrid) and three encoders on a random batch: one loss, one backward, one decode | nothing |

```bash
python examples/01_train_on_toy_data.py
python examples/02_transcribe_and_score.py
python examples/03_beam_search_with_lm.py
python examples/04_features_and_specaugment.py
python examples/05_model_zoo.py
```

The same steps as CLI commands are in the README's quick start (`stt-data`, `stt-train`,
`stt-transcribe`, `stt-eval`); the Kaggle/Colab notebook does them on real data with a GPU.
