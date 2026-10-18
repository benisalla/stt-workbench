# Data

## Manifest format

A dataset is a folder with `train.jsonl` and `val.jsonl`. Each line is one utterance:

```json
{"audio": "wavs/0001.wav", "text": "yes no yes", "duration": 2.1, "id": "0001"}
```

- `audio`: absolute, or relative to the manifest's folder; anything `soundfile` reads
  (wav, flac, ogg), mono or stereo, any sample rate (resampled on load to `--sample-rate`).
- `text`: the transcript; lower-cased and whitespace-collapsed on load (`--no-lowercase`
  is not offered; change `DataConfig.lowercase` in code if you need case).
- `duration` in seconds, used for duration bucketing and the `--max-duration-s` filter.

Write your own manifest with `stt.data.write_manifest(path, rows)` or any script.

## Built-in datasets (`stt-data`)

| name | what | size | command |
|---|---|---|---|
| `synthetic` | toy spoken language, generated locally | ~1 min of audio | `stt-data synthetic --n-train 400 --n-val 60` |
| `yesno` | 60 Hebrew "yes/no" recordings, 8 kHz (openslr 1) | 4.5 MB | `stt-data yesno` |
| `fsdd` | Free Spoken Digit Dataset, 3 000 clips, 6 speakers, 8 kHz | 15 MB | `stt-data fsdd [--clips-per-utterance 3]` |
| `librispeech` | openslr 12 subsets, with `*.trans.txt` | 337 MB – 6 GB | `stt-data librispeech --train-subset train-clean-100 --val-subset dev-clean` |
| `ljspeech` | LJ Speech 1.1, one speaker, 13 100 clips | 2.6 GB | `stt-data ljspeech` |
| `commonvoice` | a Common Voice folder (tsv + mp3), needs ffmpeg | varies | `stt-data commonvoice --root <folder>` |

Outputs go to `data/<name>/` by default (`--out`); archives are kept in `--download-dir`
(default: the same folder). `--root` points at an already-extracted LibriSpeech / LJSpeech /
Common Voice folder (for example a Kaggle input dataset) and skips the download.

### The synthetic language

`datasets.write_synthetic` makes a language with a 12-letter alphabet. Each letter has a
fixed acoustic pattern: "vowels" are a harmonic source filtered by two formants, "consonants"
a chirp with a noise burst. Words are random 2–5 letter strings, sentences 1–5 words with
200 ms pauses, played at a random speed and level with background noise, at 8 kHz. It has
no language structure at all, so a model can only do well by listening, and the CER tells
you directly whether the acoustic model and the decoder work. It takes seconds to write and
minutes to learn on a CPU; the test suite uses a 24-sentence version.

### FSDD and YESNO: real, small, low quality

Both are real recordings at 8 kHz. FSDD is split by speaker (speaker `yweweler` is held out
by default: the validation WER measures generalisation to a new voice). With
`--clips-per-utterance 3`, three digits of the same speaker are concatenated into one clip
so that the models have to *segment*, which is what distinguishes ASR from classification.
YESNO has only 60 clips (48 train / 12 val by a stable hash of the file name); it is a
check that the pipeline runs on real audio, not a benchmark.

## Tokenizers

Characters by default (`--tokenizer char`): the vocabulary is built from the training
transcripts and saved inside the checkpoint. For large datasets, train SentencePiece
pieces once:

```python
from stt.data import read_manifest
from stt.tokenizer import train_sentencepiece
texts = [u.text for u in read_manifest("data/librispeech/train.jsonl")]
train_sentencepiece(texts, "data/librispeech/bpe", vocab_size=500)
```

and pass `--tokenizer bpe --bpe-model data/librispeech/bpe.model`.

## Batching

`BucketSampler` sorts utterances by duration, cuts them into buckets of `bucket_size`
clips, shuffles inside each bucket and cuts batches from there, then shuffles the batch
order. Padding inside a batch stays small, which matters for CTC/RNN-T memory and for speed,
while the model still sees utterances in a random order.
