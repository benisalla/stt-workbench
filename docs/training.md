# Training

```bash
stt-data synthetic                                        # writes data/synthetic
stt-train --data-dir data/synthetic --sample-rate 8000 --model ctc --encoder conformer \
          --max-steps 1500 --output-dir checkpoints/ctc
```

`stt-train --help` lists every flag; the defaults are the `config.py` dataclasses.

## What a run does

1. reads the manifests, builds the tokenizer from the training transcripts;
2. builds the model (`--model`, `--encoder`, sizes) and an AdamW optimizer (β = 0.9/0.98,
   no weight decay on biases and norms);
3. each step: a duration-bucketed batch → log-mel + CMVN + SpecAugment → the model's own
   loss → gradient clipping (`--grad-clip 5`) → optimizer step with the learning rate from
   a linear warm-up (`--warmup-steps`) then cosine decay to `min_lr` at `--max-steps`;
4. every `--eval-every` steps and at the end: the validation set is decoded
   (`--decode greedy|beam`), WER / CER are computed, `last.pt` is written, and `best.pt` when
   the `--select-by` metric (wer by default) improves;
5. `log.csv` collects loss / lr / grad norm / val metrics, `results.json` the final numbers.

Everything the model needs to run again is in the checkpoint: the three configs, the
tokenizer, the weights (and the optimizer state in `last.pt`). `--no-resume` starts over.

## Mixed precision

`--precision bf16` (Ampere+ GPUs) or `fp16` (T4/P100, with a loss scaler) autocasts the
model; the CTC and transducer losses are computed in fp32 regardless, as their kernels need.

## Session limits (Kaggle / Colab)

`--time-budget-hours 11` evaluates, saves and stops before the session ends; running the
same command again resumes from `last.pt`. The notebook sets this for you.

## Choosing sizes

The defaults (`d_model 144`, 6 layers, conformer) are 3 M parameters: right for the toy
data and FSDD on a CPU, and a sensible start for LibriSpeech dev-clean on a free GPU. For
train-clean-100 or more, `--d-model 256 --n-layers 12 --batch-size 32` (about 20 M) is a
small production model. For a CPU experiment, `--d-model 96 --n-layers 4` trains at a few
steps per second.

Learning-rate notes, from the runs in `docs/evaluation.md`:
- CTC and RNN-T like `--lr 1e-3` with a short warm-up;
- attention decoders are happier with a longer warm-up (`--warmup-steps 500+`) on real data;
  the hybrid's CTC term stabilises them a lot;
- if the CTC loss plateaus while emitting only blanks for the first few hundred steps,
  that is normal: the blank is the safe answer until the encoder finds the first phonemes.

## Memory

RNN-T's joiner output is `(B, T', U+1, V)`: with characters (V ≈ 30) and 4× subsampling it
is small, but with BPE (V = 500+) and long utterances reduce `--batch-size` or use
`--subsampling-factor 4` (default). SpecAugment and bucketing cost nothing.

## Fine-tuning pretrained models

`pretrained.py` holds the wav2vec2 and Whisper wrappers; their training loop is in the
notebook (sections 6–7), because the models are large downloads and need a GPU.
