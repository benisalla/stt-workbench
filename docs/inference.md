# Inference and decoding

```bash
stt-transcribe checkpoints/ctc/best.pt clip.wav another.flac some_folder/
stt-transcribe checkpoints/ctc/best.pt clips/ --decode beam --beam-size 8 --lm data/lm.json --jsonl out.jsonl
```

Features are computed exactly as in training (minus SpecAugment); decoding depends on the
model type. All algorithms are in `decoding.py`; `decode_batch` picks the right one.

## CTC

- **Greedy**: argmax per frame, merge repeats, drop blanks. `O(T)`, almost always within a
  few % of beam search for CTC, because the distributions are peaky.
- **Prefix beam search** (Hannun 2014): keeps the best `beam` *prefixes*, each with two
  probabilities (ending in blank `p_b` and ending in a non-blank `p_nb`), so that all frame
  paths that collapse to the same text are summed. Test `test_prefix_beam_merges_paths_greedy_misses`
  shows the classic case where greedy is wrong and the beam is right.
- **With an n-gram LM** (`lm.py`): each extension by token `c` adds `lm_weight · log p_lm(c | prefix)`
  (+ `word_bonus` per token). Train one on the training transcripts:

  ```python
  from stt.lm import NGramLM
  lm = NGramLM(order=5, vocab_size=tokenizer.vocab_size).train([tokenizer.encode(t) for t in texts])
  lm.save("data/lm.json")
  ```

## Attention decoders (aed, las, hybrid)

- **Greedy**: feed `<sos>`, take the argmax, feed it back, stop at `<eos>` or `max_len`.
- **Beam search**: `beam` live prefixes ranked by summed log-probability; finished ones
  (that emitted `<eos>`) are set aside; the best finished hypothesis by length-normalised
  score wins (`score / (len + 1)^length_penalty`, so short outputs are not favoured).
- **Joint CTC/attention** (hybrid only): each extension's score is
  `(1 - λ) log p_att + λ (log p_ctc(prefix + c) - log p_ctc(prefix))`, where the CTC prefix
  probability (probability that the CTC output *starts with* the prefix) comes from
  `CTCPrefixScorer`, a forward recursion over frames kept per hypothesis. The test
  `test_prefix_scorer_sums_to_sequence_probability` checks it against brute force.
- An n-gram LM can be added to both beam searches (`--lm`).

## RNN-T

- **Greedy**: for every encoder frame, emit tokens while the joiner's argmax is not blank
  (at most `max_symbols` = 4 per frame), updating the prediction network after each token.
- **Modified beam search**: at each frame every hypothesis either emits one token or blank;
  identical token sequences are merged by log-sum-exp; the best `beam` survive. Simpler
  and nearly as good as the full Graves beam search.

## ONNX export (`stt-export`)

The encoder (and CTC head) export to a single graph: features `(1, T, 80)` + lengths →
encoder output and `ctc_log_probs`. Decode those with `decoding.ctc_greedy` or
`ctc_prefix_beam_search` on CPU. The autoregressive parts (attention decoder, prediction
network + joiner) are search procedures and stay in PyTorch. Needs `pip install 'stt-workbench[onnx]'`.
