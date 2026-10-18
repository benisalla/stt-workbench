"""Build notebooks/train_colab_kaggle.ipynb: a self-contained training notebook.

The code of src/stt/*.py is copied into the notebook at build time (imports between those
modules are removed, as everything lives in one namespace), so the notebook is identical to
the package and needs no GitHub access. Regenerate after changing src/stt:

    python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
"""

import ast
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PACKAGE = "stt"


def module_source(module: str) -> str:
    """A module's code without `from stt...` imports, __all__, __main__ and the docstring."""
    text = (ROOT / "src" / PACKAGE / f"{module}.py").read_text()
    lines = text.splitlines()
    drop: set[int] = set()
    tree = ast.parse(text)
    for node in tree.body:
        is_local_import = isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PACKAGE)
        is_all = isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__all__"
        is_main = isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
        is_docstring = node is tree.body[0] and isinstance(node, ast.Expr)
        if is_local_import or is_all or is_main or is_docstring:
            drop.update(range(node.lineno - 1, node.end_lineno))
    kept = [line for i, line in enumerate(lines) if i not in drop]
    return "\n".join(kept).strip() + "\n"


def module_doc(module: str) -> str:
    text = (ROOT / "src" / PACKAGE / f"{module}.py").read_text()
    return ast.get_docstring(ast.parse(text)) or ""


cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def module_cells(module: str, title: str) -> None:
    md(f"### {title} (same as `src/stt/{module}.py`)\n\n{module_doc(module)}")
    code(module_source(module))


# ---------------------------------------------------------------------------
md("""
# stt-workbench: training notebook (Kaggle & Colab)

Train a **speech-to-text** model on a free Kaggle or Colab GPU, with any of the approaches
of the `stt-workbench` repo:

| `MODEL` | what it is |
|---|---|
| `ctc` | encoder + CTC head: the fastest baseline |
| `aed` | attention encoder-decoder (Transformer decoder) |
| `las` | Listen-Attend-Spell: LSTM decoder with location-aware attention |
| `hybrid` | CTC + attention, joint loss and joint decoding (ESPnet default) |
| `rnnt` | RNN-Transducer: streaming-friendly |
| `wav2vec2` | fine-tune a pretrained wav2vec2 with a CTC head (Hugging Face) |
| `whisper` | fine-tune a pretrained Whisper (Hugging Face) |

with `ENCODER` = `bilstm`, `transformer` or `conformer` for the first five.

This notebook is **self-contained**: all the code is copied in from the repo, with the same
defaults, and its checkpoints work with the repo's `stt-transcribe`, `stt-eval` and
`stt-train --resume`.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Data | your dataset as input, or `DOWNLOAD_DATASET` | Google Drive, upload, or `DOWNLOAD_DATASET` |
| Session limit | 12 h per session, 30 h GPU / week | ~12 h, can disconnect when idle |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

**Just trying it?** Set `SMOKE_TEST = True`: a small synthetic dataset is written and every
cell runs in a few minutes on CPU or GPU.

**Sections:** 1 Install · 2 Config · 3 Data · 4 Model · 5 Decoding & metrics · 6 Training code
· 7 Train (live curves) · 8 Pretrained fine-tuning (wav2vec2 / Whisper) · 9 Results · 10 Download
""")

md("""
## 1. Install

Kaggle and Colab already have PyTorch, torchaudio (RNN-T loss), NumPy, SciPy, matplotlib
and `transformers`. This cell adds `soundfile` and `sentencepiece`.
""")
code("""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile>=0.12", "sentencepiece>=0.2"])

import torch

print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

md(f"""
## 2. Config

### 2.1 Config classes (same as `src/stt/config.py`)

{module_doc("config")}
""")
code(module_source("config"))

md("""
### 2.2 Settings for this run

**Data.** A dataset folder holds `train.jsonl` and `val.jsonl` manifests (one JSON object
per line: `audio`, `text`, `duration`; see the repo's `docs/data.md`). Either:

- **Kaggle:** add a dataset as input: a folder with the two manifests, or a LibriSpeech /
  LJSpeech folder (manifests are written for it); the cell below finds it;
- **Colab:** put it on Google Drive (`USE_GOOGLE_DRIVE = True`) and set `DATA_DIR`;
- **Either:** set `DOWNLOAD_DATASET` to `"yesno"` (4.5 MB), `"fsdd"` (15 MB),
  `"librispeech-dev-clean"` (337 MB) or `"librispeech-train-clean-100"` (6 GB + dev-clean).

The batch size and precision adapt to the GPU without changing what is learned.
""")
code("""
import dataclasses
import os
from pathlib import Path

MODEL = "hybrid"            # ctc | aed | las | hybrid | rnnt | wav2vec2 | whisper
ENCODER = "conformer"       # bilstm | transformer | conformer   (not used by wav2vec2 / whisper)
SMOKE_TEST = False          # True = quick check of every cell (synthetic data, small model)
DOWNLOAD_DATASET = None     # None | "yesno" | "fsdd" | "librispeech-dev-clean" | "librispeech-train-clean-100"
USE_GOOGLE_DRIVE = False    # Colab only: data + checkpoints on Drive (survive disconnects)
DATA_DIR = None             # None = find it automatically (Kaggle input, /content/data, ./data)
SAMPLE_RATE = 16000         # 8000 for yesno / fsdd / the synthetic data (set automatically for those)
TOKENIZER = "char"          # "char" | "bpe" (a SentencePiece model is trained in section 3)
BPE_VOCAB = 300
RESUME = True               # continue from last.pt in OUTPUT_DIR if it exists
TIME_BUDGET_HOURS = 11.0    # stop cleanly (after saving) before the 12 h session limit
MAX_STEPS = 20000
PRETRAINED_NAME = {"wav2vec2": "facebook/wav2vec2-base", "whisper": "openai/whisper-base"}.get(MODEL)

ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path(f"/content/drive/MyDrive/stt-workbench/{MODEL}-{ENCODER}")
elif ON_KAGGLE:
    OUTPUT_DIR = Path(f"/kaggle/working/checkpoints/{MODEL}-{ENCODER}")
elif ON_COLAB:
    OUTPUT_DIR = Path(f"/content/checkpoints/{MODEL}-{ENCODER}")
else:
    OUTPUT_DIR = Path(f"checkpoints/notebook-{MODEL}-{ENCODER}")
SCRATCH = Path("/tmp/stt_data")  # downloads (outside /kaggle/working: not counted in the 20 GB output)


def find_data_dir() -> Path | None:
    \"\"\"First folder with train.jsonl + val.jsonl, or a LibriSpeech / LJSpeech folder, in the usual places.\"\"\"
    roots = [Path("/kaggle/input"), Path("/content/data"), Path("/content/drive/MyDrive"), Path("data")]
    for root in roots:
        if not root.exists():
            continue
        candidates = [root, *sorted(p for p in root.rglob("*") if p.is_dir())][:2000]
        for path in candidates:
            if (path / "train.jsonl").exists() and (path / "val.jsonl").exists():
                return path
        for path in candidates:
            if path.name == "LibriSpeech" or (path / "metadata.csv").exists():
                return path
    return None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    gpu_gb = props.total_memory / 1e9
    precision = "bf16" if props.major >= 8 else "fp16"
    batch_size = 64 if gpu_gb >= 30 else 32
    print(f"GPU: {props.name} ({gpu_gb:.0f} GB), compute capability {props.major}.{props.minor}")
else:
    precision, batch_size = "fp32", 8
    print("No GPU: enable one in the notebook settings (CPU is fine only for SMOKE_TEST).")

model_config = ModelConfig(model=MODEL if MODEL in MODEL_TYPES else "ctc", encoder=ENCODER)
train_config = TrainConfig(
    output_dir=OUTPUT_DIR, batch_size=batch_size, precision=precision, max_steps=MAX_STEPS, warmup_steps=1000,
    num_workers=min(4, os.cpu_count() or 1), device=device.type, log_every=50, eval_every=1000,
    time_budget_hours=TIME_BUDGET_HOURS, resume=RESUME, decode="greedy",
)
if SMOKE_TEST:
    OUTPUT_DIR = OUTPUT_DIR.parent / f"smoke-{MODEL}-{ENCODER}"
    model_config = dataclasses.replace(model_config, d_model=64, n_layers=2, dec_dim=64, dec_layers=1, ffn_mult=2)
    train_config = dataclasses.replace(
        train_config, output_dir=OUTPUT_DIR, max_steps=200, warmup_steps=20, batch_size=8, log_every=10,
        eval_every=100, num_workers=0, time_budget_hours=0.5,
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("model:", MODEL, "| encoder:", ENCODER, "| output:", OUTPUT_DIR)
""")

md("## 3. Data")
module_cells("tokenizer", "3.1 Tokenizer code")
module_cells("features", "3.2 Feature code")
module_cells("data", "3.3 Manifest / dataset code")
module_cells("datasets", "3.4 Built-in dataset preparation code")
md("""
### 3.5 Prepare the data

Writes the manifests (and downloads if asked), then builds the `DataConfig` and shows a few
utterances and one log-mel spectrogram.
""")
code("""
import json

if SMOKE_TEST:
    DATA_DIR = write_synthetic(Path("synthetic_data"), n_train=200, n_val=40, seed=0)
    SAMPLE_RATE = SYNTH_SR
elif DOWNLOAD_DATASET == "yesno":
    DATA_DIR, SAMPLE_RATE = prepare_yesno(SCRATCH / "yesno"), 8000
elif DOWNLOAD_DATASET == "fsdd":
    DATA_DIR, SAMPLE_RATE = prepare_fsdd(SCRATCH / "fsdd", clips_per_utterance=3), 8000
elif DOWNLOAD_DATASET and DOWNLOAD_DATASET.startswith("librispeech-"):
    subset = DOWNLOAD_DATASET.split("-", 1)[1]
    DATA_DIR = prepare_librispeech(SCRATCH / "librispeech", train_subset=subset, val_subset="dev-clean",
                                   download_dir=SCRATCH)
elif DATA_DIR is None:
    DATA_DIR = find_data_dir()
    if DATA_DIR is None:
        raise FileNotFoundError("No data found: add a dataset as input, set DATA_DIR, DOWNLOAD_DATASET, or SMOKE_TEST = True.")
DATA_DIR = Path(DATA_DIR)
if not (DATA_DIR / "train.jsonl").exists():  # a raw LibriSpeech / LJSpeech folder: write manifests next to the outputs
    manifests = OUTPUT_DIR.parent / f"manifests-{DATA_DIR.name}"
    if DATA_DIR.name == "LibriSpeech" or any(DATA_DIR.glob("*/*.trans.txt")) or any(DATA_DIR.rglob("*.trans.txt")):
        subsets = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
        train_subset = next((s for s in subsets if s.startswith("train")), subsets[0])
        val_subset = next((s for s in subsets if s.startswith("dev")), train_subset)
        DATA_DIR = prepare_librispeech(manifests, train_subset, val_subset, root=DATA_DIR)
    elif (DATA_DIR / "metadata.csv").exists():
        DATA_DIR = prepare_ljspeech(manifests, root=DATA_DIR)
print("data:", DATA_DIR)

train_utts = read_manifest(DATA_DIR / "train.jsonl")
val_utts = read_manifest(DATA_DIR / "val.jsonl")
print(f"{len(train_utts)} train / {len(val_utts)} val utterances, "
      f"{sum(u.duration for u in train_utts) / 3600:.2f} h of training audio")
for u in train_utts[:3]:
    print(f"  {u.duration:5.2f}s  {u.text!r}")

bpe_model = None
if TOKENIZER == "bpe":
    bpe_model = train_sentencepiece([u.text for u in train_utts], OUTPUT_DIR / "bpe", vocab_size=BPE_VOCAB)
data_config = DataConfig(data_dir=DATA_DIR, sample_rate=SAMPLE_RATE, tokenizer=TOKENIZER, bpe_model=bpe_model)
tokenizer = build_tokenizer(data_config, train_utts)
print("vocabulary:", tokenizer.vocab_size, "tokens")

import matplotlib.pyplot as plt

feats = cmvn(LogMel(data_config)(torch.as_tensor(load_audio(train_utts[0].audio, SAMPLE_RATE))))
plt.figure(figsize=(10, 3))
plt.imshow(feats.T, aspect="auto", origin="lower")
plt.title(train_utts[0].text)
plt.xlabel("frame (10 ms)")
plt.ylabel("mel bin")
plt.show()
""")

md("## 4. Model")
module_cells("encoders", "4.1 Encoders")
module_cells("models", "4.2 The five recognisers")

md("## 5. Decoding, language model and metrics")
module_cells("lm", "5.1 n-gram language model")
module_cells("decoding", "5.2 Search algorithms")
module_cells("metrics", "5.3 WER / CER")

md("## 6. Training code")
module_cells("train", "6.1 Trainer (same as `src/stt/train.py`)")

md("""
## 7. Train

Runs the trainer with the settings of section 2 and draws the curves (`log.csv`) after
every evaluation. Re-running this cell after a disconnect resumes from `last.pt`.
`MODEL = "wav2vec2"` / `"whisper"` skip this section: see section 8.
""")
code("""
import csv
import time

from IPython.display import clear_output


def read_log(path):
    rows = list(csv.DictReader(open(path))) if Path(path).exists() else []
    train = [(int(r["step"]), float(r["loss"])) for r in rows if r["loss"]]
    val = [(int(r["step"]), float(r["wer"]), float(r["cer"]), float(r["val_loss"])) for r in rows if r["wer"]]
    return train, val


def plot_curves(path):
    train, val = read_log(path)
    if not train:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(*zip(*train), label="train loss")
    if val:
        axes[0].plot([v[0] for v in val], [v[3] for v in val], "o-", label="val loss")
        axes[1].plot([v[0] for v in val], [100 * v[1] for v in val], "o-", label="WER %")
        axes[1].plot([v[0] for v in val], [100 * v[2] for v in val], "s-", label="CER %")
        axes[1].set_ylim(0, min(110, max(5, 100 * max(max(v[1], v[2]) for v in val) * 1.1)))
    for ax in axes:
        ax.legend()
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


lines = []


def notebook_log(*args):
    text = " ".join(str(a) for a in args)
    lines.append(text)
    if text.startswith("  val loss"):
        clear_output(wait=True)
        plot_curves(OUTPUT_DIR / "log.csv")
        print("\\n".join(lines[-12:]))
    else:
        print(text)


if MODEL in MODEL_TYPES:
    t0 = time.time()
    results = train(data_config, model_config, train_config, log=notebook_log)
    print(f"finished in {(time.time() - t0) / 60:.1f} min:", results)
else:
    print("pretrained model selected: go to section 8")
""")

md("## 8. Pretrained fine-tuning (wav2vec2 / Whisper)")
module_cells("pretrained", "8.1 Wrappers")
md("""
### 8.2 Fine-tune

Only runs when `MODEL` is `wav2vec2` or `whisper`. The data is loaded as 16 kHz waveforms
(wav2vec2) or Whisper log-mels, in duration-bucketed batches; the optimiser is AdamW with a
small learning rate and warm-up; validation WER is computed with greedy decoding every
`FT_EVAL_EVERY` steps and the best weights are saved to `OUTPUT_DIR`.
""")
code("""
FT_STEPS = 200 if SMOKE_TEST else 3000
FT_EVAL_EVERY = 100 if SMOKE_TEST else 500
FT_LR = 3e-5 if MODEL == "wav2vec2" else 1e-5
FT_BATCH = 4 if device.type == "cpu" else 8


def waveform_batches(utts, batch_size, shuffle, seed=0):
    sampler = BucketSampler([u.duration for u in utts], batch_size, 64, shuffle, seed)
    for idx in sampler:
        waves = [torch.as_tensor(load_audio(utts[i].audio, 16000)) for i in idx]
        lengths = torch.tensor([w.numel() for w in waves])
        padded = torch.zeros(len(waves), int(lengths.max()))
        for b, w in enumerate(waves):
            padded[b, : w.numel()] = w
        yield padded, lengths, [utts[i].text for i in idx]


if MODEL in ("wav2vec2", "whisper"):
    if MODEL == "wav2vec2":
        ft = Wav2Vec2Fine(tokenizer, PRETRAINED_NAME).to(device)
    else:
        ft = WhisperFine(PRETRAINED_NAME).to(device)
    optimizer = torch.optim.AdamW([p for p in ft.parameters() if p.requires_grad], lr=FT_LR, weight_decay=0.01)
    step, best_wer, t0, ft_log = 0, float("inf"), time.time(), []

    def ft_evaluate(max_batches=None):
        ft.eval()
        refs, hyps = [], []
        for i, (waves, lengths, texts) in enumerate(waveform_batches(val_utts, FT_BATCH, False)):
            if max_batches and i >= max_batches:
                break
            if MODEL == "wav2vec2":
                hyps += ft.transcribe(waves.to(device), lengths.to(device))
            else:
                hyps += ft.transcribe(ft.features([w[:n].numpy() for w, n in zip(waves, lengths)]).to(device))
            refs += texts
        ft.train()
        return error_report(refs, hyps), list(zip(refs[:3], hyps[:3]))

    ft.train()
    while step < FT_STEPS:
        for waves, lengths, texts in waveform_batches(train_utts, FT_BATCH, True, seed=step):
            lr = FT_LR * min(1.0, (step + 1) / 100)
            for g in optimizer.param_groups:
                g["lr"] = lr
            if MODEL == "wav2vec2":
                targets = [torch.tensor(tokenizer.encode(t)) for t in texts]
                tl = torch.tensor([t.numel() for t in targets])
                padded_t = torch.zeros(len(targets), int(tl.max()), dtype=torch.long)
                for b, t in enumerate(targets):
                    padded_t[b, : t.numel()] = t
                loss = ft(waves.to(device), lengths.to(device), padded_t.to(device), tl.to(device))["loss"]
            else:
                feats = ft.features([w[:n].numpy() for w, n in zip(waves, lengths)]).to(device)
                loss = ft(feats, ft.labels(texts).to(device))["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ft.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"step {step}  loss {float(loss):.3f}  lr {lr:.1e}  {time.time() - t0:.0f}s")
            if step % FT_EVAL_EVERY == 0 or step >= FT_STEPS:
                report, examples = ft_evaluate(max_batches=5 if SMOKE_TEST else None)
                ft_log.append((step, report["wer"], report["cer"]))
                print(f"  val WER {100 * report['wer']:.1f}%  CER {100 * report['cer']:.1f}%")
                for ref, hyp in examples:
                    print(f"    ref: {ref!r}\\n    hyp: {hyp!r}")
                if report["wer"] < best_wer:
                    best_wer = report["wer"]
                    torch.save(ft.state_dict(), OUTPUT_DIR / "best_pretrained.pt")
                    (OUTPUT_DIR / "results.json").write_text(json.dumps({"model": MODEL, "name": PRETRAINED_NAME, "step": step,
                                                                          "wer": report["wer"], "cer": report["cer"]}))
            if step >= FT_STEPS:
                break
    print("best WER", best_wer)
""")

md("""
## 9. Results

For the five built-in models: greedy vs beam search (and beam + a character 5-gram LM
trained on the training transcripts), the final WER / CER table, example transcripts and
the worst alignments.
""")
code("""
if MODEL in MODEL_TYPES:
    ckpt = load_checkpoint(OUTPUT_DIR / "best.pt", device)
    model, tokenizer = ckpt["model"], ckpt["tokenizer"]
    lm = NGramLM(order=5, vocab_size=tokenizer.vocab_size).train([tokenizer.encode(u.text) for u in train_utts])
    val_loader = make_loader(SpeechDataset(data_config, val_utts, tokenizer, train=False), batch_size, False)
    max_batches = 5 if SMOKE_TEST else None
    rows = []
    for name, kwargs in [("greedy", {}), ("beam 8", {"method": "beam", "beam": 8}),
                         ("beam 8 + 5-gram LM", {"method": "beam", "beam": 8, "lm": lm, "lm_weight": 0.3})]:
        if model.cfg.model == "rnnt" and "lm" in kwargs:
            continue
        res = evaluate(model, val_loader, tokenizer, device, max_batches=max_batches, **kwargs)
        rows.append((name, res["wer"], res["cer"]))
        print(f"{name:22} WER {100 * res['wer']:6.2f}%   CER {100 * res['cer']:6.2f}%")
    print("\\nexamples (greedy):")
    res = evaluate(model, val_loader, tokenizer, device, max_batches=1)
    for ref, hyp in res["examples"]:
        print(f"  ref: {ref}\\n  hyp: {hyp}\\n")
    refs, hyps = [], []
    for batch in val_loader:
        ids = decode_batch(model, batch["feats"].to(device), batch["feat_lengths"].to(device))
        hyps += [tokenizer.decode(s) for s in ids]
        refs += batch["texts"]
        if max_batches and len(refs) >= max_batches * batch_size:
            break
    report = error_report(refs, hyps)
    worst = sorted(range(len(refs)), key=lambda i: -report["per_utterance_wer"][i])[:3]
    print("worst utterances:")
    for i in worst:
        print(format_alignment(refs[i], hyps[i]), "\\n")
    plot_curves(OUTPUT_DIR / "log.csv")
""")

md("""
## 10. Download the trained model

`best.pt` holds the weights, configs and tokenizer; use it with the repo:

```bash
stt-transcribe best.pt clip.wav --decode beam
```

On Kaggle, "Save Version" keeps `/kaggle/working`; on Colab with Drive the files are already
on Drive. Otherwise the zip below can be downloaded from the file browser.
""")
code("""
import shutil

archive = shutil.make_archive(str(OUTPUT_DIR.parent / f"stt-{MODEL}-{ENCODER}"), "zip", OUTPUT_DIR)
print("archive:", archive, f"({Path(archive).stat().st_size / 1e6:.1f} MB)")
if ON_COLAB and not USE_GOOGLE_DRIVE:
    from google.colab import files

    files.download(archive)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
