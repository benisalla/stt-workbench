"""Built-in datasets: `stt-data <name> --out data/<name>` writes audio (if needed) and manifests.

    synthetic    a deterministic toy spoken language, generated here, no download. Each letter
                 of a 12-letter alphabet is a short pattern of tones (a 2-formant "vowel" or a
                 noise burst + tone "consonant"); words are random letter strings; sentences
                 are 1-5 words with 200 ms pauses, played at random speed and level under
                 background noise at 8 kHz. It is learnable in minutes on a CPU, so it proves
                 that a model and a decoder work before any real data is downloaded.
    yesno        60 recordings of "yes"/"no" sequences (Hebrew, 8 kHz, openslr 1); 4.5 MB.
    fsdd         Free Spoken Digit Dataset: 3000 clips of a spoken digit, 6 speakers, 8 kHz;
                 ~15 MB. Split by speaker (held-out speakers in val).
    librispeech  openslr 12 tarballs (dev-clean 337 MB ... train-clean-100 6 GB) with
                 `*.trans.txt`; for Kaggle/Colab.
    ljspeech     LJ Speech (metadata.csv + wavs/, 2.6 GB), single speaker, 13 100 clips.
    commonvoice  a Common Voice `train.tsv`/`dev.tsv` + clips/*.mp3 folder (needs ffmpeg).

Manifests are `train.jsonl` / `val.jsonl` (data.py). Nothing is downloaded to anywhere but
`--out` (or `--download-dir`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.data import write_manifest

YESNO_URL = "https://www.openslr.org/resources/1/waves_yesno.tar.gz"
FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{subset}.tar.gz"
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# ------------------------------------------------------------------------------- synthetic
SYNTH_ALPHABET = "abdeiklmnost"
SYNTH_SR = 8000


def _letter_params(letter: str) -> dict:
    """Each letter is a fixed acoustic pattern. Vowels: two formants; consonants: noise + tone."""
    i = SYNTH_ALPHABET.index(letter)
    vowel = letter in "aeio"
    rng = np.random.RandomState(1000 + i)
    return {
        "vowel": vowel,
        "f1": 250 + 90 * i + rng.uniform(0, 40),
        "f2": 900 + 170 * i + rng.uniform(0, 60),
        "chirp": rng.uniform(-300, 300),  # consonants glide
        "duration": 0.09 if vowel else 0.06,
    }


def synth_letter(letter: str, rng: np.random.RandomState, speed: float) -> np.ndarray:
    p = _letter_params(letter)
    n = int(SYNTH_SR * p["duration"] / speed)
    t = np.arange(n) / SYNTH_SR
    env = np.sin(np.pi * np.arange(n) / n) ** 0.5
    if p["vowel"]:
        f0 = 110 * rng.uniform(0.85, 1.15)
        harmonics = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 12))
        wave = harmonics * (0.5 * np.sin(2 * np.pi * p["f1"] * t) + 0.5 * np.sin(2 * np.pi * p["f2"] * t)) * 0.5
        wave = wave + 0.3 * np.sin(2 * np.pi * p["f1"] * t) + 0.2 * np.sin(2 * np.pi * p["f2"] * t)
    else:
        phase = 2 * np.pi * (p["f2"] * t + 0.5 * p["chirp"] * t**2)
        wave = 0.5 * np.sin(phase) + 0.4 * rng.randn(n) * np.exp(-t * 40)
    return (wave * env).astype(np.float32)


def synth_sentence(text: str, rng: np.random.RandomState) -> np.ndarray:
    speed = rng.uniform(0.8, 1.25)
    pieces = [np.zeros(int(SYNTH_SR * rng.uniform(0.05, 0.15)), np.float32)]
    for ch in text:
        if ch == " ":
            pieces.append(np.zeros(int(SYNTH_SR * 0.2 / speed), np.float32))
        else:
            pieces.append(synth_letter(ch, rng, speed))
            pieces.append(np.zeros(int(SYNTH_SR * 0.02), np.float32))
    pieces.append(np.zeros(int(SYNTH_SR * rng.uniform(0.05, 0.15)), np.float32))
    wav = np.concatenate(pieces)
    wav = wav * rng.uniform(0.3, 0.9) / (np.abs(wav).max() + 1e-6)
    noise = rng.randn(len(wav)).astype(np.float32) * rng.uniform(0.01, 0.06)  # "low quality"
    return np.clip(wav + noise, -1, 1).astype(np.float32)


def random_sentence(rng: np.random.RandomState, max_words: int = 5) -> str:
    words = []
    for _ in range(rng.randint(1, max_words + 1)):
        words.append("".join(rng.choice(list(SYNTH_ALPHABET), size=rng.randint(2, 6))))
    return " ".join(words)


def write_synthetic(out: Path, n_train: int = 400, n_val: int = 60, seed: int = 0, max_words: int = 5) -> Path:
    """Write wavs + manifests; deterministic for a given seed."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        (out / split).mkdir(exist_ok=True)
        rows = []
        for i in range(n):
            text = random_sentence(rng, max_words)
            wav = synth_sentence(text, rng)
            path = out / split / f"{split}_{i:05d}.wav"
            sf.write(path, wav, SYNTH_SR)
            rows.append({"audio": f"{split}/{path.name}", "text": text, "duration": len(wav) / SYNTH_SR, "id": path.stem})
        write_manifest(out / f"{split}.jsonl", rows)
    return out


# ------------------------------------------------------------------------------- downloads
def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def stable_split(key: str, val_fraction: float) -> str:
    """Deterministic train/val assignment from a hash of the id."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


def prepare_yesno(out: Path, download_dir: Path | None = None, val_fraction: float = 0.2) -> Path:
    archive = download(YESNO_URL, (download_dir or out) / "waves_yesno.tar.gz")
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    wavs = sorted((out / "waves_yesno").glob("*.wav"))
    rows = {"train": [], "val": []}
    for w in wavs:
        text = " ".join("yes" if c == "1" else "no" for c in w.stem.split("_"))
        rows[stable_split(w.stem, val_fraction)].append(
            {"audio": str(w.relative_to(out)), "text": text, "duration": wav_duration(w), "id": w.stem})
    for split, split_rows in rows.items():
        write_manifest(out / f"{split}.jsonl", split_rows)
    return out


def prepare_fsdd(out: Path, download_dir: Path | None = None, val_speakers: tuple[str, ...] = ("yweweler",),
                 clips_per_utterance: int = 1, seed: int = 0) -> Path:
    """FSDD. `clips_per_utterance` > 1 concatenates several digits of one speaker into one clip
    (with short pauses) so that the task becomes a sequence task, not a classification."""
    archive = download(FSDD_URL, (download_dir or out) / "fsdd.zip")
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if m.endswith(".wav") and "/recordings/" in m]
        for m in members:
            target = out / "recordings" / Path(m).name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(m))
    wavs = sorted((out / "recordings").glob("*.wav"))
    by_speaker: dict[str, list[Path]] = {}
    for w in wavs:
        _digit, speaker, _ = w.stem.split("_", 2)
        by_speaker.setdefault(speaker, []).append(w)
    rows = {"train": [], "val": []}
    rng = random.Random(seed)
    for speaker, files in by_speaker.items():
        split = "val" if speaker in val_speakers else "train"
        rng.shuffle(files)
        if clips_per_utterance <= 1:
            for w in files:
                rows[split].append({"audio": str(w.relative_to(out)), "text": DIGITS[int(w.stem.split("_")[0])],
                                    "duration": wav_duration(w), "id": w.stem})
            continue
        (out / "sequences").mkdir(exist_ok=True)
        for i in range(0, len(files) - clips_per_utterance + 1, clips_per_utterance):
            group = files[i : i + clips_per_utterance]
            pieces, sr = [], None
            for w in group:
                wav, sr = sf.read(str(w), dtype="float32")
                pieces += [wav, np.zeros(int(sr * rng.uniform(0.1, 0.3)), np.float32)]
            wav = np.concatenate(pieces)
            path = out / "sequences" / f"{speaker}_{i // clips_per_utterance:04d}.wav"
            sf.write(path, wav, sr)
            text = " ".join(DIGITS[int(w.stem.split("_")[0])] for w in group)
            rows[split].append({"audio": str(path.relative_to(out)), "text": text, "duration": len(wav) / sr, "id": path.stem})
    for split, split_rows in rows.items():
        write_manifest(out / f"{split}.jsonl", split_rows)
    return out


def prepare_librispeech(out: Path, train_subset: str = "dev-clean", val_subset: str = "dev-clean",
                        download_dir: Path | None = None, root: Path | None = None, val_fraction: float = 0.1) -> Path:
    """`root` = an existing LibriSpeech/ folder (e.g. a Kaggle dataset); otherwise tarballs are fetched.
    If train and val subsets are the same, speakers are split (val_fraction of them held out)."""
    out.mkdir(parents=True, exist_ok=True)
    root = root or (download_dir or out) / "LibriSpeech"
    for subset in {train_subset, val_subset}:
        if (root / subset).exists():
            continue
        archive = download(LIBRISPEECH_URL.format(subset=subset), (download_dir or out) / f"{subset}.tar.gz")
        with tarfile.open(archive) as tar:
            tar.extractall(root.parent, filter="data")

    def rows_of(subset: str) -> list[dict]:
        rows = []
        for trans in sorted((root / subset).rglob("*.trans.txt")):
            for line in trans.read_text().splitlines():
                uid, text = line.split(" ", 1)
                audio = trans.parent / f"{uid}.flac"
                rows.append({"audio": str(audio), "text": text.lower(), "duration": wav_duration(audio),
                             "id": uid, "speaker": uid.split("-")[0]})
        return rows

    if train_subset == val_subset:
        rows = rows_of(train_subset)
        write_manifest(out / "train.jsonl", [r for r in rows if stable_split(r["speaker"], val_fraction) == "train"])
        write_manifest(out / "val.jsonl", [r for r in rows if stable_split(r["speaker"], val_fraction) == "val"])
    else:
        write_manifest(out / "train.jsonl", rows_of(train_subset))
        write_manifest(out / "val.jsonl", rows_of(val_subset))
    return out


def prepare_ljspeech(out: Path, root: Path | None = None, download_dir: Path | None = None, val_fraction: float = 0.05) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    root = root or (download_dir or out) / "LJSpeech-1.1"
    if not root.exists():
        archive = download(LJSPEECH_URL, (download_dir or out) / "LJSpeech-1.1.tar.bz2")
        with tarfile.open(archive) as tar:
            tar.extractall(root.parent, filter="data")
    rows = {"train": [], "val": []}
    for line in (root / "metadata.csv").read_text(encoding="utf-8").splitlines():
        uid, _, normalized = line.split("|")
        audio = root / "wavs" / f"{uid}.wav"
        rows[stable_split(uid, val_fraction)].append(
            {"audio": str(audio), "text": normalized, "duration": wav_duration(audio), "id": uid})
    for split, split_rows in rows.items():
        write_manifest(out / f"{split}.jsonl", split_rows)
    return out


def prepare_commonvoice(out: Path, root: Path, train_tsv: str = "train.tsv", val_tsv: str = "dev.tsv",
                        sample_rate: int = 16_000, max_clips: int | None = None) -> Path:
    """Converts clips/*.mp3 to wav with ffmpeg (must be installed) and writes manifests."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to decode Common Voice mp3 clips")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    for split, tsv in (("train", train_tsv), ("val", val_tsv)):
        lines = (root / tsv).read_text(encoding="utf-8").splitlines()
        header = lines[0].split("\t")
        i_path, i_text = header.index("path"), header.index("sentence")
        rows = []
        for line in lines[1 : (max_clips + 1 if max_clips else None)]:
            cols = line.split("\t")
            mp3 = root / "clips" / cols[i_path]
            wav = out / "wavs" / (Path(cols[i_path]).stem + ".wav")
            if not wav.exists():
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar",
                                str(sample_rate), str(wav)], check=True)
            rows.append({"audio": str(wav.relative_to(out)), "text": cols[i_text], "duration": wav_duration(wav),
                         "id": wav.stem})
        write_manifest(out / f"{split}.jsonl", rows)
    return out


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    return wav, sr


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Prepare a dataset: audio + train.jsonl / val.jsonl")
    p.add_argument("name", choices=["synthetic", "yesno", "fsdd", "librispeech", "ljspeech", "commonvoice"])
    p.add_argument("--out", type=Path, default=None, help="default data/<name>")
    p.add_argument("--download-dir", type=Path, default=None, help="where archives are kept (default: --out)")
    p.add_argument("--root", type=Path, default=None, help="existing dataset folder (librispeech/ljspeech/commonvoice)")
    p.add_argument("--n-train", type=int, default=400, help="synthetic: training sentences")
    p.add_argument("--n-val", type=int, default=60, help="synthetic: validation sentences")
    p.add_argument("--max-words", type=int, default=5, help="synthetic: words per sentence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clips-per-utterance", type=int, default=1, help="fsdd: digits per clip")
    p.add_argument("--train-subset", default="dev-clean", help="librispeech")
    p.add_argument("--val-subset", default="dev-clean", help="librispeech")
    p.add_argument("--max-clips", type=int, default=None, help="commonvoice: limit per split")
    args = p.parse_args(argv)
    out = args.out or Path("data") / args.name
    if args.name == "synthetic":
        write_synthetic(out, args.n_train, args.n_val, args.seed, args.max_words)
    elif args.name == "yesno":
        prepare_yesno(out, args.download_dir)
    elif args.name == "fsdd":
        prepare_fsdd(out, args.download_dir, clips_per_utterance=args.clips_per_utterance, seed=args.seed)
    elif args.name == "librispeech":
        prepare_librispeech(out, args.train_subset, args.val_subset, args.download_dir, args.root)
    elif args.name == "ljspeech":
        prepare_ljspeech(out, args.root, args.download_dir)
    elif args.name == "commonvoice":
        if args.root is None:
            raise SystemExit("--root <CommonVoice folder with train.tsv and clips/> is required")
        prepare_commonvoice(out, args.root, max_clips=args.max_clips)
    for split in ("train", "val"):
        n = sum(1 for _ in (out / f"{split}.jsonl").open())
        print(f"{split}: {n} utterances -> {out / f'{split}.jsonl'}")


__all__ = [
    "DIGITS",
    "SYNTH_ALPHABET",
    "SYNTH_SR",
    "download",
    "main",
    "prepare_commonvoice",
    "prepare_fsdd",
    "prepare_librispeech",
    "prepare_ljspeech",
    "prepare_yesno",
    "random_sentence",
    "stable_split",
    "synth_sentence",
    "write_synthetic",
]

if __name__ == "__main__":
    main()
