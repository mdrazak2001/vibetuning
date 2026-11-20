"""Held-out TEST loader for recorded whisper clips.

Looks in `vibetuning/eval_whisper/` (audio files + `sentences.txt`).
This split is never mixed into train/val from Expresso.

Any dropped format (m4a / mp3 / wav / flac / ogg / aac) is decoded and
resampled to 16 kHz mono here — the trainer does not assume WAV.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".webm"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = REPO_ROOT / "eval_whisper"
TARGET_SR = 16000


def eval_dir(path: str | Path | None = None) -> Path:
    if path is None:
        env = __import__("os").environ.get("EVAL_WHISPER_DIR")
        path = env if env else DEFAULT_EVAL_DIR
    return Path(path)


def list_audio(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def _parse_sentences(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return (by_filename, ordered_transcripts).

    Accepted line formats (blank and `#` comments skipped):
    - `clip.wav<TAB>transcript`
    - `clip.wav|transcript`
    - `clip.m4a transcript` (filename has an audio suffix)
    - `transcript` (aligned later to sorted audio)
    """
    by_name: dict[str, str] = {}
    ordered: list[str] = []
    if not path.exists():
        return by_name, ordered
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = None
        text = None
        if "\t" in line:
            left, right = line.split("\t", 1)
            name, text = left.strip(), right.strip()
        elif "|" in line:
            left, right = line.split("|", 1)
            name, text = left.strip(), right.strip()
        else:
            parts = line.split(None, 1)
            if len(parts) == 2 and Path(parts[0]).suffix.lower() in AUDIO_EXTS:
                name, text = parts[0], parts[1]
            else:
                text = line
        if not text:
            continue
        ordered.append(text)
        if name:
            by_name[Path(name).name] = text
            by_name[Path(name).stem] = text
    return by_name, ordered


def is_eval_ready(directory: str | Path | None = None) -> bool:
    d = eval_dir(directory)
    return d.is_dir() and bool(list_audio(d)) and (d / "sentences.txt").exists()


def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.reshape(-1)


def _resample(y, sr: int, target: int = TARGET_SR):
    import numpy as np

    if sr == target:
        return y.astype(np.float32)
    n_out = int(round(len(y) * target / float(sr)))
    if n_out < 1:
        return y.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)


def _ffmpeg_wav_bytes(path: Path) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not on PATH")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "wav",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg produced empty audio for {path}")
    return proc.stdout


def load_audio_16k_mono(path: str | Path) -> dict:
    """Decode any supported container to float32 mono 16 kHz.

    Order: ffmpeg (m4a/mp3/aac) → soundfile (wav/flac) → librosa.
    """
    import numpy as np

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    y = None
    sr = None

    # Compressed phone recordings: ffmpeg first (soundfile cannot read m4a).
    if suffix in {".m4a", ".mp3", ".aac", ".ogg", ".webm"} or shutil.which("ffmpeg"):
        try:
            raw = _ffmpeg_wav_bytes(path)
            import soundfile as sf

            y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except FileNotFoundError:
            y = None
        except Exception:
            y = None

    if y is None:
        try:
            import soundfile as sf

            y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            y = None

    if y is None:
        try:
            import librosa

            y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode {path} as audio. Install ffmpeg and soundfile, "
                f"or convert to wav. Last error: {exc!r}"
            ) from exc

    y = _to_mono(y)
    sr = int(sr)
    if sr != TARGET_SR:
        y = _resample(y, sr, TARGET_SR)
        sr = TARGET_SR
    peak = float(max(abs(float(y.min(initial=0.0))), abs(float(y.max(initial=0.0)))))
    if peak > 1.0:
        y = y / peak
    return {"array": np.asarray(y, dtype=np.float32), "sampling_rate": sr}


def load_eval_whisper(directory: str | Path | None = None) -> list[dict]:
    """Return `[{audio_path, text}, ...]` for a held-out recorded set.

    Empty list if the user has not dropped clips yet.
    """
    d = eval_dir(directory)
    if not d.is_dir():
        return []
    audio = list_audio(d)
    by_name, ordered = _parse_sentences(d / "sentences.txt")
    rows: list[dict] = []
    for i, wav in enumerate(audio):
        text = by_name.get(wav.name) or by_name.get(wav.stem)
        if text is None and i < len(ordered) and not by_name:
            text = ordered[i]
        if not text:
            continue
        rows.append({"audio_path": str(wav.resolve()), "text": text})
    return rows


def load_eval_arrays(directory: str | Path | None = None) -> list[dict]:
    """Like load_eval_whisper, plus decoded `audio` (16 kHz mono)."""
    rows = []
    for item in load_eval_whisper(directory):
        rows.append(
            {
                "audio_path": item["audio_path"],
                "text": item["text"],
                "audio": load_audio_16k_mono(item["audio_path"]),
            }
        )
    return rows
