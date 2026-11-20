"""Held-out TEST loader for recorded whisper clips.

Looks in `vibetuning/eval_whisper/` (audio files + `sentences.txt`).
This split is never mixed into train/val from Expresso.
"""

from __future__ import annotations

from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = REPO_ROOT / "eval_whisper"


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
    - `clip.wav transcript` (filename has an audio suffix)
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


def load_eval_whisper(directory: str | Path | None = None) -> list[dict]:
    """Return `[{audio_path, text}, ...]` for the held-out recorded set.

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
