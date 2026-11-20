#!/usr/bin/env python3
"""M2 dataset forge: Expresso read/whisper → 16 kHz mono clips + manifests.

Runs on Boat CPU (vibetuner). Does not create a GPU. Does not download the
36 GB nyuuzyou/asmr archive. Pulls only style==whisper rows from the public
ylacombe/expresso mirror (CC-BY-NC-4.0, ungated) via the HF datasets-server
so we never materialize the 5.8 GB full-read parquet.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEPS = ["numpy", "soundfile"]
DATASET = "ylacombe/expresso"
CONFIG = "read"
SPLIT = "train"
STYLE = "whisper"
TARGET_SR = 16000
MAX_SEC = 30.0
MIN_SEC = 0.4
MIN_TEXT_CHARS = 2
VAL_FRACTION = 0.10
SEED = 42
SEARCH_PAGE = 100
ROWS_PAGE = 100
DL_WORKERS = 6
HTTP_TIMEOUT = 60
USER_AGENT = "vibetuning-m2-forge/1.0"

GARBAGE_TEXT = re.compile(
    r"^(?:\[(?:inaudible|music|silence|noise)\]|</?unk>|…+|\.+)$",
    re.IGNORECASE,
)
HAS_LETTER = re.compile(r"[A-Za-z]")


def log(msg: str) -> None:
    print(msg, flush=True)


def venv_python() -> Path:
    override = os.environ.get("M2_VENV")
    if override:
        return Path(override) / "bin" / "python"
    home = Path.home()
    for cand in (
        home / "vibetuning" / ".venv",
        home / ".venv-m2",
        home / ".venv-jl",
    ):
        py = cand / "bin" / "python"
        if py.exists():
            return py
    return home / "vibetuning" / ".venv" / "bin" / "python"


def install_deps() -> None:
    py = venv_python()
    venv = py.parent.parent
    if not py.exists():
        log(f"[m2] creating venv {venv}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    # venv `python` is often a symlink to the system interpreter; compare prefixes.
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        log(f"[m2] re-exec {py} (sys.prefix={sys.prefix})")
        os.execv(str(py), [str(py), *sys.argv])
    log(f"[m2] pip install: {' '.join(DEPS)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *DEPS])
    log("[m2] pip install done")


def repo_root() -> Path:
    env = os.environ.get("M2_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def headers() -> dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def http_json(url: str, retries: int = 6) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers())
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            last = exc
            # Index still building, or brief 5xx.
            if exc.code in {429, 500, 502, 503, 504} or "index is loading" in body:
                sleep = min(30, 2 ** attempt) + attempt
                log(f"[m2] retry {attempt + 1}/{retries} {exc.code} {url} sleep={sleep}s {body!r}")
                time.sleep(sleep)
                continue
            raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
        except Exception as exc:  # noqa: BLE001
            last = exc
            sleep = min(20, 1.5 ** attempt)
            log(f"[m2] retry {attempt + 1}/{retries} {type(exc).__name__} {url}")
            time.sleep(sleep)
    raise RuntimeError(f"failed {url}: {last!r}")


def http_bytes(url: str, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers())
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(12, 1.8 ** attempt))
    raise RuntimeError(f"download failed {url}: {last!r}")


def audio_url(audio) -> str | None:
    if audio is None:
        return None
    if isinstance(audio, str) and audio.startswith("http"):
        return audio
    if isinstance(audio, dict):
        for key in ("src", "path", "url"):
            val = audio.get(key)
            if isinstance(val, str) and val.startswith("http"):
                return val
        inner = audio.get("bytes") or audio.get("array")
        if inner is not None:
            return None  # decoded payload, handled elsewhere
        return audio_url(audio.get("audio"))
    if isinstance(audio, list) and audio:
        return audio_url(audio[0])
    return None


def collect_via_search() -> list[dict]:
    rows: list[dict] = []
    offset = 0
    total = None
    while True:
        qs = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "query": STYLE,
                "offset": offset,
                "length": SEARCH_PAGE,
            }
        )
        data = http_json(f"https://datasets-server.huggingface.co/search?{qs}")
        total = data.get("num_rows_total", total)
        page = data.get("rows") or []
        if not page:
            break
        for item in page:
            row = item.get("row") or {}
            if row.get("style") != STYLE:
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "text": row.get("text") or "",
                    "speaker_id": row.get("speaker_id"),
                    "style": row.get("style"),
                    "audio_url": audio_url(row.get("audio")),
                    "row_idx": item.get("row_idx"),
                }
            )
        offset += len(page)
        log(f"[m2] search offset={offset}/{total} kept={len(rows)}")
        if total is not None and offset >= total:
            break
        if len(page) < SEARCH_PAGE:
            break
    return rows


def collect_via_rows_scan() -> list[dict]:
    """Fallback: walk the viewer pages and keep style==whisper only."""
    rows: list[dict] = []
    offset = 0
    total = None
    while True:
        qs = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": ROWS_PAGE,
            }
        )
        data = http_json(f"https://datasets-server.huggingface.co/rows?{qs}")
        total = data.get("num_rows_total") or data.get("num_rows") or total
        page = data.get("rows") or []
        if not page:
            break
        for item in page:
            row = item.get("row") or {}
            if row.get("style") != STYLE:
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "text": row.get("text") or "",
                    "speaker_id": row.get("speaker_id"),
                    "style": row.get("style"),
                    "audio_url": audio_url(row.get("audio")),
                    "row_idx": item.get("row_idx"),
                }
            )
        offset += len(page)
        if offset % 1000 == 0:
            log(f"[m2] rows scan offset={offset}/{total} whisper={len(rows)}")
        if total is not None and offset >= total:
            break
        if len(page) < ROWS_PAGE:
            break
    return rows


def clean_text(text: str) -> str | None:
    t = (text or "").strip()
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    # Expresso marks emphasis with *word*; keep the word, drop the markers.
    t = t.replace("*", "")
    if len(t) < MIN_TEXT_CHARS:
        return None
    if not HAS_LETTER.search(t):
        return None
    if GARBAGE_TEXT.match(t):
        return None
    return t


def to_mono_16k(y, sr: int):
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.reshape(-1)
    if sr != TARGET_SR:
        n_out = int(round(len(y) * TARGET_SR / float(sr)))
        if n_out < 1:
            return None, TARGET_SR
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = TARGET_SR
    peak = float(max(abs(y.min(initial=0.0)), abs(y.max(initial=0.0))))
    if peak > 1.0:
        y = y / peak
    return y, sr


def rms(y) -> float:
    import numpy as np

    if y is None or len(y) == 0:
        return 0.0
    return float((np.mean(np.square(y.astype("float64")))) ** 0.5)


def write_wav(path: Path, y, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    clipped = np.clip(y, -1.0, 1.0)
    sf.write(str(path), clipped, sr, subtype="PCM_16")


def process_one(meta: dict, audio_dir: Path) -> dict | None:
    import soundfile as sf

    text = clean_text(meta["text"])
    if text is None:
        return {"skip": "garbage_text", "id": meta.get("id")}
    url = meta.get("audio_url")
    if not url:
        return {"skip": "no_audio_url", "id": meta.get("id")}
    raw = http_bytes(url)
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    y, sr = to_mono_16k(y, int(sr))
    if y is None:
        return {"skip": "resample_fail", "id": meta.get("id")}
    duration = len(y) / float(sr)
    if duration > MAX_SEC:
        return {"skip": "too_long", "id": meta.get("id"), "duration": duration}
    if duration < MIN_SEC:
        return {"skip": "too_short", "id": meta.get("id"), "duration": duration}
    energy = rms(y)
    if energy < 1e-4:
        return {"skip": "silent", "id": meta.get("id"), "duration": duration}
    clip_id = str(meta["id"]).replace("/", "_")
    rel = f"audio/{clip_id}.wav"
    out = audio_dir / f"{clip_id}.wav"
    write_wav(out, y, sr)
    return {
        "audio_path": rel,
        "text": text,
        "id": clip_id,
        "speaker_id": meta.get("speaker_id"),
        "duration_sec": round(duration, 3),
        "rms": round(energy, 6),
        "source_row_idx": meta.get("row_idx"),
    }


def split_train_val(kept: list[dict]) -> tuple[list[dict], list[dict]]:
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        by_spk[str(row.get("speaker_id") or "unk")].append(row)
    rng = random.Random(SEED)
    train: list[dict] = []
    val: list[dict] = []
    for spk, items in sorted(by_spk.items()):
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * VAL_FRACTION))) if len(items) >= 10 else max(0, len(items) // 10)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"audio_path": row["audio_path"], "text": row["text"]}, ensure_ascii=False) + "\n")


def render_card(stats: dict) -> str:
    return f"""# vibetuning whisper ASR dataset card (M2)

Held-out **TEST** is *not* this set. TEST = user-recorded clips in `eval_whisper/`
(audio + `sentences.txt`), loaded by `recipes/sft_whisper/eval_loader.py`.

## Source used

| Field | Value |
| --- | --- |
| Primary | [ylacombe/expresso](https://huggingface.co/datasets/ylacombe/expresso) `read` split, `style == whisper` |
| Official paper / demo | [arXiv:2308.05725](https://arxiv.org/abs/2308.05725) · [speechbot.github.io/expresso](https://speechbot.github.io/expresso/) · [facebookresearch/textlesslib expresso](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset) |
| License | **CC BY-NC 4.0** (non-commercial). HF card tag `cc-by-nc-4.0`. |
| Gating | **Ungated / public.** `facebook/expresso` does not exist on HF (API: missing or auth-walled). No terms gate on `ylacombe/expresso`; `HF_TOKEN` not required. |
| Label provenance | **Human transcripts** for Expresso *read* speech (paper + dataset card: "The transcriptions of the read speech are also provided."). Not model-generated. |
| Upstream whisper hours | 79 min read + 86 min improvised. Improvised dialogues are **not** in the `read` config and are omitted (no matching transcripts in that subset). |
| Card scrape | context.dev `POST /v1/web/scrape` of the HF dataset card (2026-11-20). |

## Built subset (this repo)

| Stat | Value |
| --- | --- |
| Clips kept | {stats["n_clips"]} |
| Total hours | {stats["hours"]:.3f} |
| Avg duration | {stats["avg_duration_sec"]:.2f} s |
| Min / max duration | {stats["min_duration_sec"]:.2f} / {stats["max_duration_sec"]:.2f} s |
| Sample rate | 16 kHz mono PCM16 WAV |
| Max clip length | ≤ {MAX_SEC:.0f} s (longer dropped) |
| Train / val | {stats["n_train"]} / {stats["n_val"]} (seed {SEED}, ~{VAL_FRACTION:.0%} val, stratified by `speaker_id`) |
| Speakers | {stats["speakers"]} |
| Dropped | {json.dumps(stats["dropped"], sort_keys=True)} |
| Audio on disk | `data/expresso_whisper/audio/` (gitignored; lives on vibetuner) |
| Manifests | `data/expresso_whisper/manifest.jsonl`, `manifest_train.jsonl`, `manifest_val.jsonl` |

Manifest schema: `{{"audio_path": "audio/<id>.wav", "text": "<transcript>"}}`.

## Discovery not used

### `facebook/expresso`

HF datasets-server: dataset does not exist or is not accessible without auth. context.dev scrape of the HF URL hit an anti-bot wall. **Not a usable source.**

### `nyuuzyou/asmr` — considered, **not downloaded**

- Card: https://huggingface.co/datasets/nyuuzyou/asmr
- **License: unspecified** (HF tag `other`). Do not treat as open.
- **Labels are distillation, not human-verified.** `text` is machine-generated by [`distil-whisper/distil-large-v3.5`](https://huggingface.co/distil-whisper/distil-large-v3.5).
- Audio lives only in `dataset.tar.zst` (**36.3 GB**). `meta.csv` is 26.4 MB of filename+pseudo-labels; the parquet viewer is metadata-only.
- Hard rule: never download the full tar. Expresso was accessible, so asmr was not pulled (not even `meta.csv` beyond the card).

### WhispReal / WhispSynth (M0 paper hits)

[WhispSynth](https://arxiv.org/abs/2603.14853v1) and [WhispEar](https://arxiv.org/abs/2603.08046v1) are papers that *point at* whisper corpora (Expresso BY-NC, wTIMIT, …). They are not a separately licensed clip dump we can stream. No extra download.

## Caveats

- **CC BY-NC 4.0**: fine for this hackathon / research demo; not for a commercial product without a separate license from the Expresso authors.
- Read-speech whisper, studio 48 kHz originally, 4 speakers (ex01–ex04). Domain gap vs phone-mic / true ASMR is expected — that is why TEST is user-recorded whisper.
- Emphasis markers (`*word*`) stripped from text.
- Empty / non-alphabetic / placeholder transcripts dropped.
- Do not mix asmr pseudo-labels into train without relabeling.

## How it was built

On Boat sandbox `vibetuner` (`bx_6nvrrech`), CPU only, ₹0 GPU: `scripts/forge_expresso.py` listed whisper rows through the HF datasets-server (search, rows-scan fallback), downloaded those wavs only, resampled to 16 kHz mono, wrote manifests + this card.
"""


def copy_samples(kept: list[dict], audio_dir: Path, sample_dir: Path) -> list[dict]:
    import shutil

    sample_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(kept, key=lambda r: r["duration_sec"])
    picks = []
    if ranked:
        # short, median, longer — prefer distinct speakers when possible
        candidates = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
        seen_spk = set()
        for row in candidates:
            picks.append(row)
            seen_spk.add(row.get("speaker_id"))
        for row in ranked:
            if len(picks) >= 3:
                break
            if row.get("speaker_id") not in seen_spk:
                picks.append(row)
                seen_spk.add(row.get("speaker_id"))
        # unique by id, cap 3
        uniq = []
        seen = set()
        for row in picks:
            if row["id"] in seen:
                continue
            uniq.append(row)
            seen.add(row["id"])
            if len(uniq) == 3:
                break
        picks = uniq
    lines = []
    for i, row in enumerate(picks, 1):
        src = audio_dir / Path(row["audio_path"]).name
        dst = sample_dir / f"sample_{i}_{row['id']}.wav"
        shutil.copy2(src, dst)
        lines.append(
            {
                "n": i,
                "id": row["id"],
                "speaker_id": row.get("speaker_id"),
                "duration_sec": row["duration_sec"],
                "rms": row["rms"],
                "text": row["text"],
                "wav": str(dst.name),
            }
        )
    (sample_dir / "transcripts.json").write_text(json.dumps(lines, indent=2), encoding="utf-8")
    txt = []
    for item in lines:
        txt.append(
            f"sample_{item['n']}: {item['wav']}\n"
            f"  id={item['id']} speaker={item['speaker_id']} dur={item['duration_sec']}s rms={item['rms']}\n"
            f"  text: {item['text']}\n"
        )
    (sample_dir / "transcripts.txt").write_text("\n".join(txt), encoding="utf-8")
    return lines


def main() -> int:
    started = time.time()
    install_deps()

    root = repo_root()
    out = Path(os.environ.get("M2_OUT", str(root / "data" / "expresso_whisper")))
    audio_dir = out / "audio"
    sample_dir = root / "data" / "samples"
    card_path = root / "data" / "dataset_card.md"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    log(f"[m2] root={root} out={out}")
    log("[m2] collecting whisper rows (search first)")
    metas: list[dict] = []
    try:
        metas = collect_via_search()
        log(f"[m2] search kept {len(metas)} whisper rows")
    except Exception as exc:  # noqa: BLE001
        log(f"[m2] search failed ({exc!r}); scanning rows")
    if len(metas) < 500:
        log("[m2] search too small; scanning viewer rows")
        metas = collect_via_rows_scan()
        log(f"[m2] rows scan kept {len(metas)} whisper rows")

    # de-dupe by id
    by_id = {}
    for m in metas:
        if m.get("id"):
            by_id[m["id"]] = m
    metas = list(by_id.values())
    log(f"[m2] unique whisper ids={len(metas)}")
    if not metas:
        raise SystemExit("no whisper rows found")

    kept: list[dict] = []
    dropped: Counter[str] = Counter()
    log(f"[m2] downloading + resampling {len(metas)} clips workers={DL_WORKERS}")
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futs = {pool.submit(process_one, m, audio_dir): m["id"] for m in metas}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                dropped["download_error"] += 1
                log(f"[m2] error id={futs[fut]} {exc!r}")
                result = None
            if result is None:
                dropped["unknown"] += 1
            elif result.get("skip"):
                dropped[str(result["skip"])] += 1
            else:
                kept.append(result)
            if done % 50 == 0 or done == len(metas):
                log(f"[m2] processed {done}/{len(metas)} kept={len(kept)} dropped={dict(dropped)}")

    if not kept:
        raise SystemExit("no clips kept")

    kept.sort(key=lambda r: r["id"])
    train, val = split_train_val(kept)
    durations = [r["duration_sec"] for r in kept]
    hours = sum(durations) / 3600.0
    speakers = dict(Counter(r.get("speaker_id") for r in kept))
    stats = {
        "source": f"{DATASET}:{CONFIG}:{SPLIT}:style={STYLE}",
        "license": "CC-BY-NC-4.0",
        "gated": False,
        "label_provenance": "human (Expresso read transcripts)",
        "n_listed": len(metas),
        "n_clips": len(kept),
        "n_train": len(train),
        "n_val": len(val),
        "hours": round(hours, 4),
        "avg_duration_sec": round(sum(durations) / len(durations), 3),
        "min_duration_sec": round(min(durations), 3),
        "max_duration_sec": round(max(durations), 3),
        "sample_rate": TARGET_SR,
        "speakers": speakers,
        "dropped": dict(dropped),
        "elapsed_sec": round(time.time() - started, 1),
        "gpu_spend_inr": 0,
    }

    write_jsonl(out / "manifest.jsonl", kept)
    write_jsonl(out / "manifest_train.jsonl", train)
    write_jsonl(out / "manifest_val.jsonl", val)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    # sidecar metadata for debugging (not the trainer schema)
    with (out / "clips.jsonl").open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    samples = copy_samples(kept, audio_dir, sample_dir)
    stats["spot_check"] = samples
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    card_path.write_text(render_card(stats), encoding="utf-8")

    log("========== M2 STATS ==========")
    log(json.dumps(stats, indent=2))
    log("========== SPOT CHECK ==========")
    for item in samples:
        log(
            f"SAMPLE {item['n']} file={item['wav']} id={item['id']} "
            f"spk={item['speaker_id']} dur={item['duration_sec']}s rms={item['rms']}"
        )
        log(f"  TEXT: {item['text']}")
    log(f"[m2] card -> {card_path}")
    log(f"[m2] manifests -> {out}")
    log("[m2] GPU spend ₹0")
    log("[m2] complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"[m2] FATAL: {exc!r}")
        raise
