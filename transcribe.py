"""
transcribe.py - makes a transcript for recordings that don't have one, locally with Whisper.

Only runs in mode "all" (see pipeline.py). Downloaded Teams transcripts are always
preferred; this is the fallback for recordings without one.

How it works:
- faster-whisper (an optimized open-source Whisper) reads the audio track of the .mp4 or
  .m4a directly; no conversion step needed.
- Voice activity detection skips silence, which also makes it faster.
- The result is written as a .vtt to data/processed/<video_id>/transcript.vtt, in the same
  format ingest.py already reads. data/input is never modified.
- Whisper does not know who is speaking, so these transcripts have no speaker names.

Speed on a 4-core laptop with the "small" model: roughly 20-40 minutes per hour of audio.
Accents (e.g. Indian English) are handled well by Whisper; "medium" is more accurate but
about 2-3x slower. Set WHISPER_MODEL in .env.

Setup (once, only when you start using mode "all"):
  uv add faster-whisper
  (the first run downloads the model, ~500 MB for "small")

Usage:
  uv run transcribe.py --video <video_id>   # one recording
  uv run transcribe.py                      # every included recording without a transcript
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config
import store

log = logging.getLogger("transcribe")


def _ts(seconds: float) -> str:
    """Seconds -> VTT timestamp, e.g. 62.345 -> '00:01:02.345'."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def audio_16k(src: Path) -> np.ndarray:
    """The sound track as 16 kHz mono samples, decoded by ffmpeg. Whisper gets the samples, not
    the file: the PyAV that faster-whisper uses to open files breaks with some versions
    ("open() got an unexpected keyword argument 'metadata_errors'")."""
    r = subprocess.run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
                        "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read the sound of {src.name}: "
                           f"{r.stderr.decode('utf-8', errors='replace')[:300]}")
    audio = np.frombuffer(r.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"{src.name} has no sound track")
    return audio


def needs_transcript(row: dict) -> bool:
    """True for a recording with neither a downloaded nor a generated transcript (and not
    already found to be in another language)."""
    if (store.video_dir(row["video_id"]) / "language.json").exists():
        return False
    return bool(row.get("video_file")) and store.transcript_path(row) is None


def transcribe_row(row: dict, model=None) -> str:
    """Transcribe one manifest row. `model` can be passed in to reuse it across files."""
    if not needs_transcript(row):
        return "skipped"
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("faster-whisper is not installed. Run: uv add faster-whisper") from e

    src = config.INPUT_DIR / row["video_file"]
    if not src.exists():
        raise FileNotFoundError(f"Recording not found: {src}")
    model = model or WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    t0 = time.perf_counter()
    # language detected from the first 30 s; recordings in other languages than KNOWLEDGE_LANGS
    # are not transcribed at all (the segments are a generator: nothing is done until we iterate)
    segments, info = model.transcribe(audio_16k(src), language=None, vad_filter=True, beam_size=5)
    if info.language not in config.KNOWLEDGE_LANGS:
        store.write_json(store.video_dir(row["video_id"]) / "language.json",
                         {"lang": info.language, "probability": round(info.language_probability, 2)})
        log.info("%s: language %s (%.0f%%), not in KNOWLEDGE_LANGS=%s - not transcribed", src.name,
                 info.language, info.language_probability * 100, ",".join(config.KNOWLEDGE_LANGS))
        return "other language"

    lines = ["WEBVTT", ""]
    n = 0
    for seg in segments:   # a generator: transcription happens while we iterate
        text = seg.text.strip()
        if text:
            lines += [f"{_ts(seg.start)} --> {_ts(seg.end)}", text, ""]
            n += 1
    if n == 0:
        raise RuntimeError(f"Whisper found no speech in {src.name}")
    out = store.generated_transcript(row["video_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("%s: %s cues from %.0f min of audio in %.0f min -> %s",
             row["video_id"], n, info.duration / 60, (time.perf_counter() - t0) / 60, out)
    return "done"


def run(rows: list[dict]) -> dict[str, int]:
    """Transcribe all rows that need it; the model is loaded once."""
    todo = [r for r in rows if needs_transcript(r)]
    counts: dict[str, int] = {"skipped": len(rows) - len(todo)}
    if not todo:
        return counts
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error("%s recordings need a transcript, but faster-whisper is not installed. "
                  "Run: uv add faster-whisper", len(todo))
        counts["failed"] = len(todo)
        return counts
    log.info("Transcribing %s recordings with Whisper '%s' (slow on CPU)", len(todo), config.WHISPER_MODEL)
    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    for row in todo:
        try:
            result = transcribe_row(row, model)
        except Exception as e:
            log.error("%s: FAILED - %s", row.get("video_file"), e)
            result = "failed"
        counts[result] = counts.get(result, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video_id from the manifest (default: all included rows)")
    args = ap.parse_args()
    config.setup_logging()
    rows = store.included_rows()
    if args.video:
        rows = [r for r in rows if r["video_id"] == args.video]
    counts = run(rows)
    log.info("Finished: %s", ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
