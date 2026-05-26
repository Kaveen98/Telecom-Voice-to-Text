"""
Transcript file storage helpers.

Transcripts are written under TRANSCRIPT_OUTPUT_DIR/YYYY.MM.DD using the
completion/import time, not any date embedded in the source audio filename.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BASE_DIR, TRANSCRIPT_DATE_FORMAT, TRANSCRIPT_OUTPUT_DIR, app_now


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WHITESPACE = re.compile(r"\s+")


def _sanitize_component(value: Any, fallback: str, max_len: int = 120) -> str:
    text = str(value or "").strip()
    text = _INVALID_FILENAME_CHARS.sub("-", text)
    text = _WHITESPACE.sub("-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip(" .-_")
    if not text:
        text = fallback
    return text[:max_len]


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def _base_output_dir(output_dir: str | Path | None = None) -> Path:
    if output_dir is None:
        return TRANSCRIPT_OUTPUT_DIR
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _write_unique_text(date_dir: Path, filename: str, transcript: str) -> Path:
    candidate = date_dir / filename
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1

    while True:
        try:
            with candidate.open("x", encoding="utf-8") as f:
                f.write(transcript)
            return candidate
        except FileExistsError:
            counter += 1
            candidate = date_dir / f"{stem}-{counter}{suffix}"


def save_transcript_text(
    audio_path: str | Path,
    transcript: str,
    model: str,
    mode: str,
    call_id: int | None = None,
    saved_at: datetime | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """
    Save transcript text and return DB-ready metadata.

    transcript_file_path is stored relative to the repository root when the
    configured output directory is inside the project.
    """
    saved_dt = saved_at or app_now()
    output_date = saved_dt.strftime(TRANSCRIPT_DATE_FORMAT)
    timestamp = saved_dt.strftime("%Y%m%d-%H%M%S-%f")

    base_dir = _base_output_dir(output_dir)
    date_dir = base_dir / output_date
    date_dir.mkdir(parents=True, exist_ok=True)

    audio_stem = _sanitize_component(Path(audio_path).stem, "audio")
    mode_slug = _sanitize_component(mode, "mode", max_len=40)
    model_slug = _sanitize_component(model, "model", max_len=80)
    # call_id is accepted for callers that naturally have it, but the filename
    # stays focused on source, mode, model, and completion/import timestamp.
    _ = call_id
    filename = f"{audio_stem}__{mode_slug}__{model_slug}__{timestamp}.txt"
    transcript_path = _write_unique_text(date_dir, filename, transcript)

    return {
        "transcript_file_path": _relative_path(transcript_path),
        "transcript_saved_at": saved_dt.isoformat(),
        "transcript_output_date": output_date,
    }


def resolve_transcript_path(stored_path: str) -> Path:
    """Resolve a stored relative or absolute transcript path."""
    path = Path(stored_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()
