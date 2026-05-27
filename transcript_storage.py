"""
Compatibility wrappers for transcript output storage.

The TXT/JSON saving implementation now lives in gemini_flash_stt.py. This file
keeps the older function names available until watcher.py is refactored.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def save_transcript_text(
    audio_path: str | Path,
    transcript: str,
    model: str,
    mode: str,
    call_id: int | None = None,
    saved_at: datetime | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compatibility wrapper around gemini_flash_stt.save_transcription_outputs().

    The return value preserves legacy keys such as transcript_file_path while
    adding metadata_json_path for the new sidecar JSON output.
    """
    from gemini_flash_stt import save_transcription_outputs

    result: dict[str, Any] = {
        "transcript": transcript,
        "model": model,
        "mode": mode,
        "status": "completed" if transcript else "empty",
    }
    if call_id is not None:
        result["call_id"] = call_id

    return save_transcription_outputs(
        result=result,
        audio_path=audio_path,
        output_root=output_dir,
        saved_at=saved_at,
    )


def resolve_transcript_path(stored_path: str) -> Path:
    """Compatibility wrapper for resolving stored transcript/metadata paths."""
    from gemini_flash_stt import resolve_transcript_path as _resolve

    return _resolve(stored_path)
