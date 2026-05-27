"""
Shared runtime configuration for the transcription pipeline.

Values are read from .env when present, but existing process environment values
always win. Paths are resolved relative to the repository root unless absolute.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


BASE_DIR = Path(__file__).resolve().parent


def load_env() -> None:
    """Load .env from the project root without overriding existing env vars."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def _env_path(name: str, default: str) -> Path:
    raw_value = os.getenv(name, default).strip() or default
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


load_env()

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Colombo").strip() or "Asia/Colombo"
TRANSCRIPTIONS_DIR = _env_path(
    "TRANSCRIPTIONS_DIR",
    os.getenv("TRANSCRIPT_OUTPUT_DIR", "transcriptions"),
)
# Backward-compatible name used by watcher.py until its cleanup pass.
TRANSCRIPT_OUTPUT_DIR = TRANSCRIPTIONS_DIR
TRANSCRIPT_DATE_FORMAT = (
    os.getenv("TRANSCRIPT_DATE_FORMAT", "%Y.%m.%d").strip() or "%Y.%m.%d"
)
_timezone_warning_shown = False


def _app_timezone():
    global _timezone_warning_shown
    try:
        return ZoneInfo(APP_TIMEZONE)
    except ZoneInfoNotFoundError:
        if APP_TIMEZONE == "Asia/Colombo":
            return timezone(timedelta(hours=5, minutes=30), name="Asia/Colombo")
        if not _timezone_warning_shown:
            print(
                f"WARNING: APP_TIMEZONE '{APP_TIMEZONE}' is invalid or unavailable. "
                "Using UTC as fallback.",
                file=sys.stderr,
            )
            _timezone_warning_shown = True
        return timezone.utc


def app_now() -> datetime:
    """Return the current application time as a timezone-aware datetime."""
    return datetime.now(_app_timezone())
