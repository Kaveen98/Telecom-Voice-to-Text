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


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float = 0.0) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int = 0) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _cost_limit_db_failure_policy() -> str:
    policy = os.getenv("COST_LIMIT_DB_FAILURE_POLICY", "block").strip().lower()
    return policy if policy in {"block", "allow"} else "block"


load_env()

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Colombo").strip() or "Asia/Colombo"
INPUT_INCOMING_DIR = _env_path("INPUT_INCOMING_DIR", "input_audio/incoming")
INPUT_PROCESSING_DIR = _env_path("INPUT_PROCESSING_DIR", "input_audio/processing")
INPUT_COMPLETED_DIR = _env_path("INPUT_COMPLETED_DIR", "input_audio/completed")
INPUT_FAILED_DIR = _env_path("INPUT_FAILED_DIR", "input_audio/failed")
INPUT_DEFERRED_DIR = _env_path("INPUT_DEFERRED_DIR", "input_audio/deferred")
TRANSCRIPTIONS_DIR = _env_path(
    "TRANSCRIPTIONS_DIR",
    os.getenv("TRANSCRIPT_OUTPUT_DIR", "transcriptions"),
)
# Backward-compatible name used by watcher.py until its cleanup pass.
TRANSCRIPT_OUTPUT_DIR = TRANSCRIPTIONS_DIR
LOG_DIR = _env_path("LOG_DIR", "logs")
TRANSCRIPT_DATE_FORMAT = (
    os.getenv("TRANSCRIPT_DATE_FORMAT", "%Y.%m.%d").strip() or "%Y.%m.%d"
)
DAILY_COST_LIMIT_ENABLED = _env_bool("DAILY_COST_LIMIT_ENABLED", default=False)
DAILY_COST_LIMIT_LKR = max(0.0, _env_float("DAILY_COST_LIMIT_LKR", default=0.0))
DAILY_COST_WARNING_PERCENT = max(
    0,
    min(100, _env_int("DAILY_COST_WARNING_PERCENT", default=80)),
)
COST_LIMIT_PREFLIGHT_ENABLED = _env_bool(
    "COST_LIMIT_PREFLIGHT_ENABLED",
    default=True,
)
COST_LIMIT_DB_FAILURE_POLICY = _cost_limit_db_failure_policy()
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
