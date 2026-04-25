"""
Central runtime configuration for the transcription pipeline.

The defaults preserve the local developer workflow. Linux deployments can
override paths and service settings with environment variables or a .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_GOOGLE_LOCATION = "us-central1"
DEFAULT_LKR_RATE = 316.0
DEFAULT_PREPROCESS_PROFILE = "ffmpeg_silenceremove_v1"
NO_TRIM_PREPROCESS_PROFILE = "no_trim"


def load_env(env_path: Path | None = None) -> None:
    """
    Load KEY=VALUE pairs from .env without overriding existing environment.

    This keeps compatibility with the previous local .env behavior while
    allowing systemd/Docker/host environment variables to take precedence.
    """
    path = env_path or PROJECT_ROOT / ".env"
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(raw: str | None, default: Path, *, base: Path = PROJECT_ROOT) -> Path:
    if raw is None or raw.strip() == "":
        return default.resolve()

    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_log_file(raw: str | None, log_dir: Path) -> Path:
    if raw is None or raw.strip() == "":
        return (log_dir / "watcher.log").resolve()

    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = log_dir / path
    return path.resolve()


def resolve_credentials_path(raw: str | None = None) -> Path | None:
    value = raw if raw is not None else os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if value is None or value.strip() == "":
        return None

    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


load_env()

_DATA_DIR_CONFIGURED = bool(os.getenv("STT_DATA_DIR", "").strip())
DATA_DIR = _resolve_path(os.getenv("STT_DATA_DIR"), PROJECT_ROOT)

INPUT_DIR = _resolve_path(
    os.getenv("STT_INPUT_DIR"),
    DATA_DIR / "input_audio" if _DATA_DIR_CONFIGURED else PROJECT_ROOT / "input_audio",
)
OUTPUT_DIR = _resolve_path(
    os.getenv("STT_OUTPUT_DIR"),
    DATA_DIR / "output" if _DATA_DIR_CONFIGURED else PROJECT_ROOT / "output",
)
DB_PATH = _resolve_path(
    os.getenv("STT_DB_PATH"),
    DATA_DIR / "calls.db" if _DATA_DIR_CONFIGURED else PROJECT_ROOT / "calls.db",
)

LOG_DIR = _resolve_path(
    os.getenv("STT_LOG_DIR"),
    DATA_DIR / "logs" if _DATA_DIR_CONFIGURED else PROJECT_ROOT,
)
LOG_FILE = _resolve_log_file(os.getenv("STT_LOG_FILE"), LOG_DIR)

MODEL_NAME = os.getenv("STT_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
LKR_RATE = _get_float("STT_LKR_RATE", DEFAULT_LKR_RATE)

STRIP_SILENCE = _get_bool("STT_STRIP_SILENCE", default=True)
SILENCE_START_THRESHOLD_DB = _get_float("STT_SILENCE_START_THRESHOLD_DB", -40.0)
SILENCE_STOP_THRESHOLD_DB = _get_float("STT_SILENCE_STOP_THRESHOLD_DB", -40.0)
SILENCE_START_DURATION = _get_float("STT_SILENCE_START_DURATION", 0.3)
SILENCE_STOP_DURATION = _get_float("STT_SILENCE_STOP_DURATION", 0.5)
PREPROCESS_PROFILE = (
    os.getenv("STT_PREPROCESS_PROFILE", DEFAULT_PREPROCESS_PROFILE).strip()
    or DEFAULT_PREPROCESS_PROFILE
)


def get_effective_preprocess_profile(strip_silence: bool | None = None) -> str:
    """Return the profile name used for dedupe and call metadata."""
    strip_enabled = STRIP_SILENCE if strip_silence is None else strip_silence
    return PREPROCESS_PROFILE if strip_enabled else NO_TRIM_PREPROCESS_PROFILE


EFFECTIVE_PREPROCESS_PROFILE = get_effective_preprocess_profile()

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
GOOGLE_CLOUD_LOCATION = (
    os.getenv("GOOGLE_CLOUD_LOCATION")
    or os.getenv("STT_GEMINI_LOCATION")
    or DEFAULT_GOOGLE_LOCATION
).strip()
GOOGLE_APPLICATION_CREDENTIALS = resolve_credentials_path()

DASHBOARD_HOST = os.getenv("STT_DASHBOARD_HOST", "0.0.0.0").strip() or "0.0.0.0"
DASHBOARD_PORT = _get_int("STT_DASHBOARD_PORT", 5050)
ENABLE_RESET = _get_bool("STT_ENABLE_RESET", default=False)


def apply_google_credentials_env(credentials_path: Path | None = None) -> None:
    """Normalize GOOGLE_APPLICATION_CREDENTIALS to an absolute path if set."""
    path = credentials_path if credentials_path is not None else resolve_credentials_path()
    if path is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)


def validate_google_runtime() -> tuple[str | None, str, str]:
    """
    Validate non-network Google runtime configuration.

    A local credentials JSON is optional because production Linux hosts may use
    Application Default Credentials from an attached service account.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", GOOGLE_CLOUD_PROJECT).strip()
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("STT_GEMINI_LOCATION")
        or GOOGLE_CLOUD_LOCATION
    ).strip()
    credentials_path = resolve_credentials_path()

    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set.")
    if not location:
        raise RuntimeError("GOOGLE_CLOUD_LOCATION is not set.")
    if credentials_path is not None and not credentials_path.exists():
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is set but the file does not exist: "
            f"{credentials_path}"
        )

    apply_google_credentials_env(credentials_path)
    return str(credentials_path) if credentials_path is not None else None, project_id, location
