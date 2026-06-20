"""
gemini_flash_stt.py
Standalone reusable Sinhala/English/Tamil transcription module using Gemini.

Usage:
    python gemini_flash_stt.py input_audio/sample_call.mp3
    python gemini_flash_stt.py input_audio/sample_call.mp3 --save
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
# Media tools are resolved with shutil.which and invoked without a shell.
import subprocess  # nosec B404
import sys
import tempfile
import time
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from google import genai
from google.genai import types

from config import BASE_DIR, TRANSCRIPTIONS_DIR, app_now


# ── Settings — edit these ─────────────────────────────────────────────────────
MODEL_NAME       = "gemini-2.5-flash"   # ← change to switch models
DEFAULT_LOCATION = "us-central1"        # ← change to asia-south1 for Sri Lanka
LKR_RATE_FALLBACK = 316.0              # ← used if live exchange rate API is unreachable
STRIP_SILENCE    = True                # ← set False to disable silence stripping
PROVIDER = "google"
API_SURFACE = "vertex_ai"
COST_CALCULATION_VERSION = "estimated-v1"
PRICING_VERSION = "manual-2026-04"
PRICING_SOURCE = "configured pricing constants"

# ── Live USD/LKR exchange rate ────────────────────────────────────────────────
# Fetched once per hour from open.er-api.com (free, no API key needed).
# Falls back to LKR_RATE_FALLBACK if network is unavailable.
_lkr_rate_cache: float | None = None
_lkr_rate_fetched_at: float = 0.0          # epoch seconds
_LKR_CACHE_TTL = 3600                      # refresh every 60 minutes


def fetch_lkr_rate() -> float:
    """
    Return the current USD → LKR exchange rate.
    Result is cached for one hour so the API is not called on every transcription.
    Falls back to LKR_RATE_FALLBACK if the request fails.
    """
    import json

    global _lkr_rate_cache, _lkr_rate_fetched_at

    now = time.time()
    if _lkr_rate_cache is not None and (now - _lkr_rate_fetched_at) < _LKR_CACHE_TTL:
        return _lkr_rate_cache

    try:
        url = "https://open.er-api.com/v6/latest/USD"
        # This endpoint is a fixed HTTPS URL, never caller-controlled.
        with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
            data = json.loads(resp.read().decode())

        rate = float(data["rates"]["LKR"])
        _lkr_rate_cache    = rate
        _lkr_rate_fetched_at = now
        print(f"[info] Live USD/LKR rate fetched: {rate:.2f}")
        return rate

    except Exception as exc:
        if _lkr_rate_cache is not None:
            # Use stale cached value rather than falling back to hardcoded rate
            print(f"[warning] Exchange rate refresh failed ({exc}). Using cached rate: {_lkr_rate_cache:.2f}")
            return _lkr_rate_cache

        print(f"[warning] Exchange rate fetch failed ({exc}). Using fallback: {LKR_RATE_FALLBACK}")
        return LKR_RATE_FALLBACK


# Convenience alias for display/storage fallbacks. Live rate is fetched per call.
LKR_RATE = LKR_RATE_FALLBACK

# API retry settings (for transient network errors on server)
API_MAX_RETRIES  = 3
API_RETRY_DELAY  = 5   # seconds between retries

# ── Vertex AI Gemini pricing (USD per 1M tokens) — verified April 2026 ────────
# Source: GCP billing account CSV
# Flash models: audio_input > text_input (different rates — split matters)
# Pro models:   audio_input == text_input (same rate — split doesn't matter)
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash-lite": {"text_input": 0.10, "audio_input": 0.50,  "output": 0.40},
    "gemini-2.5-flash":      {"text_input": 0.30, "audio_input": 1.00,  "output": 2.50},
    "gemini-2.5-pro":        {"text_input": 1.25, "audio_input": 1.25,  "output": 10.00},
    "gemini-3-flash":        {"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3.0-flash":      {"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3-flash-preview":{"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3-pro":          {"text_input": 2.00, "audio_input": 2.00,  "output": 12.00},
    "gemini-3.0-pro":        {"text_input": 2.00, "audio_input": 2.00,  "output": 12.00},
    "gemini-3.1-flash-lite": {"text_input": 0.25, "audio_input": 0.80,  "output": 1.50},
    "gemini-3.1-flash":      {"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3.1-pro":        {"text_input": 2.00, "audio_input": 2.00,  "output": 12.00},
}

# 26 tok/sec = standard audio file upload (verified from real API responses)
# 258 tok/sec = Live API / video streaming only — NOT used here
_AUDIO_TOKENS_PER_SECOND = 26

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WHITESPACE = re.compile(r"\s+")


def get_model_pricing(model_name: str) -> dict[str, float]:
    """
    Return pricing for model_name. Longest-prefix match so
    'gemini-2.5-flash-lite' wins over 'gemini-2.5-flash'.
    """
    normalised = model_name.lower().strip()
    for key in sorted(_MODEL_PRICING, key=len, reverse=True):
        if normalised.startswith(key):
            return _MODEL_PRICING[key]

    print(
        f"[warning] No pricing found for '{model_name}'. "
        "Cost will show as $0. Add it to _MODEL_PRICING."
    )
    return {"text_input": 0.0, "audio_input": 0.0, "output": 0.0}


def _sanitize_filename_component(
    value: Any,
    fallback: str,
    max_len: int = 140,
) -> str:
    text = str(value or "").strip()
    text = _INVALID_FILENAME_CHARS.sub("-", text)
    text = _WHITESPACE.sub("-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip(" .-_")
    if not text:
        text = fallback
    return text[:max_len]


def _base_transcriptions_dir(output_root: str | Path | None = None) -> Path:
    load_env()
    raw_value = output_root or os.getenv("TRANSCRIPTIONS_DIR") or TRANSCRIPTIONS_DIR
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(resolved)


def resolve_transcript_path(stored_path: str) -> Path:
    """Resolve a stored relative or absolute transcript/metadata path."""
    path = Path(stored_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _unique_output_paths(date_dir: Path, stem: str) -> tuple[Path, Path]:
    counter = 1
    while True:
        suffix = "" if counter == 1 else f"-{counter}"
        txt_path = date_dir / f"{stem}{suffix}.txt"
        json_path = date_dir / f"{stem}{suffix}.json"
        if not txt_path.exists() and not json_path.exists():
            return txt_path, json_path
        counter += 1


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            if method_name == "model_dump":
                return _json_safe(method(mode="json", exclude_none=True), depth + 1)
            return _json_safe(method(), depth + 1)
        except TypeError:
            try:
                return _json_safe(method(), depth + 1)
            # Optional third-party serializers may fail; try the next fallback.
            except Exception:  # nosec B112
                continue
        # The object is optional metadata; continue to the next representation.
        except Exception:  # nosec B112
            continue

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        public_attrs = {
            key: item
            for key, item in attrs.items()
            if not key.startswith("_") and not callable(item)
        }
        if public_attrs:
            return _json_safe(public_attrs, depth + 1)

    return str(value)


def _configured_vertex_location() -> str:
    return os.getenv("STT_GEMINI_LOCATION", DEFAULT_LOCATION).strip() or DEFAULT_LOCATION


def _camel_case(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_serializable_or(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        safe_value = _json_safe(value)
        json.dumps(safe_value, ensure_ascii=False, allow_nan=False)
        return safe_value
    except Exception:
        return default


def _json_dict_or_empty(value: Any) -> dict[str, Any]:
    safe_value = _json_serializable_or(value, {})
    return safe_value if isinstance(safe_value, dict) else {}


def _usage_metadata_value(
    usage: Any,
    raw_usage: Mapping[str, Any],
    field_name: str,
) -> Any:
    try:
        value = getattr(usage, field_name, None) if usage is not None else None
    except Exception:
        value = None
    if value is not None:
        return value

    if field_name in raw_usage:
        return raw_usage[field_name]

    camel_name = _camel_case(field_name)
    if camel_name in raw_usage:
        return raw_usage[camel_name]

    return None


def _usage_metadata_to_dict(usage: Any) -> dict[str, Any]:
    return _json_dict_or_empty(usage)


def _metadata_from_result(
    result: dict[str, Any],
    audio_path: str | Path | None,
    txt_path: Path,
    json_path: Path,
    saved_dt: datetime,
    output_date: str,
) -> dict[str, Any]:
    source_audio_path = (
        str(audio_path)
        if audio_path is not None
        else str(result.get("audio_path", "") or "")
    )
    original_file_name = Path(source_audio_path).name if source_audio_path else "audio"
    duration = float(result.get("duration_seconds", 0) or 0)
    silence_removed = float(result.get("silence_removed_seconds", 0) or 0)
    original_duration = float(
        result.get("original_duration_seconds", duration + silence_removed) or 0
    )
    submitted_duration = float(
        result.get("submitted_duration_seconds", duration) or 0
    )
    silence_ratio = float(result.get("silence_removed_ratio", 0) or 0)
    if not silence_ratio and original_duration > 0:
        silence_ratio = silence_removed / original_duration

    metadata = {
        "original_file_name": original_file_name,
        "source_audio_path": source_audio_path,
        "transcript_txt_path": _relative_or_absolute(txt_path),
        "metadata_json_path": _relative_or_absolute(json_path),
        "transcript_file_path": _relative_or_absolute(txt_path),
        "transcribed_at": saved_dt.isoformat(),
        "transcript_saved_at": saved_dt.isoformat(),
        "transcript_output_date": output_date,
        "provider": result.get("provider", PROVIDER),
        "api_surface": result.get("api_surface", API_SURFACE),
        "vertex_location": result.get("vertex_location") or _configured_vertex_location(),
        "cost_calculation_version": result.get(
            "cost_calculation_version",
            COST_CALCULATION_VERSION,
        ),
        "pricing_version": result.get("pricing_version", PRICING_VERSION),
        "pricing_source": result.get("pricing_source", PRICING_SOURCE),
        "model": result.get("model", MODEL_NAME),
        "language": result.get("language", ""),
        "languages_detected": result.get("languages_detected", []),
        "duration_seconds": duration,
        "original_duration_seconds": original_duration,
        "submitted_duration_seconds": submitted_duration,
        "silence_removed_seconds": silence_removed,
        "silence_removed_ratio": silence_ratio,
        "status": result.get("status") or ("completed" if result.get("success", True) else "empty"),
        "database_save_status": result.get("database_save_status", ""),
        "input_tokens": result.get("input_tokens", 0),
        "audio_tokens": result.get("audio_tokens", 0),
        "text_input_tokens": result.get("text_input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "thoughts_tokens": result.get("thoughts_tokens", 0),
        "billed_output_tokens": result.get("billed_output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "raw_usage_metadata": result.get("raw_usage_metadata", {}),
        "prompt_tokens_details": result.get("prompt_tokens_details", []),
        "candidates_tokens_details": result.get("candidates_tokens_details", []),
        "cache_tokens_details": result.get("cache_tokens_details", []),
        "cached_content_token_count": result.get("cached_content_token_count", 0),
        "tool_use_prompt_token_count": result.get("tool_use_prompt_token_count", 0),
        "audio_input_cost_usd": result.get("audio_input_cost_usd", 0),
        "text_input_cost_usd": result.get("text_input_cost_usd", 0),
        "output_cost_usd": result.get("output_cost_usd", 0),
        "total_cost_usd": result.get("total_cost_usd", 0),
        "total_cost_lkr": result.get("total_cost_lkr", 0),
        "lkr_rate": result.get("lkr_rate", LKR_RATE),
    }
    if result.get("file_hash"):
        metadata["file_hash"] = result["file_hash"]
    if result.get("error_message"):
        metadata["error_message"] = result["error_message"]
    return {key: _json_safe(value) for key, value in metadata.items()}


def save_transcription_outputs(
    result: dict[str, Any],
    audio_path: str | Path | None = None,
    output_root: str | Path | None = None,
    saved_at: datetime | None = None,
) -> dict[str, Any]:
    """
    Save transcript TXT plus sidecar JSON metadata without using a database.

    TXT files are the primary output. JSON files store operational metadata
    beside the TXT. MySQL/database saving happens later and independently, so
    this function must work even when database storage is disabled or broken.
    """
    transcript = str(result.get("transcript", ""))
    if "transcript" not in result:
        raise RuntimeError("Cannot save transcription output: result has no transcript.")

    saved_dt = saved_at or app_now()
    output_date = saved_dt.strftime("%Y-%m-%d")
    timestamp = saved_dt.strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = _base_transcriptions_dir(output_root)
    date_dir = (base_dir / output_date).resolve()
    try:
        date_dir.relative_to(base_dir)
    except ValueError as exc:
        raise RuntimeError("Transcript output path escaped the configured root.") from exc

    date_dir.mkdir(parents=True, exist_ok=True)

    source = audio_path if audio_path is not None else result.get("audio_path", "audio")
    audio_stem = _sanitize_filename_component(Path(str(source)).stem, "audio")
    file_stem = f"{audio_stem}__transcribed_{timestamp}"
    txt_path, json_path = _unique_output_paths(date_dir, file_stem)

    try:
        txt_path.write_text(transcript, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to save transcript TXT: {exc}") from exc

    metadata = _metadata_from_result(
        result=result,
        audio_path=audio_path,
        txt_path=txt_path,
        json_path=json_path,
        saved_dt=saved_dt,
        output_date=output_date,
    )

    try:
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to save transcript metadata JSON: {exc}") from exc

    return {
        **metadata,
        "transcript_txt_path": _relative_or_absolute(txt_path),
        "metadata_json_path": _relative_or_absolute(json_path),
        "transcript_file_path": _relative_or_absolute(txt_path),
        "transcript_saved_at": saved_dt.isoformat(),
        "transcript_output_date": output_date,
        "original_file_name": metadata["original_file_name"],
        "status": metadata["status"],
    }


# Prevent repeated client creation in the same Python process
_GENAI_CLIENT: genai.Client | None = None


def load_env() -> None:
    """Load .env file from project root. Existing env vars are preserved."""
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"

    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    # Resolve relative credentials path
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if creds and not os.path.isabs(creds):
        resolved = (base_dir / creds).resolve()
        if resolved.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)


def validate_setup() -> tuple[str, str, str]:
    """Validate required env config. Returns (credentials_path, project_id, location)."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    project_id       = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location         = os.getenv("STT_GEMINI_LOCATION", DEFAULT_LOCATION).strip()

    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    if not os.path.exists(credentials_path):
        raise RuntimeError(f"Credentials file not found: {credentials_path}")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set.")

    return credentials_path, project_id, location


def get_genai_client(project_id: str, location: str) -> genai.Client:
    """Create the Google Gen AI client once per process."""
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT
    _GENAI_CLIENT = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )
    return _GENAI_CLIENT


def ensure_ffmpeg_available() -> dict[str, str]:
    """Return absolute paths for working ffmpeg and ffprobe executables."""
    resolved_binaries: dict[str, str] = {}
    for binary in ("ffmpeg", "ffprobe"):
        resolved = shutil.which(binary)
        if not resolved:
            raise RuntimeError(
                f"{binary} is not installed or not on PATH. "
                "Install with: sudo apt install ffmpeg"
            )
        resolved = str(Path(resolved).resolve())
        try:
            # Absolute executable, fixed argument list, and no command shell.
            result = subprocess.run(  # nosec B603
                [resolved, "-version"],
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"{binary} is not installed or not on PATH. "
                "Install with: sudo apt install ffmpeg"
            )
        if result.returncode != 0:
            raise RuntimeError(f"{binary} is installed but not working correctly.")
        resolved_binaries[binary] = resolved
    return resolved_binaries


def _get_wav_duration(wav_path: Path) -> float:
    """Return duration in seconds of a WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def load_audio_as_wav(
    file_path: str,
    strip_silence: bool | None = None,
) -> tuple[bytes, float, float]:
    """
    Convert audio file to 16kHz mono WAV. Optionally strip silence.

    Returns:
        (wav_bytes, duration_after_stripping_seconds, silence_removed_seconds)
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    media_binaries = ensure_ffmpeg_available()
    ffmpeg_path = media_binaries["ffmpeg"]
    do_strip = STRIP_SILENCE if strip_silence is None else strip_silence

    # Use TemporaryDirectory — safer than mktemp, auto-cleaned even on crash
    with tempfile.TemporaryDirectory(prefix="slt_stt_") as tmp_dir:
        tmp_raw   = Path(tmp_dir) / "raw.wav"
        tmp_final = Path(tmp_dir) / "final.wav"

        # Step 1 — convert to 16kHz mono WAV
        # Absolute executable, argument list, and shell=False prevent injection.
        result = subprocess.run(  # nosec B603
            [ffmpeg_path, "-y", "-i", str(input_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp_raw)],
            capture_output=True, text=True, check=False, shell=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")

        original_duration = _get_wav_duration(tmp_raw)

        if do_strip:
            # Step 2 — strip silence (start, end, interior gaps > 0.5s)
            silence_filter = (
                "silenceremove="
                "start_periods=1:start_duration=0.3:start_threshold=-40dB:"
                "stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB"
            )
            # Reuse the same resolved executable and shell-free argument list.
            result2 = subprocess.run(  # nosec B603
                [ffmpeg_path, "-y", "-i", str(tmp_raw),
                 "-af", silence_filter, "-ar", "16000", "-ac", "1",
                 "-f", "wav", str(tmp_final)],
                capture_output=True, text=True, check=False, shell=False,
            )
            if result2.returncode != 0:
                # Fallback: use unstripped audio
                stripped_duration = original_duration
                wav_bytes = tmp_raw.read_bytes()
            else:
                stripped_duration = _get_wav_duration(tmp_final)
                wav_bytes = tmp_final.read_bytes()
        else:
            stripped_duration = original_duration
            wav_bytes = tmp_raw.read_bytes()

        silence_removed = max(0.0, original_duration - stripped_duration)
        return wav_bytes, stripped_duration, silence_removed


def _parse_languages(raw_text: str) -> tuple[str, list[str]]:
    """
    Strip LANGUAGES: footer and return (clean_transcript, ["Sinhala", ...]).
    Falls back to inferring from [SI]/[EN]/[TA] tags if footer is missing.
    """
    lines = raw_text.strip().splitlines()
    languages: list[str] = []
    clean_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("LANGUAGES:"):
            for lang in stripped[len("LANGUAGES:"):].strip().split(","):
                lang = lang.strip().title()
                if lang and lang not in languages:
                    languages.append(lang)
        else:
            clean_lines.append(line)

    if not languages:
        tag_map = {"[SI]": "Sinhala", "[EN]": "English", "[TA]": "Tamil"}
        for tag, name in tag_map.items():
            if tag in raw_text and name not in languages:
                languages.append(name)

    return "\n".join(clean_lines).strip(), languages


def transcribe_wav_bytes(
    client: genai.Client,
    wav_bytes: bytes,
    duration_seconds: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    """
    Transcribe WAV audio bytes using Gemini with automatic retry on failure.

    Each transcript line is prefixed: [SI] Sinhala  [EN] English  [TA] Tamil

    Returns:
        (transcript_text, usage_info)
    """
    prompt = (
        "You are a professional transcriptionist for a Sri Lankan telecom call center.\n"
        "Transcribe this phone call audio exactly as spoken.\n"
        "The call may contain Sinhala, English, and/or Tamil.\n\n"
        "Rules:\n"
        "- Prefix EVERY line or utterance with its language tag:\n"
        "    [SI] for Sinhala   [EN] for English   [TA] for Tamil\n"
        "- Keep Sinhala in Sinhala Unicode script (e.g., ආයුබෝවන්)\n"
        "- Keep English words in English when spoken in English\n"
        "- Keep Tamil in Tamil Unicode script (e.g., வணக்கம்)\n"
        "- Do NOT translate between languages\n"
        "- Do NOT summarize\n"
        "- Do NOT add speaker labels or explanations beyond the language tag\n"
        "- If a part is unclear, transcribe only what is reasonably audible\n"
        "- If the audio is silent or contains no speech, output nothing\n"
        "- On the very last line write exactly: "
        "LANGUAGES: <comma-separated list of languages detected>"
    )

    audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")

    # Retry loop — handles transient network/API errors on production server
    last_error: Exception | None = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[audio_part, prompt],
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt < API_MAX_RETRIES:
                print(f"  API attempt {attempt} failed: {exc}. "
                      f"Retrying in {API_RETRY_DELAY}s...")
                time.sleep(API_RETRY_DELAY)
    else:
        raise RuntimeError(
            f"API failed after {API_MAX_RETRIES} attempts: {last_error}"
        )

    raw_text = (getattr(response, "text", "") or "").strip()
    transcript, languages_detected = _parse_languages(raw_text)

    usage = getattr(response, "usage_metadata", None)
    raw_usage_metadata = _usage_metadata_to_dict(usage)
    input_tokens = _safe_int(
        _usage_metadata_value(usage, raw_usage_metadata, "prompt_token_count")
    )
    output_tokens = _safe_int(
        _usage_metadata_value(usage, raw_usage_metadata, "candidates_token_count")
    )
    thoughts_tokens = _safe_int(
        _usage_metadata_value(usage, raw_usage_metadata, "thoughts_token_count")
    )
    total_tokens = _safe_int(
        _usage_metadata_value(usage, raw_usage_metadata, "total_token_count")
    )
    cached_content_token_count = _safe_int(
        _usage_metadata_value(
            usage,
            raw_usage_metadata,
            "cached_content_token_count",
        )
    )
    tool_use_prompt_token_count = _safe_int(
        _usage_metadata_value(
            usage,
            raw_usage_metadata,
            "tool_use_prompt_token_count",
        )
    )
    prompt_tokens_details = _json_serializable_or(
        _usage_metadata_value(usage, raw_usage_metadata, "prompt_tokens_details"),
        [],
    )
    candidates_tokens_details = _json_serializable_or(
        _usage_metadata_value(usage, raw_usage_metadata, "candidates_tokens_details"),
        [],
    )
    cache_tokens_details = _json_serializable_or(
        _usage_metadata_value(usage, raw_usage_metadata, "cache_tokens_details"),
        [],
    )

    pricing = get_model_pricing(MODEL_NAME)
    p_text  = pricing["text_input"]
    p_audio = pricing["audio_input"]
    p_out   = pricing["output"]

    # Split input tokens into audio vs text
    # Google returns one combined prompt_token_count (audio + text together).
    # We estimate audio portion using 26 tok/sec (standard file upload rate).
    # 258/sec is only for Live API streaming — not used here.
    estimated_audio_tokens = min(int(duration_seconds * _AUDIO_TOKENS_PER_SECOND), input_tokens)
    text_input_tokens      = max(input_tokens - estimated_audio_tokens, 0)
    actual_tok_per_sec     = (input_tokens / duration_seconds) if duration_seconds > 0 else 0

    # Google bills thinking tokens as output tokens
    billed_output_tokens = output_tokens + thoughts_tokens

    # Cost breakdown (USD)
    audio_input_cost = (estimated_audio_tokens / 1_000_000) * p_audio
    text_input_cost  = (text_input_tokens      / 1_000_000) * p_text
    input_cost       = audio_input_cost + text_input_cost
    output_cost      = (billed_output_tokens   / 1_000_000) * p_out
    total_cost       = input_cost + output_cost

    return transcript, {
        "input_tokens":         input_tokens,
        "audio_tokens":         estimated_audio_tokens,
        "text_input_tokens":    text_input_tokens,
        "output_tokens":        output_tokens,          # response text only
        "thoughts_tokens":      thoughts_tokens,
        "billed_output_tokens": billed_output_tokens,   # what Google charges
        "total_tokens":         total_tokens,
        "raw_usage_metadata":   raw_usage_metadata,
        "prompt_tokens_details": prompt_tokens_details,
        "candidates_tokens_details": candidates_tokens_details,
        "cache_tokens_details": cache_tokens_details,
        "cached_content_token_count": cached_content_token_count,
        "tool_use_prompt_token_count": tool_use_prompt_token_count,
        "actual_tok_per_sec":   actual_tok_per_sec,
        "audio_input_cost_usd": audio_input_cost,
        "text_input_cost_usd":  text_input_cost,
        "input_cost_usd":       input_cost,
        "output_cost_usd":      output_cost,
        "total_cost_usd":       total_cost,
        "provider":             PROVIDER,
        "api_surface":          API_SURFACE,
        "cost_calculation_version": COST_CALCULATION_VERSION,
        "pricing_version":      PRICING_VERSION,
        "pricing_source":       PRICING_SOURCE,
        "languages_detected":   languages_detected,
    }


def transcribe_audio_file(audio_path: str) -> dict[str, Any]:
    """
    Main entry point for watcher.py and archived utility scripts.

    Args:
        audio_path: Path to input audio file (mp3, wav, m4a, ogg, flac, etc.)

    Returns dict with transcript, tokens, costs, languages, durations.
    """
    load_env()
    _, project_id, location = validate_setup()
    client = get_genai_client(project_id, location)

    wav_bytes, duration_seconds, silence_removed_seconds = load_audio_as_wav(audio_path)

    start_time = time.time()
    transcript, usage = transcribe_wav_bytes(client, wav_bytes, duration_seconds)
    elapsed_seconds = time.time() - start_time

    # Fetch live exchange rate at transcription time (cached for 1 hour)
    lkr_rate = fetch_lkr_rate()

    return {
        "transcript":              transcript,
        "model":                   MODEL_NAME,
        "vertex_location":         location,
        "audio_path":              str(Path(audio_path).resolve()),
        "duration_seconds":        duration_seconds,
        "silence_removed_seconds": silence_removed_seconds,
        "elapsed_seconds":         elapsed_seconds,
        "success":                 bool(transcript),
        "lkr_rate":                lkr_rate,
        "total_cost_lkr":          usage.get("total_cost_usd", 0.0) * lkr_rate,
        **usage,
    }


def save_transcript(audio_path: str, transcript: str, output_dir: str = "output") -> str:
    """
    Save transcript to TXT/JSON output files and return the TXT path.

    The output_dir argument is kept for compatibility. The default now follows
    TRANSCRIPTIONS_DIR; a custom output_dir gets the same dated subfolder layout
    under that base directory.
    """
    custom_output_dir = None if output_dir == "output" else output_dir
    saved_info = save_transcription_outputs(
        result={
            "transcript": transcript,
            "model": MODEL_NAME,
            "mode": "manual",
            "status": "completed" if transcript else "empty",
        },
        audio_path=audio_path,
        output_root=custom_output_dir,
    )
    return str(resolve_transcript_path(saved_info["transcript_txt_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="SLT Gemini transcription tool")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--save", action="store_true",
                        help="Save transcript to the dated transcript output folder")
    parser.add_argument(
        "--print-transcript",
        action="store_true",
        help="Print full transcript text to the console",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"SLT Transcription  |  {MODEL_NAME}")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    try:
        result = transcribe_audio_file(args.audio)

        print(f"Model      : {result['model']}")
        print(f"Audio time : {result['duration_seconds']:.1f}s  "
              f"(silence removed: {result.get('silence_removed_seconds', 0):.1f}s)")
        print(f"Languages  : {', '.join(result.get('languages_detected', [])) or 'unknown'}")
        print(f"Run time   : {result['elapsed_seconds']:.1f}s")
        print("-" * 60)

        tok_per_sec = result.get("actual_tok_per_sec", 0)
        thoughts    = result.get("thoughts_tokens", 0)
        billed_out  = result.get("billed_output_tokens", result["output_tokens"])

        print(f"Tokens (total)  : {result['total_tokens']:,}")
        print(f"  Input tokens  : {result['input_tokens']:,}  "
              f"(actual rate: {tok_per_sec:.1f} tok/s)")
        print(f"    Audio input : ~{result['audio_tokens']:,}  (estimated from duration)")
        print(f"    Text input  : ~{result['text_input_tokens']:,}  (prompt)")
        print(f"  Output tokens : {result['output_tokens']:,}  (response text)")
        if thoughts:
            print(f"  Thinking tok  : {thoughts:,}  (billed as output by Google)")
            print(f"  Billed output : {billed_out:,}  (response + thinking)")
        print("-" * 60)

        _p   = get_model_pricing(result['model'])
        _usd = result['total_cost_usd']
        _ai  = result['audio_input_cost_usd']
        _ti  = result['text_input_cost_usd']
        _oc  = result['output_cost_usd']
        display_lkr_rate = float(
            result.get("lkr_rate", LKR_RATE_FALLBACK) or LKR_RATE_FALLBACK
        )

        print(f"Pricing : audio=${_p['audio_input']}/1M  "
              f"text=${_p['text_input']}/1M  "
              f"output=${_p['output']}/1M  [{result['model']}]")
        print("-" * 60)
        print(
            f"  {'Component':<18} {'USD':>12}   "
            f"{'LKR (x' + str(int(display_lkr_rate)) + ')':>12}"
        )
        print(f"  {'-'*44}")
        print(
            f"  {'Audio input':<18} ${_ai:>11.6f}   "
            f"Rs.{_ai * display_lkr_rate:>9.4f}"
        )
        print(
            f"  {'Text input':<18} ${_ti:>11.6f}   "
            f"Rs.{_ti * display_lkr_rate:>9.4f}"
        )
        oc_note = f"  (resp {result['output_tokens']:,} + thinking {thoughts:,})" if thoughts else ""
        print(
            f"  {'Output':<18} ${_oc:>11.6f}   "
            f"Rs.{_oc * display_lkr_rate:>9.4f}{oc_note}"
        )
        print(f"  {'-'*44}")
        print(
            f"  {'TOTAL':<18} ${_usd:>11.6f}   "
            f"Rs.{_usd * display_lkr_rate:>9.4f}"
        )
        print("=" * 60)

        if result["success"] and args.print_transcript:
            print("\nTRANSCRIPT:\n")
            print(result["transcript"])
        elif result["success"]:
            print("\nTranscript returned. Full text is not printed by default.")
        else:
            print("\nNo transcript returned.")

        if args.save and result["transcript"]:
            saved_path = save_transcript(args.audio, result["transcript"])
            print(f"\nSaved transcript to: {saved_path}")

    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\nSETUP ERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
