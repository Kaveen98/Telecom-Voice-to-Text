"""
gemini_flash_stt.py
Standalone reusable Sinhala/English transcription module using Gemini 2.5 Flash.

Usage:
    python gemini_flash_stt.py input_audio/sample_call.mp3
    python gemini_flash_stt.py input_audio/sample_call.mp3 --save
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from config import (
    DEFAULT_GOOGLE_LOCATION,
    LKR_RATE,
    MODEL_NAME,
    OUTPUT_DIR,
    SILENCE_START_DURATION,
    SILENCE_START_THRESHOLD_DB,
    SILENCE_STOP_DURATION,
    SILENCE_STOP_THRESHOLD_DB,
    STRIP_SILENCE,
    apply_google_credentials_env,
    get_effective_preprocess_profile,
    load_env,
    validate_google_runtime,
)

DEFAULT_LOCATION = DEFAULT_GOOGLE_LOCATION

# ── Vertex AI Gemini pricing table (USD per 1M tokens) — updated April 2026 ──
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
#
# Keys are model name prefixes (matched with str.startswith).
# audio_input: Vertex AI bills audio at 258 tokens/sec at a higher rate.
# thinking tokens on Pro models are billed as output tokens.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    # ── Gemini 2.5 ────────────────────────────────────────────────────────
    # Prices verified against your GCP billing account CSV — April 2026
    # Flash models: audio_input > text_input (audio costs more)
    # Pro models:   audio_input == text_input (same rate)
    "gemini-2.5-flash-lite": {"text_input": 0.10, "audio_input": 0.50,  "output": 0.40},
    "gemini-2.5-flash":      {"text_input": 0.30, "audio_input": 1.00,  "output": 2.50},
    "gemini-2.5-pro":        {"text_input": 1.25, "audio_input": 1.25,  "output": 10.00},  # fixed: was 3.00
    # ── Gemini 3 / 3.0 ────────────────────────────────────────────────────
    "gemini-3-flash":        {"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3.0-flash":      {"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3-flash-preview":{"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3-pro":          {"text_input": 2.00, "audio_input": 2.00,  "output": 12.00},  # fixed: was 4.00
    "gemini-3.0-pro":        {"text_input": 2.00, "audio_input": 2.00,  "output": 12.00},  # fixed: was 4.00
    # ── Gemini 3.1 ────────────────────────────────────────────────────────
    "gemini-3.1-flash-lite": {"text_input": 0.25, "audio_input": 0.80,  "output": 1.50},
    "gemini-3.1-flash":      {"text_input": 0.50, "audio_input": 1.00,  "output": 3.00},
    "gemini-3.1-pro":        {"text_input": 2.00, "audio_input": 2.00,  "output": 12.00},
}

# Audio token rate used by Vertex AI for billing
_AUDIO_TOKENS_PER_SECOND = 258


@dataclass(frozen=True)
class AudioPreprocessResult:
    wav_bytes: bytes
    original_duration_seconds: float
    submitted_duration_seconds: float
    silence_removed_seconds: float
    silence_removed_ratio: float
    strip_silence_enabled: bool
    preprocess_profile: str
    silence_filter: str


def get_model_pricing(model_name: str) -> dict[str, float]:
    """
    Return the pricing dict for *model_name*.

    Matching is done longest-prefix-first so that e.g. 'gemini-2.5-flash-lite'
    is preferred over 'gemini-2.5-flash'.
    Falls back to a warning and zero prices if the model is unknown.
    """
    normalised = model_name.lower().strip()
    # Sort keys by length descending so longer (more specific) keys win
    for key in sorted(_MODEL_PRICING, key=len, reverse=True):
        if normalised.startswith(key):
            return _MODEL_PRICING[key]

    print(
        f"⚠  WARNING: No pricing found for model '{model_name}'. "
        "Cost will show as $0.000000. "
        "Add it to _MODEL_PRICING in the script."
    )
    return {"text_input": 0.0, "audio_input": 0.0, "output": 0.0}


# Prevent repeated client creation in the same Python process
_GENAI_CLIENT: genai.Client | None = None


def validate_setup() -> tuple[str | None, str, str]:
    """
    Validate required environment configuration.
    Returns:
        (credentials_path, project_id, location)
    """
    load_env()
    apply_google_credentials_env()
    return validate_google_runtime()


def get_genai_client(project_id: str, location: str) -> genai.Client:
    """
    Create the Google Gen AI client once per process.
    """
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


def ensure_media_tools_available() -> None:
    """Ensure ffmpeg and ffprobe are installed and available on PATH."""
    missing = [
        binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing required media tool(s) on PATH: " + ", ".join(missing)
        )

    for binary in ("ffmpeg", "ffprobe"):
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{binary} is installed but not working correctly.")


def ensure_ffmpeg_available() -> None:
    """
    Ensure ffmpeg/ffprobe are installed and available on PATH.

    The function name is kept for backward compatibility with existing callers.
    """
    ensure_media_tools_available()


def _get_wav_duration(wav_path: Path) -> float:
    """Return duration in seconds of a WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _format_silence_db(value: float) -> str:
    return f"{value:g}dB"


def _format_filter_number(value: float) -> str:
    return f"{value:g}"


def build_silence_filter() -> str:
    """Build the configured FFmpeg silenceremove filter."""
    return (
        "silenceremove="
        f"start_periods=1:"
        f"start_duration={_format_filter_number(SILENCE_START_DURATION)}:"
        f"start_threshold={_format_silence_db(SILENCE_START_THRESHOLD_DB)}:"
        f"stop_periods=-1:"
        f"stop_duration={_format_filter_number(SILENCE_STOP_DURATION)}:"
        f"stop_threshold={_format_silence_db(SILENCE_STOP_THRESHOLD_DB)}"
    )


def load_audio_as_wav_with_metadata(
    file_path: str,
    strip_silence: bool | None = None,
) -> AudioPreprocessResult:
    """
    Convert an input audio file to 16kHz mono WAV using ffmpeg.
    Optionally strip leading/trailing/interior silence to reduce token cost.

    Silence billing fact: Vertex AI bills audio at 258 tokens/sec for the
    FULL duration — silence included. Stripping silence before upload
    directly reduces the token count and therefore the API cost.

    Args:
        file_path:      Path to input audio file.
        strip_silence:  If True, remove silence segments with ffmpeg before
                        sending to the API. Recommended for call center audio
                        where hold music and silence are common.

    Returns:
        AudioPreprocessResult with WAV bytes and duration/silence metadata.
    """
    input_path = Path(file_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ensure_ffmpeg_available()
    strip_enabled = STRIP_SILENCE if strip_silence is None else strip_silence
    silence_filter = build_silence_filter() if strip_enabled else ""

    with tempfile.TemporaryDirectory(prefix="slt_stt_") as tmp_dir:
        tmp_raw = Path(tmp_dir) / "raw.wav"
        tmp_final = Path(tmp_dir) / "final.wav"

        # Step 1: Convert to 16kHz mono WAV
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp_raw)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")

        original_duration = _get_wav_duration(tmp_raw)

        if strip_enabled:
            # Step 2: Strip silence. stop_periods=-1 preserves the current
            # behavior of removing interior silence as well as edge silence.
            result2 = subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_raw),
                 "-af", silence_filter, "-ar", "16000", "-ac", "1",
                 "-f", "wav", str(tmp_final)],
                capture_output=True, text=True, check=False,
            )
            if result2.returncode != 0:
                # Fall back to unstripped audio if filter fails
                tmp_final = tmp_raw
                stripped_duration = original_duration
            else:
                stripped_duration = _get_wav_duration(tmp_final)
        else:
            tmp_final = tmp_raw
            stripped_duration = original_duration

        silence_removed = max(0.0, original_duration - stripped_duration)
        silence_removed_ratio = (
            silence_removed / original_duration if original_duration > 0 else 0.0
        )

        with tmp_final.open("rb") as f:
            wav_bytes = f.read()

        return AudioPreprocessResult(
            wav_bytes=wav_bytes,
            original_duration_seconds=original_duration,
            submitted_duration_seconds=stripped_duration,
            silence_removed_seconds=silence_removed,
            silence_removed_ratio=silence_removed_ratio,
            strip_silence_enabled=strip_enabled,
            preprocess_profile=get_effective_preprocess_profile(strip_enabled),
            silence_filter=silence_filter,
        )


def load_audio_as_wav(
    file_path: str,
    strip_silence: bool | None = None,
) -> tuple[bytes, float, float]:
    """
    Backward-compatible wrapper returning the historical tuple:
    (wav_bytes, submitted_duration_seconds, silence_removed_seconds).
    """
    result = load_audio_as_wav_with_metadata(file_path, strip_silence)
    return (
        result.wav_bytes,
        result.submitted_duration_seconds,
        result.silence_removed_seconds,
    )


def _parse_languages(raw_text: str) -> tuple[str, list[str]]:
    """
    Strip the LANGUAGES: footer line from the transcript and return
    (clean_transcript, ["Sinhala", "English", ...]).
    """
    lines = raw_text.strip().splitlines()
    languages: list[str] = []
    clean_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("LANGUAGES:"):
            lang_part = stripped[len("LANGUAGES:"):].strip()
            for lang in lang_part.split(","):
                lang = lang.strip().title()
                if lang and lang not in languages:
                    languages.append(lang)
        else:
            clean_lines.append(line)

    # Also infer languages from inline tags [SI] [EN] [TA] if footer missing
    if not languages:
        tag_map = {"[SI]": "Sinhala", "[EN]": "English", "[TA]": "Tamil"}
        for tag, name in tag_map.items():
            if tag in raw_text and name not in languages:
                languages.append(name)

    return "\n".join(clean_lines).strip(), languages


def _provider_usage_to_json(usage: Any) -> str:
    """Serialize provider usage metadata when the SDK exposes it."""
    if usage is None:
        return ""

    data: Any
    if hasattr(usage, "model_dump"):
        try:
            data = usage.model_dump(mode="json", exclude_none=True)
        except TypeError:
            data = usage.model_dump()
    elif hasattr(usage, "to_json_dict"):
        data = usage.to_json_dict()
    elif hasattr(usage, "to_dict"):
        data = usage.to_dict()
    else:
        data = {
            "prompt_token_count": getattr(usage, "prompt_token_count", 0) or 0,
            "candidates_token_count": getattr(usage, "candidates_token_count", 0) or 0,
            "thoughts_token_count": getattr(usage, "thoughts_token_count", 0) or 0,
            "total_token_count": getattr(usage, "total_token_count", 0) or 0,
        }

    return json.dumps(data, ensure_ascii=False, default=str)


def transcribe_wav_bytes(
    client: genai.Client,
    wav_bytes: bytes,
    duration_seconds: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    """
    Transcribe WAV audio bytes using Gemini.

    Each line of the transcript is prefixed with a language tag:
      [SI] = Sinhala   [EN] = English   [TA] = Tamil

    Args:
        client:           Authenticated Gen AI client.
        wav_bytes:        Raw WAV audio data.
        duration_seconds: Audio duration used to split audio vs text token costs.

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
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[audio_part, prompt],
    )

    raw_text = (getattr(response, "text", "") or "").strip()
    transcript, languages_detected = _parse_languages(raw_text)

    usage = getattr(response, "usage_metadata", None)
    input_tokens    = getattr(usage, "prompt_token_count",     0) or 0
    output_tokens   = getattr(usage, "candidates_token_count", 0) or 0
    thoughts_tokens = getattr(usage, "thoughts_token_count",  0) or 0
    total_tokens    = getattr(usage, "total_token_count",      0) or 0
    provider_usage_json = _provider_usage_to_json(usage)

    # Look up live pricing for whichever model is active
    pricing = get_model_pricing(MODEL_NAME)
    p_text  = pricing["text_input"]
    p_audio = pricing["audio_input"]
    p_out   = pricing["output"]

    # ── Input token split ─────────────────────────────────────────────────
    # Google returns one combined prompt_token_count that covers both audio
    # and the text prompt.  We estimate the audio portion from the duration
    # using Vertex AI's documented rate for standard audio file upload
    # (~25-30 tok/sec empirically; 258/sec is the Live-API / video rate).
    # We cap at actual input_tokens so we never over-attribute to audio.
    estimated_audio_tokens = min(int(duration_seconds * _AUDIO_TOKENS_PER_SECOND), input_tokens)
    text_input_tokens      = max(input_tokens - estimated_audio_tokens, 0)

    # Actual tok/sec observed from the real API response (for display only)
    actual_tok_per_sec = (input_tokens / duration_seconds) if duration_seconds > 0 else 0

    # ── Output token billing (FIX: thinking tokens are billed as output) ──
    # Google bills thinking tokens at the same output rate as response tokens.
    # candidates_token_count = response text only (NOT including thinking).
    # thoughts_token_count   = internal reasoning tokens (billed as output).
    billed_output_tokens = output_tokens + thoughts_tokens

    # ── Cost breakdown (USD) ──────────────────────────────────────────────
    audio_input_cost = (estimated_audio_tokens / 1_000_000) * p_audio
    text_input_cost  = (text_input_tokens      / 1_000_000) * p_text
    input_cost       = audio_input_cost + text_input_cost
    output_cost      = (billed_output_tokens   / 1_000_000) * p_out
    total_cost       = input_cost + output_cost

    usage_info = {
        "input_tokens":          input_tokens,
        "audio_tokens":          estimated_audio_tokens,
        "text_input_tokens":     text_input_tokens,
        "output_tokens":         output_tokens,         # response text only
        "thoughts_tokens":       thoughts_tokens,
        "billed_output_tokens":  billed_output_tokens,  # what Google actually charges
        "total_tokens":          total_tokens,
        "provider_usage_json":    provider_usage_json,
        "provider_input_tokens":  input_tokens,
        "provider_output_tokens": output_tokens,
        "provider_thoughts_tokens": thoughts_tokens,
        "provider_total_tokens":  total_tokens,
        "estimated_audio_tokens": estimated_audio_tokens,
        "actual_tok_per_sec":    actual_tok_per_sec,
        "audio_input_cost_usd":  audio_input_cost,
        "text_input_cost_usd":   text_input_cost,
        "input_cost_usd":        input_cost,
        "output_cost_usd":       output_cost,
        "total_cost_usd":        total_cost,
        "pricing_source":        "local_table",
        "pricing_model":         MODEL_NAME,
        "languages_detected":    languages_detected,
    }

    return transcript, usage_info


def transcribe_audio_file(
    audio_path: str,
    strip_silence: bool | None = None,
) -> dict[str, Any]:
    """
    Reusable function for other systems.

    Args:
        audio_path:     Path to an input audio file (mp3, wav, m4a, etc.).
        strip_silence:  Optional per-call override. None keeps the configured
                        production default.

    Returns:
        {
            "transcript": str,
            "model": str,
            "audio_path": str,
            "duration_seconds": float,
            "silence_removed_seconds": float,
            "elapsed_seconds": float,
            "success": bool,
            "languages_detected": list[str],
            "input_tokens": int,
            "audio_tokens": int,
            "text_input_tokens": int,
            "output_tokens": int,
            "thoughts_tokens": int,
            "total_tokens": int,
            "audio_input_cost_usd": float,
            "text_input_cost_usd": float,
            "input_cost_usd": float,
            "output_cost_usd": float,
            "total_cost_usd": float,
        }
    """
    load_env()
    _, project_id, location = validate_setup()
    client = get_genai_client(project_id, location)

    audio = load_audio_as_wav_with_metadata(audio_path, strip_silence)

    start_time = time.time()
    transcript, usage = transcribe_wav_bytes(
        client,
        audio.wav_bytes,
        audio.submitted_duration_seconds,
    )
    elapsed_seconds = time.time() - start_time

    return {
        "transcript": transcript,
        "model": MODEL_NAME,
        "audio_path": str(Path(audio_path).resolve()),
        # Backward compatibility: duration_seconds remains submitted duration.
        "duration_seconds": audio.submitted_duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "success": bool(transcript),
        **{
            key: value
            for key, value in asdict(audio).items()
            if key != "wav_bytes"
        },
        **usage,
    }


def save_transcript(
    audio_path: str,
    transcript: str,
    output_dir: str | Path | None = None,
) -> str:
    """
    Save transcript into the output directory.
    Model name is included in the filename so switching models never
    overwrites a previous transcript for the same audio file.
    e.g. call_gemini-2.5-flash_transcript.txt
         call_gemini-2.5-pro_transcript.txt
    """
    input_path  = Path(audio_path)
    output_path = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    model_slug      = MODEL_NAME.replace("/", "-")
    transcript_file = output_path / f"{input_path.stem}_{model_slug}_transcript.txt"
    transcript_file.write_text(transcript, encoding="utf-8")

    return str(transcript_file.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini 2.5 Flash transcription tool"
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save transcript to the output/ folder",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Gemini 2.5 Flash STT")
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
        # ── Token breakdown ───────────────────────────────────────────────
        tok_per_sec = result.get("actual_tok_per_sec", 0)
        thoughts    = result.get("thoughts_tokens", 0)
        billed_out  = result.get("billed_output_tokens", result["output_tokens"])
        print(f"Tokens (total)  : {result['total_tokens']:,}")
        print(f"  Input tokens  : {result['input_tokens']:,}  "
              f"(actual rate: {tok_per_sec:.1f} tok/s for this audio)")
        print(f"    Audio input : ~{result['audio_tokens']:,}  (estimated from duration)")
        print(f"    Text input  : ~{result['text_input_tokens']:,}  (prompt)")
        print(f"  Output tokens : {result['output_tokens']:,}  (response text)")
        if thoughts:
            print(f"  Thinking tok  : {thoughts:,}  ← also billed as output by Google")
            print(f"  Billed output : {billed_out:,}  (response + thinking)")
        print("-" * 60)
        # ── Cost breakdown — rates auto-selected from MODEL_NAME ──────────
        _p   = get_model_pricing(result['model'])
        _usd = result['total_cost_usd']
        _ai  = result['audio_input_cost_usd']
        _ti  = result['text_input_cost_usd']
        _oc  = result['output_cost_usd']
        print(f"Pricing used    : audio=${_p['audio_input']}/1M  "
              f"text=${_p['text_input']}/1M  "
              f"output=${_p['output']}/1M  [{result['model']}]")
        print(f"{'─'*60}")
        print(f"  {'Component':<18} {'USD':>12}   {'LKR (×' + str(int(LKR_RATE)) + ')':>12}")
        print(f"  {'─'*44}")
        print(f"  {'Audio input':<18} ${_ai:>11.6f}   Rs.{_ai * LKR_RATE:>9.4f}")
        print(f"  {'Text input':<18} ${_ti:>11.6f}   Rs.{_ti * LKR_RATE:>9.4f}")
        oc_note = f"  (resp {result['output_tokens']:,} + thinking {thoughts:,})" if thoughts else ""
        print(f"  {'Output':<18} ${_oc:>11.6f}   Rs.{_oc * LKR_RATE:>9.4f}{oc_note}")
        print(f"  {'─'*44}")
        print(f"  {'TOTAL':<18} ${_usd:>11.6f}   Rs.{_usd * LKR_RATE:>9.4f}")
        print("=" * 60)

        if result["success"]:
            print("\nTRANSCRIPT:\n")
            print(result["transcript"])
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
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
