"""
gemini_flash_stt.py
Standalone reusable Sinhala/English transcription module using Gemini 2.5 Flash.

Usage:
    python gemini_flash_stt.py input_audio/sample_call.mp3
    python gemini_flash_stt.py input_audio/sample_call.mp3 --save
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_NAME = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"

# Prevent repeated vertex initialization in the same Python process
_VERTEX_INITIALIZED = False


def load_env() -> None:
    """
    Load environment variables from a local .env file in the project root.
    Existing environment variables are preserved.
    """
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
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            os.environ.setdefault(key, value)

    # Resolve relative credentials path against project root
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if creds and not os.path.isabs(creds):
        resolved = (base_dir / creds).resolve()
        if resolved.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)


def validate_setup() -> tuple[str, str, str]:
    """
    Validate required environment configuration.
    Returns:
        (credentials_path, project_id, location)
    """
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.getenv("STT_GEMINI_LOCATION", DEFAULT_LOCATION).strip()

    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    if not os.path.exists(credentials_path):
        raise RuntimeError(f"Credentials file not found: {credentials_path}")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set.")

    return credentials_path, project_id, location


def connect_to_vertex_ai(project_id: str, location: str) -> None:
    """
    Initialize Vertex AI once per process.
    """
    global _VERTEX_INITIALIZED

    if _VERTEX_INITIALIZED:
        return

    import vertexai

    vertexai.init(project=project_id, location=location)
    _VERTEX_INITIALIZED = True


def ensure_ffmpeg_available() -> None:
    """
    Ensure ffmpeg is installed and available on PATH.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not available on PATH."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError("ffmpeg is installed but not working correctly.")


def load_audio_as_wav(file_path: str) -> tuple[bytes, float]:
    """
    Convert an input audio file to 16kHz mono WAV using ffmpeg.

    Returns:
        (wav_bytes, duration_seconds)
    """
    input_path = Path(file_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ensure_ffmpeg_available()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            str(tmp_wav_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

        with tmp_wav_path.open("rb") as f:
            wav_bytes = f.read()

        with wave.open(str(tmp_wav_path), "rb") as wav_file:
            duration_seconds = wav_file.getnframes() / wav_file.getframerate()

        return wav_bytes, duration_seconds

    finally:
        if tmp_wav_path.exists():
            tmp_wav_path.unlink()


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    """
    Transcribe WAV audio bytes using Gemini 2.5 Flash.
    """
    from vertexai.generative_models import GenerativeModel, Part

    prompt = (
        "You are a professional transcriptionist. "
        "Transcribe this phone call audio exactly as spoken. "
        "Rules:\n"
        "- Output only the transcription text\n"
        "- Keep Sinhala in Sinhala Unicode script\n"
        "- Keep English words when spoken in English\n"
        "- Do not translate\n"
        "- Do not summarize\n"
        "- Do not add speaker labels, headings, notes, or explanations\n"
        "- If a part is unclear, transcribe only what is reasonably audible\n"
        "- If the audio is silent, output nothing"
    )

    model = GenerativeModel(MODEL_NAME)
    audio_part = Part.from_data(data=wav_bytes, mime_type="audio/wav")
    response = model.generate_content([audio_part, prompt])

    return (getattr(response, "text", "") or "").strip()


def transcribe_audio_file(audio_path: str) -> dict[str, Any]:
    """
    Reusable function for other systems.

    Args:
        audio_path: path to an input audio file (mp3, wav, m4a, etc.)

    Returns:
        {
            "transcript": str,
            "model": str,
            "audio_path": str,
            "duration_seconds": float,
            "elapsed_seconds": float,
            "success": bool,
        }
    """
    load_env()
    _, project_id, location = validate_setup()
    connect_to_vertex_ai(project_id, location)

    wav_bytes, duration_seconds = load_audio_as_wav(audio_path)

    start_time = time.time()
    transcript = transcribe_wav_bytes(wav_bytes)
    elapsed_seconds = time.time() - start_time

    return {
        "transcript": transcript,
        "model": MODEL_NAME,
        "audio_path": str(Path(audio_path).resolve()),
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "success": bool(transcript),
    }


def save_transcript(audio_path: str, transcript: str, output_dir: str = "output") -> str:
    """
    Save transcript into the output directory using the input audio filename.
    """
    input_path = Path(audio_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    transcript_file = output_path / f"{input_path.stem}_transcript.txt"
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
        print(f"Audio time : {result['duration_seconds']:.1f}s")
        print(f"Run time   : {result['elapsed_seconds']:.1f}s")
        print("-" * 60)

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