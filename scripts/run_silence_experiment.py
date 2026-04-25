"""
Compare no-trim and current FFmpeg silence-trim transcription runs.

This is an experiment helper only. It does not change production defaults,
write call rows to the database, or modify .env/config files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini_flash_stt import transcribe_audio_file

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
DEFAULT_OUTPUT = Path("experiments/silence_experiment_results.csv")
TRANSCRIPTS_DIR = Path("experiments/transcripts")
PROFILE_STRIP_SILENCE = {
    "no_trim": False,
    "current_trim": True,
}
CSV_COLUMNS = [
    "source_file",
    "profile",
    "success",
    "error",
    "original_duration_seconds",
    "submitted_duration_seconds",
    "silence_removed_seconds",
    "silence_removed_ratio",
    "provider_input_tokens",
    "provider_output_tokens",
    "provider_thoughts_tokens",
    "provider_total_tokens",
    "estimated_audio_tokens",
    "billed_output_tokens",
    "actual_tok_per_sec",
    "estimated_cost_usd",
    "pricing_source",
    "preprocess_profile",
    "silence_filter",
    "language_tags",
    "transcript_length_chars",
    "transcript_preview",
    "manual_quality_score",
    "manual_notes",
    "missing_words_flag",
]


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--limit must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("--limit must be at least 1")
    return value


def _parse_profiles(raw: str) -> list[str]:
    profiles: list[str] = []
    for part in raw.split(","):
        profile = part.strip()
        if not profile:
            continue
        if profile not in PROFILE_STRIP_SILENCE:
            allowed = ", ".join(PROFILE_STRIP_SILENCE)
            raise argparse.ArgumentTypeError(
                f"Unknown profile {profile!r}. Allowed profiles: {allowed}"
            )
        if profile not in profiles:
            profiles.append(profile)
    if not profiles:
        raise argparse.ArgumentTypeError("At least one profile is required")
    return profiles


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _collect_audio_files(inputs: list[str], limit: int) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            candidates = [path]
        elif path.is_file():
            raise ValueError(f"Unsupported audio extension: {path}")
        else:
            continue

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(resolved)
            if len(files) >= limit:
                return files

    return files


def _safe_transcript_path(source_file: Path, profile: str) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_file.stem).strip("._")
    if not safe_stem:
        safe_stem = "audio"
    path_hash = hashlib.sha1(str(source_file.resolve()).encode("utf-8")).hexdigest()[:10]
    return TRANSCRIPTS_DIR / f"{safe_stem}_{path_hash}_{profile}.txt"


def _language_tags(raw: Any) -> str:
    if isinstance(raw, list):
        return ", ".join(str(item) for item in raw)
    if raw is None:
        return ""
    return str(raw)


def _preview(transcript: str, limit: int = 240) -> str:
    return " ".join(transcript.split())[:limit]


def _result_row(source_file: Path, profile: str, result: dict[str, Any]) -> dict[str, Any]:
    transcript = result.get("transcript", "") or ""
    return {
        "source_file": _display_path(source_file),
        "profile": profile,
        "success": bool(result.get("success")),
        "error": "",
        "original_duration_seconds": result.get("original_duration_seconds", ""),
        "submitted_duration_seconds": result.get(
            "submitted_duration_seconds",
            result.get("duration_seconds", ""),
        ),
        "silence_removed_seconds": result.get("silence_removed_seconds", ""),
        "silence_removed_ratio": result.get("silence_removed_ratio", ""),
        "provider_input_tokens": result.get(
            "provider_input_tokens",
            result.get("input_tokens", ""),
        ),
        "provider_output_tokens": result.get(
            "provider_output_tokens",
            result.get("output_tokens", ""),
        ),
        "provider_thoughts_tokens": result.get(
            "provider_thoughts_tokens",
            result.get("thoughts_tokens", ""),
        ),
        "provider_total_tokens": result.get(
            "provider_total_tokens",
            result.get("total_tokens", ""),
        ),
        "estimated_audio_tokens": result.get(
            "estimated_audio_tokens",
            result.get("audio_tokens", ""),
        ),
        "billed_output_tokens": result.get("billed_output_tokens", ""),
        "actual_tok_per_sec": result.get("actual_tok_per_sec", ""),
        "estimated_cost_usd": result.get("total_cost_usd", ""),
        "pricing_source": result.get("pricing_source", ""),
        "preprocess_profile": result.get("preprocess_profile", ""),
        "silence_filter": result.get("silence_filter", ""),
        "language_tags": _language_tags(result.get("languages_detected")),
        "transcript_length_chars": len(transcript),
        "transcript_preview": _preview(transcript),
        "manual_quality_score": "",
        "manual_notes": "",
        "missing_words_flag": "",
    }


def _error_row(source_file: Path, profile: str, error: Exception) -> dict[str, Any]:
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "source_file": _display_path(source_file),
            "profile": profile,
            "success": False,
            "error": f"{error.__class__.__name__}: {error}",
        }
    )
    return row


def _run_one(source_file: Path, profile: str) -> dict[str, Any]:
    strip_silence = PROFILE_STRIP_SILENCE[profile]
    result = transcribe_audio_file(str(source_file), strip_silence=strip_silence)
    transcript_path = _safe_transcript_path(source_file, profile)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(result.get("transcript", "") or "", encoding="utf-8")
    return _result_row(source_file, profile, result)


def _print_dry_run(files: list[Path], profiles: list[str]) -> None:
    print(f"Dry run: {len(files)} file(s), {len(profiles)} profile(s)")
    print(f"Provider calls that would run: {len(files) * len(profiles)}")
    for source_file in files:
        for profile in profiles:
            mode = "strip_silence=true" if PROFILE_STRIP_SILENCE[profile] else "strip_silence=false"
            print(f"- {_display_path(source_file)} | {profile} | {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare no-trim and current FFmpeg silence-trim STT metadata."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more audio files or folders containing audio files.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=25,
        help="Maximum number of source audio files to test. Default: 25.",
    )
    parser.add_argument(
        "--profiles",
        type=_parse_profiles,
        default=_parse_profiles("no_trim,current_trim"),
        help="Comma-separated profiles to run. Default: no_trim,current_trim.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files/profiles without making provider calls or writing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = _collect_audio_files(args.input, args.limit)
    profiles: list[str] = args.profiles

    if not files:
        print("No supported audio files found.")
        return 1

    if args.dry_run:
        _print_dry_run(files, profiles)
        return 0

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(files) * len(profiles)
    completed = 0
    print(f"Running {total_runs} provider transcription call(s).")
    print(f"CSV output: {output_path}")
    print(f"Transcripts: {TRANSCRIPTS_DIR}")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for source_file in files:
            for profile in profiles:
                completed += 1
                print(f"[{completed}/{total_runs}] {_display_path(source_file)} | {profile}")
                try:
                    row = _run_one(source_file, profile)
                except Exception as exc:
                    row = _error_row(source_file, profile, exc)
                writer.writerow(row)
                csv_file.flush()

    print("Experiment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
