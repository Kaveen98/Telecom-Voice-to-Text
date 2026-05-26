"""
Reusable Vertex AI Gemini batch-processing backend.

The module keeps the original CLI shape, but the core steps are now callable
from Flask routes or other Python code:

prepare -> submit -> check status -> import results
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from google.cloud import storage as gcs
except ImportError:  # pragma: no cover - validated when a GCS action is used
    gcs = None

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic import JobServiceClient
    from google.cloud.aiplatform_v1.types import (
        BatchPredictionJob,
        GcsDestination,
        GcsSource,
    )
except ImportError:  # pragma: no cover - validated when a Vertex action is used
    aiplatform = None
    JobServiceClient = None
    BatchPredictionJob = None
    GcsDestination = None
    GcsSource = None

from config import (
    BASE_DIR,
    BATCH_AUDIO_PREFIX,
    BATCH_GCS_BUCKET,
    BATCH_INPUT_DIR,
    BATCH_JOBS_PREFIX,
    BATCH_LOCAL_WORK_DIR,
    BATCH_LOCATION,
    BATCH_MODEL,
    BATCH_OUTPUT_PREFIX,
    app_now,
)
from transcript_storage import save_transcript_text


SUPPORTED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
POLL_INTERVAL_S = 60
LKR_RATE = 305.0
_BATCH_AUDIO_TOKENS_PER_SECOND = 25
FFPROBE_TIMEOUT_S = 20

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {
        "standard_audio_input_per_1m": 1.00,
        "standard_text_input_per_1m": 0.30,
        "standard_text_output_per_1m": 2.50,
        "priority_audio_input_per_1m": 1.80,
        "priority_text_input_per_1m": 0.54,
        "priority_text_output_per_1m": 4.50,
        "batch_audio_input_per_1m": 0.50,
        "batch_text_input_per_1m": 0.15,
        "batch_text_output_per_1m": 1.25,
    },
    "gemini-3-flash-preview": {
        "standard_audio_input_per_1m": 1.00,
        "standard_text_input_per_1m": 0.50,
        "standard_text_output_per_1m": 3.00,
        "priority_audio_input_per_1m": 1.80,
        "priority_text_input_per_1m": 0.90,
        "priority_text_output_per_1m": 5.40,
        "batch_audio_input_per_1m": 0.50,
        "batch_text_input_per_1m": 0.25,
        "batch_text_output_per_1m": 1.50,
    },
}

TRANSCRIPTION_PROMPT = (
    "You are a professional transcriptionist for a Sri Lankan telecom call center.\n"
    "Transcribe this phone call audio exactly as spoken.\n"
    "The call may contain Sinhala, English, and/or Tamil.\n\n"
    "Rules:\n"
    "- Prefix EVERY line or utterance with its language tag:\n"
    "    [SI] for Sinhala   [EN] for English   [TA] for Tamil\n"
    "- Keep Sinhala in Sinhala Unicode script\n"
    "- Keep English words in English when spoken in English\n"
    "- Keep Tamil in Tamil Unicode script\n"
    "- Do NOT translate. Do NOT summarize.\n"
    "- If a part is unclear, transcribe only what is reasonably audible\n"
    "- If the audio is silent, output nothing\n"
    "- On the very last line write: LANGUAGES: <comma-separated languages detected>"
)


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def _job_key() -> str:
    return f"batch_{app_now().strftime('%Y%m%d_%H%M%S_%f')}"


def _normalise_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _join_gcs_prefix(prefix: str, job_key: str) -> str:
    return f"{_normalise_prefix(prefix)}/{job_key}"


def _sanitize_gcs_object_component(value: str, fallback: str = "audio") -> str:
    text = Path(value).name.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip(" .-_")
    return text or fallback


def _batch_audio_object_name(batch_item_id: int, local_path: Path) -> str:
    return f"{int(batch_item_id)}__{_sanitize_gcs_object_component(local_path.name)}"


def _item_id_from_gcs_uri(gcs_uri: str) -> int | None:
    object_name = Path(gcs_uri or "").name
    prefix, marker, _ = object_name.partition("__")
    if marker and prefix.isdigit():
        return int(prefix)
    return None


def _original_filename_from_gcs_uri(gcs_uri: str) -> str:
    object_name = Path(gcs_uri or "").name
    prefix, marker, original = object_name.partition("__")
    if marker and prefix.isdigit() and original:
        return original
    return object_name


def _require_gcs_bucket(bucket_name: str | None = None) -> str:
    bucket = (bucket_name or BATCH_GCS_BUCKET).strip()
    if not bucket:
        raise RuntimeError(
            "BATCH_GCS_BUCKET is not set. Add it to .env or pass --bucket."
        )
    return bucket


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def _validate_google_setup(location: str | None = None) -> tuple[str, str]:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if credentials_path:
        creds = _resolve_path(credentials_path)
        if creds.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds)
            credentials_path = str(creds)

    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    if not Path(credentials_path).exists():
        raise RuntimeError(f"Credentials file not found: {credentials_path}")

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set.")

    return project_id, (location or BATCH_LOCATION)


def _require_gcs_client() -> Any:
    if gcs is None:
        raise RuntimeError(
            "google-cloud-storage is not installed. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        )
    return gcs.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))


def _aiplatform_endpoint(location: str) -> str:
    if location == "global":
        return "aiplatform.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def _require_job_client(location: str) -> Any:
    if (
        aiplatform is None
        or JobServiceClient is None
        or BatchPredictionJob is None
        or GcsDestination is None
        or GcsSource is None
    ):
        raise RuntimeError(
            "google-cloud-aiplatform is not installed. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        )
    return JobServiceClient(
        client_options={"api_endpoint": _aiplatform_endpoint(location)}
    )


def get_model_pricing(model: str | None = None) -> dict[str, float]:
    """Return explicit Gemini pricing for a model using longest-prefix match."""
    model_name = (model or BATCH_MODEL).lower().strip()
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if model_name.startswith(key):
            return MODEL_PRICING[key]
    raise ValueError(
        f"No batch pricing configured for model '{model or BATCH_MODEL}'."
    )


def estimate_audio_tokens(duration_seconds: float) -> int:
    return max(0, int(float(duration_seconds or 0) * _BATCH_AUDIO_TOKENS_PER_SECOND))


def estimate_batch_cost_usd(
    audio_duration_seconds: float,
    output_tokens: int = 0,
    model: str | None = None,
) -> dict[str, float | int]:
    pricing = get_model_pricing(model)
    audio_tokens = estimate_audio_tokens(audio_duration_seconds)
    audio_cost = (
        audio_tokens / 1_000_000 * pricing["batch_audio_input_per_1m"]
    )
    output_cost = (
        max(0, int(output_tokens or 0))
        / 1_000_000
        * pricing["batch_text_output_per_1m"]
    )
    return {
        "audio_tokens": audio_tokens,
        "audio_input_cost_usd": audio_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": audio_cost + output_cost,
    }


def _parse_languages(raw_text: str) -> tuple[str, list[str]]:
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


def list_ready_audio_files(input_dir: str | Path | None = None) -> list[Path]:
    audio_dir = _resolve_path(input_dir or BATCH_INPUT_DIR)
    if not audio_dir.exists():
        return []
    return sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXT
    )


def get_audio_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    explicit = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac",
        ".opus": "audio/opus",
    }
    if suffix in explicit:
        return explicit[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def measure_audio_duration(path: str | Path) -> tuple[float, str]:
    audio_path = Path(path)
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=FFPROBE_TIMEOUT_S,
        )
    except FileNotFoundError:
        return 0.0, "ffprobe is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return 0.0, f"ffprobe timed out after {FFPROBE_TIMEOUT_S}s."
    except (OSError, subprocess.CalledProcessError) as exc:
        return 0.0, f"ffprobe failed: {exc}"

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        return 0.0, f"ffprobe returned {result.returncode}: {message}"[:500]

    try:
        return max(0.0, float(result.stdout.strip())), ""
    except ValueError:
        return 0.0, f"ffprobe returned invalid duration: {result.stdout.strip()!r}"


def measure_audio_duration_seconds(path: str | Path) -> float:
    duration_seconds, _ = measure_audio_duration(path)
    return duration_seconds


def upload_audio_to_gcs(
    local_path: Path,
    bucket_name: str,
    gcs_prefix: str = "batch-audio",
    object_name: str | None = None,
) -> str:
    client = _require_gcs_client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{_normalise_prefix(gcs_prefix)}/{object_name or local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}"


def upload_jsonl_to_gcs(
    jsonl_path: Path,
    bucket_name: str,
    gcs_prefix: str = "batch-jobs",
) -> str:
    client = _require_gcs_client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{_normalise_prefix(gcs_prefix)}/{jsonl_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(jsonl_path))
    return f"gs://{bucket_name}/{blob_name}"


def download_results_from_gcs(
    bucket_name: str,
    output_prefix: str,
) -> list[dict[str, Any]]:
    client = _require_gcs_client()
    bucket = client.bucket(bucket_name)
    results: list[dict[str, Any]] = []

    for blob in bucket.list_blobs(prefix=_normalise_prefix(output_prefix)):
        if not blob.name.endswith(".jsonl"):
            continue
        content = blob.download_as_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                results.append({"error": f"Invalid JSON output line: {line[:120]}"})
    return results


def build_batch_jsonl(
    audio_uris: list[
        tuple[str, str] | tuple[str, str, str] | dict[str, Any]
    ],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in audio_uris:
            if isinstance(item, dict):
                gcs_uri = item["gcs_uri"]
                filename = item["filename"]
                mime_type = item.get("mime_type", "application/octet-stream")
                batch_item_id = item.get("batch_item_id", "")
            elif len(item) == 3:
                gcs_uri, filename, mime_type = item
                batch_item_id = ""
            else:
                gcs_uri, filename = item
                mime_type = "application/octet-stream"
                batch_item_id = ""

            record = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "fileData": {
                                        "mimeType": mime_type,
                                        "fileUri": gcs_uri,
                                    }
                                },
                                {"text": TRANSCRIPTION_PROMPT},
                            ],
                        }
                    ]
                },
                "_metadata": {
                    "batch_item_id": str(batch_item_id) if batch_item_id else "",
                    "filename": filename,
                    "gcs_uri": gcs_uri,
                    "mime_type": mime_type,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def submit_batch_job(
    project_id: str,
    location: str,
    input_gcs_uri: str,
    output_gcs_prefix: str,
    job_name: str | None = None,
    model: str | None = None,
) -> str:
    client = _require_job_client(location)
    if aiplatform is not None:
        aiplatform.init(project=project_id, location=location)

    display_name = job_name or _job_key()
    model_name = model or BATCH_MODEL
    parent = f"projects/{project_id}/locations/{location}"

    job = BatchPredictionJob(
        display_name=display_name,
        model=f"publishers/google/models/{model_name}",
        input_config=BatchPredictionJob.InputConfig(
            instances_format="jsonl",
            gcs_source=GcsSource(uris=[input_gcs_uri]),
        ),
        output_config=BatchPredictionJob.OutputConfig(
            predictions_format="jsonl",
            gcs_destination=GcsDestination(output_uri_prefix=output_gcs_prefix),
        ),
    )
    response = client.create_batch_prediction_job(
        parent=parent,
        batch_prediction_job=job,
    )
    return response.name


def _get_vertex_job(project_id: str, location: str, job_resource_name: str) -> Any:
    client = _require_job_client(location)
    return client.get_batch_prediction_job(name=job_resource_name)


def poll_batch_job(
    project_id: str,
    location: str,
    job_resource_name: str,
    timeout_hours: float = 26.0,
) -> str:
    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    deadline = time.time() + timeout_hours * 3600
    last_state = ""

    while time.time() < deadline:
        state = _get_vertex_job(project_id, location, job_resource_name).state.name
        if state != last_state:
            print(f"  [{time.strftime('%H:%M:%S')}] Status: {state}")
            last_state = state
        if state in terminal_states:
            return state
        time.sleep(POLL_INTERVAL_S)

    return "TIMEOUT"


def _normalise_vertex_state(state: str) -> str:
    mapping = {
        "JOB_STATE_PENDING": "SUBMITTED",
        "JOB_STATE_QUEUED": "SUBMITTED",
        "JOB_STATE_RUNNING": "RUNNING",
        "JOB_STATE_SUCCEEDED": "SUCCEEDED",
        "JOB_STATE_FAILED": "FAILED",
        "JOB_STATE_CANCELLED": "CANCELLED",
        "JOB_STATE_EXPIRED": "EXPIRED",
        "JOB_STATE_PAUSED": "PAUSED",
    }
    return mapping.get(state, state.replace("JOB_STATE_", "") or "UNKNOWN")


def _load_db_helpers() -> dict[str, Any]:
    import database

    return {
        "add_batch_item": database.add_batch_item,
        "create_batch_job": database.create_batch_job,
        "get_batch_item_by_id_or_filename_or_gcs_uri": database.get_batch_item_by_id_or_filename_or_gcs_uri,
        "get_batch_job": database.get_batch_job,
        "list_batch_items": database.list_batch_items,
        "mark_batch_job_counts": database.mark_batch_job_counts,
        "save_call": database.save_call,
        "try_mark_batch_item_importing": database.try_mark_batch_item_importing,
        "try_mark_batch_job_submitting": database.try_mark_batch_job_submitting,
        "update_batch_item": database.update_batch_item,
        "update_batch_job": database.update_batch_job,
        "update_call_transcript_file": database.update_call_transcript_file,
    }


def _create_prepared_batch_from_files(
    audio_files: list[Path],
    bucket_name: str | None = None,
    model: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    db = _load_db_helpers()
    bucket = _require_gcs_bucket(bucket_name)
    batch_model = model or BATCH_MODEL
    batch_location = location or BATCH_LOCATION
    get_model_pricing(batch_model)
    _validate_google_setup(batch_location)

    if not audio_files:
        raise RuntimeError("No ready audio files found for batch preparation.")

    job_key = _job_key()
    audio_prefix = _join_gcs_prefix(BATCH_AUDIO_PREFIX, job_key)
    jobs_prefix = _join_gcs_prefix(BATCH_JOBS_PREFIX, job_key)
    output_prefix = _join_gcs_prefix(BATCH_OUTPUT_PREFIX, job_key)
    output_gcs_prefix = f"gs://{bucket}/{output_prefix}"
    work_dir = BATCH_LOCAL_WORK_DIR / job_key
    work_dir.mkdir(parents=True, exist_ok=True)

    job_id = db["create_batch_job"](
        job_key=job_key,
        status="PREPARING",
        model=batch_model,
        location=batch_location,
        bucket=bucket,
        audio_prefix=audio_prefix,
        jobs_prefix=jobs_prefix,
        output_prefix=output_prefix,
        output_gcs_prefix=output_gcs_prefix,
    )

    jsonl_records: list[dict[str, str]] = []
    try:
        for audio_path in audio_files:
            duration_s, duration_warning = measure_audio_duration(audio_path)
            cost_info = estimate_batch_cost_usd(duration_s, model=batch_model)
            mime_type = get_audio_mime_type(audio_path)
            warning = duration_warning if duration_warning else ""

            item_id = db["add_batch_item"](
                batch_job_id=job_id,
                filename=audio_path.name,
                local_audio_path=_relative_path(audio_path),
                mime_type=mime_type,
                duration_seconds=duration_s,
                estimated_audio_tokens=int(cost_info["audio_tokens"]),
                estimated_cost_usd=float(cost_info["total_cost_usd"]),
                status="UPLOADING",
                error_message=warning,
            )

            gcs_uri = upload_audio_to_gcs(
                audio_path,
                bucket,
                audio_prefix,
                object_name=_batch_audio_object_name(item_id, audio_path),
            )
            db["update_batch_item"](
                item_id,
                gcs_audio_uri=gcs_uri,
                status="UPLOADED",
            )
            jsonl_records.append(
                {
                    "batch_item_id": str(item_id),
                    "gcs_uri": gcs_uri,
                    "filename": audio_path.name,
                    "mime_type": mime_type,
                }
            )

        jsonl_path = work_dir / f"{job_key}.jsonl"
        build_batch_jsonl(jsonl_records, jsonl_path)
        input_gcs_uri = upload_jsonl_to_gcs(jsonl_path, bucket, jobs_prefix)
        db["mark_batch_job_counts"](job_id)
        db["update_batch_job"](
            job_id,
            status="PREPARED",
            input_jsonl_path=_relative_path(jsonl_path),
            input_jsonl_gcs_uri=input_gcs_uri,
            output_gcs_prefix=output_gcs_prefix,
        )
    except Exception as exc:
        db["update_batch_job"](job_id, status="FAILED", error_message=str(exc))
        raise

    return {
        "status": "PREPARED",
        "batch_job_id": job_id,
        "job_key": job_key,
        "file_count": len(audio_files),
        "input_jsonl_gcs_uri": input_gcs_uri,
        "output_gcs_prefix": output_gcs_prefix,
    }


def prepare_batch_job(
    input_dir: str | Path | None = None,
    model: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    audio_files = list_ready_audio_files(input_dir)
    return _create_prepared_batch_from_files(
        audio_files,
        bucket_name=BATCH_GCS_BUCKET,
        model=model,
        location=location,
    )


def submit_prepared_batch(batch_job_id: int) -> dict[str, Any]:
    db = _load_db_helpers()
    job = db["get_batch_job"](batch_job_id)
    if not job:
        raise RuntimeError(f"Batch job not found: {batch_job_id}")
    if job.get("vertex_job_name"):
        return {
            "status": job.get("status", ""),
            "batch_job_id": batch_job_id,
            "vertex_job_name": job["vertex_job_name"],
            "already_submitted": True,
        }
    if not job.get("input_jsonl_gcs_uri"):
        raise RuntimeError("Batch job has no uploaded JSONL input URI.")

    submitting_job = db["try_mark_batch_job_submitting"](batch_job_id)
    if not submitting_job:
        current = db["get_batch_job"](batch_job_id) or job
        return {
            "status": current.get("status", ""),
            "batch_job_id": batch_job_id,
            "vertex_job_name": current.get("vertex_job_name", ""),
            "already_submitted": bool(current.get("vertex_job_name")),
            "submit_started": current.get("status") == "SUBMITTING",
            "submitted": False,
            "message": "Batch job is not in PREPARED state or is already submitting/submitted.",
        }

    try:
        project_id, location = _validate_google_setup(
            submitting_job.get("location") or BATCH_LOCATION
        )
        vertex_job_name = submit_batch_job(
            project_id=project_id,
            location=location,
            input_gcs_uri=submitting_job["input_jsonl_gcs_uri"],
            output_gcs_prefix=submitting_job["output_gcs_prefix"],
            job_name=submitting_job["job_key"],
            model=submitting_job.get("model") or BATCH_MODEL,
        )
    except Exception as exc:
        db["update_batch_job"](
            batch_job_id,
            status="FAILED",
            error_message=f"Vertex batch submission failed: {exc}",
        )
        raise

    now = app_now().isoformat()
    db["update_batch_job"](
        batch_job_id,
        status="SUBMITTED",
        vertex_job_name=vertex_job_name,
        submitted_at=now,
        error_message="",
    )
    for item in db["list_batch_items"](batch_job_id):
        if item.get("status") in {"PENDING", "UPLOADED"}:
            db["update_batch_item"](item["id"], status="SUBMITTED")

    return {
        "status": "SUBMITTED",
        "batch_job_id": batch_job_id,
        "vertex_job_name": vertex_job_name,
        "submitted": True,
    }


def check_batch_status(batch_job_id: int) -> dict[str, Any]:
    db = _load_db_helpers()
    job = db["get_batch_job"](batch_job_id)
    if not job:
        raise RuntimeError(f"Batch job not found: {batch_job_id}")
    if not job.get("vertex_job_name"):
        raise RuntimeError("Batch job has not been submitted yet.")

    project_id, location = _validate_google_setup(job.get("location") or BATCH_LOCATION)
    vertex_job = _get_vertex_job(project_id, location, job["vertex_job_name"])
    raw_state = vertex_job.state.name
    status = _normalise_vertex_state(raw_state)
    now = app_now().isoformat()
    updates: dict[str, Any] = {
        "status": status,
        "last_checked_at": now,
    }
    if status in {"SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"}:
        updates["completed_at"] = now
    db["update_batch_job"](batch_job_id, **updates)

    return {
        "status": status,
        "raw_state": raw_state,
        "batch_job_id": batch_job_id,
        "vertex_job_name": job["vertex_job_name"],
    }


def _metadata_from_result(item: dict[str, Any]) -> dict[str, str]:
    metadata = item.get("_metadata") or {}
    if not metadata:
        metadata = item.get("instance", {}).get("_metadata") or {}
    if not metadata:
        metadata = item.get("request", {}).get("_metadata") or {}

    batch_item_id = str(
        metadata.get("batch_item_id")
        or metadata.get("batchItemId")
        or metadata.get("item_id")
        or ""
    )
    filename = metadata.get("filename", "")
    gcs_uri = metadata.get("gcs_uri", "")
    if not gcs_uri:
        request = item.get("request") or item.get("instance", {}).get("request") or {}
        contents = request.get("contents", [])
        parts = contents[0].get("parts", []) if contents else []
        for part in parts:
            file_data = part.get("fileData") or part.get("file_data") or {}
            if file_data.get("fileUri") or file_data.get("file_uri"):
                gcs_uri = file_data.get("fileUri") or file_data.get("file_uri")
                break
    if not batch_item_id and gcs_uri:
        parsed_item_id = _item_id_from_gcs_uri(gcs_uri)
        if parsed_item_id is not None:
            batch_item_id = str(parsed_item_id)
    if not filename and gcs_uri:
        filename = _original_filename_from_gcs_uri(gcs_uri)
    return {
        "batch_item_id": batch_item_id,
        "filename": filename,
        "gcs_uri": gcs_uri,
    }


def _response_from_result(item: dict[str, Any]) -> dict[str, Any]:
    response = item.get("response") or item.get("prediction") or item.get("predictions") or {}
    if isinstance(response, list):
        return response[0] if response and isinstance(response[0], dict) else {}
    return response if isinstance(response, dict) else {}


def _error_from_result(item: dict[str, Any], response: dict[str, Any]) -> str:
    candidates = [
        item.get("status"),
        item.get("error"),
        response.get("error"),
        item.get("message"),
        response.get("message"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            parts = [
                str(candidate.get("code") or "").strip(),
                str(candidate.get("status") or "").strip(),
                str(candidate.get("message") or "").strip(),
            ]
            details = candidate.get("details")
            if details:
                parts.append(json.dumps(details, ensure_ascii=False))
            message = " ".join(part for part in parts if part)
            return message or json.dumps(candidate, ensure_ascii=False)
        return str(candidate)
    return ""


def _candidate_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    return parts[0].get("text", "")


def _usage_value(usage: dict[str, Any], camel: str, snake: str) -> int:
    return int(usage.get(camel, usage.get(snake, 0)) or 0)


def import_batch_results(batch_job_id: int, force: bool = False) -> dict[str, Any]:
    db = _load_db_helpers()
    job = db["get_batch_job"](batch_job_id)
    if not job:
        raise RuntimeError(f"Batch job not found: {batch_job_id}")
    importable_statuses = {"SUCCEEDED", "IMPORTED", "IMPORTED_WITH_ERRORS"}
    if job.get("status") not in importable_statuses and not force:
        raise RuntimeError("Batch job must be SUCCEEDED before importing results.")

    bucket = job.get("bucket", "")
    output_prefix = job.get("output_prefix", "")
    if not bucket or not output_prefix:
        raise RuntimeError("Batch job is missing GCS output configuration.")

    results = download_results_from_gcs(bucket, output_prefix)
    imported = 0
    already_imported = 0
    failed = 0
    unmatched = 0
    skipped = 0

    for result_item in results:
        metadata = _metadata_from_result(result_item)
        response = _response_from_result(result_item)
        item_error = _error_from_result(result_item, response)
        batch_item = db["get_batch_item_by_id_or_filename_or_gcs_uri"](
            batch_job_id,
            filename=metadata.get("filename", ""),
            gcs_audio_uri=metadata.get("gcs_uri", ""),
            item_id=metadata.get("batch_item_id", ""),
        )
        if not batch_item:
            unmatched += 1
            continue
        if batch_item.get("status") == "IMPORTED" and batch_item.get("call_id"):
            already_imported += 1
            continue
        if batch_item.get("status") == "IMPORTING":
            skipped += 1
            continue

        importing_item = db["try_mark_batch_item_importing"](batch_item["id"])
        if not importing_item:
            current_item = db["get_batch_item_by_id_or_filename_or_gcs_uri"](
                batch_job_id,
                item_id=batch_item["id"],
            )
            if current_item and current_item.get("status") == "IMPORTED" and current_item.get("call_id"):
                already_imported += 1
            else:
                skipped += 1
            continue
        batch_item = importing_item

        if item_error:
            db["update_batch_item"](
                batch_item["id"],
                status="FAILED",
                error_message=item_error,
            )
            failed += 1
            continue

        raw_text = _candidate_text(response)
        if not raw_text:
            db["update_batch_item"](
                batch_item["id"],
                status="FAILED",
                error_message="No candidate text returned.",
            )
            failed += 1
            continue

        call_id: int | None = None
        try:
            transcript, languages = _parse_languages(raw_text)
            usage = response.get("usageMetadata", response.get("usage_metadata", {}))
            out_tok = _usage_value(usage, "candidatesTokenCount", "candidates_token_count")
            input_tokens = _usage_value(usage, "promptTokenCount", "prompt_token_count")
            thoughts_tokens = _usage_value(usage, "thoughtsTokenCount", "thoughts_token_count")
            total_tokens = _usage_value(usage, "totalTokenCount", "total_token_count")
            duration_s = float(batch_item.get("duration_seconds") or 0)
            cost_info = estimate_batch_cost_usd(
                duration_s,
                output_tokens=out_tok,
                model=job.get("model") or BATCH_MODEL,
            )
            pricing = get_model_pricing(job.get("model") or BATCH_MODEL)
            audio_tokens = int(cost_info["audio_tokens"])
            text_input_tokens = max(input_tokens - audio_tokens, 0)
            text_input_cost = (
                text_input_tokens / 1_000_000 * pricing["batch_text_input_per_1m"]
            )
            output_cost = float(cost_info["output_cost_usd"])
            audio_cost = float(cost_info["audio_input_cost_usd"])
            total_cost = audio_cost + text_input_cost + output_cost

            local_audio_path = batch_item.get("local_audio_path") or metadata.get("filename", "")
            call_result = {
                "audio_path": local_audio_path,
                "transcript": transcript,
                "model": job.get("model") or BATCH_MODEL,
                "duration_seconds": duration_s,
                "silence_removed_seconds": 0.0,
                "languages_detected": languages,
                "input_tokens": input_tokens,
                "audio_tokens": audio_tokens,
                "text_input_tokens": text_input_tokens,
                "output_tokens": out_tok,
                "thoughts_tokens": thoughts_tokens,
                "billed_output_tokens": out_tok + thoughts_tokens,
                "total_tokens": total_tokens,
                "audio_input_cost_usd": audio_cost,
                "text_input_cost_usd": text_input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
            }
            call_id = db["save_call"](call_result, lkr_rate=LKR_RATE, batch_mode=True)
            saved_info = save_transcript_text(
                audio_path=local_audio_path,
                transcript=transcript,
                model=job.get("model") or BATCH_MODEL,
                mode="batch",
                call_id=call_id,
            )
            db["update_call_transcript_file"](call_id, saved_info)
            db["update_batch_item"](
                batch_item["id"],
                status="IMPORTED",
                call_id=call_id,
                transcript_file_path=saved_info["transcript_file_path"],
                error_message="",
            )
            imported += 1
        except Exception as exc:
            update_fields: dict[str, Any] = {
                "status": "IMPORT_ERROR",
                "error_message": f"Import failed: {exc}",
            }
            if call_id is not None:
                update_fields["call_id"] = call_id
            db["update_batch_item"](batch_item["id"], **update_fields)
            failed += 1

    db["mark_batch_job_counts"](batch_job_id)
    final_status = (
        "IMPORTED"
        if failed == 0 and unmatched == 0 and skipped == 0
        else "IMPORTED_WITH_ERRORS"
    )
    db["update_batch_job"](
        batch_job_id,
        status=final_status,
        imported_at=app_now().isoformat(),
    )

    return {
        "batch_job_id": batch_job_id,
        "result_records": len(results),
        "imported": imported,
        "already_imported": already_imported,
        "failed": failed,
        "unmatched": unmatched,
        "skipped": skipped,
    }


def process_batch_results(
    results: list[dict[str, Any]],
    original_files: dict[str, Path],
) -> None:
    """Legacy CLI result importer for direct, non-durable use."""
    db = _load_db_helpers()
    for result_item in results:
        metadata = _metadata_from_result(result_item)
        response = _response_from_result(result_item)
        filename = metadata.get("filename") or "unknown"
        error = _error_from_result(result_item, response)
        if error:
            print(f"  No output for {filename}: {error}")
            continue

        raw_text = _candidate_text(response)
        if not raw_text:
            print(f"  No output for: {filename}")
            continue

        transcript, languages = _parse_languages(raw_text)
        usage = response.get("usageMetadata", response.get("usage_metadata", {}))
        out_tok = _usage_value(usage, "candidatesTokenCount", "candidates_token_count")
        input_tokens = _usage_value(usage, "promptTokenCount", "prompt_token_count")
        thoughts_tokens = _usage_value(usage, "thoughtsTokenCount", "thoughts_token_count")
        total_tokens = _usage_value(usage, "totalTokenCount", "total_token_count")
        local_path = original_files.get(filename)
        duration_s = measure_audio_duration_seconds(local_path) if local_path else 0.0
        cost_info = estimate_batch_cost_usd(duration_s, out_tok)
        audio_tokens = int(cost_info["audio_tokens"])

        call_result = {
            "audio_path": str(local_path) if local_path else filename,
            "transcript": transcript,
            "model": BATCH_MODEL,
            "duration_seconds": duration_s,
            "silence_removed_seconds": 0.0,
            "languages_detected": languages,
            "input_tokens": input_tokens,
            "audio_tokens": audio_tokens,
            "text_input_tokens": max(input_tokens - audio_tokens, 0),
            "output_tokens": out_tok,
            "thoughts_tokens": thoughts_tokens,
            "billed_output_tokens": out_tok + thoughts_tokens,
            "total_tokens": total_tokens,
            "audio_input_cost_usd": float(cost_info["audio_input_cost_usd"]),
            "text_input_cost_usd": 0,
            "output_cost_usd": float(cost_info["output_cost_usd"]),
            "total_cost_usd": float(cost_info["total_cost_usd"]),
        }
        call_id = db["save_call"](call_result, lkr_rate=LKR_RATE, batch_mode=True)
        saved_info = save_transcript_text(
            audio_path=local_path if local_path else filename,
            transcript=transcript,
            model=BATCH_MODEL,
            mode="batch",
            call_id=call_id,
        )
        db["update_call_transcript_file"](call_id, saved_info)
        print(f"  Imported {filename} | DB#{call_id}")


def wait_for_batch_job(batch_job_id: int, timeout_hours: float = 26.0) -> str:
    terminal = {"SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"}
    deadline = time.time() + timeout_hours * 3600
    last_status = ""

    while time.time() < deadline:
        info = check_batch_status(batch_job_id)
        status = info["status"]
        if status != last_status:
            print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}")
            last_status = status
        if status in terminal:
            return status
        time.sleep(POLL_INTERVAL_S)

    return "TIMEOUT"


def run_batch(
    audio_files: list[Path],
    bucket_name: str,
    wait: bool = True,
) -> None:
    audio_files = [Path(p) for p in audio_files]
    prepared = _create_prepared_batch_from_files(
        audio_files,
        bucket_name=bucket_name,
        model=BATCH_MODEL,
        location=BATCH_LOCATION,
    )
    batch_job_id = prepared["batch_job_id"]
    submitted = submit_prepared_batch(batch_job_id)

    print(f"Batch job id: {batch_job_id}")
    print(f"Vertex job: {submitted['vertex_job_name']}")
    if not wait:
        return

    final_status = wait_for_batch_job(batch_job_id)
    if final_status != "SUCCEEDED":
        print(f"Batch job ended with status: {final_status}")
        return

    result = import_batch_results(batch_job_id)
    print(
        "Batch import complete: "
        f"{result['imported']} imported, "
        f"{result['already_imported']} already imported, "
        f"{result['failed']} failed"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit a Vertex AI Gemini batch transcription job"
    )
    parser.add_argument("audio_files", nargs="+", help="Audio files to transcribe")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit job and exit without waiting for results",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.audio_files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("ERROR: Files not found:", [str(p) for p in missing])
        sys.exit(1)

    run_batch(paths, args.bucket, wait=not args.no_wait)
