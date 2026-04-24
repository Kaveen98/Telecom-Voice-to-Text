"""
batch_processor.py
Vertex AI Batch Prediction for Gemini — 50% cheaper, async processing.

═══════════════════════════════════════════════════════════════════════
HOW BATCH PROCESSING WORKS (explained plainly)
═══════════════════════════════════════════════════════════════════════

REAL-TIME (what watcher.py does):
  You send ONE audio → wait → get transcript back immediately (~5–30s).
  Billed at full price. Good for live call-center monitoring.

BATCH (what this file does):
  You collect MANY audios → pack them into one job → submit → go home →
  Google processes them in the background (up to 24 hours) → results
  appear in a GCS bucket → you pick them up.
  Billed at 50% of normal price. Good for overnight/bulk processing.

TIMING:
  - Queue time: unpredictable (depends on Google's idle capacity)
  - Processing: up to 24 hours after the job starts running
  - After 24 h: incomplete requests are cancelled (you pay only for done)
  - NOT suitable if you need the transcript within minutes

PRICING:
  Batch = 50% of real-time rates.
  Example on Gemini 2.5 Flash:
    Real-time audio input:  $1.00 / 1M tokens
    Batch audio input:      $0.50 / 1M tokens
  So a 3-min call that costs Rs.14.56 real-time costs Rs.7.28 in batch.

INPUT METHOD:
  - Each audio file must be uploaded to a GCS bucket
  - A JSONL file is created: one JSON object per line, one line per call
  - That JSONL is also uploaded to GCS
  - The batch job is submitted pointing at the JSONL

OUTPUT METHOD:
  - Google writes a JSONL file back to GCS (same or different bucket)
  - Each output line corresponds to one input line
  - You download and parse the output JSONL

SYNCHRONISED vs ASYNCHRONISED:
  Real-time  = SYNCHRONOUS  → you wait for the response in the same process
  Batch      = ASYNCHRONOUS → you submit, disconnect, check back later
  This script handles the async part: submit, poll status, fetch results.

═══════════════════════════════════════════════════════════════════════

Requires:
    pip install google-cloud-storage --break-system-packages
    pip install google-cloud-aiplatform --break-system-packages
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from config import DATA_DIR, GOOGLE_CLOUD_PROJECT, LKR_RATE, OUTPUT_DIR

try:
    from google.cloud import storage as gcs
except ImportError:
    print("ERROR: google-cloud-storage not installed.")
    print("Run:  pip install google-cloud-storage --break-system-packages")
    sys.exit(1)

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic import JobServiceClient
    from google.cloud.aiplatform_v1.types import (
        BatchPredictionJob,
        GcsDestination,
        GcsSource,
    )
except ImportError:
    print("ERROR: google-cloud-aiplatform not installed.")
    print("Run:  pip install google-cloud-aiplatform --break-system-packages")
    sys.exit(1)

from database import save_call
from gemini_flash_stt import (
    MODEL_NAME,
    _AUDIO_TOKENS_PER_SECOND,
    get_model_pricing,
    load_env,
    validate_setup,
)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_DISCOUNT  = 0.50   # 50% off real-time price
BATCH_TEMP_DIR  = DATA_DIR / "tmp"
POLL_INTERVAL_S = 60     # seconds between status checks

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


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _gcs_client() -> gcs.Client:
    return gcs.Client(
        project=GOOGLE_CLOUD_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )


def upload_audio_to_gcs(
    local_path: Path,
    bucket_name: str,
    gcs_prefix: str = "batch-audio",
) -> str:
    """Upload audio file and return gs:// URI."""
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{gcs_prefix}/{local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"  Uploaded: {local_path.name} → {uri}")
    return uri


def upload_jsonl_to_gcs(
    jsonl_path: Path,
    bucket_name: str,
    gcs_prefix: str = "batch-jobs",
) -> str:
    """Upload JSONL input file and return gs:// URI."""
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{gcs_prefix}/{jsonl_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(jsonl_path))
    return f"gs://{bucket_name}/{blob_name}"


def download_results_from_gcs(
    bucket_name: str,
    output_prefix: str,
) -> list[dict]:
    """Download all JSONL output files from a GCS prefix and return parsed lines."""
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    results: list[dict] = []

    for blob in bucket.list_blobs(prefix=output_prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        content = blob.download_as_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


# ── JSONL builder ─────────────────────────────────────────────────────────────

def build_batch_jsonl(
    audio_uris: list[tuple[str, str]],   # [(gcs_uri, original_filename), ...]
    output_path: Path,
) -> Path:
    """
    Write a JSONL file where each line is one Gemini batch request.

    Format required by Vertex AI Gemini batch API:
    {
      "request": {
        "contents": [
          {"role": "user", "parts": [
            {"fileData": {"mimeType": "audio/wav", "fileUri": "gs://..."}},
            {"text": "<prompt>"}
          ]}
        ]
      },
      "_metadata": {"filename": "original.mp3"}
    }
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for gcs_uri, filename in audio_uris:
            record = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "fileData": {
                                        "mimeType": "audio/wav",
                                        "fileUri": gcs_uri,
                                    }
                                },
                                {"text": TRANSCRIPTION_PROMPT},
                            ],
                        }
                    ]
                },
                "_metadata": {"filename": filename},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  JSONL written: {output_path}  ({len(audio_uris)} requests)")
    return output_path


# ── Batch job submission ───────────────────────────────────────────────────────

def submit_batch_job(
    project_id: str,
    location: str,
    input_gcs_uri: str,
    output_gcs_prefix: str,
    job_name: str | None = None,
) -> str:
    """
    Submit a Vertex AI Batch Prediction job.
    Returns the job resource name (used to poll status).
    """
    aiplatform.init(project=project_id, location=location)

    if job_name is None:
        job_name = f"slt-transcription-{uuid.uuid4().hex[:8]}"

    client = JobServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )

    job = BatchPredictionJob(
        display_name=job_name,
        model=f"publishers/google/models/{MODEL_NAME}",
        input_config=BatchPredictionJob.InputConfig(
            instances_format="jsonl",
            gcs_source=GcsSource(uris=[input_gcs_uri]),
        ),
        output_config=BatchPredictionJob.OutputConfig(
            predictions_format="jsonl",
            gcs_destination=GcsDestination(output_uri_prefix=output_gcs_prefix),
        ),
    )

    parent = f"projects/{project_id}/locations/{location}"
    response = client.create_batch_prediction_job(
        parent=parent, batch_prediction_job=job
    )

    print(f"  Batch job submitted: {response.name}")
    print(f"  State: {response.state.name}")
    return response.name


def poll_batch_job(
    project_id: str,
    location: str,
    job_resource_name: str,
    timeout_hours: float = 26.0,
) -> str:
    """
    Poll until the batch job completes or times out.
    Returns the final state string.
    """
    client = JobServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )

    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                       "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    deadline = time.time() + timeout_hours * 3600
    last_state = ""

    print(f"\n  Polling job status every {POLL_INTERVAL_S}s ...")
    print("  (Batch jobs can take up to 24 hours — this is normal)\n")

    while time.time() < deadline:
        job = client.get_batch_prediction_job(name=job_resource_name)
        state = job.state.name

        if state != last_state:
            print(f"  [{time.strftime('%H:%M:%S')}] Status: {state}")
            last_state = state

        if state in terminal_states:
            return state

        time.sleep(POLL_INTERVAL_S)

    return "TIMEOUT"


# ── Result parsing ─────────────────────────────────────────────────────────────

def _estimate_cost(duration_seconds: float, output_tokens: int) -> dict[str, float]:
    """Estimate batch cost (50% of real-time rates)."""
    pricing = get_model_pricing(MODEL_NAME)
    audio_tokens = int(duration_seconds * _AUDIO_TOKENS_PER_SECOND)
    audio_cost   = (audio_tokens  / 1_000_000) * pricing["audio_input"] * BATCH_DISCOUNT
    output_cost  = (output_tokens / 1_000_000) * pricing["output"]      * BATCH_DISCOUNT
    return {
        "audio_tokens":       audio_tokens,
        "audio_input_cost_usd": audio_cost,
        "output_cost_usd":    output_cost,
        "total_cost_usd":     audio_cost + output_cost,
    }


def process_batch_results(
    results: list[dict],
    original_files: dict[str, Path],   # {filename: local_path}
) -> None:
    """
    Parse batch output, save transcripts and DB records.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for item in results:
        metadata   = item.get("_metadata", {})
        filename   = metadata.get("filename", "unknown")
        response   = item.get("response", {})
        candidates = response.get("candidates", [])

        if not candidates:
            print(f"  ⚠  No output for: {filename}")
            continue

        raw_text  = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        usage     = response.get("usageMetadata", {})
        out_tok   = usage.get("candidatesTokenCount", 0)

        # Parse language tags from transcript
        from gemini_flash_stt import _parse_languages
        transcript, languages = _parse_languages(raw_text)

        local_path  = original_files.get(filename)
        duration_s  = 0.0   # would need ffprobe to get exact duration post-hoc
        cost_info   = _estimate_cost(duration_s, out_tok)

        result = {
            "audio_path":             str(local_path) if local_path else filename,
            "transcript":             transcript,
            "model":                  MODEL_NAME,
            "duration_seconds":       duration_s,
            "silence_removed_seconds": 0.0,
            "languages_detected":     languages,
            "input_tokens":           usage.get("promptTokenCount", 0),
            "output_tokens":          out_tok,
            "thoughts_tokens":        usage.get("thoughtsTokenCount", 0),
            "total_tokens":           usage.get("totalTokenCount", 0),
            "text_input_tokens":      0,
            **cost_info,
        }

        call_id = save_call(result, lkr_rate=LKR_RATE, batch_mode=True)

        stem = Path(filename).stem
        out_file = OUTPUT_DIR / f"{stem}_transcript.txt"
        out_file.write_text(transcript, encoding="utf-8")

        print(
            f"  ✅  {filename}  |  {out_tok:,} output tokens  "
            f"|  ${cost_info['total_cost_usd']:.5f}  "
            f"(Rs.{cost_info['total_cost_usd']*LKR_RATE:.3f})  "
            f"|  DB#{call_id}"
        )


# ── Main CLI ──────────────────────────────────────────────────────────────────

def run_batch(
    audio_files: list[Path],
    bucket_name: str,
    wait: bool = True,
) -> None:
    """
    Full pipeline: upload → build JSONL → submit → poll → parse.

    Args:
        audio_files:  List of local audio file paths.
        bucket_name:  GCS bucket name (must already exist).
        wait:         If True, block until job completes. If False, just submit.
    """
    load_env()
    _, project_id, location = validate_setup()

    job_id      = uuid.uuid4().hex[:8]
    BATCH_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path  = BATCH_TEMP_DIR / f"batch_input_{job_id}.jsonl"
    out_prefix  = f"gs://{bucket_name}/batch-output/{job_id}"

    print(f"\n{'='*60}")
    print(f"SLT Batch Transcription Job  [{job_id}]")
    print(f"  Files    : {len(audio_files)}")
    print(f"  Bucket   : {bucket_name}")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Discount : {int(BATCH_DISCOUNT*100)}% off real-time pricing")
    print(f"{'='*60}\n")

    # 1. Upload audio files
    print("Step 1/4  Uploading audio files to GCS ...")
    original_files: dict[str, Path] = {}
    audio_uris: list[tuple[str, str]] = []
    for af in audio_files:
        uri = upload_audio_to_gcs(af, bucket_name)
        audio_uris.append((uri, af.name))
        original_files[af.name] = af

    # 2. Build JSONL
    print("\nStep 2/4  Building JSONL input file ...")
    build_batch_jsonl(audio_uris, jsonl_path)
    input_gcs_uri = upload_jsonl_to_gcs(jsonl_path, bucket_name)
    jsonl_path.unlink(missing_ok=True)

    # 3. Submit job
    print("\nStep 3/4  Submitting batch prediction job ...")
    job_name = submit_batch_job(project_id, location, input_gcs_uri, out_prefix)

    if not wait:
        print(f"\n  Job submitted. Resource name saved for later polling:")
        print(f"  {job_name}")
        return

    # 4. Poll until done
    print("\nStep 4/4  Waiting for job to complete ...")
    final_state = poll_batch_job(project_id, location, job_name)

    if final_state != "JOB_STATE_SUCCEEDED":
        print(f"\n  ❌  Job ended with state: {final_state}")
        return

    print("\n  ✅  Job succeeded! Downloading results ...")
    bucket_only  = bucket_name
    output_prefix = f"batch-output/{job_id}"
    results = download_results_from_gcs(bucket_only, output_prefix)
    print(f"  Found {len(results)} result records.\n")
    process_batch_results(results, original_files)
    print(f"\n{'='*60}")
    print("Batch job complete. Transcripts saved to output/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit a Vertex AI Gemini batch transcription job"
    )
    parser.add_argument(
        "audio_files", nargs="+", help="Audio files to transcribe"
    )
    parser.add_argument(
        "--bucket", required=True, help="GCS bucket name (must exist)"
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit job and exit without waiting for results"
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.audio_files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("ERROR: Files not found:", [str(p) for p in missing])
        sys.exit(1)

    run_batch(paths, args.bucket, wait=not args.no_wait)
