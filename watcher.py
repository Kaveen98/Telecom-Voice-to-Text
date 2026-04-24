"""
watcher.py
Durable folder-watcher auto-transcription pipeline.

Usage:
    python watcher.py                        # uses input_audio/ lifecycle dirs
    python watcher.py --input /path/to/base  # uses /path/to/base/incoming
    python watcher.py --batch                # preserve existing batch_mode flag

Drop audio into input_audio/incoming/. The watcher registers each stable file
as a durable SQLite job, then a worker claims jobs from the database and moves
files through processing/, completed/, failed/, and archive/.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

try:
    from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("ERROR: watchdog is not installed.")
    print("Run:  pip install watchdog --break-system-packages")
    sys.exit(1)

from config import INPUT_DIR, LKR_RATE, LOG_FILE, MODEL_NAME, OUTPUT_DIR
from database import (
    JOB_FINALIZING,
    JOB_INVALID,
    JOB_FAILED,
    JOB_PENDING,
    JOB_RETRYING,
    JOB_SKIPPED_DUPLICATE,
    JOB_TRANSCRIBED,
    claim_next_job,
    create_or_get_job,
    get_call,
    get_job,
    get_jobs_needing_finalization,
    init_db,
    mark_job_failed,
    mark_job_invalid,
    mark_job_retrying,
    mark_job_transcribed,
    recover_stale_jobs,
    save_call_and_mark_finalizing,
    update_job_paths,
)
from gemini_flash_stt import transcribe_audio_file

# -- Configuration -----------------------------------------------------------
DEFAULT_BASE_DIR = INPUT_DIR
SUPPORTED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
TEMP_PARTIAL_EXT = {".tmp", ".part", ".crdownload"}

STABLE_CHECKS = 3
STABLE_INTERVAL_S = 1.0
STABLE_TIMEOUT_S = 120.0
WORKER_IDLE_SLEEP_S = 5.0
STALE_JOB_MINUTES = 60
MAX_ATTEMPTS = 3

PROVIDER = "vertex_gemini"
PREPROCESS_PROFILE = "default"

# -- Logging ----------------------------------------------------------------
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("watcher")

_wake_event = threading.Event()
_stop_event = threading.Event()


def lifecycle_dirs(base_dir: Path) -> dict[str, Path]:
    """
    Return the Phase 1 folder lifecycle under *base_dir*.

    If a caller passes an incoming/ folder directly, treat its parent as the
    lifecycle root so manual invocations remain forgiving.
    """
    base = base_dir.resolve()
    if base.name.lower() == "incoming":
        base = base.parent

    return {
        "base": base,
        "incoming": base / "incoming",
        "processing": base / "processing",
        "completed": base / "completed",
        "failed": base / "failed",
        "archive": base / "archive",
    }


def ensure_lifecycle_dirs(dirs: dict[str, Path]) -> None:
    for key in ("incoming", "processing", "completed", "failed", "archive"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def wait_until_file_stable(
    path: Path,
    checks: int = STABLE_CHECKS,
    interval_s: float = STABLE_INTERVAL_S,
    timeout_s: float = STABLE_TIMEOUT_S,
) -> bool:
    """Wait until size/mtime are unchanged for several observations."""
    deadline = time.time() + timeout_s
    last_fingerprint: tuple[int, int] | None = None
    stable_count = 0

    while time.time() < deadline:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False

        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if fingerprint == last_fingerprint:
            stable_count += 1
            if stable_count >= checks:
                return True
        else:
            stable_count = 0
            last_fingerprint = fingerprint

        time.sleep(interval_s)

    return False


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for i in range(1, 10_000):
        numbered = dest_dir / f"{stem}_{i}{suffix}"
        if not numbered.exists():
            return numbered

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return dest_dir / f"{stem}_{timestamp}{suffix}"


def safe_move(src: Path, dest_dir: Path, filename: str | None = None) -> Path:
    """Move a file without overwriting an existing destination."""
    src = src.resolve()
    dest = _unique_destination(dest_dir, filename or src.name).resolve()
    if src == dest:
        return src
    shutil.move(str(src), str(dest))
    return dest


def _job_filename(job: dict) -> str:
    return f"{job['id']}_{job['original_filename']}"


def _short_hash(file_hash: str | None) -> str:
    return (file_hash or "")[:12]


def register_audio_file(
    path: Path,
    dirs: dict[str, Path],
    batch_mode: bool = False,
) -> dict | None:
    """Register a stable incoming audio file as a durable database job."""
    path = path.resolve()

    if path.suffix.lower() in TEMP_PARTIAL_EXT:
        log.info("Ignored temporary/partial file: %s", path.name)
        return None

    if not wait_until_file_stable(path):
        log.warning("File did not become stable before timeout: %s", path)
        return None

    if not path.exists():
        log.warning("File disappeared before registration: %s", path)
        return None

    if path.suffix.lower() not in SUPPORTED_EXT:
        unsupported_path = safe_move(path, dirs["failed"] / "unsupported")
        log.warning(
            "Unsupported file moved without provider call | file=%s failed_path=%s",
            path.name,
            unsupported_path,
        )
        return None

    stat = path.stat()
    file_hash = compute_sha256(path)
    transcription_mode = "batch" if batch_mode else "realtime"

    job = create_or_get_job(
        file_hash=file_hash,
        original_filename=path.name,
        original_extension=path.suffix.lower(),
        file_size_bytes=stat.st_size,
        incoming_path=str(path),
        provider=PROVIDER,
        model=MODEL_NAME,
        transcription_mode=transcription_mode,
        preprocess_profile=PREPROCESS_PROFILE,
        max_attempts=MAX_ATTEMPTS,
    )

    if job["status"] == JOB_SKIPPED_DUPLICATE:
        duplicate_dir = dirs["archive"] / "duplicates"
        archived = safe_move(path, duplicate_dir, _job_filename(job))
        update_job_paths(job["id"], archive_path=str(archived))
        log.info(
            "Duplicate skipped | job_id=%s duplicate_of=%s hash=%s file=%s archive=%s",
            job["id"],
            job.get("duplicate_of_job_id"),
            _short_hash(file_hash),
            path.name,
            archived,
        )
        return job

    log.info(
        "Job registered | job_id=%s status=%s hash=%s file=%s size=%s",
        job["id"],
        job["status"],
        _short_hash(file_hash),
        path.name,
        stat.st_size,
    )
    _wake_event.set()
    return job


class _AudioDropHandler(FileSystemEventHandler):
    """Registers stable audio files created or moved into incoming/."""

    def __init__(self, dirs: dict[str, Path], batch_mode: bool = False) -> None:
        self.dirs = dirs
        self.batch_mode = batch_mode

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        register_audio_file(Path(event.src_path), self.dirs, self.batch_mode)

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        register_audio_file(Path(event.dest_path), self.dirs, self.batch_mode)


def _existing_audio_path(job: dict, include_completed: bool = False) -> Path | None:
    keys = ["processing_path", "incoming_path"]
    if include_completed:
        keys.insert(0, "completed_path")
    for key in keys:
        value = job.get(key)
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path
    return None


def _ensure_processing_file(job: dict, dirs: dict[str, Path]) -> Path:
    current_path = _existing_audio_path(job)
    if current_path is None:
        raise FileNotFoundError(
            f"No audio file found for job {job['id']} ({job['original_filename']})"
        )

    if current_path.suffix.lower() not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported audio extension: {current_path.suffix}")

    if current_path.resolve().parent == dirs["processing"].resolve():
        processing_path = current_path.resolve()
    else:
        processing_path = safe_move(
            current_path,
            dirs["processing"],
            _job_filename(job),
        )

    update_job_paths(job["id"], processing_path=str(processing_path))
    return processing_path


def classify_error(exc: Exception) -> tuple[str, bool, bool]:
    """Return (error_type, retryable, invalid_file)."""
    msg = str(exc).lower()

    invalid_keywords = (
        "audio file not found",
        "file not found",
        "unsupported audio extension",
        "ffmpeg conversion failed",
        "invalid data",
        "moov atom not found",
        "could not find codec",
        "failed to read frame size",
    )
    setup_keywords = (
        "google_application_credentials",
        "google_cloud_project",
        "credentials file not found",
        "ffmpeg is not installed",
        "not available on path",
    )
    transient_keywords = (
        "timeout",
        "deadline",
        "temporarily",
        "unavailable",
        "service unavailable",
        "connection",
        "network",
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "quota",
    )

    if isinstance(exc, FileNotFoundError) or any(k in msg for k in invalid_keywords):
        return "invalid_audio", False, True
    if any(k in msg for k in setup_keywords):
        return "setup_error", False, False
    if any(k in msg for k in transient_keywords):
        return "transient_provider_error", True, False
    return exc.__class__.__name__, True, False


def calculate_next_attempt(attempt_count: int) -> datetime:
    if attempt_count <= 1:
        delay = timedelta(minutes=5)
    elif attempt_count == 2:
        delay = timedelta(minutes=30)
    else:
        delay = timedelta(hours=2)
    return datetime.now() + delay


def _move_failed_file(job: dict, dirs: dict[str, Path]) -> str | None:
    current_path = _existing_audio_path(job, include_completed=True)
    if current_path is None:
        return None
    failed_path = safe_move(current_path, dirs["failed"], _job_filename(job))
    update_job_paths(job["id"], failed_path=str(failed_path))
    return str(failed_path)


def _transcript_path_for_call(job: dict, call: dict) -> Path:
    existing = job.get("transcript_path")
    if existing:
        return Path(existing)

    model_slug = (call.get("model") or MODEL_NAME or "unknown").replace("/", "-")
    return OUTPUT_DIR / f"{Path(job['original_filename']).stem}_{model_slug}_transcript.txt"


def _find_completed_audio(job: dict, dirs: dict[str, Path]) -> Path | None:
    candidates = []
    if job.get("completed_path"):
        candidates.append(Path(job["completed_path"]))
    candidates.append(dirs["completed"] / _job_filename(job))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _audio_for_finalization(job: dict, dirs: dict[str, Path]) -> Path:
    completed = _find_completed_audio(job, dirs)
    if completed is not None:
        update_job_paths(job["id"], completed_path=str(completed))
        return completed

    current_path = _existing_audio_path(job)
    if current_path is None:
        raise FileNotFoundError(
            f"No audio file available to finalize job {job['id']}"
        )

    completed_path = safe_move(current_path, dirs["completed"], _job_filename(job))
    update_job_paths(job["id"], completed_path=str(completed_path))
    return completed_path


def _schedule_finalization_retry(
    job: dict,
    exc: Exception,
    tb: str | None = None,
) -> None:
    attempts = int(job.get("attempt_count", 0))
    max_attempts = int(job.get("max_attempts", MAX_ATTEMPTS))
    message = str(exc)

    if attempts < max_attempts:
        next_attempt = calculate_next_attempt(max(attempts, 1))
        mark_job_retrying(
            job["id"],
            "finalization_error",
            message,
            next_attempt.isoformat(timespec="seconds"),
            traceback=tb,
        )
        log.warning(
            "Finalization retry scheduled | job_id=%s attempt=%s/%s next_attempt=%s error=%s",
            job["id"],
            attempts,
            max_attempts,
            next_attempt.isoformat(timespec="seconds"),
            message,
        )
    else:
        mark_job_failed(
            job["id"],
            "finalization_error",
            message,
            failed_path=job.get("failed_path"),
            traceback=tb,
        )
        log.error(
            "Finalization failed permanently | job_id=%s attempts=%s/%s error=%s",
            job["id"],
            attempts,
            max_attempts,
            message,
        )


def _finalize_saved_call(job: dict, dirs: dict[str, Path]) -> bool:
    """
    Complete transcript writing and file movement from a saved calls row.

    This function is intentionally idempotent. It must never call the provider.
    """
    call_id = job.get("call_id")
    if not call_id:
        return False

    try:
        call = get_call(int(call_id))
        if call is None:
            raise RuntimeError(f"Saved call row not found: {call_id}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        transcript_path = _transcript_path_for_call(job, call)
        if not transcript_path.exists():
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(call.get("transcript") or "", encoding="utf-8")

        completed_path = _audio_for_finalization(job, dirs)
        mark_job_transcribed(
            job["id"],
            int(call_id),
            completed_path=str(completed_path),
            transcript_path=str(transcript_path.resolve()),
            success=bool(call.get("transcript")),
        )
        log.info(
            "Saved call finalized | job_id=%s call_id=%s completed=%s transcript=%s",
            job["id"],
            call_id,
            completed_path,
            transcript_path,
        )
        return True
    except Exception as exc:
        _schedule_finalization_retry(job, exc, traceback.format_exc())
        return False


def _process_job(job: dict, dirs: dict[str, Path], batch_mode: bool = False) -> None:
    job_id = job["id"]
    log.info(
        "Job claimed | job_id=%s attempt=%s/%s file=%s hash=%s",
        job_id,
        job["attempt_count"],
        job["max_attempts"],
        job["original_filename"],
        _short_hash(job.get("file_hash")),
    )

    try:
        if job.get("call_id"):
            _finalize_saved_call(job, dirs)
            return

        processing_path = _ensure_processing_file(job, dirs)
        log.info("Transcription started | job_id=%s file=%s", job_id, processing_path)

        started = time.time()
        result = transcribe_audio_file(str(processing_path))
        result["filename"] = job["original_filename"]
        call_id = save_call_and_mark_finalizing(
            job_id,
            result,
            lkr_rate=LKR_RATE,
            batch_mode=batch_mode,
        )
        finalizing_job = get_job(job_id) or {**job, "call_id": call_id}
        finalized = _finalize_saved_call(finalizing_job, dirs)

        elapsed = time.time() - started
        log.info(
            "Job provider result saved | job_id=%s call_id=%s finalized=%s "
            "success=%s duration=%.1fs cost_usd=%.6f",
            job_id,
            call_id,
            finalized,
            bool(result.get("success")),
            elapsed,
            result.get("total_cost_usd", 0.0),
        )

    except Exception as exc:
        error_type, retryable, invalid_file = classify_error(exc)
        tb = traceback.format_exc()
        message = str(exc)
        latest_job = get_job(job_id) or job

        if latest_job.get("call_id"):
            _finalize_saved_call(latest_job, dirs)
            return

        if invalid_file:
            failed_path = _move_failed_file(latest_job, dirs)
            mark_job_invalid(job_id, message, failed_path=failed_path, traceback=tb)
            log.error(
                "Job invalid | job_id=%s error_type=%s file=%s error=%s",
                job_id,
                error_type,
                latest_job.get("original_filename"),
                message,
            )
            return

        attempts = int(latest_job.get("attempt_count", job.get("attempt_count", 1)))
        max_attempts = int(latest_job.get("max_attempts", MAX_ATTEMPTS))

        if retryable and attempts < max_attempts:
            next_attempt = calculate_next_attempt(attempts)
            mark_job_retrying(
                job_id,
                error_type,
                message,
                next_attempt.isoformat(timespec="seconds"),
                traceback=tb,
            )
            log.warning(
                "Job retry scheduled | job_id=%s attempt=%s/%s next_attempt=%s "
                "error_type=%s error=%s",
                job_id,
                attempts,
                max_attempts,
                next_attempt.isoformat(timespec="seconds"),
                error_type,
                message,
            )
            return

        failed_path = _move_failed_file(latest_job, dirs)
        mark_job_failed(job_id, error_type, message, failed_path=failed_path, traceback=tb)
        log.error(
            "Job failed | job_id=%s attempts=%s/%s error_type=%s error=%s",
            job_id,
            attempts,
            max_attempts,
            error_type,
            message,
        )


def _recover_processing_file(path: Path, dirs: dict[str, Path]) -> None:
    prefix, _, _ = path.name.partition("_")
    if not prefix.isdigit():
        recovered = safe_move(path, dirs["incoming"], f"recovered_{path.name}")
        log.warning("Recovery moved orphan processing file back to incoming | path=%s", recovered)
        return

    job = get_job(int(prefix))
    if not job:
        recovered = safe_move(path, dirs["incoming"], f"recovered_{path.name}")
        log.warning("Recovery moved unknown processing file back to incoming | path=%s", recovered)
        return

    update_job_paths(job["id"], processing_path=str(path.resolve()))
    if job["status"] == JOB_TRANSCRIBED:
        completed = safe_move(path, dirs["completed"], path.name)
        update_job_paths(job["id"], completed_path=str(completed))
        log.info(
            "Recovery moved terminal processing file to completed | job_id=%s path=%s",
            job["id"],
            completed,
        )
    elif job["status"] in {JOB_FAILED, JOB_INVALID}:
        failed = safe_move(path, dirs["failed"], path.name)
        update_job_paths(job["id"], failed_path=str(failed))
        log.info(
            "Recovery moved terminal processing file to failed | job_id=%s path=%s",
            job["id"],
            failed,
        )
    elif job["status"] == JOB_SKIPPED_DUPLICATE:
        archived = safe_move(path, dirs["archive"] / "duplicates", path.name)
        update_job_paths(job["id"], archive_path=str(archived))
        log.info(
            "Recovery moved duplicate processing file to archive | job_id=%s path=%s",
            job["id"],
            archived,
        )
    else:
        log.info("Recovery linked processing file | job_id=%s path=%s", job["id"], path)


def recover_filesystem_state(dirs: dict[str, Path], batch_mode: bool = False) -> None:
    recovered = recover_stale_jobs(stale_after_minutes=STALE_JOB_MINUTES)
    if recovered:
        log.info("Recovered stale processing jobs | count=%s", recovered)

    for job in get_jobs_needing_finalization():
        _finalize_saved_call(job, dirs)

    for path in sorted(dirs["processing"].iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
            _recover_processing_file(path, dirs)

    for path in sorted(dirs["incoming"].iterdir()):
        if path.is_file():
            register_audio_file(path, dirs, batch_mode=batch_mode)


def _worker(dirs: dict[str, Path], batch_mode: bool = False) -> None:
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"
    log.info("Worker started | worker_id=%s", worker_id)

    while not _stop_event.is_set():
        job = claim_next_job(worker_id, stale_after_minutes=STALE_JOB_MINUTES)
        if job is None:
            _wake_event.wait(WORKER_IDLE_SLEEP_S)
            _wake_event.clear()
            continue

        _process_job(job, dirs, batch_mode=batch_mode)

    log.info("Worker stopped | worker_id=%s", worker_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-transcribe audio files dropped into incoming/."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(DEFAULT_BASE_DIR),
        help=(
            "Lifecycle base folder containing incoming/, processing/, completed/, "
            f"failed/, archive/ (default: {DEFAULT_BASE_DIR})"
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Preserve existing behavior by marking completed call rows as batch_mode",
    )
    args = parser.parse_args()

    dirs = lifecycle_dirs(Path(args.input))
    ensure_lifecycle_dirs(dirs)
    init_db()

    log.info("=" * 60)
    log.info("SLT Call Center - Durable Auto-Transcription Watcher")
    log.info("  Base     : %s", dirs["base"])
    log.info("  Incoming : %s", dirs["incoming"])
    log.info("  Output   : %s", OUTPUT_DIR)
    log.info("  Mode     : %s", "batch flag" if args.batch else "real-time")
    log.info("  Model    : %s", MODEL_NAME)
    log.info("  LKR rate : %s", LKR_RATE)
    log.info("  Supported: %s", ", ".join(sorted(SUPPORTED_EXT)))
    log.info("=" * 60)

    recover_filesystem_state(dirs, batch_mode=args.batch)

    worker_thread = threading.Thread(
        target=_worker,
        kwargs={"dirs": dirs, "batch_mode": args.batch},
        daemon=True,
    )
    worker_thread.start()

    observer = Observer()
    observer.schedule(_AudioDropHandler(dirs, args.batch), str(dirs["incoming"]), recursive=False)
    observer.start()
    log.info("Watching for new files in incoming/. Press Ctrl+C to stop.\n")

    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("\nStopping watcher...")
    finally:
        _stop_event.set()
        _wake_event.set()
        observer.stop()
        observer.join()
        worker_thread.join()
        log.info("Watcher stopped. Goodbye.")


if __name__ == "__main__":
    main()
