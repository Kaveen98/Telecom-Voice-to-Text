"""
watcher.py
Windows Server realtime transcription watcher.

Runtime command:
    python watcher.py

Workflow:
    incoming -> processing -> Gemini transcription -> TXT/JSON outputs
    -> optional MySQL metadata -> completed

TXT transcript files and JSON metadata files are the primary output. MySQL is
secondary metadata/index storage only, so database failures are logged as
warnings and never trigger another Gemini call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    BASE_DIR,
    INPUT_COMPLETED_DIR,
    INPUT_FAILED_DIR,
    INPUT_INCOMING_DIR,
    INPUT_PROCESSING_DIR,
    LOG_DIR,
    TRANSCRIPTIONS_DIR,
    app_now,
)
from database import (
    is_database_enabled,
    is_file_already_processed,
    save_failure_record,
    save_transcription_record,
)


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".opus",
}

DEFAULT_STABLE_CHECK_SECONDS = 5.0
DEFAULT_STABLE_CHECK_INTERVAL = 1.0
DEFAULT_STABLE_MAX_WAIT_SECONDS = 300.0

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WHITESPACE = re.compile(r"\s+")
_CREDENTIAL_JSON_PATTERN = re.compile(
    r"[\w:./\\ -]*credentials[\w./\\ -]*\.json",
    re.IGNORECASE,
)

log = logging.getLogger("watcher")
_file_queue: queue.Queue[Path | None] = queue.Queue()
_shutdown = threading.Event()
_queued_paths: set[Path] = set()
_queued_lock = threading.Lock()


@dataclass(frozen=True)
class RuntimePaths:
    incoming: Path
    processing: Path
    completed: Path
    failed: Path
    transcriptions: Path
    logs: Path


@dataclass(frozen=True)
class DatabaseStatus:
    status: str
    record_id: int | None = None
    error: str | None = None


class StableFileError(RuntimeError):
    """Raised when an incoming file cannot be safely processed."""


class ShutdownDuringStableCheck(RuntimeError):
    """Raised when shutdown is requested before a queued file is moved."""


class _DailyFileHandler(logging.Handler):
    """Write logs to logs/watcher_YYYY-MM-DD.log and rotate at midnight."""

    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        self.log_dir = log_dir
        self._active_date = ""
        self._file_handler: logging.FileHandler | None = None

    def _ensure_file_handler(self) -> logging.FileHandler:
        current_date = app_now().strftime("%Y-%m-%d")
        if self._file_handler is not None and current_date == self._active_date:
            return self._file_handler

        if self._file_handler is not None:
            self._file_handler.close()

        self._active_date = current_date
        log_path = self.log_dir / f"watcher_{current_date}.log"
        self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
        self._file_handler.setLevel(self.level)
        if self.formatter is not None:
            self._file_handler.setFormatter(self.formatter)
        return self._file_handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_file_handler().emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None
        super().close()


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _resolve_runtime_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _runtime_paths(incoming_override: str | None = None) -> RuntimePaths:
    return RuntimePaths(
        incoming=_resolve_runtime_path(incoming_override or INPUT_INCOMING_DIR),
        processing=_resolve_runtime_path(INPUT_PROCESSING_DIR),
        completed=_resolve_runtime_path(INPUT_COMPLETED_DIR),
        failed=_resolve_runtime_path(INPUT_FAILED_DIR),
        transcriptions=_resolve_runtime_path(TRANSCRIPTIONS_DIR),
        logs=_resolve_runtime_path(LOG_DIR),
    )


def _ensure_runtime_folders(paths: RuntimePaths) -> None:
    # These folders are runtime state, not source-controlled data.
    for folder in (
        paths.incoming,
        paths.processing,
        paths.completed,
        paths.failed,
        paths.transcriptions,
        paths.logs,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = _DailyFileHandler(log_dir)
    file_handler.setFormatter(formatter)

    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    log.addHandler(stream_handler)
    log.addHandler(file_handler)


def _sanitize_filename_component(value: Any, fallback: str = "audio") -> str:
    text = str(value or "").strip()
    text = _INVALID_FILENAME_CHARS.sub("-", text)
    text = _WHITESPACE.sub("-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip(" .-_")
    return (text or fallback)[:140]


def _timestamp(value: Any = None) -> str:
    dt = value if value is not None else app_now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def _output_date(value: Any = None) -> str:
    dt = value if value is not None else app_now()
    return dt.strftime("%Y-%m-%d")


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def _audio_archive_path(root: Path, original_file_name: str, marker: str, when: Any) -> Path:
    date_dir = root / _output_date(when)
    date_dir.mkdir(parents=True, exist_ok=True)

    original = Path(original_file_name)
    safe_stem = _sanitize_filename_component(original.stem)
    suffix = original.suffix.lower()
    file_stem = f"{safe_stem}__{marker}_{_timestamp(when)}"
    return _unique_path(date_dir, file_stem, suffix)


def _processing_path(processing_dir: Path, original_file_name: str, when: Any) -> Path:
    processing_dir.mkdir(parents=True, exist_ok=True)
    original = Path(original_file_name)
    safe_stem = _sanitize_filename_component(original.stem)
    suffix = original.suffix.lower()
    file_stem = f"{safe_stem}__processing_{_timestamp(when)}"
    return _unique_path(processing_dir, file_stem, suffix)


def _safe_move(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination


def _path_for_storage(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(resolved)


def _stored_path_to_absolute(stored_path: str | None) -> Path | None:
    if not stored_path:
        return None
    path = Path(stored_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _safe_error_message(exc: BaseException, max_len: int = 500) -> str:
    message = str(exc) or exc.__class__.__name__
    for key in (
        "MYSQL_PASSWORD",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
    ):
        value = os.getenv(key, "").strip()
        if value:
            message = message.replace(value, f"<redacted:{key}>")

    message = _CREDENTIAL_JSON_PATTERN.sub("<redacted:credential-json>", message)
    message = " ".join(message.split())
    if len(message) > max_len:
        message = f"{message[:max_len]}..."
    return message


def _wait_for_stable_file(path: Path) -> None:
    # Windows copies may expose the destination file before the copy finishes.
    # Wait until size and mtime remain unchanged for the configured window.
    stable_seconds = _env_float(
        "WATCHER_STABLE_CHECK_SECONDS",
        DEFAULT_STABLE_CHECK_SECONDS,
    )
    check_interval = max(
        0.25,
        _env_float("WATCHER_STABLE_CHECK_INTERVAL", DEFAULT_STABLE_CHECK_INTERVAL),
    )
    max_wait = max(stable_seconds, _env_float(
        "WATCHER_STABLE_MAX_WAIT_SECONDS",
        DEFAULT_STABLE_MAX_WAIT_SECONDS,
    ))
    deadline = time.monotonic() + max_wait
    last_signature: tuple[int, int] | None = None
    unchanged_since: float | None = None

    while True:
        if _shutdown.is_set():
            raise ShutdownDuringStableCheck(
                "shutdown requested before file became stable"
            )

        if not path.exists():
            raise FileNotFoundError(f"Incoming file disappeared: {path.name}")

        try:
            stat_result = path.stat()
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise StableFileError(
                    f"file could not be inspected: {_safe_error_message(exc)}"
                ) from exc
            time.sleep(check_interval)
            continue

        if stat_result.st_size <= 0:
            if time.monotonic() >= deadline:
                raise StableFileError("file is empty")
            last_signature = None
            unchanged_since = None
            time.sleep(check_interval)
            continue

        signature = (stat_result.st_size, stat_result.st_mtime_ns)
        current = time.monotonic()
        file_age = max(0.0, time.time() - stat_result.st_mtime)
        if signature != last_signature:
            last_signature = signature
            unchanged_since = current
        elif (
            unchanged_since is not None
            and current - unchanged_since >= stable_seconds
            and file_age >= stable_seconds
        ):
            log.info("Stable file confirmed: %s (%s bytes)", path.name, stat_result.st_size)
            return

        if current >= deadline:
            raise StableFileError(
                f"file did not become stable within {max_wait:.1f} seconds"
            )
        time.sleep(check_interval)


def _compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _is_duplicate_file(file_hash: str, original_file_name: str, audio_path: Path) -> bool:
    if not file_hash or not is_database_enabled():
        return False
    try:
        return is_file_already_processed(
            file_hash=file_hash,
            original_file_name=original_file_name,
            audio_path=str(audio_path),
        )
    except Exception as exc:
        log.warning(
            "Optional MySQL duplicate check failed for %s: %s",
            original_file_name,
            _safe_error_message(exc),
        )
        return False


def _update_metadata_json(
    saved_info: dict[str, Any],
    updates: dict[str, Any],
) -> None:
    metadata_path = _stored_path_to_absolute(saved_info.get("metadata_json_path"))
    if metadata_path is None:
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(updates)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning(
            "Transcript metadata JSON update failed for %s: %s",
            metadata_path.name,
            _safe_error_message(exc),
        )


def _save_optional_database_record(record: dict[str, Any]) -> DatabaseStatus:
    # TXT/JSON files are already saved before this function is called.
    if not is_database_enabled():
        log.info("MySQL metadata disabled; TXT/JSON outputs remain saved.")
        return DatabaseStatus(status="disabled")

    try:
        result = save_transcription_record(record)
    except Exception as exc:
        safe_message = _safe_error_message(exc)
        log.warning("MySQL metadata warning: %s", safe_message)
        return DatabaseStatus(status="failed", error=safe_message)

    if result.success:
        log.info("MySQL metadata saved: record id %s", result.record_id)
        return DatabaseStatus(status="saved", record_id=result.record_id)

    if result.disabled:
        log.info("MySQL metadata disabled; TXT/JSON outputs remain saved.")
        return DatabaseStatus(status="disabled", error=result.error)

    log.warning("MySQL metadata warning: %s", result.error or "write failed")
    return DatabaseStatus(status="failed", error=result.error)


def _save_optional_failure_record(
    original_file_name: str,
    failed_audio_path: Path | None,
    file_hash: str,
    exc: BaseException,
) -> None:
    if not is_database_enabled():
        return

    try:
        result = save_failure_record(
            original_file_name=original_file_name,
            file_hash=file_hash,
            audio_failed_path=_path_for_storage(failed_audio_path) if failed_audio_path else "",
            error_message=f"{exc.__class__.__name__}: {_safe_error_message(exc)}",
            status="failed",
            mode="realtime",
        )
    except Exception as db_exc:
        log.warning(
            "Optional MySQL failure record write failed: %s",
            _safe_error_message(db_exc),
        )
        return

    if not result.success and not result.disabled:
        log.warning("Optional MySQL failure record warning: %s", result.error)


def _write_failure_error_file(
    failed_audio_path: Path,
    original_file_name: str,
    exc: BaseException,
    when: Any,
) -> Path:
    error_path = failed_audio_path.with_suffix(".error.txt")
    content = "\n".join(
        [
            f"original_file_name: {original_file_name}",
            f"failure_timestamp: {when.isoformat()}",
            f"error_type: {exc.__class__.__name__}",
            f"error_message: {_safe_error_message(exc)}",
            "",
        ]
    )
    error_path.write_text(content, encoding="utf-8")
    return error_path


def _move_to_failed(
    paths: RuntimePaths,
    current_path: Path,
    original_file_name: str,
    exc: BaseException,
) -> Path | None:
    failed_at = app_now()
    if not current_path.exists():
        log.error(
            "Cannot move failed file because it is missing: %s",
            original_file_name,
        )
        return None

    failed_path = _audio_archive_path(paths.failed, original_file_name, "failed", failed_at)
    failed_path = _safe_move(current_path, failed_path)
    log.error(
        "Moved failed audio to %s after %s: %s",
        _path_for_storage(failed_path),
        exc.__class__.__name__,
        _safe_error_message(exc),
    )

    try:
        error_path = _write_failure_error_file(
            failed_audio_path=failed_path,
            original_file_name=original_file_name,
            exc=exc,
            when=failed_at,
        )
        log.info("Failure details written to %s", _path_for_storage(error_path))
    except Exception as error_file_exc:
        log.warning(
            "Could not write failure error file for %s: %s",
            original_file_name,
            _safe_error_message(error_file_exc),
        )

    return failed_path


def _process_candidate(path: Path, paths: RuntimePaths) -> None:
    original_path = path
    original_file_name = path.name
    file_hash = ""
    processing_path: Path | None = None

    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        log.info("Ignoring unsupported file: %s", path.name)
        return

    try:
        log.info("Detected file: %s", original_file_name)
        _wait_for_stable_file(path)

        processing_at = app_now()
        processing_path = _processing_path(paths.processing, original_file_name, processing_at)
        processing_path = _safe_move(path, processing_path)
        log.info("Moved to processing: %s", _path_for_storage(processing_path))

        try:
            file_hash = _compute_file_hash(processing_path)
        except Exception as hash_exc:
            log.warning(
                "Could not compute file hash for %s: %s",
                original_file_name,
                _safe_error_message(hash_exc),
            )

        if _is_duplicate_file(file_hash, original_file_name, processing_path):
            duplicate_at = app_now()
            duplicate_path = _audio_archive_path(
                paths.completed,
                original_file_name,
                "duplicate",
                duplicate_at,
            )
            duplicate_path = _safe_move(processing_path, duplicate_path)
            log.info(
                "Duplicate already recorded in MySQL; moved audio to %s without a new Gemini call.",
                _path_for_storage(duplicate_path),
            )
            return

        log.info("Transcription started: %s", original_file_name)
        # Lazy import keeps --help and --dry-run free of Gemini setup/API side effects.
        from gemini_flash_stt import save_transcription_outputs, transcribe_audio_file

        result = transcribe_audio_file(str(processing_path))
        result.update(
            {
                "original_file_name": original_file_name,
                "file_hash": file_hash,
                "audio_input_path": _path_for_storage(original_path),
                "audio_processing_path": _path_for_storage(processing_path),
                "mode": "realtime",
                "status": "completed",
            }
        )

        saved_info = save_transcription_outputs(
            result=result,
            audio_path=original_path,
        )
        transcript_txt = saved_info.get("transcript_txt_path", "")
        metadata_json = saved_info.get("metadata_json_path", "")
        log.info("TXT transcript saved: %s", transcript_txt)
        log.info("JSON metadata saved: %s", metadata_json)

        completed_at = app_now()
        completed_path = _audio_archive_path(
            paths.completed,
            original_file_name,
            "transcribed",
            completed_at,
        )

        record = {
            **result,
            **saved_info,
            "database_save_status": "pending",
            "audio_completed_path": _path_for_storage(completed_path),
            "transcribed_at": saved_info.get("transcript_saved_at"),
        }
        db_status = _save_optional_database_record(record)
        _update_metadata_json(
            saved_info,
            {
                "database_save_status": db_status.status,
                "database_record_id": db_status.record_id,
                "database_error": db_status.error or "",
                "audio_input_path": _path_for_storage(original_path),
                "audio_processing_path": _path_for_storage(processing_path),
                "audio_completed_path": _path_for_storage(completed_path),
                "file_hash": file_hash,
            },
        )

        completed_path = _safe_move(processing_path, completed_path)
        log.info("Moved completed audio to %s", _path_for_storage(completed_path))

        duration = float(result.get("duration_seconds", 0) or 0)
        languages = ", ".join(result.get("languages_detected", [])) or "unknown"
        total_tokens = int(result.get("total_tokens", 0) or 0)
        total_cost = float(result.get("total_cost_usd", 0) or 0)
        elapsed = float(result.get("elapsed_seconds", 0) or 0)
        log.info(
            "Completed %s | duration=%.1fs | languages=%s | tokens=%s | cost_usd=%.6f | api_elapsed=%.1fs | db=%s",
            original_file_name,
            duration,
            languages,
            f"{total_tokens:,}",
            total_cost,
            elapsed,
            db_status.status,
        )

    except ShutdownDuringStableCheck:
        log.info("Shutdown requested; leaving file in incoming: %s", original_file_name)
    except FileNotFoundError as exc:
        log.warning("%s", _safe_error_message(exc))
    except Exception as exc:
        current_path = processing_path if processing_path and processing_path.exists() else path
        failed_path = _move_to_failed(paths, current_path, original_file_name, exc)
        _save_optional_failure_record(original_file_name, failed_path, file_hash, exc)


def _queue_path_key(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _enqueue_candidate(path: Path) -> None:
    if _shutdown.is_set():
        return

    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        log.info("Ignoring unsupported file: %s", path.name)
        return

    key = _queue_path_key(path)
    with _queued_lock:
        if key in _queued_paths:
            return
        _queued_paths.add(key)

    log.info("Queued audio file: %s", path.name)
    _file_queue.put(key)


def _mark_dequeued(path: Path) -> None:
    key = _queue_path_key(path)
    with _queued_lock:
        _queued_paths.discard(key)


def _worker(paths: RuntimePaths) -> None:
    """Sequential worker: one audio file is processed at a time."""
    while True:
        path = _file_queue.get()
        try:
            if path is None:
                return
            if _shutdown.is_set():
                log.info("Shutdown requested; leaving queued file in incoming: %s", path.name)
                continue
            _process_candidate(path, paths)
        finally:
            if path is not None:
                _mark_dequeued(path)
            _file_queue.task_done()


def _handle_signal(signum: int, frame: object) -> None:
    _ = frame
    log.info("Received signal %s. Stopping watcher after current work.", signum)
    _shutdown.set()


def _load_watchdog() -> tuple[Any, Any]:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        log.error("watchdog is not installed. Run: python -m pip install -r requirements.txt")
        raise SystemExit(1)
    return FileSystemEventHandler, Observer


def _start_observer(paths: RuntimePaths) -> Any:
    FileSystemEventHandler, Observer = _load_watchdog()

    class AudioDropHandler(FileSystemEventHandler):  # type: ignore[misc, valid-type]
        """Queue files created or moved into the incoming folder."""

        def on_created(self, event: Any) -> None:
            if not event.is_directory:
                _enqueue_candidate(Path(event.src_path))

        def on_moved(self, event: Any) -> None:
            if not event.is_directory:
                _enqueue_candidate(Path(event.dest_path))

    observer = Observer()
    observer.schedule(AudioDropHandler(), str(paths.incoming), recursive=False)
    observer.start()
    return observer


def _queue_existing_files(paths: RuntimePaths) -> None:
    existing_files = sorted(
        path
        for path in paths.incoming.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

    if not existing_files:
        log.info("Incoming folder has no supported audio files.")
        return

    log.info("Found %s existing supported audio file(s).", len(existing_files))
    for path in existing_files:
        _enqueue_candidate(path)


def _log_startup(paths: RuntimePaths, dry_run: bool) -> None:
    log.info("=" * 60)
    log.info("Telecom Voice-to-Text realtime watcher")
    log.info("Incoming      : %s", paths.incoming)
    log.info("Processing    : %s", paths.processing)
    log.info("Completed     : %s", paths.completed)
    log.info("Failed        : %s", paths.failed)
    log.info("Transcriptions: %s", paths.transcriptions)
    log.info("Logs          : %s", paths.logs)
    log.info("Supported     : %s", ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS)))
    log.info(
        "Stable check  : %.1fs unchanged, %.1fs interval, %.1fs max wait",
        _env_float("WATCHER_STABLE_CHECK_SECONDS", DEFAULT_STABLE_CHECK_SECONDS),
        max(
            0.25,
            _env_float("WATCHER_STABLE_CHECK_INTERVAL", DEFAULT_STABLE_CHECK_INTERVAL),
        ),
        _env_float("WATCHER_STABLE_MAX_WAIT_SECONDS", DEFAULT_STABLE_MAX_WAIT_SECONDS),
    )
    log.info("MySQL metadata: %s", "enabled" if is_database_enabled() else "disabled")
    log.info("Dry run       : %s", "yes" if dry_run else "no")
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously transcribe audio dropped into input_audio/incoming."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help=f"Incoming folder override (default: {INPUT_INCOMING_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create folders, configure logging, print startup config, then exit.",
    )
    args = parser.parse_args()

    paths = _runtime_paths(args.input)
    _ensure_runtime_folders(paths)
    _setup_logging(paths.logs)
    _log_startup(paths, dry_run=args.dry_run)

    if args.dry_run:
        log.info("Dry run complete. No files were processed and no Gemini call was made.")
        return

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    worker_thread = threading.Thread(target=_worker, args=(paths,), daemon=False)
    worker_thread.start()

    observer = _start_observer(paths)
    _queue_existing_files(paths)
    log.info("Watcher running. Press Ctrl+C to stop.")

    try:
        while not _shutdown.is_set():
            time.sleep(1)
    finally:
        log.info("Stopping filesystem observer...")
        observer.stop()
        observer.join()

        log.info("Waiting for active transcription to finish...")
        _file_queue.put(None)
        worker_thread.join()
        log.info("Watcher stopped cleanly.")


if __name__ == "__main__":
    main()
