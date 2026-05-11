"""
watcher.py
Linux folder-watcher auto-transcription pipeline.

Usage:
    python watcher.py                        # watch default input_audio/ folder
    python watcher.py --input /path/to/dir   # watch custom folder
    python watcher.py --batch                # queue for batch mode instead of real-time

Drop any audio file (mp3, wav, m4a, ogg, flac, aac, opus) into the watched
folder and it will be automatically transcribed, logged to the database, and
saved as a .txt file in output/.

On a Linux server, run as a systemd service — see slt-watcher.service

Requires:
    pip install watchdog --break-system-packages
"""
from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path

try:
    from watchdog.events import FileCreatedEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("ERROR: watchdog is not installed.")
    print("Run:  pip install watchdog --break-system-packages")
    sys.exit(1)

from database import save_call
from gemini_flash_stt import LKR_RATE, _AUDIO_TOKENS_PER_SECOND, transcribe_audio_file

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_WATCH_DIR = Path(__file__).parent / "input_audio"
OUTPUT_DIR        = Path(__file__).parent / "output"
SUPPORTED_EXT     = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
FILE_SETTLE_SECS  = 1.5     # Wait for file write to finish before processing
MAX_RETRIES       = 2       # Retry failed transcriptions before giving up

# ── Logging — writes to both terminal and watcher.log ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "watcher.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("watcher")

# ── Shared queue between filesystem watcher thread and worker thread ──────────
_file_queue: queue.Queue[Path | None] = queue.Queue()
_shutdown   = threading.Event()


class _AudioDropHandler(FileSystemEventHandler):
    """Fires when a new file appears in the watched folder."""

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in SUPPORTED_EXT:
            time.sleep(FILE_SETTLE_SECS)   # let the file finish writing
            log.info(f"📥  Detected: {path.name}")
            _file_queue.put(path)


def _worker(batch_mode: bool = False) -> None:
    """
    Pulls files off the queue one at a time.
    Retries failed transcriptions up to MAX_RETRIES times.
    Stops cleanly when it receives None (shutdown signal).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    while True:
        audio_path = _file_queue.get()
        if audio_path is None:      # shutdown sentinel
            break

        log.info(f"🎙  Transcribing: {audio_path.name}")
        t0 = time.time()

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = transcribe_audio_file(str(audio_path))
                lkr_rate = result.get("lkr_rate", LKR_RATE)   # live rate from result
                call_id = save_call(result, lkr_rate=lkr_rate, batch_mode=batch_mode)

                # Save transcript — model name in filename prevents overwriting
                model_slug      = result.get("model", "unknown").replace("/", "-")
                transcript_path = OUTPUT_DIR / f"{audio_path.stem}_{model_slug}_transcript.txt"
                transcript_path.write_text(result["transcript"], encoding="utf-8")

                elapsed   = time.time() - t0
                silence_s = result.get("silence_removed_seconds", 0)

                # Silence savings: stripped_seconds × tok/s ÷ 1M × audio_rate × LKR
                # Uses 26 tok/sec (standard file upload rate, not 258 Live API rate)
                silence_saved_rs = (
                    silence_s * _AUDIO_TOKENS_PER_SECOND / 1_000_000 * 1.0 * lkr_rate
                )

                log.info(
                    f"✅  Done: {audio_path.name}\n"
                    f"       Duration   : {result['duration_seconds']:.1f}s  "
                    f"(silence stripped: {silence_s:.1f}s → saved ≈ Rs.{silence_saved_rs:.4f})\n"
                    f"       Languages  : {', '.join(result.get('languages_detected', [])) or 'unknown'}\n"
                    f"       Tokens     : {result['total_tokens']:,} total  "
                    f"(audio={result['audio_tokens']:,}  text={result['text_input_tokens']:,}  "
                    f"out={result['output_tokens']:,})\n"
                    f"       Cost       : ${result['total_cost_usd']:.6f}  "
                    f"(Rs. {result['total_cost_usd'] * lkr_rate:.4f})  [rate: {lkr_rate:.2f}]\n"
                    f"       Transcript : {transcript_path}\n"
                    f"       DB id      : {call_id}  |  processed in {elapsed:.1f}s"
                )
                last_error = None
                break   # success — stop retrying

            except FileNotFoundError as exc:
                log.error(f"File not found: {exc}")
                break   # don't retry missing files
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    log.warning(
                        f"Attempt {attempt} failed for {audio_path.name}: {exc}. "
                        f"Retrying ({attempt}/{MAX_RETRIES})..."
                    )
                    time.sleep(5)

        if last_error is not None:
            log.error(
                f"❌  Failed after {MAX_RETRIES} attempts: {audio_path.name} — {last_error}"
            )

        _file_queue.task_done()


def _handle_signal(signum: int, frame: object) -> None:
    """
    Handle SIGTERM and SIGINT cleanly.
    systemd sends SIGTERM when stopping the service — this ensures the
    current transcription finishes before the process exits.
    """
    log.info(f"Received signal {signum}. Shutting down gracefully...")
    _shutdown.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-transcribe audio files dropped into a folder."
    )
    parser.add_argument(
        "--input", "-i",
        default=str(DEFAULT_WATCH_DIR),
        help=f"Folder to watch (default: {DEFAULT_WATCH_DIR})",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Mark processed calls as batch_mode in the database",
    )
    args = parser.parse_args()

    watch_dir = Path(args.input)
    watch_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Register signal handlers — required for systemd SIGTERM on Linux server
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    log.info("=" * 60)
    log.info("SLT Call Center — Auto-Transcription Watcher")
    log.info(f"  Watching : {watch_dir}")
    log.info(f"  Output   : {OUTPUT_DIR}")
    log.info(f"  Mode     : {'batch' if args.batch else 'real-time'}")
    log.info(f"  LKR rate : {LKR_RATE}")
    log.info(f"  Retries  : {MAX_RETRIES} per file")
    log.info("  Supported: " + ", ".join(sorted(SUPPORTED_EXT)))
    log.info("=" * 60)

    # Scan for files already in folder before watcher started
    already_there = sorted([
        p for p in watch_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])
    if already_there:
        log.info(f"Found {len(already_there)} existing file(s) — queuing them:")
        for p in already_there:
            log.info(f"  + {p.name}")
            _file_queue.put(p)
    else:
        log.info("No existing files. Drop audio files into the folder to begin.")

    log.info("Watching for new files... (SIGTERM or Ctrl+C to stop)\n")

    # Start worker thread
    worker_thread = threading.Thread(
        target=_worker, kwargs={"batch_mode": args.batch}, daemon=True
    )
    worker_thread.start()

    # Start filesystem observer (uses inotify on Linux — zero CPU while waiting)
    observer = Observer()
    observer.schedule(_AudioDropHandler(), str(watch_dir), recursive=False)
    observer.start()

    # Main loop — waits for shutdown signal
    while not _shutdown.is_set():
        time.sleep(1)

    # Graceful shutdown
    log.info("Stopping filesystem observer...")
    observer.stop()
    observer.join()

    log.info("Waiting for current transcription to finish...")
    _file_queue.put(None)   # signal worker to exit after current job
    worker_thread.join()

    log.info("Watcher stopped cleanly.")


if __name__ == "__main__":
    main()
