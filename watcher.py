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

Requires:
    pip install watchdog --break-system-packages
"""
from __future__ import annotations

import argparse
import logging
import queue
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
from gemini_flash_stt import transcribe_audio_file

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_WATCH_DIR = Path(__file__).parent / "input_audio"
OUTPUT_DIR        = Path(__file__).parent / "output"
LKR_RATE          = 316.0   # Update to current USD→LKR exchange rate
SUPPORTED_EXT     = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
FILE_SETTLE_SECS  = 1.5     # Wait for file write to finish before processing

# ── Logging ───────────────────────────────────────────────────────────────────
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

# ── Queue shared between watcher thread and worker thread ────────────────────
_file_queue: queue.Queue[Path | None] = queue.Queue()


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
    Pulls audio files off the queue one at a time, transcribes them,
    saves to the database, and writes the transcript to output/.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    while True:
        audio_path = _file_queue.get()
        if audio_path is None:          # sentinel — time to stop
            break

        log.info(f"🎙  Transcribing: {audio_path.name}")
        t0 = time.time()

        try:
            result = transcribe_audio_file(str(audio_path))
            call_id = save_call(result, lkr_rate=LKR_RATE, batch_mode=batch_mode)

            # Write transcript file — model name included so switching models
            # never overwrites a previous transcript for the same audio file.
            # e.g. call_gemini-2.5-flash_transcript.txt
            #      call_gemini-2.5-pro_transcript.txt
            model_slug    = result.get("model", "unknown").replace("/", "-")
            transcript_path = OUTPUT_DIR / f"{audio_path.stem}_{model_slug}_transcript.txt"
            transcript_path.write_text(result["transcript"], encoding="utf-8")

            elapsed = time.time() - t0
            silence_s = result.get("silence_removed_seconds", 0)
            silence_saved_rs = silence_s / 1000 * 258 / 1_000_000 * 1.0 * LKR_RATE  # audio token savings

            log.info(
                f"✅  Done: {audio_path.name}\n"
                f"       Duration   : {result['duration_seconds']:.1f}s  "
                f"(silence stripped: {silence_s:.1f}s → saved ≈ Rs.{silence_saved_rs:.4f})\n"
                f"       Languages  : {', '.join(result.get('languages_detected', [])) or 'unknown'}\n"
                f"       Tokens     : {result['total_tokens']:,} total  "
                f"(audio={result['audio_tokens']:,}  text={result['text_input_tokens']:,}  "
                f"out={result['output_tokens']:,})\n"
                f"       Cost       : ${result['total_cost_usd']:.6f}  "
                f"(Rs. {result['total_cost_usd'] * LKR_RATE:.4f})\n"
                f"       Transcript : {transcript_path}\n"
                f"       DB id      : {call_id}  |  processed in {elapsed:.1f}s"
            )

        except FileNotFoundError as exc:
            log.error(f"File not found: {exc}")
        except RuntimeError as exc:
            log.error(f"Transcription failed for {audio_path.name}: {exc}")
        except Exception as exc:
            log.exception(f"Unexpected error for {audio_path.name}: {exc}")
        finally:
            _file_queue.task_done()


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

    log.info("=" * 60)
    log.info("SLT Call Center — Auto-Transcription Watcher")
    log.info(f"  Watching : {watch_dir}")
    log.info(f"  Output   : {OUTPUT_DIR}")
    log.info(f"  Mode     : {'batch' if args.batch else 'real-time'}")
    log.info(f"  LKR rate : {LKR_RATE}")
    log.info("  Supported: " + ", ".join(sorted(SUPPORTED_EXT)))
    log.info("=" * 60)

    # ── Scan for files already sitting in the folder ──────────────────────
    # The watcher only catches NEW files. If you dropped files before
    # starting, this picks them all up automatically.
    already_there = sorted([
        p for p in watch_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])
    if already_there:
        log.info(f"Found {len(already_there)} existing file(s) in folder — queuing them all:")
        for p in already_there:
            log.info(f"  + {p.name}")
            _file_queue.put(p)
    else:
        log.info("No existing files found. Drop audio files into the folder to begin.")

    log.info("Watching for new files... Press Ctrl+C to stop.\n")

    # Start worker thread
    worker_thread = threading.Thread(
        target=_worker, kwargs={"batch_mode": args.batch}, daemon=True
    )
    worker_thread.start()

    # Start filesystem observer (uses inotify on Linux — no polling)
    observer = Observer()
    observer.schedule(_AudioDropHandler(), str(watch_dir), recursive=False)
    observer.start()

    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("\nStopping watcher...")
        observer.stop()
        _file_queue.put(None)   # signal worker to finish

    observer.join()
    worker_thread.join()
    log.info("Watcher stopped. Goodbye.")


if __name__ == "__main__":
    main()
