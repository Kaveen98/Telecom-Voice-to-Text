"""
database.py
SQLite logger for per-call transcription records.
Every call processed by gemini_flash_stt is saved here so the dashboard
and billing reconciliation always have per-call detail.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(__file__).parent / "calls.db"

JOB_PENDING = "pending"
JOB_PROCESSING = "processing"
JOB_FINALIZING = "finalizing"
JOB_TRANSCRIBED = "transcribed"
JOB_FAILED = "failed"
JOB_RETRYING = "retrying"
JOB_SKIPPED_DUPLICATE = "skipped_duplicate"
JOB_INVALID = "invalid"

ACTIVE_JOB_STATUSES = {JOB_PENDING, JOB_PROCESSING, JOB_FINALIZING, JOB_RETRYING}

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    filename                 TEXT    NOT NULL,
    audio_path               TEXT,
    duration_seconds         REAL    DEFAULT 0,
    silence_removed_seconds  REAL    DEFAULT 0,
    model                    TEXT,
    input_tokens             INTEGER DEFAULT 0,
    audio_tokens             INTEGER DEFAULT 0,
    text_input_tokens        INTEGER DEFAULT 0,
    output_tokens            INTEGER DEFAULT 0,
    thoughts_tokens          INTEGER DEFAULT 0,
    total_tokens             INTEGER DEFAULT 0,
    audio_input_cost_usd     REAL    DEFAULT 0,
    text_input_cost_usd      REAL    DEFAULT 0,
    output_cost_usd          REAL    DEFAULT 0,
    total_cost_usd           REAL    DEFAULT 0,
    total_cost_lkr           REAL    DEFAULT 0,
    lkr_rate                 REAL    DEFAULT 305,
    languages_detected       TEXT    DEFAULT '',
    transcript               TEXT    DEFAULT '',
    batch_mode               INTEGER DEFAULT 0,
    processed_at             TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_date ON calls (processed_at);

CREATE TABLE IF NOT EXISTS transcription_jobs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,

    file_hash            TEXT    NOT NULL,
    original_filename    TEXT    NOT NULL,
    original_extension   TEXT,
    file_size_bytes      INTEGER DEFAULT 0,
    duplicate_of_job_id  INTEGER,

    incoming_path        TEXT,
    processing_path      TEXT,
    completed_path       TEXT,
    failed_path          TEXT,
    archive_path         TEXT,
    transcript_path      TEXT,

    provider             TEXT    NOT NULL DEFAULT 'vertex_gemini',
    model                TEXT,
    transcription_mode   TEXT    NOT NULL DEFAULT 'realtime',
    preprocess_profile   TEXT    DEFAULT 'default',

    status               TEXT    NOT NULL,
    priority             INTEGER DEFAULT 100,
    attempt_count        INTEGER NOT NULL DEFAULT 0,
    max_attempts         INTEGER NOT NULL DEFAULT 3,
    locked_by            TEXT,
    locked_at            TEXT,
    next_attempt_at      TEXT,

    call_id              INTEGER,
    success              INTEGER DEFAULT 0,

    discovered_at        TEXT    NOT NULL,
    queued_at            TEXT,
    started_at           TEXT,
    completed_at         TEXT,
    failed_at            TEXT,
    updated_at           TEXT    NOT NULL,

    last_error_type      TEXT,
    last_error_message   TEXT,
    last_traceback       TEXT,

    FOREIGN KEY (duplicate_of_job_id) REFERENCES transcription_jobs(id),
    FOREIGN KEY (call_id) REFERENCES calls(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_next_attempt
ON transcription_jobs (status, next_attempt_at, priority, id);

CREATE INDEX IF NOT EXISTS idx_jobs_locked_at
ON transcription_jobs (locked_at);

CREATE INDEX IF NOT EXISTS idx_jobs_file_hash
ON transcription_jobs (file_hash);

CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_unique_hash_profile
ON transcription_jobs (
    file_hash, provider, model, transcription_mode, preprocess_profile
)
WHERE duplicate_of_job_id IS NULL;
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent writes
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with _connect() as conn:
        conn.executescript(_CREATE_SQL)


def save_call(
    result: dict[str, Any],
    lkr_rate: float = 305.0,
    batch_mode: bool = False,
) -> int:
    """
    Persist one transcription result.  Returns the new row id.
    """
    init_db()
    with _connect() as conn:
        return _insert_call(conn, result, lkr_rate, batch_mode)


def _insert_call(
    conn: sqlite3.Connection,
    result: dict[str, Any],
    lkr_rate: float,
    batch_mode: bool,
) -> int:
    total_usd = result.get("total_cost_usd", 0.0)
    langs = result.get("languages_detected", [])
    langs_str = ", ".join(langs) if isinstance(langs, list) else str(langs)

    cur = conn.execute(
        """
        INSERT INTO calls (
            filename, audio_path,
            duration_seconds, silence_removed_seconds,
            model,
            input_tokens, audio_tokens, text_input_tokens,
            output_tokens, thoughts_tokens, total_tokens,
            audio_input_cost_usd, text_input_cost_usd, output_cost_usd,
            total_cost_usd, total_cost_lkr, lkr_rate,
            languages_detected, transcript, batch_mode, processed_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            result.get("filename")
            or Path(result.get("audio_path", "unknown")).name,
            result.get("audio_path", ""),
            result.get("duration_seconds", 0),
            result.get("silence_removed_seconds", 0),
            result.get("model", ""),
            result.get("input_tokens", 0),
            result.get("audio_tokens", 0),
            result.get("text_input_tokens", 0),
            result.get("output_tokens", 0),
            result.get("thoughts_tokens", 0),
            result.get("total_tokens", 0),
            result.get("audio_input_cost_usd", 0),
            result.get("text_input_cost_usd", 0),
            result.get("output_cost_usd", 0),
            total_usd,
            total_usd * lkr_rate,
            lkr_rate,
            langs_str,
            result.get("transcript", ""),
            1 if batch_mode else 0,
            datetime.now().isoformat(),
        ),
    )
    return cur.lastrowid


def save_call_and_mark_finalizing(
    job_id: int,
    result: dict[str, Any],
    lkr_rate: float = 305.0,
    batch_mode: bool = False,
) -> int:
    """
    Atomically save the provider result and mark the job as finalizing.

    This removes the crash window where a call row exists but the job has no
    call_id, which would otherwise allow a restart to call the provider again.
    """
    init_db()
    now = _now()
    with _connect() as conn:
        call_id = _insert_call(conn, result, lkr_rate, batch_mode)
        cur = conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?,
                call_id = ?,
                success = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = NULL,
                last_error_type = NULL,
                last_error_message = NULL,
                last_traceback = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (
                JOB_FINALIZING,
                call_id,
                1 if result.get("success") else 0,
                now,
                job_id,
            ),
        )
        if cur.rowcount != 1:
            raise RuntimeError(f"Job not found while marking finalizing: {job_id}")
        return call_id


def mark_job_finalizing(job_id: int, call_id: int, success: bool = True) -> None:
    """Mark a job as having a saved call row but incomplete file finalization."""
    init_db()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?,
                call_id = ?,
                success = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (JOB_FINALIZING, call_id, 1 if success else 0, now, job_id),
        )


def create_or_get_job(
    *,
    file_hash: str,
    original_filename: str,
    incoming_path: str,
    file_size_bytes: int = 0,
    provider: str = "vertex_gemini",
    model: str = "",
    transcription_mode: str = "realtime",
    preprocess_profile: str = "default",
    original_extension: str = "",
    priority: int = 100,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """
    Create a durable transcription job or record the file as a duplicate.

    A duplicate is scoped by content hash plus processing profile, so the same
    audio may still be reprocessed intentionally with a different model/mode.
    """
    init_db()
    now = _now()

    with _connect() as conn:
        existing_path = conn.execute(
            f"""
            SELECT * FROM transcription_jobs
            WHERE incoming_path = ?
              AND status IN ({','.join('?' for _ in ACTIVE_JOB_STATUSES)})
            ORDER BY id DESC
            LIMIT 1
            """,
            [incoming_path, *sorted(ACTIVE_JOB_STATUSES)],
        ).fetchone()
        if existing_path:
            return dict(existing_path)

        duplicate = conn.execute(
            """
            SELECT * FROM transcription_jobs
            WHERE file_hash = ?
              AND provider = ?
              AND COALESCE(model, '') = COALESCE(?, '')
              AND transcription_mode = ?
              AND preprocess_profile = ?
              AND duplicate_of_job_id IS NULL
            ORDER BY id ASC
            LIMIT 1
            """,
            (
                file_hash,
                provider,
                model,
                transcription_mode,
                preprocess_profile,
            ),
        ).fetchone()

        if duplicate:
            cur = conn.execute(
                """
                INSERT INTO transcription_jobs (
                    file_hash, original_filename, original_extension,
                    file_size_bytes, duplicate_of_job_id,
                    incoming_path, provider, model, transcription_mode,
                    preprocess_profile, status, priority, max_attempts,
                    discovered_at, queued_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    file_hash,
                    original_filename,
                    original_extension,
                    file_size_bytes,
                    duplicate["id"],
                    incoming_path,
                    provider,
                    model,
                    transcription_mode,
                    preprocess_profile,
                    JOB_SKIPPED_DUPLICATE,
                    priority,
                    max_attempts,
                    now,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM transcription_jobs WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            return dict(row)

        try:
            cur = conn.execute(
                """
                INSERT INTO transcription_jobs (
                    file_hash, original_filename, original_extension,
                    file_size_bytes, incoming_path, provider, model,
                    transcription_mode, preprocess_profile, status, priority,
                    max_attempts, discovered_at, queued_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    file_hash,
                    original_filename,
                    original_extension,
                    file_size_bytes,
                    incoming_path,
                    provider,
                    model,
                    transcription_mode,
                    preprocess_profile,
                    JOB_PENDING,
                    priority,
                    max_attempts,
                    now,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            duplicate = conn.execute(
                """
                SELECT * FROM transcription_jobs
                WHERE file_hash = ?
                  AND provider = ?
                  AND COALESCE(model, '') = COALESCE(?, '')
                  AND transcription_mode = ?
                  AND preprocess_profile = ?
                  AND duplicate_of_job_id IS NULL
                ORDER BY id ASC
                LIMIT 1
                """,
                (
                    file_hash,
                    provider,
                    model,
                    transcription_mode,
                    preprocess_profile,
                ),
            ).fetchone()
            if duplicate is None:
                raise
            cur = conn.execute(
                """
                INSERT INTO transcription_jobs (
                    file_hash, original_filename, original_extension,
                    file_size_bytes, duplicate_of_job_id,
                    incoming_path, provider, model, transcription_mode,
                    preprocess_profile, status, priority, max_attempts,
                    discovered_at, queued_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    file_hash,
                    original_filename,
                    original_extension,
                    file_size_bytes,
                    duplicate["id"],
                    incoming_path,
                    provider,
                    model,
                    transcription_mode,
                    preprocess_profile,
                    JOB_SKIPPED_DUPLICATE,
                    priority,
                    max_attempts,
                    now,
                    now,
                    now,
                ),
            )

        row = conn.execute(
            "SELECT * FROM transcription_jobs WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)


def get_job(job_id: int) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        return _row_to_dict(
            conn.execute(
                "SELECT * FROM transcription_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        )


def get_call(call_id: int) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        return _row_to_dict(
            conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        )


def get_jobs_needing_finalization() -> list[dict[str, Any]]:
    """
    Return jobs with saved call rows whose filesystem finalization may need
    to be completed after a restart.
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM transcription_jobs
            WHERE call_id IS NOT NULL
              AND status IN (?, ?, ?)
            ORDER BY updated_at ASC, id ASC
            """,
            (JOB_PROCESSING, JOB_FINALIZING, JOB_RETRYING),
        ).fetchall()
        return [dict(row) for row in rows]


def update_job_paths(
    job_id: int,
    *,
    incoming_path: str | None = None,
    processing_path: str | None = None,
    completed_path: str | None = None,
    failed_path: str | None = None,
    archive_path: str | None = None,
    transcript_path: str | None = None,
) -> None:
    """Update one or more lifecycle paths for a job."""
    init_db()
    updates: list[str] = []
    values: list[Any] = []
    for column, value in {
        "incoming_path": incoming_path,
        "processing_path": processing_path,
        "completed_path": completed_path,
        "failed_path": failed_path,
        "archive_path": archive_path,
        "transcript_path": transcript_path,
    }.items():
        if value is not None:
            updates.append(f"{column} = ?")
            values.append(value)

    if not updates:
        return

    updates.append("updated_at = ?")
    values.append(_now())
    values.append(job_id)

    with _connect() as conn:
        conn.execute(
            f"UPDATE transcription_jobs SET {', '.join(updates)} WHERE id = ?",
            values,
        )


def claim_next_job(
    worker_id: str,
    stale_after_minutes: int = 60,
) -> dict[str, Any] | None:
    """
    Atomically claim the next pending/due job. Provider work must happen after
    this transaction commits.
    """
    init_db()
    recover_stale_jobs(stale_after_minutes=stale_after_minutes)
    now = _now()

    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM transcription_jobs
            WHERE status = ?
               OR (status = ? AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
            ORDER BY priority ASC, COALESCE(next_attempt_at, queued_at, discovered_at) ASC, id ASC
            LIMIT 1
            """,
            (JOB_PENDING, JOB_RETRYING, now),
        ).fetchone()

        if row is None:
            conn.commit()
            return None

        conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?,
                attempt_count = attempt_count + 1,
                locked_by = ?,
                locked_at = ?,
                started_at = COALESCE(started_at, ?),
                updated_at = ?,
                last_error_type = NULL,
                last_error_message = NULL,
                last_traceback = NULL
            WHERE id = ?
            """,
            (
                JOB_PROCESSING,
                worker_id,
                now,
                now,
                now,
                row["id"],
            ),
        )
        claimed = conn.execute(
            "SELECT * FROM transcription_jobs WHERE id = ?", (row["id"],)
        ).fetchone()
        conn.commit()
        return dict(claimed)


def mark_job_transcribed(
    job_id: int,
    call_id: int,
    completed_path: str,
    transcript_path: str,
    archive_path: str | None = None,
    success: bool = True,
) -> None:
    init_db()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?,
                call_id = ?,
                success = ?,
                completed_path = ?,
                archive_path = COALESCE(?, archive_path),
                transcript_path = ?,
                completed_at = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (
                JOB_TRANSCRIBED,
                call_id,
                1 if success else 0,
                completed_path,
                archive_path,
                transcript_path,
                now,
                now,
                job_id,
            ),
        )


def mark_job_retrying(
    job_id: int,
    error_type: str,
    error_message: str,
    next_attempt_at: str,
    traceback: str | None = None,
) -> None:
    init_db()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = ?,
                last_error_type = ?,
                last_error_message = ?,
                last_traceback = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                JOB_RETRYING,
                next_attempt_at,
                error_type,
                error_message,
                traceback,
                now,
                job_id,
            ),
        )


def mark_job_failed(
    job_id: int,
    error_type: str,
    error_message: str,
    failed_path: str | None = None,
    traceback: str | None = None,
) -> None:
    init_db()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?,
                failed_path = COALESCE(?, failed_path),
                failed_at = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = NULL,
                last_error_type = ?,
                last_error_message = ?,
                last_traceback = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                JOB_FAILED,
                failed_path,
                now,
                error_type,
                error_message,
                traceback,
                now,
                job_id,
            ),
        )


def mark_job_invalid(
    job_id: int,
    reason: str,
    failed_path: str | None = None,
    traceback: str | None = None,
) -> None:
    init_db()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?,
                failed_path = COALESCE(?, failed_path),
                failed_at = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = NULL,
                last_error_type = 'invalid_file',
                last_error_message = ?,
                last_traceback = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                JOB_INVALID,
                failed_path,
                now,
                reason,
                traceback,
                now,
                job_id,
            ),
        )


def recover_stale_jobs(stale_after_minutes: int = 60) -> int:
    """Release stale processing jobs so a restarted worker can retry them."""
    init_db()
    now_dt = datetime.now()
    now = now_dt.isoformat(timespec="seconds")
    cutoff = (now_dt - timedelta(minutes=stale_after_minutes)).isoformat(
        timespec="seconds"
    )

    with _connect() as conn:
        to_retry = conn.execute(
            """
            SELECT id FROM transcription_jobs
            WHERE status = ?
              AND (locked_at IS NULL OR locked_at <= ?)
              AND call_id IS NULL
              AND attempt_count < max_attempts
            """,
            (JOB_PROCESSING, cutoff),
        ).fetchall()
        to_fail = conn.execute(
            """
            SELECT id FROM transcription_jobs
            WHERE status = ?
              AND (locked_at IS NULL OR locked_at <= ?)
              AND call_id IS NULL
              AND attempt_count >= max_attempts
            """,
            (JOB_PROCESSING, cutoff),
        ).fetchall()

        conn.executemany(
            """
            UPDATE transcription_jobs
            SET status = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = ?,
                last_error_type = 'stale_processing_job',
                last_error_message = 'Recovered stale processing job after restart',
                updated_at = ?
            WHERE id = ?
            """,
            [(JOB_RETRYING, now, now, row["id"]) for row in to_retry],
        )
        conn.executemany(
            """
            UPDATE transcription_jobs
            SET status = ?,
                locked_by = NULL,
                locked_at = NULL,
                next_attempt_at = NULL,
                failed_at = ?,
                last_error_type = 'stale_processing_job',
                last_error_message = 'Stale processing job exceeded max attempts',
                updated_at = ?
            WHERE id = ?
            """,
            [(JOB_FAILED, now, now, row["id"]) for row in to_fail],
        )

        return len(to_retry) + len(to_fail)


def get_job_summary() -> dict:
    """Return a compact operational summary for watcher/dashboard use."""
    init_db()
    with _connect() as conn:
        counts = conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM transcription_jobs
            GROUP BY status
            ORDER BY status
            """
        ).fetchall()
        recent_failures = conn.execute(
            """
            SELECT id, original_filename, status, attempt_count, max_attempts,
                   last_error_type, last_error_message, updated_at
            FROM transcription_jobs
            WHERE status IN (?, ?)
            ORDER BY updated_at DESC, id DESC
            LIMIT 10
            """,
            (JOB_FAILED, JOB_INVALID),
        ).fetchall()

    return {
        "counts": {row["status"]: row["count"] for row in counts},
        "recent_failures": [dict(row) for row in recent_failures],
    }


def reset_db() -> None:
    """
    Delete dashboard/job records. The table structure stays intact.

    Dashboard reset is destructive; clear jobs before calls so duplicate
    detection does not keep suppressing files after call rows are removed.
    """
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM transcription_jobs")
        conn.execute("DELETE FROM calls")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='transcription_jobs'")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='calls'")
        conn.commit()


def get_dashboard_data(model_filter: str = "all") -> dict:
    """
    Return all data needed by the dashboard in one query round-trip.
    model_filter: "all" or a specific model name e.g. "gemini-2.5-flash"
    """
    init_db()
    today = datetime.now().strftime("%Y-%m-%d")

    # Build WHERE clauses based on model filter
    model_where_today = "processed_at LIKE ?"
    model_where_all   = "1=1"
    params_today      = [f"{today}%"]
    params_all: list  = []

    if model_filter != "all":
        model_where_today += " AND model = ?"
        model_where_all    = "model = ?"
        params_today.append(model_filter)
        params_all.append(model_filter)

    with _connect() as conn:
        # ── Today's aggregates ────────────────────────────────────────────
        today_row = conn.execute(
            f"""
            SELECT
                COUNT(*)                        AS calls_today,
                COALESCE(SUM(total_cost_usd),0) AS cost_usd,
                COALESCE(SUM(total_cost_lkr),0) AS cost_lkr,
                COALESCE(SUM(total_tokens),0)   AS tokens_total,
                COALESCE(SUM(audio_tokens),0)   AS tokens_audio,
                COALESCE(SUM(output_tokens),0)  AS tokens_output,
                COALESCE(SUM(duration_seconds),0)        AS audio_seconds,
                COALESCE(SUM(silence_removed_seconds),0) AS silence_removed,
                SUM(CASE WHEN batch_mode=1 THEN 1 ELSE 0 END) AS batch_calls,
                SUM(CASE WHEN batch_mode=0 THEN 1 ELSE 0 END) AS realtime_calls
            FROM calls WHERE {model_where_today}
            """,
            params_today,
        ).fetchone()

        # ── Language breakdown for today ──────────────────────────────────
        lang_rows = conn.execute(
            f"SELECT languages_detected FROM calls WHERE {model_where_today}",
            params_today,
        ).fetchall()
        lang_counts: dict[str, int] = {"Sinhala": 0, "English": 0, "Tamil": 0}
        for lr in lang_rows:
            for lang in (lr["languages_detected"] or "").split(","):
                lang = lang.strip().title()
                if lang in lang_counts:
                    lang_counts[lang] += 1

        # ── Silence saved today ───────────────────────────────────────────
        silence_row = conn.execute(
            f"""
            SELECT COALESCE(SUM(silence_removed_seconds),0) AS total_silence_s
            FROM calls WHERE {model_where_today}
            """,
            params_today,
        ).fetchone()

        # ── Per-model breakdown (always all models, for the filter tabs) ──
        model_rows = conn.execute(
            """
            SELECT
                model,
                COUNT(*)                        AS calls,
                COALESCE(SUM(total_cost_usd),0) AS cost_usd,
                COALESCE(SUM(total_cost_lkr),0) AS cost_lkr,
                COALESCE(SUM(total_tokens),0)   AS tokens
            FROM calls
            WHERE processed_at LIKE ?
            GROUP BY model
            ORDER BY calls DESC
            """,
            (f"{today}%",),
        ).fetchall()

        # ── Recent 20 calls ───────────────────────────────────────────────
        recent = conn.execute(
            f"""
            SELECT filename, duration_seconds, silence_removed_seconds,
                   model, total_tokens, total_cost_usd, total_cost_lkr,
                   languages_detected, processed_at, batch_mode,
                   CASE WHEN LENGTH(transcript)>0 THEN 1 ELSE 0 END AS success
            FROM calls
            WHERE {model_where_all}
            ORDER BY id DESC LIMIT 20
            """,
            params_all,
        ).fetchall()

        # ── All-time totals ───────────────────────────────────────────────
        totals = conn.execute(
            f"""
            SELECT
                COUNT(*)                        AS total_calls,
                COALESCE(SUM(total_cost_usd),0) AS total_cost_usd,
                COALESCE(SUM(total_cost_lkr),0) AS total_cost_lkr,
                COALESCE(SUM(total_tokens),0)   AS total_tokens
            FROM calls WHERE {model_where_all}
            """,
            params_all,
        ).fetchone()

        # ── Daily cost for last 14 days ───────────────────────────────────
        daily = conn.execute(
            f"""
            SELECT
                SUBSTR(processed_at,1,10)       AS day,
                COUNT(*)                        AS calls,
                COALESCE(SUM(total_cost_lkr),0) AS cost_lkr
            FROM calls
            WHERE {model_where_all}
            GROUP BY SUBSTR(processed_at,1,10)
            ORDER BY day DESC
            LIMIT 14
            """,
            params_all,
        ).fetchall()

        # ── All distinct models ever used (for filter dropdown) ───────────
        all_models = conn.execute(
            "SELECT DISTINCT model FROM calls ORDER BY model"
        ).fetchall()

    return {
        "today":         dict(today_row) if today_row else {},
        "languages":     lang_counts,
        "recent":        [dict(r) for r in recent],
        "totals":        dict(totals) if totals else {},
        "daily":         [dict(d) for d in daily],
        "silence_saved_s": silence_row["total_silence_s"] if silence_row else 0,
        "model_breakdown": [dict(m) for m in model_rows],
        "all_models":    [r["model"] for r in all_models],
        "active_filter": model_filter,
    }
