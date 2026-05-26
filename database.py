"""
database.py
PostgreSQL logger for per-call transcription records.
Every call processed by gemini_flash_stt is saved here so the dashboard
and billing reconciliation always have per-call detail.

Setup
-----
1. Install driver:
       pip install psycopg2-binary --break-system-packages

2. Create the database (run once as postgres superuser):
       sudo -u postgres createuser slt
       sudo -u postgres createdb slt_calls -O slt
       sudo -u postgres psql -c "ALTER USER slt WITH PASSWORD 'yourpassword';"

3. Add to your .env file:
       DATABASE_URL=postgresql://slt:yourpassword@localhost:5432/slt_calls

That's it — init_db() creates the table automatically on first run.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import app_now

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    print("ERROR: psycopg2 is not installed.")
    print("Run:  pip install psycopg2-binary --break-system-packages")
    raise


# ── Connection ────────────────────────────────────────────────────────────────

_DB_URL: str | None = None
_pool:   psycopg2.pool.ThreadedConnectionPool | None = None
_db_initialized = False   # avoid CREATE TABLE on every call


def _get_db_url() -> str:
    """Read DATABASE_URL from env or .env file."""
    global _DB_URL
    if _DB_URL:
        return _DB_URL

    # Load .env file (same pattern as gemini_flash_stt.py)
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key == "DATABASE_URL" and "DATABASE_URL" not in os.environ:
                    os.environ["DATABASE_URL"] = val

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set.\n"
            "Add this to your .env file:\n"
            "  DATABASE_URL=postgresql://slt:yourpassword@localhost:5432/slt_calls"
        )
    _DB_URL = url
    return url


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return (or lazily create) the shared thread-safe connection pool."""
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=_get_db_url(),
        )
    return _pool


@contextmanager
def _connect() -> Iterator[psycopg2.extensions.connection]:
    """
    Borrow a connection from the pool.
    Commits on success, rolls back on error, returns to pool when done.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = False
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    id                       SERIAL           PRIMARY KEY,
    filename                 TEXT             NOT NULL,
    audio_path               TEXT,
    duration_seconds         DOUBLE PRECISION DEFAULT 0,
    silence_removed_seconds  DOUBLE PRECISION DEFAULT 0,
    model                    TEXT,
    input_tokens             INTEGER          DEFAULT 0,
    audio_tokens             INTEGER          DEFAULT 0,
    text_input_tokens        INTEGER          DEFAULT 0,
    output_tokens            INTEGER          DEFAULT 0,
    thoughts_tokens          INTEGER          DEFAULT 0,
    billed_output_tokens     INTEGER          DEFAULT 0,
    total_tokens             INTEGER          DEFAULT 0,
    audio_input_cost_usd     DOUBLE PRECISION DEFAULT 0,
    text_input_cost_usd      DOUBLE PRECISION DEFAULT 0,
    output_cost_usd          DOUBLE PRECISION DEFAULT 0,
    total_cost_usd           DOUBLE PRECISION DEFAULT 0,
    total_cost_lkr           DOUBLE PRECISION DEFAULT 0,
    lkr_rate                 DOUBLE PRECISION DEFAULT 316,
    languages_detected       TEXT             DEFAULT '',
    transcript               TEXT             DEFAULT '',
    transcript_file_path     TEXT             DEFAULT '',
    transcript_saved_at      TEXT             DEFAULT '',
    transcript_output_date   TEXT             DEFAULT '',
    batch_mode               INTEGER          DEFAULT 0,
    processed_at             TEXT             NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_date ON calls (processed_at);
"""

_CREATE_BATCH_SQL = """
CREATE TABLE IF NOT EXISTS batch_jobs (
    id                       SERIAL           PRIMARY KEY,
    job_key                  TEXT             UNIQUE NOT NULL,
    status                   TEXT             DEFAULT 'PREPARED',
    model                    TEXT             DEFAULT '',
    location                 TEXT             DEFAULT '',
    bucket                   TEXT             DEFAULT '',
    audio_prefix             TEXT             DEFAULT '',
    jobs_prefix              TEXT             DEFAULT '',
    output_prefix            TEXT             DEFAULT '',
    input_jsonl_path         TEXT             DEFAULT '',
    input_jsonl_gcs_uri      TEXT             DEFAULT '',
    output_gcs_prefix        TEXT             DEFAULT '',
    vertex_job_name          TEXT             DEFAULT '',
    file_count               INTEGER          DEFAULT 0,
    total_duration_seconds   DOUBLE PRECISION DEFAULT 0,
    estimated_audio_tokens   INTEGER          DEFAULT 0,
    estimated_cost_usd       DOUBLE PRECISION DEFAULT 0,
    submitted_at             TEXT             DEFAULT '',
    last_checked_at          TEXT             DEFAULT '',
    completed_at             TEXT             DEFAULT '',
    imported_at              TEXT             DEFAULT '',
    error_message            TEXT             DEFAULT '',
    created_at               TEXT             DEFAULT '',
    updated_at               TEXT             DEFAULT ''
);
CREATE TABLE IF NOT EXISTS batch_items (
    id                       SERIAL           PRIMARY KEY,
    batch_job_id             INTEGER          REFERENCES batch_jobs(id) ON DELETE CASCADE,
    filename                 TEXT             DEFAULT '',
    local_audio_path         TEXT             DEFAULT '',
    gcs_audio_uri            TEXT             DEFAULT '',
    mime_type                TEXT             DEFAULT '',
    duration_seconds         DOUBLE PRECISION DEFAULT 0,
    estimated_audio_tokens   INTEGER          DEFAULT 0,
    estimated_cost_usd       DOUBLE PRECISION DEFAULT 0,
    status                   TEXT             DEFAULT 'PENDING',
    call_id                  INTEGER,
    transcript_file_path     TEXT             DEFAULT '',
    error_message            TEXT             DEFAULT '',
    created_at               TEXT             DEFAULT '',
    updated_at               TEXT             DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_batch_items_job ON batch_items (batch_job_id);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs (status);
"""

# Columns added after initial release — ADD COLUMN IF NOT EXISTS is safe to run always
_OPTIONAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("billed_output_tokens", "INTEGER DEFAULT 0"),
    ("transcript_file_path", "TEXT DEFAULT ''"),
    ("transcript_saved_at", "TEXT DEFAULT ''"),
    ("transcript_output_date", "TEXT DEFAULT ''"),
)

_BATCH_JOB_COLUMNS: tuple[tuple[str, str], ...] = (
    ("job_key", "TEXT UNIQUE NOT NULL"),
    ("status", "TEXT DEFAULT 'PREPARED'"),
    ("model", "TEXT DEFAULT ''"),
    ("location", "TEXT DEFAULT ''"),
    ("bucket", "TEXT DEFAULT ''"),
    ("audio_prefix", "TEXT DEFAULT ''"),
    ("jobs_prefix", "TEXT DEFAULT ''"),
    ("output_prefix", "TEXT DEFAULT ''"),
    ("input_jsonl_path", "TEXT DEFAULT ''"),
    ("input_jsonl_gcs_uri", "TEXT DEFAULT ''"),
    ("output_gcs_prefix", "TEXT DEFAULT ''"),
    ("vertex_job_name", "TEXT DEFAULT ''"),
    ("file_count", "INTEGER DEFAULT 0"),
    ("total_duration_seconds", "DOUBLE PRECISION DEFAULT 0"),
    ("estimated_audio_tokens", "INTEGER DEFAULT 0"),
    ("estimated_cost_usd", "DOUBLE PRECISION DEFAULT 0"),
    ("submitted_at", "TEXT DEFAULT ''"),
    ("last_checked_at", "TEXT DEFAULT ''"),
    ("completed_at", "TEXT DEFAULT ''"),
    ("imported_at", "TEXT DEFAULT ''"),
    ("error_message", "TEXT DEFAULT ''"),
    ("created_at", "TEXT DEFAULT ''"),
    ("updated_at", "TEXT DEFAULT ''"),
)

_BATCH_ITEM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("batch_job_id", "INTEGER REFERENCES batch_jobs(id) ON DELETE CASCADE"),
    ("filename", "TEXT DEFAULT ''"),
    ("local_audio_path", "TEXT DEFAULT ''"),
    ("gcs_audio_uri", "TEXT DEFAULT ''"),
    ("mime_type", "TEXT DEFAULT ''"),
    ("duration_seconds", "DOUBLE PRECISION DEFAULT 0"),
    ("estimated_audio_tokens", "INTEGER DEFAULT 0"),
    ("estimated_cost_usd", "DOUBLE PRECISION DEFAULT 0"),
    ("status", "TEXT DEFAULT 'PENDING'"),
    ("call_id", "INTEGER"),
    ("transcript_file_path", "TEXT DEFAULT ''"),
    ("error_message", "TEXT DEFAULT ''"),
    ("created_at", "TEXT DEFAULT ''"),
    ("updated_at", "TEXT DEFAULT ''"),
)

_BATCH_JOB_UPDATE_FIELDS = {column for column, _ in _BATCH_JOB_COLUMNS} - {"job_key"}
_BATCH_ITEM_UPDATE_FIELDS = {column for column, _ in _BATCH_ITEM_COLUMNS} - {"batch_job_id"}


def _ensure_columns(conn: psycopg2.extensions.connection) -> None:
    """Migrate existing tables: add new columns without touching old data."""
    with conn.cursor() as cur:
        for column, definition in _OPTIONAL_COLUMNS:
            cur.execute(
                f"ALTER TABLE calls ADD COLUMN IF NOT EXISTS {column} {definition}"
            )
        for column, definition in _BATCH_JOB_COLUMNS:
            cur.execute(
                f"ALTER TABLE batch_jobs ADD COLUMN IF NOT EXISTS {column} {definition}"
            )
        for column, definition in _BATCH_ITEM_COLUMNS:
            cur.execute(
                f"ALTER TABLE batch_items ADD COLUMN IF NOT EXISTS {column} {definition}"
            )


def init_db() -> None:
    """Create tables and apply schema migrations (safe to call repeatedly)."""
    global _db_initialized
    if _db_initialized:
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_SQL)
            cur.execute(_CREATE_BATCH_SQL)
        _ensure_columns(conn)
    _db_initialized = True


# ── Write ─────────────────────────────────────────────────────────────────────

def save_call(
    result: dict[str, Any],
    lkr_rate: float = 316.0,
    batch_mode: bool = False,
) -> int:
    """
    Persist one transcription result. Returns the new row id.
    Prefers result["lkr_rate"] (live rate from fetch_lkr_rate) over the
    lkr_rate argument.
    """
    init_db()
    total_usd  = result.get("total_cost_usd", 0.0)
    saved_rate = result.get("lkr_rate", lkr_rate)
    total_lkr  = result.get("total_cost_lkr", total_usd * saved_rate)
    langs      = result.get("languages_detected", [])
    langs_str  = ", ".join(langs) if isinstance(langs, list) else str(langs)

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO calls (
                    filename, audio_path,
                    duration_seconds, silence_removed_seconds,
                    model,
                    input_tokens, audio_tokens, text_input_tokens,
                    output_tokens, thoughts_tokens, billed_output_tokens, total_tokens,
                    audio_input_cost_usd, text_input_cost_usd, output_cost_usd,
                    total_cost_usd, total_cost_lkr, lkr_rate,
                    languages_detected, transcript, batch_mode, processed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    Path(result.get("audio_path", "unknown")).name,
                    result.get("audio_path", ""),
                    result.get("duration_seconds", 0),
                    result.get("silence_removed_seconds", 0),
                    result.get("model", ""),
                    result.get("input_tokens", 0),
                    result.get("audio_tokens", 0),
                    result.get("text_input_tokens", 0),
                    result.get("output_tokens", 0),
                    result.get("thoughts_tokens", 0),
                    result.get("billed_output_tokens", 0),
                    result.get("total_tokens", 0),
                    result.get("audio_input_cost_usd", 0),
                    result.get("text_input_cost_usd", 0),
                    result.get("output_cost_usd", 0),
                    total_usd,
                    total_lkr,
                    saved_rate,
                    langs_str,
                    result.get("transcript", ""),
                    1 if batch_mode else 0,
                    app_now().isoformat(),
                ),
            )
            return cur.fetchone()[0]   # RETURNING id


def update_call_transcript_file(call_id: int, saved_info: dict[str, Any]) -> None:
    """Attach saved transcript file metadata to an existing call row."""
    init_db()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE calls
                SET transcript_file_path = %s,
                    transcript_saved_at = %s,
                    transcript_output_date = %s
                WHERE id = %s
                """,
                (
                    saved_info.get("transcript_file_path", ""),
                    saved_info.get("transcript_saved_at", ""),
                    saved_info.get("transcript_output_date", ""),
                    call_id,
                ),
            )


def reset_db() -> None:
    """Delete all call records and reset the auto-increment counter."""
    init_db()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE calls RESTART IDENTITY")


# ── Read ──────────────────────────────────────────────────────────────────────

def get_call_transcript(call_id: int) -> dict[str, Any] | None:
    """Return transcript text and file metadata for one call."""
    init_db()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, filename, transcript, transcript_file_path,
                       transcript_saved_at, transcript_output_date
                FROM calls
                WHERE id = %s
                """,
                (call_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def _fetch_one(cur: psycopg2.extensions.cursor) -> dict[str, Any] | None:
    row = cur.fetchone()
    return dict(row) if row else None


def _now_text() -> str:
    return app_now().isoformat()


def create_batch_job(
    job_key: str,
    status: str = "PREPARED",
    model: str = "",
    location: str = "",
    bucket: str = "",
    audio_prefix: str = "",
    jobs_prefix: str = "",
    output_prefix: str = "",
    **fields: Any,
) -> int:
    """Create a durable batch job row and return its id."""
    init_db()
    now = _now_text()
    extra_fields = {k: v for k, v in fields.items() if k in _BATCH_JOB_UPDATE_FIELDS}
    values = {
        "job_key": job_key,
        "status": status,
        "model": model,
        "location": location,
        "bucket": bucket,
        "audio_prefix": audio_prefix,
        "jobs_prefix": jobs_prefix,
        "output_prefix": output_prefix,
        "created_at": fields.pop("created_at", now),
        "updated_at": fields.pop("updated_at", now),
        **extra_fields,
    }
    columns = list(values.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO batch_jobs ({column_sql})
                VALUES ({placeholders})
                RETURNING id
                """,
                [values[column] for column in columns],
            )
            return cur.fetchone()[0]


def add_batch_item(
    batch_job_id: int,
    filename: str = "",
    local_audio_path: str = "",
    gcs_audio_uri: str = "",
    mime_type: str = "",
    duration_seconds: float = 0.0,
    estimated_audio_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    status: str = "PENDING",
    **fields: Any,
) -> int:
    """Create a durable batch item row and return its id."""
    init_db()
    now = _now_text()
    extra_fields = {k: v for k, v in fields.items() if k in _BATCH_ITEM_UPDATE_FIELDS}
    values = {
        "batch_job_id": batch_job_id,
        "filename": filename,
        "local_audio_path": local_audio_path,
        "gcs_audio_uri": gcs_audio_uri,
        "mime_type": mime_type,
        "duration_seconds": duration_seconds,
        "estimated_audio_tokens": estimated_audio_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "status": status,
        "created_at": fields.pop("created_at", now),
        "updated_at": fields.pop("updated_at", now),
        **extra_fields,
    }
    columns = list(values.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO batch_items ({column_sql})
                VALUES ({placeholders})
                RETURNING id
                """,
                [values[column] for column in columns],
            )
            return cur.fetchone()[0]


def get_batch_job(job_id: int) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM batch_jobs WHERE id = %s", (job_id,))
            return _fetch_one(cur)


def get_batch_job_by_key(job_key: str) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM batch_jobs WHERE job_key = %s", (job_key,))
            return _fetch_one(cur)


def list_batch_jobs(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM batch_jobs ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def list_batch_items(batch_job_id: int) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM batch_items WHERE batch_job_id = %s ORDER BY id",
                (batch_job_id,),
            )
            return [dict(row) for row in cur.fetchall()]


def _update_row(
    table: str,
    row_id: int,
    allowed_fields: set[str],
    fields: dict[str, Any],
) -> None:
    updates = {k: v for k, v in fields.items() if k in allowed_fields}
    updates["updated_at"] = fields.get("updated_at", _now_text())
    if not updates:
        return

    assignments = ", ".join(f"{column} = %s" for column in updates)
    params = list(updates.values()) + [row_id]
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {table} SET {assignments} WHERE id = %s",
                params,
            )


def update_batch_job(job_id: int, **fields: Any) -> None:
    init_db()
    _update_row("batch_jobs", job_id, _BATCH_JOB_UPDATE_FIELDS, fields)


def update_batch_item(item_id: int, **fields: Any) -> None:
    init_db()
    _update_row("batch_items", item_id, _BATCH_ITEM_UPDATE_FIELDS, fields)


def try_mark_batch_job_submitting(job_id: int) -> dict[str, Any] | None:
    """Atomically claim a prepared batch job for Vertex submission."""
    init_db()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE batch_jobs
                SET status = 'SUBMITTING',
                    error_message = '',
                    updated_at = %s
                WHERE id = %s
                  AND status = 'PREPARED'
                  AND COALESCE(vertex_job_name, '') = ''
                RETURNING *
                """,
                (_now_text(), job_id),
            )
            return _fetch_one(cur)


def try_mark_batch_item_importing(item_id: int) -> dict[str, Any] | None:
    """Atomically claim one batch item before creating a calls row."""
    init_db()
    importable_statuses = [
        "PENDING",
        "UPLOADED",
        "SUBMITTED",
        "SUCCEEDED",
        "FAILED",
        "IMPORT_ERROR",
    ]
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE batch_items
                SET status = 'IMPORTING',
                    error_message = '',
                    updated_at = %s
                WHERE id = %s
                  AND status = ANY(%s)
                  AND (call_id IS NULL OR call_id = 0)
                RETURNING *
                """,
                (_now_text(), item_id, importable_statuses),
            )
            return _fetch_one(cur)


def get_batch_item_by_id_or_filename_or_gcs_uri(
    batch_job_id: int,
    filename: str = "",
    gcs_audio_uri: str = "",
    item_id: int | str | None = None,
) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            parsed_item_id: int | None = None
            if item_id not in (None, ""):
                try:
                    parsed_item_id = int(item_id)
                except (TypeError, ValueError):
                    parsed_item_id = None
            if parsed_item_id is not None:
                cur.execute(
                    """
                    SELECT * FROM batch_items
                    WHERE batch_job_id = %s AND id = %s
                    LIMIT 1
                    """,
                    (batch_job_id, parsed_item_id),
                )
                row = _fetch_one(cur)
                if row:
                    return row
            if gcs_audio_uri:
                cur.execute(
                    """
                    SELECT * FROM batch_items
                    WHERE batch_job_id = %s AND gcs_audio_uri = %s
                    LIMIT 1
                    """,
                    (batch_job_id, gcs_audio_uri),
                )
                row = _fetch_one(cur)
                if row:
                    return row
            if filename:
                cur.execute(
                    """
                    SELECT * FROM batch_items
                    WHERE batch_job_id = %s AND filename = %s
                    LIMIT 1
                    """,
                    (batch_job_id, filename),
                )
                return _fetch_one(cur)
    return None


def get_batch_item_by_filename_or_gcs_uri(
    batch_job_id: int,
    filename: str = "",
    gcs_audio_uri: str = "",
) -> dict[str, Any] | None:
    return get_batch_item_by_id_or_filename_or_gcs_uri(
        batch_job_id,
        filename=filename,
        gcs_audio_uri=gcs_audio_uri,
    )


def mark_batch_job_counts(job_id: int) -> None:
    """Refresh aggregate file/duration/token/cost counts from batch_items."""
    init_db()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE batch_jobs
                SET file_count = stats.file_count,
                    total_duration_seconds = stats.total_duration_seconds,
                    estimated_audio_tokens = stats.estimated_audio_tokens,
                    estimated_cost_usd = stats.estimated_cost_usd,
                    updated_at = %s
                FROM (
                    SELECT
                        COUNT(*) AS file_count,
                        COALESCE(SUM(duration_seconds), 0) AS total_duration_seconds,
                        COALESCE(SUM(estimated_audio_tokens), 0) AS estimated_audio_tokens,
                        COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                    FROM batch_items
                    WHERE batch_job_id = %s
                ) AS stats
                WHERE batch_jobs.id = %s
                """,
                (_now_text(), job_id, job_id),
            )


def get_dashboard_data(model_filter: str = "all", date: str = "") -> dict:
    """
    Return all data needed by the dashboard in one query round-trip.
    model_filter: "all" or a specific model name e.g. "gemini-2.5-flash"
    date: "YYYY-MM-DD" to view a specific day, defaults to today
    """
    init_db()
    today_str     = app_now().strftime("%Y-%m-%d")
    selected_date = date if date else today_str
    is_today      = selected_date == today_str

    # Parameterised WHERE clauses — no f-string interpolation for values (SQL injection safe)
    where_day = "processed_at LIKE %s"
    where_all = "TRUE"
    params_day: list = [f"{selected_date}%"]
    params_all: list = []

    if model_filter != "all":
        where_day += " AND model = %s"
        where_all  = "model = %s"
        params_day.append(model_filter)
        params_all.append(model_filter)

    where_day_recent  = "processed_at LIKE %s"
    params_day_recent: list = [f"{selected_date}%"]
    if model_filter != "all":
        where_day_recent += " AND model = %s"
        params_day_recent.append(model_filter)

    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # ── Selected day's aggregates ─────────────────────────────────
            cur.execute(
                f"""
                SELECT
                    COUNT(*)                                  AS calls_today,
                    COALESCE(SUM(total_cost_usd),0)           AS cost_usd,
                    COALESCE(SUM(total_cost_lkr),0)           AS cost_lkr,
                    COALESCE(SUM(total_tokens),0)             AS tokens_total,
                    COALESCE(SUM(audio_tokens),0)             AS tokens_audio,
                    COALESCE(SUM(output_tokens),0)            AS tokens_output,
                    COALESCE(SUM(billed_output_tokens),0)     AS tokens_billed_output,
                    COALESCE(SUM(duration_seconds),0)         AS audio_seconds,
                    COALESCE(SUM(silence_removed_seconds),0)  AS silence_removed,
                    SUM(CASE WHEN batch_mode=1 THEN 1 ELSE 0 END) AS batch_calls,
                    SUM(CASE WHEN batch_mode=0 THEN 1 ELSE 0 END) AS realtime_calls
                FROM calls WHERE {where_day}
                """,
                params_day,
            )
            today_row = cur.fetchone()

            # ── Language breakdown ────────────────────────────────────────
            cur.execute(
                f"SELECT languages_detected FROM calls WHERE {where_day}",
                params_day,
            )
            lang_rows = cur.fetchall()
            lang_counts: dict[str, int] = {"Sinhala": 0, "English": 0, "Tamil": 0}
            for lr in lang_rows:
                for lang in (lr["languages_detected"] or "").split(","):
                    lang = lang.strip().title()
                    if lang in lang_counts:
                        lang_counts[lang] += 1

            # ── Silence saved ─────────────────────────────────────────────
            cur.execute(
                f"""
                SELECT COALESCE(SUM(silence_removed_seconds),0) AS total_silence_s
                FROM calls WHERE {where_day}
                """,
                params_day,
            )
            silence_row = cur.fetchone()

            # ── Per-model breakdown for selected day ──────────────────────
            cur.execute(
                """
                SELECT
                    model,
                    COUNT(*)                        AS calls,
                    COALESCE(SUM(total_cost_usd),0) AS cost_usd,
                    COALESCE(SUM(total_cost_lkr),0) AS cost_lkr,
                    COALESCE(SUM(total_tokens),0)   AS tokens
                FROM calls
                WHERE processed_at LIKE %s
                GROUP BY model
                ORDER BY calls DESC
                """,
                (f"{selected_date}%",),
            )
            model_rows = cur.fetchall()

            # ── Recent 20 calls for selected day ──────────────────────────
            cur.execute(
                f"""
                SELECT id, filename, duration_seconds, silence_removed_seconds,
                       model, total_tokens, total_cost_usd, total_cost_lkr,
                       billed_output_tokens, languages_detected, processed_at,
                       batch_mode, transcript_file_path, transcript_saved_at,
                       transcript_output_date,
                       CASE
                           WHEN COALESCE(transcript, '') <> ''
                             OR COALESCE(transcript_file_path, '') <> ''
                           THEN 1 ELSE 0
                       END AS success
                FROM calls
                WHERE {where_day_recent}
                ORDER BY id DESC LIMIT 20
                """,
                params_day_recent,
            )
            recent = cur.fetchall()

            # ── All-time totals ───────────────────────────────────────────
            cur.execute(
                f"""
                SELECT
                    COUNT(*)                        AS total_calls,
                    COALESCE(SUM(total_cost_usd),0) AS total_cost_usd,
                    COALESCE(SUM(total_cost_lkr),0) AS total_cost_lkr,
                    COALESCE(SUM(total_tokens),0)   AS total_tokens
                FROM calls WHERE {where_all}
                """,
                params_all or None,
            )
            totals = cur.fetchone()

            # ── Daily cost for last 14 days ───────────────────────────────
            cur.execute(
                f"""
                SELECT
                    SUBSTR(processed_at, 1, 10)     AS day,
                    COUNT(*)                        AS calls,
                    COALESCE(SUM(total_cost_lkr),0) AS cost_lkr
                FROM calls
                WHERE {where_all}
                GROUP BY SUBSTR(processed_at, 1, 10)
                ORDER BY day DESC
                LIMIT 14
                """,
                params_all or None,
            )
            daily = cur.fetchall()

            # ── All distinct models (for filter tabs) ─────────────────────
            cur.execute("SELECT DISTINCT model FROM calls ORDER BY model")
            all_models = cur.fetchall()

            # ── All distinct dates (for date picker) ──────────────────────
            cur.execute(
                "SELECT DISTINCT SUBSTR(processed_at, 1, 10) AS day "
                "FROM calls ORDER BY day DESC"
            )
            all_dates = cur.fetchall()

    return {
        "today":           dict(today_row) if today_row else {},
        "languages":       lang_counts,
        "recent":          [dict(r) for r in recent],
        "totals":          dict(totals) if totals else {},
        "daily":           [dict(d) for d in daily],
        "silence_saved_s": silence_row["total_silence_s"] if silence_row else 0,
        "model_breakdown": [dict(m) for m in model_rows],
        "all_models":      [r["model"] for r in all_models],
        "all_dates":       [r["day"] for r in all_dates],
        "active_filter":   model_filter,
        "selected_date":   selected_date,
        "is_today":        is_today,
    }
