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
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

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
    batch_mode               INTEGER          DEFAULT 0,
    processed_at             TEXT             NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_date ON calls (processed_at);
"""

# Columns added after initial release — ADD COLUMN IF NOT EXISTS is safe to run always
_OPTIONAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("billed_output_tokens", "INTEGER DEFAULT 0"),
)


def _ensure_columns(conn: psycopg2.extensions.connection) -> None:
    """Migrate existing tables: add new columns without touching old data."""
    with conn.cursor() as cur:
        for column, definition in _OPTIONAL_COLUMNS:
            cur.execute(
                f"ALTER TABLE calls ADD COLUMN IF NOT EXISTS {column} {definition}"
            )


def init_db() -> None:
    """Create tables and apply schema migrations (safe to call repeatedly)."""
    global _db_initialized
    if _db_initialized:
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_SQL)
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
                    datetime.now().isoformat(),
                ),
            )
            return cur.fetchone()[0]   # RETURNING id


def reset_db() -> None:
    """Delete all call records and reset the auto-increment counter."""
    init_db()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE calls RESTART IDENTITY")


# ── Read ──────────────────────────────────────────────────────────────────────

def get_dashboard_data(model_filter: str = "all", date: str = "") -> dict:
    """
    Return all data needed by the dashboard in one query round-trip.
    model_filter: "all" or a specific model name e.g. "gemini-2.5-flash"
    date: "YYYY-MM-DD" to view a specific day, defaults to today
    """
    init_db()
    selected_date = date if date else datetime.now().strftime("%Y-%m-%d")
    is_today      = selected_date == datetime.now().strftime("%Y-%m-%d")

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
                SELECT filename, duration_seconds, silence_removed_seconds,
                       model, total_tokens, total_cost_usd, total_cost_lkr,
                       billed_output_tokens, languages_detected, processed_at,
                       batch_mode,
                       CASE WHEN LENGTH(transcript)>0 THEN 1 ELSE 0 END AS success
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
