"""
database.py
SQLite logger for per-call transcription records.
Every call processed by gemini_flash_stt is saved here so the dashboard
and billing reconciliation always have per-call detail.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "calls.db"

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
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent writes
    return conn


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
    total_usd = result.get("total_cost_usd", 0.0)
    langs = result.get("languages_detected", [])
    langs_str = ", ".join(langs) if isinstance(langs, list) else str(langs)

    with _connect() as conn:
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


def reset_db() -> None:
    """Delete all call records. The table structure stays intact."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM calls")
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
