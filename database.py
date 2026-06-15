"""
database.py
Optional MySQL metadata/index storage for realtime transcriptions.

TXT transcript files and JSON metadata files are the primary output for this
project. MySQL is secondary: it helps search, dashboarding, and reconciliation,
but database failures must not be treated as transcription failures. This module
does not call Gemini, does not manage transcript files, and must never log
passwords or other secrets.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping

from config import app_now

try:
    import mysql.connector
except ImportError as exc:  # Import compatibility: fail only when DB is used.
    mysql = None  # type: ignore[assignment]
    _MYSQL_IMPORT_ERROR: ImportError | None = exc
else:
    _MYSQL_IMPORT_ERROR = None


log = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Base class for database-layer errors."""


class DatabaseDisabledError(DatabaseError):
    """Raised when a direct connection is requested while DB is disabled."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when database environment configuration is invalid."""


class DatabaseUnavailableError(DatabaseError):
    """Raised when MySQL cannot be reached or used."""


@dataclass(frozen=True)
class DatabaseWriteResult:
    """Structured result for optional metadata writes."""

    success: bool
    record_id: int | None = None
    error: str | None = None
    disabled: bool = False


_db_initialized = False
_warned_messages: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned_messages:
        log.warning(message)
        _warned_messages.add(key)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _db_backend() -> str:
    return (os.getenv("DB_BACKEND", "mysql").strip() or "mysql").lower()


def is_database_enabled() -> bool:
    """
    Return True only when MySQL metadata storage is explicitly enabled.

    DB_ENABLED defaults to false for safety. The sample .env enables MySQL for
    deployments that have configured a dedicated, non-root MySQL user.
    """
    if not _env_bool("DB_ENABLED", default=False):
        return False

    backend = _db_backend()
    if backend != "mysql":
        _warn_once(
            "unsupported_backend",
            f"Database backend '{backend}' is not supported on this branch; "
            "treating database storage as disabled.",
        )
        return False

    return True


def _mysql_config() -> dict[str, Any]:
    if not is_database_enabled():
        raise DatabaseDisabledError("Database metadata storage is disabled.")

    if _MYSQL_IMPORT_ERROR is not None:
        raise DatabaseUnavailableError(
            "mysql-connector-python is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from _MYSQL_IMPORT_ERROR

    try:
        port = int(os.getenv("MYSQL_PORT", "3306").strip() or "3306")
    except ValueError as exc:
        raise DatabaseConfigurationError("MYSQL_PORT must be an integer.") from exc

    try:
        timeout = int(os.getenv("MYSQL_CONNECT_TIMEOUT", "10").strip() or "10")
    except ValueError as exc:
        raise DatabaseConfigurationError(
            "MYSQL_CONNECT_TIMEOUT must be an integer."
        ) from exc

    return {
        "host": os.getenv("MYSQL_HOST", "localhost").strip() or "localhost",
        "port": port,
        "database": (
            os.getenv("MYSQL_DATABASE", "telecom_voice_to_text").strip()
            or "telecom_voice_to_text"
        ),
        "user": os.getenv("MYSQL_USER", "telecom_app").strip() or "telecom_app",
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "connection_timeout": timeout,
        "autocommit": False,
        "charset": "utf8mb4",
        "use_unicode": True,
    }


def get_connection():
    """
    Open a MySQL connection.

    The returned connection must be closed by the caller. Errors intentionally
    avoid including passwords or full DSNs.
    """
    try:
        return mysql.connector.connect(**_mysql_config())  # type: ignore[union-attr]
    except (DatabaseDisabledError, DatabaseConfigurationError):
        raise
    except Exception as exc:
        raise DatabaseUnavailableError(
            "MySQL connection failed. Check MySQL configuration and availability."
        ) from exc


@contextmanager
def _connect() -> Iterator[Any]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        finally:
            raise
    finally:
        conn.close()


@contextmanager
def _cursor(conn: Any, dictionary: bool = False) -> Iterator[Any]:
    cur = conn.cursor(dictionary=dictionary)
    try:
        yield cur
    finally:
        cur.close()


_CREATE_TRANSCRIPTIONS_SQL = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    original_file_name VARCHAR(512) NOT NULL,
    file_hash VARCHAR(128) DEFAULT NULL,
    audio_input_path TEXT,
    audio_processing_path TEXT,
    audio_completed_path TEXT,
    audio_failed_path TEXT,
    transcript_txt_path TEXT,
    metadata_json_path TEXT,
    status VARCHAR(64) NOT NULL DEFAULT 'completed',
    mode VARCHAR(32) NOT NULL DEFAULT 'realtime',
    transcribed_at DATETIME DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    model_name VARCHAR(128) DEFAULT NULL,
    language VARCHAR(255) DEFAULT NULL,
    duration_seconds DOUBLE DEFAULT 0,
    original_duration_seconds DOUBLE DEFAULT 0,
    submitted_duration_seconds DOUBLE DEFAULT 0,
    silence_removed_seconds DOUBLE DEFAULT 0,
    silence_removed_ratio DOUBLE DEFAULT 0,
    provider_input_tokens BIGINT DEFAULT 0,
    provider_audio_tokens BIGINT DEFAULT 0,
    provider_text_input_tokens BIGINT DEFAULT 0,
    provider_output_tokens BIGINT DEFAULT 0,
    provider_thoughts_tokens BIGINT DEFAULT 0,
    provider_billed_output_tokens BIGINT DEFAULT 0,
    provider_total_tokens BIGINT DEFAULT 0,
    audio_input_cost_usd DECIMAL(18,8) DEFAULT 0,
    text_input_cost_usd DECIMAL(18,8) DEFAULT 0,
    output_cost_usd DECIMAL(18,8) DEFAULT 0,
    estimated_cost_usd DECIMAL(18,8) DEFAULT 0,
    estimated_cost_lkr DECIMAL(18,4) DEFAULT 0,
    lkr_rate DECIMAL(18,6) DEFAULT 0,
    transcript_saved_at DATETIME DEFAULT NULL,
    transcript_output_date VARCHAR(32) DEFAULT NULL,
    error_message TEXT,
    INDEX idx_transcriptions_status (status),
    INDEX idx_transcriptions_transcribed_at (transcribed_at),
    INDEX idx_transcriptions_original_file_name (original_file_name),
    INDEX idx_transcriptions_file_hash (file_hash)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""


def init_database() -> bool:
    """
    Create the MySQL metadata table when database storage is enabled.

    Returns False when DB storage is disabled. Raises a clear database-layer
    error when MySQL is enabled but unavailable.
    """
    global _db_initialized
    if _db_initialized:
        return True
    if not is_database_enabled():
        return False

    try:
        with _connect() as conn:
            with _cursor(conn) as cur:
                cur.execute(_CREATE_TRANSCRIPTIONS_SQL)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseUnavailableError(
            f"MySQL schema initialization failed: {exc.__class__.__name__}"
        ) from exc

    _db_initialized = True
    return True


def init_db() -> bool:
    """Compatibility wrapper for older callers."""
    return init_database()


def _disabled_result() -> DatabaseWriteResult:
    return DatabaseWriteResult(
        success=False,
        error="Database metadata storage is disabled.",
        disabled=True,
    )


def _failure_result(exc: Exception) -> DatabaseWriteResult:
    return DatabaseWriteResult(
        success=False,
        error=f"{exc.__class__.__name__}: database metadata write failed.",
    )


def _as_mapping(record: Mapping[str, Any] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    data = dict(record or {})
    data.update(kwargs)
    return data


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_mysql_datetime(value: Any = None) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            dt = app_now()
    else:
        dt = app_now()

    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _optional_mysql_datetime(value: Any = None) -> datetime | None:
    if not value:
        return None
    return _to_mysql_datetime(value)


def _language_string(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _path_name(data: Mapping[str, Any]) -> str:
    filename = data.get("original_file_name") or data.get("filename")
    if filename:
        return str(filename)

    path_value = (
        data.get("audio_input_path")
        or data.get("audio_path")
        or data.get("audio_processing_path")
        or "unknown"
    )
    return Path(str(path_value)).name or "unknown"


def _build_record_values(data: Mapping[str, Any]) -> tuple[Any, ...]:
    duration = _to_float(data.get("duration_seconds"))
    silence_removed = _to_float(data.get("silence_removed_seconds"))
    original_duration = _to_float(
        data.get("original_duration_seconds"),
        default=duration + silence_removed if silence_removed else duration,
    )
    submitted_duration = _to_float(
        data.get("submitted_duration_seconds"),
        default=duration,
    )
    ratio = _to_float(data.get("silence_removed_ratio"))
    if not ratio and original_duration > 0:
        ratio = silence_removed / original_duration

    total_cost_usd = _to_float(
        data.get("estimated_cost_usd", data.get("total_cost_usd"))
    )
    total_cost_lkr = _to_float(
        data.get("estimated_cost_lkr", data.get("total_cost_lkr"))
    )
    lkr_rate = _to_float(data.get("lkr_rate"))
    if not total_cost_lkr and total_cost_usd and lkr_rate:
        total_cost_lkr = total_cost_usd * lkr_rate

    mode = str(data.get("mode") or ("batch" if data.get("batch_mode") else "realtime"))
    status = str(data.get("status") or "completed")

    return (
        _path_name(data),
        data.get("file_hash") or None,
        data.get("audio_input_path") or data.get("audio_path") or "",
        data.get("audio_processing_path") or "",
        data.get("audio_completed_path") or "",
        data.get("audio_failed_path") or "",
        data.get("transcript_txt_path") or data.get("transcript_file_path") or "",
        data.get("metadata_json_path") or "",
        status,
        mode,
        _to_mysql_datetime(data.get("transcribed_at") or data.get("processed_at")),
        data.get("model_name") or data.get("model") or "",
        data.get("language") or _language_string(data.get("languages_detected")),
        duration,
        original_duration,
        submitted_duration,
        silence_removed,
        ratio,
        _to_int(data.get("provider_input_tokens", data.get("input_tokens"))),
        _to_int(data.get("provider_audio_tokens", data.get("audio_tokens"))),
        _to_int(data.get("provider_text_input_tokens", data.get("text_input_tokens"))),
        _to_int(data.get("provider_output_tokens", data.get("output_tokens"))),
        _to_int(data.get("provider_thoughts_tokens", data.get("thoughts_tokens"))),
        _to_int(
            data.get(
                "provider_billed_output_tokens",
                data.get("billed_output_tokens"),
            )
        ),
        _to_int(data.get("provider_total_tokens", data.get("total_tokens"))),
        _to_float(data.get("audio_input_cost_usd")),
        _to_float(data.get("text_input_cost_usd")),
        _to_float(data.get("output_cost_usd")),
        total_cost_usd,
        total_cost_lkr,
        lkr_rate,
        _optional_mysql_datetime(data.get("transcript_saved_at")),
        data.get("transcript_output_date") or "",
        data.get("error_message") or "",
    )


_INSERT_TRANSCRIPTION_SQL = """
INSERT INTO transcriptions (
    original_file_name,
    file_hash,
    audio_input_path,
    audio_processing_path,
    audio_completed_path,
    audio_failed_path,
    transcript_txt_path,
    metadata_json_path,
    status,
    mode,
    transcribed_at,
    model_name,
    language,
    duration_seconds,
    original_duration_seconds,
    submitted_duration_seconds,
    silence_removed_seconds,
    silence_removed_ratio,
    provider_input_tokens,
    provider_audio_tokens,
    provider_text_input_tokens,
    provider_output_tokens,
    provider_thoughts_tokens,
    provider_billed_output_tokens,
    provider_total_tokens,
    audio_input_cost_usd,
    text_input_cost_usd,
    output_cost_usd,
    estimated_cost_usd,
    estimated_cost_lkr,
    lkr_rate,
    transcript_saved_at,
    transcript_output_date,
    error_message
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""


def save_transcription_record(
    record: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> DatabaseWriteResult:
    """
    Save one transcription metadata row.

    Full transcript text is intentionally not stored in MySQL. Store the TXT path
    and metadata JSON path instead.
    """
    if not is_database_enabled():
        return _disabled_result()

    data = _as_mapping(record, kwargs)
    try:
        init_database()
        with _connect() as conn:
            with _cursor(conn) as cur:
                cur.execute(_INSERT_TRANSCRIPTION_SQL, _build_record_values(data))
                record_id = int(cur.lastrowid or 0)
        return DatabaseWriteResult(success=True, record_id=record_id)
    except Exception as exc:
        log.warning("Optional MySQL metadata write failed: %s", exc.__class__.__name__)
        return _failure_result(exc)


def save_failure_record(
    original_file_name: str = "",
    error_message: str = "",
    **kwargs: Any,
) -> DatabaseWriteResult:
    """Save metadata for a file that failed before completion."""
    data = dict(kwargs)
    data["original_file_name"] = original_file_name or data.get("original_file_name", "")
    data["error_message"] = error_message or data.get("error_message", "")
    data["status"] = data.get("status") or "failed"
    return save_transcription_record(data)


def is_file_already_processed(
    file_hash: str = "",
    original_file_name: str = "",
    audio_path: str = "",
) -> bool:
    """Return True when a completed metadata row already exists for the file."""
    if not is_database_enabled():
        return False

    file_name = original_file_name or (Path(audio_path).name if audio_path else "")
    clauses: list[str] = ["status = 'completed'"]
    params: list[Any] = []
    if file_hash:
        clauses.append("file_hash = %s")
        params.append(file_hash)
    if file_name:
        clauses.append("original_file_name = %s")
        params.append(file_name)
    if len(clauses) == 1:
        return False

    try:
        init_database()
        where = " AND ".join(clauses)
        with _connect() as conn:
            with _cursor(conn) as cur:
                cur.execute(
                    f"SELECT id FROM transcriptions WHERE {where} LIMIT 1",
                    tuple(params),
                )
                return cur.fetchone() is not None
    except Exception as exc:
        log.warning(
            "Optional MySQL duplicate check failed: %s", exc.__class__.__name__
        )
        return False


def save_call(
    result: dict[str, Any],
    lkr_rate: float = 316.0,
    batch_mode: bool = False,
) -> int:
    """
    Compatibility wrapper for older callers.

    Returns the MySQL row id on success, or 0 when MySQL is disabled or
    unavailable. Returning 0 prevents an optional DB outage from causing the
    caller to retry a paid Gemini transcription.
    """
    payload = dict(result)
    payload["lkr_rate"] = payload.get("lkr_rate", lkr_rate)
    payload["batch_mode"] = batch_mode
    payload["mode"] = "batch" if batch_mode else payload.get("mode", "realtime")
    write_result = save_transcription_record(payload)
    return int(write_result.record_id or 0)


def update_call_transcript_file(call_id: int, saved_info: dict[str, Any]) -> None:
    """Compatibility wrapper to attach TXT/JSON output paths to a row."""
    if not call_id or call_id <= 0 or not is_database_enabled():
        return

    try:
        init_database()
        with _connect() as conn:
            with _cursor(conn) as cur:
                cur.execute(
                    """
                    UPDATE transcriptions
                    SET transcript_txt_path = %s,
                        metadata_json_path = %s,
                        transcript_saved_at = %s,
                        transcript_output_date = %s
                    WHERE id = %s
                    """,
                    (
                        saved_info.get("transcript_file_path", "")
                        or saved_info.get("transcript_txt_path", ""),
                        saved_info.get("metadata_json_path", ""),
                        _optional_mysql_datetime(saved_info.get("transcript_saved_at")),
                        saved_info.get("transcript_output_date", ""),
                        call_id,
                    ),
                )
    except Exception as exc:
        log.warning(
            "Optional MySQL transcript path update failed: %s",
            exc.__class__.__name__,
        )


def reset_db() -> None:
    """
    Dangerous compatibility function, disabled by default.

    Set ALLOW_DANGEROUS_DB_RESET=true only in an intentional local maintenance
    session. This must never run accidentally in production.
    """
    if not _env_bool("ALLOW_DANGEROUS_DB_RESET", default=False):
        raise RuntimeError(
            "Dangerous database reset is disabled. Set "
            "ALLOW_DANGEROUS_DB_RESET=true only for explicit maintenance."
        )

    init_database()
    with _connect() as conn:
        with _cursor(conn) as cur:
            cur.execute("DELETE FROM transcriptions")
            cur.execute("ALTER TABLE transcriptions AUTO_INCREMENT = 1")


def _empty_dashboard_data(
    model_filter: str = "all",
    date_value: str = "",
    error: str = "",
) -> dict[str, Any]:
    today_str = app_now().strftime("%Y-%m-%d")
    selected_date = date_value if date_value else today_str
    selected_month = _empty_month_summary(selected_date)
    data: dict[str, Any] = {
        "today": {
            "calls_today": 0,
            "cost_usd": 0,
            "cost_lkr": 0,
            "tokens_total": 0,
            "tokens_audio": 0,
            "tokens_output": 0,
            "tokens_billed_output": 0,
            "audio_seconds": 0,
            "silence_removed": 0,
            "batch_calls": 0,
            "realtime_calls": 0,
        },
        "languages": {"Sinhala": 0, "English": 0, "Tamil": 0},
        "recent": [],
        "totals": {
            "total_calls": 0,
            "total_cost_usd": 0,
            "total_cost_lkr": 0,
            "total_tokens": 0,
        },
        "month": selected_month,
        "monthly": [],
        "daily": [],
        "silence_saved_s": 0,
        "model_breakdown": [],
        "all_models": [],
        "all_dates": [],
        "active_filter": model_filter,
        "selected_date": selected_date,
        "is_today": selected_date == today_str,
        "database_enabled": is_database_enabled(),
    }
    if error:
        data["database_error"] = error
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _safe_dict(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {key: _json_safe(value) for key, value in dict(row).items()}


def _safe_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_safe_dict(row) for row in rows]


def _selected_month_start(selected_date: str) -> date:
    try:
        selected = date.fromisoformat(selected_date)
    except ValueError:
        selected = app_now().date()
    return selected.replace(day=1)


def _next_month_start(month_start: date) -> date:
    if month_start.month == 12:
        return date(month_start.year + 1, 1, 1)
    return date(month_start.year, month_start.month + 1, 1)


def _month_label(month_key: str) -> str:
    try:
        return datetime.strptime(month_key, "%Y-%m").strftime("%B %Y")
    except ValueError:
        return month_key


def _empty_month_summary(selected_date: str) -> dict[str, Any]:
    month_start = _selected_month_start(selected_date)
    month_key = month_start.strftime("%Y-%m")
    return {
        "month": month_key,
        "label": _month_label(month_key),
        "calls": 0,
        "audio_seconds": 0,
        "tokens": 0,
        "cost_usd": 0,
        "cost_lkr": 0,
    }


def _month_summary(
    month_start: date,
    row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = _empty_month_summary(month_start.isoformat())
    if row:
        safe_row = _safe_dict(row)
        for key in ("calls", "audio_seconds", "tokens", "cost_usd", "cost_lkr"):
            summary[key] = safe_row.get(key, 0) or 0
    return summary


def _monthly_history_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    monthly: list[dict[str, Any]] = []
    for row in rows:
        safe_row = _safe_dict(row)
        month_key = str(safe_row.get("month", "") or "")
        monthly.append(
            {
                "month": month_key,
                "label": _month_label(month_key),
                "calls": safe_row.get("calls", 0) or 0,
                "audio_seconds": safe_row.get("audio_seconds", 0) or 0,
                "tokens": safe_row.get("tokens", 0) or 0,
                "cost_usd": safe_row.get("cost_usd", 0) or 0,
                "cost_lkr": safe_row.get("cost_lkr", 0) or 0,
            }
        )
    return monthly


def get_call_transcript(call_id: int) -> dict[str, Any] | None:
    """Return transcript file metadata for one call; transcript text is file-first."""
    if not is_database_enabled():
        return None

    try:
        init_database()
        with _connect() as conn:
            with _cursor(conn, dictionary=True) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        original_file_name AS filename,
                        '' AS transcript,
                        transcript_txt_path AS transcript_file_path,
                        transcript_saved_at,
                        transcript_output_date
                    FROM transcriptions
                    WHERE id = %s
                    """,
                    (call_id,),
                )
                row = cur.fetchone()
                return _safe_dict(row) if row else None
    except Exception as exc:
        log.warning(
            "Optional MySQL transcript lookup failed: %s", exc.__class__.__name__
        )
        return None


def _where_for_day(model_filter: str, selected_date: str) -> tuple[str, list[Any]]:
    where = "DATE(transcribed_at) = %s"
    params: list[Any] = [selected_date]
    if model_filter != "all":
        where += " AND model_name = %s"
        params.append(model_filter)
    return where, params


def _where_for_month(
    model_filter: str,
    month_start: date,
) -> tuple[str, list[Any]]:
    next_month = _next_month_start(month_start)
    where = "transcribed_at >= %s AND transcribed_at < %s"
    params: list[Any] = [
        datetime(month_start.year, month_start.month, 1),
        datetime(next_month.year, next_month.month, 1),
    ]
    if model_filter != "all":
        where += " AND model_name = %s"
        params.append(model_filter)
    return where, params


def _where_all(model_filter: str) -> tuple[str, list[Any]]:
    if model_filter == "all":
        return "1 = 1", []
    return "model_name = %s", [model_filter]


def get_dashboard_data(model_filter: str = "all", date: str = "") -> dict[str, Any]:
    """
    Compatibility dashboard data query.

    The dashboard is optional on this branch. If MySQL is disabled or
    unavailable, return an empty dashboard-shaped payload instead of failing.
    """
    empty = _empty_dashboard_data(model_filter=model_filter, date_value=date)
    if not is_database_enabled():
        return empty

    selected_date = empty["selected_date"]
    month_start = _selected_month_start(selected_date)
    where_day, params_day = _where_for_day(model_filter, selected_date)
    where_month, params_month = _where_for_month(model_filter, month_start)
    where_all, params_all = _where_all(model_filter)

    try:
        init_database()
        with _connect() as conn:
            with _cursor(conn, dictionary=True) as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS calls_today,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr,
                        COALESCE(SUM(provider_total_tokens), 0) AS tokens_total,
                        COALESCE(SUM(provider_audio_tokens), 0) AS tokens_audio,
                        COALESCE(SUM(provider_output_tokens), 0) AS tokens_output,
                        COALESCE(SUM(provider_billed_output_tokens), 0)
                            AS tokens_billed_output,
                        COALESCE(SUM(duration_seconds), 0) AS audio_seconds,
                        COALESCE(SUM(silence_removed_seconds), 0)
                            AS silence_removed,
                        SUM(CASE WHEN mode = 'batch' THEN 1 ELSE 0 END)
                            AS batch_calls,
                        SUM(CASE WHEN mode <> 'batch' THEN 1 ELSE 0 END)
                            AS realtime_calls
                    FROM transcriptions
                    WHERE {where_day}
                    """,
                    tuple(params_day),
                )
                today_row = _safe_dict(cur.fetchone())

                cur.execute(
                    f"SELECT language FROM transcriptions WHERE {where_day}",
                    tuple(params_day),
                )
                lang_rows = _safe_rows(cur.fetchall())

                lang_counts = {"Sinhala": 0, "English": 0, "Tamil": 0}
                for row in lang_rows:
                    for lang in str(row.get("language", "") or "").split(","):
                        lang = lang.strip().title()
                        if lang in lang_counts:
                            lang_counts[lang] += 1

                cur.execute(
                    f"""
                    SELECT COALESCE(SUM(silence_removed_seconds), 0)
                        AS total_silence_s
                    FROM transcriptions
                    WHERE {where_day}
                    """,
                    tuple(params_day),
                )
                silence_row = _safe_dict(cur.fetchone())

                cur.execute(
                    """
                    SELECT
                        model_name AS model,
                        COUNT(*) AS calls,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr,
                        COALESCE(SUM(provider_total_tokens), 0) AS tokens
                    FROM transcriptions
                    WHERE DATE(transcribed_at) = %s
                    GROUP BY model_name
                    ORDER BY calls DESC
                    """,
                    (selected_date,),
                )
                model_rows = _safe_rows(cur.fetchall())

                cur.execute(
                    f"""
                    SELECT
                        id,
                        original_file_name AS filename,
                        duration_seconds,
                        silence_removed_seconds,
                        model_name AS model,
                        provider_total_tokens AS total_tokens,
                        estimated_cost_usd AS total_cost_usd,
                        estimated_cost_lkr AS total_cost_lkr,
                        provider_billed_output_tokens AS billed_output_tokens,
                        language AS languages_detected,
                        transcribed_at AS processed_at,
                        CASE WHEN mode = 'batch' THEN 1 ELSE 0 END AS batch_mode,
                        transcript_txt_path AS transcript_file_path,
                        transcript_saved_at,
                        transcript_output_date,
                        CASE
                            WHEN COALESCE(transcript_txt_path, '') <> ''
                            THEN 1 ELSE 0
                        END AS success
                    FROM transcriptions
                    WHERE {where_day}
                    ORDER BY id DESC
                    LIMIT 20
                    """,
                    tuple(params_day),
                )
                recent = _safe_rows(cur.fetchall())

                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_calls,
                        COALESCE(SUM(estimated_cost_usd), 0) AS total_cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS total_cost_lkr,
                        COALESCE(SUM(provider_total_tokens), 0) AS total_tokens
                    FROM transcriptions
                    WHERE {where_all}
                    """,
                    tuple(params_all),
                )
                totals = _safe_dict(cur.fetchone())

                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS calls,
                        COALESCE(SUM(duration_seconds), 0) AS audio_seconds,
                        COALESCE(SUM(provider_total_tokens), 0) AS tokens,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr
                    FROM transcriptions
                    WHERE {where_month}
                    """,
                    tuple(params_month),
                )
                month = _month_summary(month_start, cur.fetchone())

                cur.execute(
                    f"""
                    SELECT
                        CONCAT(
                            YEAR(transcribed_at),
                            '-',
                            LPAD(MONTH(transcribed_at), 2, '0')
                        ) AS month,
                        COUNT(*) AS calls,
                        COALESCE(SUM(duration_seconds), 0) AS audio_seconds,
                        COALESCE(SUM(provider_total_tokens), 0) AS tokens,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr
                    FROM transcriptions
                    WHERE {where_all}
                        AND transcribed_at IS NOT NULL
                    GROUP BY YEAR(transcribed_at), MONTH(transcribed_at)
                    ORDER BY YEAR(transcribed_at) DESC, MONTH(transcribed_at) DESC
                    LIMIT 12
                    """,
                    tuple(params_all),
                )
                monthly = _monthly_history_rows(cur.fetchall())

                cur.execute(
                    f"""
                    SELECT
                        DATE(transcribed_at) AS day,
                        COUNT(*) AS calls,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr,
                        COALESCE(SUM(provider_total_tokens), 0) AS tokens,
                        COALESCE(SUM(duration_seconds), 0) AS audio_seconds
                    FROM transcriptions
                    WHERE {where_all}
                    GROUP BY DATE(transcribed_at)
                    ORDER BY day DESC
                    LIMIT 90
                    """,
                    tuple(params_all),
                )
                daily = _safe_rows(cur.fetchall())

                cur.execute(
                    "SELECT DISTINCT model_name AS model "
                    "FROM transcriptions "
                    "WHERE COALESCE(model_name, '') <> '' "
                    "ORDER BY model_name"
                )
                all_models = _safe_rows(cur.fetchall())

                cur.execute(
                    "SELECT DISTINCT DATE(transcribed_at) AS day "
                    "FROM transcriptions "
                    "WHERE transcribed_at IS NOT NULL "
                    "ORDER BY day DESC"
                )
                all_dates = _safe_rows(cur.fetchall())

        return {
            "today": today_row or empty["today"],
            "languages": lang_counts,
            "recent": recent,
            "totals": totals or empty["totals"],
            "month": month,
            "monthly": monthly,
            "daily": daily,
            "silence_saved_s": silence_row.get("total_silence_s", 0),
            "model_breakdown": model_rows,
            "all_models": [row["model"] for row in all_models],
            "all_dates": [row["day"] for row in all_dates],
            "active_filter": model_filter,
            "selected_date": selected_date,
            "is_today": empty["is_today"],
            "database_enabled": True,
        }
    except Exception as exc:
        log.warning("Optional MySQL dashboard query failed: %s", exc.__class__.__name__)
        return _empty_dashboard_data(
            model_filter=model_filter,
            date_value=date,
            error=f"{exc.__class__.__name__}: dashboard database query failed.",
        )
