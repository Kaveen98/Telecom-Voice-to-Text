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
import json
import os
from calendar import monthrange
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping

from config import (
    COST_LIMIT_DB_FAILURE_POLICY,
    DAILY_COST_LIMIT_ENABLED,
    DAILY_COST_LIMIT_LKR,
    DAILY_COST_WARNING_PERCENT,
    app_now,
)

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
        config = _mysql_config()
        return mysql.connector.connect(**config)  # type: ignore[union-attr]
    except (DatabaseDisabledError, DatabaseConfigurationError, DatabaseUnavailableError):
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
    provider VARCHAR(64) DEFAULT 'google',
    api_surface VARCHAR(64) DEFAULT 'vertex_ai',
    vertex_location VARCHAR(128) DEFAULT NULL,
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
    cached_content_token_count BIGINT DEFAULT 0,
    tool_use_prompt_token_count BIGINT DEFAULT 0,
    raw_usage_metadata_json LONGTEXT DEFAULT NULL,
    prompt_tokens_details_json LONGTEXT DEFAULT NULL,
    candidates_tokens_details_json LONGTEXT DEFAULT NULL,
    cache_tokens_details_json LONGTEXT DEFAULT NULL,
    audio_input_cost_usd DECIMAL(18,8) DEFAULT 0,
    text_input_cost_usd DECIMAL(18,8) DEFAULT 0,
    output_cost_usd DECIMAL(18,8) DEFAULT 0,
    estimated_cost_usd DECIMAL(18,8) DEFAULT 0,
    estimated_cost_lkr DECIMAL(18,4) DEFAULT 0,
    cost_calculation_version VARCHAR(64) DEFAULT NULL,
    pricing_version VARCHAR(64) DEFAULT NULL,
    pricing_source VARCHAR(255) DEFAULT NULL,
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

_TRANSCRIPTIONS_SCHEMA_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("provider", "VARCHAR(64) DEFAULT 'google'"),
    ("api_surface", "VARCHAR(64) DEFAULT 'vertex_ai'"),
    ("vertex_location", "VARCHAR(128) DEFAULT NULL"),
    ("cost_calculation_version", "VARCHAR(64) DEFAULT NULL"),
    ("pricing_version", "VARCHAR(64) DEFAULT NULL"),
    ("pricing_source", "VARCHAR(255) DEFAULT NULL"),
    # Store JSON as LONGTEXT for compatibility across MySQL/MariaDB versions.
    ("raw_usage_metadata_json", "LONGTEXT DEFAULT NULL"),
    ("prompt_tokens_details_json", "LONGTEXT DEFAULT NULL"),
    ("candidates_tokens_details_json", "LONGTEXT DEFAULT NULL"),
    ("cache_tokens_details_json", "LONGTEXT DEFAULT NULL"),
    ("cached_content_token_count", "BIGINT DEFAULT 0"),
    ("tool_use_prompt_token_count", "BIGINT DEFAULT 0"),
)


def _mysql_identifier(identifier: str) -> str:
    if not identifier or not all(ch.isalnum() or ch == "_" for ch in identifier):
        raise ValueError("Unsafe MySQL identifier.")
    return f"`{identifier}`"


def get_existing_columns(cur: Any, table_name: str = "transcriptions") -> set[str]:
    """Return existing column names for a MySQL table."""
    cur.execute(f"SHOW COLUMNS FROM {_mysql_identifier(table_name)}")
    columns: set[str] = set()
    for row in cur.fetchall():
        column_name = row.get("Field") if isinstance(row, Mapping) else row[0]
        columns.add(str(column_name).lower())
    return columns


def ensure_column(
    cur: Any,
    existing_columns: set[str],
    column_name: str,
    column_definition: str,
    table_name: str = "transcriptions",
) -> bool:
    """Add a missing column without changing existing columns or data."""
    column_key = column_name.lower()
    if column_key in existing_columns:
        return False

    cur.execute(
        f"ALTER TABLE {_mysql_identifier(table_name)} "
        f"ADD COLUMN {_mysql_identifier(column_name)} {column_definition}"
    )
    existing_columns.add(column_key)
    log.info("MySQL schema migration added column %s.%s", table_name, column_name)
    return True


def migrate_database_schema(cur: Any) -> list[str]:
    """Apply idempotent, non-destructive schema additions."""
    existing_columns = get_existing_columns(cur, "transcriptions")
    applied: list[str] = []
    for column_name, column_definition in _TRANSCRIPTIONS_SCHEMA_MIGRATIONS:
        if ensure_column(cur, existing_columns, column_name, column_definition):
            applied.append(column_name)

    if applied:
        log.info(
            "MySQL schema migration applied %s column(s): %s",
            len(applied),
            ", ".join(applied),
        )
    else:
        log.info("MySQL schema migration checked: no column changes needed.")
    return applied


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
                migrate_database_schema(cur)
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


def _json_text_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_text_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_text_safe(item) for item in value]

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            if method_name == "model_dump":
                return _json_text_safe(method(mode="json", exclude_none=True))
            return _json_text_safe(method())
        except TypeError:
            try:
                return _json_text_safe(method())
            # Optional third-party serializers may fail; try the next fallback.
            except Exception:  # nosec B112
                continue
        # The object is optional metadata; continue to the next representation.
        except Exception:  # nosec B112
            continue

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        public_attrs = {
            key: item
            for key, item in attrs.items()
            if not key.startswith("_") and not callable(item)
        }
        if public_attrs:
            return _json_text_safe(public_attrs)

    return str(value)


def _to_json_text(value: Any) -> str | None:
    if value is None or value == "":
        return None

    try:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                value = json.loads(stripped)
            except (TypeError, ValueError):
                value = stripped

        return json.dumps(
            _json_text_safe(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except Exception:
        return None


def _json_record_value(
    data: Mapping[str, Any],
    column_key: str,
    source_key: str,
) -> str | None:
    value = data.get(column_key)
    if value is None:
        value = data.get(source_key)
    return _to_json_text(value)


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
        data.get("provider") or "google",
        data.get("api_surface") or "vertex_ai",
        data.get("vertex_location") or None,
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
        _to_int(data.get("cached_content_token_count")),
        _to_int(data.get("tool_use_prompt_token_count")),
        _json_record_value(data, "raw_usage_metadata_json", "raw_usage_metadata"),
        _json_record_value(
            data,
            "prompt_tokens_details_json",
            "prompt_tokens_details",
        ),
        _json_record_value(
            data,
            "candidates_tokens_details_json",
            "candidates_tokens_details",
        ),
        _json_record_value(data, "cache_tokens_details_json", "cache_tokens_details"),
        _to_float(data.get("audio_input_cost_usd")),
        _to_float(data.get("text_input_cost_usd")),
        _to_float(data.get("output_cost_usd")),
        total_cost_usd,
        total_cost_lkr,
        data.get("cost_calculation_version") or None,
        data.get("pricing_version") or None,
        data.get("pricing_source") or None,
        lkr_rate,
        _optional_mysql_datetime(data.get("transcript_saved_at")),
        data.get("transcript_output_date") or "",
        data.get("error_message") or "",
    )


_TRANSCRIPTION_COLUMNS: tuple[str, ...] = (
    "original_file_name",
    "file_hash",
    "audio_input_path",
    "audio_processing_path",
    "audio_completed_path",
    "audio_failed_path",
    "transcript_txt_path",
    "metadata_json_path",
    "status",
    "mode",
    "transcribed_at",
    "provider",
    "api_surface",
    "vertex_location",
    "model_name",
    "language",
    "duration_seconds",
    "original_duration_seconds",
    "submitted_duration_seconds",
    "silence_removed_seconds",
    "silence_removed_ratio",
    "provider_input_tokens",
    "provider_audio_tokens",
    "provider_text_input_tokens",
    "provider_output_tokens",
    "provider_thoughts_tokens",
    "provider_billed_output_tokens",
    "provider_total_tokens",
    "cached_content_token_count",
    "tool_use_prompt_token_count",
    "raw_usage_metadata_json",
    "prompt_tokens_details_json",
    "candidates_tokens_details_json",
    "cache_tokens_details_json",
    "audio_input_cost_usd",
    "text_input_cost_usd",
    "output_cost_usd",
    "estimated_cost_usd",
    "estimated_cost_lkr",
    "cost_calculation_version",
    "pricing_version",
    "pricing_source",
    "lkr_rate",
    "transcript_saved_at",
    "transcript_output_date",
    "error_message",
)

# Identifiers come only from the fixed tuple above; every record value uses %s.
_INSERT_TRANSCRIPTION_SQL = (
    "INSERT INTO transcriptions ("  # nosec B608
    + ", ".join(_TRANSCRIPTION_COLUMNS)
    + ") VALUES ("
    + ", ".join(["%s"] * len(_TRANSCRIPTION_COLUMNS))
    + ")"
)

# The same fixed identifiers are used for updates; values still use %s.
_UPDATE_TRANSCRIPTION_SQL = (
    "UPDATE transcriptions SET "  # nosec B608
    + ", ".join(f"{column} = %s" for column in _TRANSCRIPTION_COLUMNS)
    + " WHERE id = %s"
)


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


def update_transcription_record(
    record_id: int,
    record: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> DatabaseWriteResult:
    """Replace a reserved metadata row with its final state and measurements."""
    if not is_database_enabled():
        return _disabled_result()
    if not record_id or record_id <= 0:
        return DatabaseWriteResult(success=False, error="Invalid metadata record id.")

    data = _as_mapping(record, kwargs)
    try:
        init_database()
        with _connect() as conn:
            with _cursor(conn) as cur:
                values = _build_record_values(data) + (record_id,)
                cur.execute(_UPDATE_TRANSCRIPTION_SQL, values)
                if cur.rowcount != 1:
                    raise DatabaseUnavailableError(
                        "Reserved metadata row was not found during finalization."
                    )
        return DatabaseWriteResult(success=True, record_id=record_id)
    except Exception as exc:
        log.warning("Optional MySQL metadata update failed: %s", exc.__class__.__name__)
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
                # Clauses are fixed above; file hash/name remain %s parameters.
                cur.execute(
                    f"SELECT id FROM transcriptions WHERE {where} LIMIT 1",  # nosec B608
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
    start_date: str = "",
    end_date: str = "",
    error: str = "",
) -> dict[str, Any]:
    today_str = app_now().strftime("%Y-%m-%d")
    selected_date = date_value if date_value else today_str
    selected_month = _empty_month_summary(selected_date)
    cost_period, _, _ = _resolve_cost_period(start_date, end_date)
    empty_cost_summary = _empty_cost_totals(finalized=True)
    safety_error = error
    if not safety_error and DAILY_COST_LIMIT_ENABLED and not is_database_enabled():
        safety_error = "Database metadata storage is disabled."
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
        "cost_period": cost_period,
        "range_total": dict(empty_cost_summary),
        "rolling_costs": {
            "last_7_days": dict(empty_cost_summary),
            "last_30_days": dict(empty_cost_summary),
            "last_90_days": dict(empty_cost_summary),
        },
        "projected_month_end": {
            "month_to_date_cost_lkr": 0,
            "projected_cost_lkr": 0,
            "elapsed_days": 0,
            "days_in_month": 0,
        },
        "daily_by_month": [],
        "silence_saved_s": 0,
        "model_breakdown": [],
        "all_models": [],
        "all_dates": [],
        "active_filter": model_filter,
        "selected_date": selected_date,
        "is_today": selected_date == today_str,
        "database_enabled": is_database_enabled(),
        "daily_cost_safety": build_daily_cost_safety_status(error=safety_error),
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


def _parse_date(value: str) -> date | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _date_start(value: date) -> datetime:
    return datetime(value.year, value.month, value.day)


def _coerce_usage_date(target_date: Any = None) -> date:
    if target_date is None:
        return app_now().date()
    if isinstance(target_date, datetime):
        return target_date.date()
    if isinstance(target_date, date):
        return target_date
    parsed = _parse_date(str(target_date))
    return parsed or app_now().date()


def _daily_cost_usage_from_cursor(cur: Any, target_date: Any = None) -> dict[str, Any]:
    usage_date = _coerce_usage_date(target_date)
    start_dt = _date_start(usage_date)
    end_dt = _date_start(usage_date + timedelta(days=1))
    cur.execute(
        """
        SELECT
            COUNT(*) AS calls,
            COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr,
            COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
            COALESCE(SUM(provider_total_tokens), 0) AS tokens,
            COALESCE(SUM(duration_seconds), 0) AS audio_seconds
        FROM transcriptions
        WHERE transcribed_at >= %s
            AND transcribed_at < %s
        """,
        (start_dt, end_dt),
    )
    row = _safe_dict(cur.fetchone())
    return {
        "date": usage_date.isoformat(),
        "calls": _to_int(row.get("calls")),
        "cost_lkr": _to_float(row.get("cost_lkr")),
        "cost_usd": _to_float(row.get("cost_usd")),
        "tokens": _to_int(row.get("tokens")),
        "audio_seconds": _to_float(row.get("audio_seconds")),
    }


def get_daily_cost_usage(target_date: Any = None) -> dict[str, Any]:
    """
    Return estimated API cost usage for one app-local calendar day.

    Raises a database-layer exception when MySQL metadata cannot be checked.
    Watcher safety code uses this distinction to avoid paid Gemini calls when
    COST_LIMIT_DB_FAILURE_POLICY=block.
    """
    if not is_database_enabled():
        raise DatabaseDisabledError("Database metadata storage is disabled.")

    try:
        init_database()
        with _connect() as conn:
            with _cursor(conn, dictionary=True) as cur:
                return _daily_cost_usage_from_cursor(cur, target_date)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseUnavailableError(
            f"MySQL daily cost usage query failed: {exc.__class__.__name__}"
        ) from exc


def build_daily_cost_safety_status(
    usage: Mapping[str, Any] | None = None,
    estimated_next_cost_lkr: float = 0.0,
    error: str = "",
) -> dict[str, Any]:
    usage_date = str((usage or {}).get("date") or app_now().date().isoformat())
    configured = DAILY_COST_LIMIT_ENABLED and DAILY_COST_LIMIT_LKR > 0
    limit_lkr = DAILY_COST_LIMIT_LKR if DAILY_COST_LIMIT_LKR > 0 else 0.0
    warning_percent = DAILY_COST_WARNING_PERCENT
    db_failure_policy = COST_LIMIT_DB_FAILURE_POLICY

    base = {
        "date": usage_date,
        "enabled": bool(configured),
        "limit_lkr": limit_lkr,
        "used_lkr": 0.0,
        "remaining_lkr": limit_lkr,
        "usage_percent": 0.0,
        "warning_percent": warning_percent,
        "warning": False,
        "blocked": False,
        "allowed": True,
        "status": "disabled",
        "reason": "Daily cost limit is disabled.",
        "db_failure_policy": db_failure_policy,
        "estimated_next_cost_lkr": max(0.0, estimated_next_cost_lkr),
        "calls": 0,
        "cost_usd": 0.0,
        "tokens": 0,
        "audio_seconds": 0.0,
        "error": "",
    }

    if not configured:
        return base

    if error:
        return {
            **base,
            "status": "db_unavailable",
            "blocked": True,
            "allowed": False,
            "warning": True,
            "reason": (
                "Daily cost limit cannot verify MySQL accounting; "
                "paid requests are blocked fail-closed."
            ),
            "error": error,
        }

    usage = usage or {}
    used_lkr = _to_float(usage.get("cost_lkr"))
    estimated_next = max(0.0, estimated_next_cost_lkr)
    projected_lkr = used_lkr + estimated_next
    remaining_lkr = max(limit_lkr - used_lkr, 0.0)
    usage_percent = (used_lkr / limit_lkr * 100) if limit_lkr > 0 else 0.0
    projected_percent = (
        projected_lkr / limit_lkr * 100
        if limit_lkr > 0 and estimated_next > 0
        else usage_percent
    )
    warning_threshold = limit_lkr * (warning_percent / 100)
    blocked = used_lkr >= limit_lkr or (
        estimated_next > 0 and projected_lkr > limit_lkr
    )
    warning = used_lkr >= warning_threshold or projected_lkr >= warning_threshold

    if used_lkr >= limit_lkr:
        reason = "Daily estimated Gemini API cost limit reached."
    elif estimated_next > 0 and projected_lkr > limit_lkr:
        reason = "Estimated next file may exceed the daily cost limit."
    elif warning:
        reason = "Daily estimated Gemini API cost is above the warning threshold."
    else:
        reason = "Daily estimated Gemini API cost is below the configured limit."

    return {
        **base,
        "status": "blocked" if blocked else "warning" if warning else "ok",
        "used_lkr": used_lkr,
        "remaining_lkr": remaining_lkr,
        "usage_percent": usage_percent,
        "projected_usage_percent": projected_percent,
        "warning": warning,
        "blocked": blocked,
        "allowed": not blocked,
        "reason": reason,
        "calls": _to_int(usage.get("calls")),
        "cost_usd": _to_float(usage.get("cost_usd")),
        "tokens": _to_int(usage.get("tokens")),
        "audio_seconds": _to_float(usage.get("audio_seconds")),
    }


def _range_label(start: date, end: date) -> str:
    if start == end:
        return start.isoformat()
    if start.day == 1 and end == _next_month_start(start.replace(day=1)) - timedelta(days=1):
        return _month_label(start.strftime("%Y-%m"))
    return f"{start.isoformat()} to {end.isoformat()}"


def _resolve_cost_period(
    start_date: str = "",
    end_date: str = "",
) -> tuple[dict[str, str], datetime, datetime]:
    today = app_now().date()
    parsed_start = _parse_date(start_date)
    parsed_end = _parse_date(end_date)

    if parsed_start is None and parsed_end is None:
        start = today.replace(day=1)
        end = _next_month_start(start) - timedelta(days=1)
    elif parsed_start is not None and parsed_end is None:
        start = parsed_start
        end = parsed_start
    elif parsed_start is None and parsed_end is not None:
        start = parsed_end.replace(day=1)
        end = parsed_end
    else:
        start = parsed_start or today
        end = parsed_end or start

    if start > end:
        start, end = end, start

    end_exclusive = end + timedelta(days=1)
    period = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "label": _range_label(start, end),
    }
    return period, _date_start(start), _date_start(end_exclusive)


def _where_for_range(
    model_filter: str,
    start_dt: datetime,
    end_exclusive_dt: datetime,
) -> tuple[str, list[Any]]:
    where = "transcribed_at >= %s AND transcribed_at < %s"
    params: list[Any] = [start_dt, end_exclusive_dt]
    if model_filter != "all":
        where += " AND model_name = %s"
        params.append(model_filter)
    return where, params


def _empty_cost_totals(finalized: bool = False) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "calls": 0,
        "audio_seconds": 0,
        "tokens": 0,
        "cost_usd": 0,
        "cost_lkr": 0,
    }
    if finalized:
        return _finalize_cost_totals(totals)
    return totals


def _finalize_cost_totals(totals: Mapping[str, Any]) -> dict[str, Any]:
    calls = _to_int(totals.get("calls"))
    audio_seconds = _to_float(totals.get("audio_seconds"))
    audio_minutes = audio_seconds / 60 if audio_seconds else 0
    tokens = _to_int(totals.get("tokens"))
    cost_usd = _to_float(totals.get("cost_usd"))
    cost_lkr = _to_float(totals.get("cost_lkr"))
    return {
        "calls": calls,
        "audio_seconds": audio_seconds,
        "audio_minutes": audio_minutes,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "cost_lkr": cost_lkr,
        "avg_cost_per_call_lkr": cost_lkr / calls if calls else 0,
        "avg_cost_per_audio_min_lkr": cost_lkr / audio_minutes if audio_minutes else 0,
    }


def _cost_summary(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return _empty_cost_totals(finalized=True)
    safe_row = _safe_dict(row)
    return _finalize_cost_totals(
        {
            "calls": safe_row.get("calls", 0) or 0,
            "audio_seconds": safe_row.get("audio_seconds", 0) or 0,
            "tokens": safe_row.get("tokens", 0) or 0,
            "cost_usd": safe_row.get("cost_usd", 0) or 0,
            "cost_lkr": safe_row.get("cost_lkr", 0) or 0,
        }
    )


def _add_cost_totals(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    target["calls"] = _to_int(target.get("calls")) + _to_int(source.get("calls"))
    target["audio_seconds"] = _to_float(target.get("audio_seconds")) + _to_float(
        source.get("audio_seconds")
    )
    target["tokens"] = _to_int(target.get("tokens")) + _to_int(source.get("tokens"))
    target["cost_usd"] = _to_float(target.get("cost_usd")) + _to_float(
        source.get("cost_usd")
    )
    target["cost_lkr"] = _to_float(target.get("cost_lkr")) + _to_float(
        source.get("cost_lkr")
    )


def _daily_by_month(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    by_month: dict[str, dict[str, Any]] = {}
    range_totals = _empty_cost_totals()

    for row in rows:
        summary = _cost_summary(row)
        day = str(_safe_dict(row).get("day", "") or "")
        month_key = day[:7]
        if not month_key:
            continue

        group = by_month.get(month_key)
        if group is None:
            group = {
                "month": month_key,
                "label": _month_label(month_key),
                "rows": [],
                "_subtotal": _empty_cost_totals(),
            }
            by_month[month_key] = group
            groups.append(group)

        group["rows"].append({"day": day, **summary})
        _add_cost_totals(group["_subtotal"], summary)
        _add_cost_totals(range_totals, summary)

    finalized_groups: list[dict[str, Any]] = []
    for group in groups:
        finalized_groups.append(
            {
                "month": group["month"],
                "label": group["label"],
                "rows": group["rows"],
                "subtotal": _finalize_cost_totals(group["_subtotal"]),
            }
        )

    return finalized_groups, _finalize_cost_totals(range_totals)


def _query_cost_summary(cur: Any, where: str, params: list[Any]) -> dict[str, Any]:
    # Callers pass WHERE fragments from the fixed internal builders below.
    cur.execute(
        f"""
        SELECT
            COUNT(*) AS calls,
            COALESCE(SUM(duration_seconds), 0) AS audio_seconds,
            COALESCE(SUM(provider_total_tokens), 0) AS tokens,
            COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
            COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr
        FROM transcriptions
        WHERE {where}
        """,  # nosec B608
        tuple(params),
    )
    return _cost_summary(cur.fetchone())


def _rolling_costs(cur: Any, model_filter: str) -> dict[str, Any]:
    today = app_now().date()
    end_exclusive = _date_start(today + timedelta(days=1))
    results: dict[str, Any] = {}
    for label, days in (
        ("last_7_days", 7),
        ("last_30_days", 30),
        ("last_90_days", 90),
    ):
        start = today - timedelta(days=days - 1)
        where, params = _where_for_range(model_filter, _date_start(start), end_exclusive)
        results[label] = _query_cost_summary(cur, where, params)
    return results


def _projected_month_end(
    cur: Any,
    model_filter: str,
    selected_month_start: date,
) -> dict[str, Any]:
    today = app_now().date()
    next_month = _next_month_start(selected_month_start)
    days_in_month = monthrange(
        selected_month_start.year,
        selected_month_start.month,
    )[1]

    if today < selected_month_start:
        elapsed_days = 0
        month_to_date_cost = 0.0
    else:
        if today >= next_month:
            elapsed_days = days_in_month
            end_exclusive = next_month
        else:
            elapsed_days = max(today.day, 1)
            end_exclusive = today + timedelta(days=1)

        where, params = _where_for_range(
            model_filter,
            _date_start(selected_month_start),
            _date_start(end_exclusive),
        )
        month_to_date_cost = _query_cost_summary(cur, where, params)["cost_lkr"]

    projected = (
        (month_to_date_cost / elapsed_days) * days_in_month
        if elapsed_days > 0 and month_to_date_cost > 0
        else 0
    )
    return {
        "month_to_date_cost_lkr": month_to_date_cost,
        "projected_cost_lkr": projected,
        "elapsed_days": elapsed_days,
        "days_in_month": days_in_month,
    }


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
                # Fixed WHERE fragment; model/date values stay in params_day.
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
    selected = _parse_date(selected_date) or app_now().date()
    return _where_for_range(
        model_filter,
        _date_start(selected),
        _date_start(selected + timedelta(days=1)),
    )


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


def get_dashboard_data(
    model_filter: str = "all",
    date: str = "",
    start_date: str = "",
    end_date: str = "",
) -> dict[str, Any]:
    """
    Compatibility dashboard data query.

    The dashboard is optional on this branch. If MySQL is disabled or
    unavailable, return an empty dashboard-shaped payload instead of failing.
    """
    empty = _empty_dashboard_data(
        model_filter=model_filter,
        date_value=date,
        start_date=start_date,
        end_date=end_date,
    )
    if not is_database_enabled():
        return empty

    selected_date = empty["selected_date"]
    month_start = _selected_month_start(selected_date)
    cost_period, range_start_dt, range_end_exclusive_dt = _resolve_cost_period(
        start_date,
        end_date,
    )
    where_day, params_day = _where_for_day(model_filter, selected_date)
    where_day_all_models, params_day_all_models = _where_for_day("all", selected_date)
    where_month, params_month = _where_for_month(model_filter, month_start)
    where_all, params_all = _where_all(model_filter)
    where_range, params_range = _where_for_range(
        model_filter,
        range_start_dt,
        range_end_exclusive_dt,
    )
    # Every interpolated WHERE fragment above is selected from fixed SQL text;
    # model and date values are carried separately in the matching params list.

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
                    """,  # nosec B608
                    tuple(params_day),
                )
                today_row = _safe_dict(cur.fetchone())

                # Fixed WHERE fragment; model/date values stay in params_day.
                cur.execute(
                    f"SELECT language FROM transcriptions WHERE {where_day}",  # nosec B608
                    tuple(params_day),
                )
                lang_rows = _safe_rows(cur.fetchall())

                lang_counts = {"Sinhala": 0, "English": 0, "Tamil": 0}
                for row in lang_rows:
                    for lang in str(row.get("language", "") or "").split(","):
                        lang = lang.strip().title()
                        if lang in lang_counts:
                            lang_counts[lang] += 1

                # Fixed WHERE fragment; model/date values stay in params_day.
                cur.execute(
                    f"""
                    SELECT COALESCE(SUM(silence_removed_seconds), 0)
                        AS total_silence_s
                    FROM transcriptions
                    WHERE {where_day}
                    """,  # nosec B608
                    tuple(params_day),
                )
                silence_row = _safe_dict(cur.fetchone())

                # Fixed WHERE fragment with no external model value interpolated.
                cur.execute(
                    f"""
                    SELECT
                        model_name AS model,
                        COUNT(*) AS calls,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr,
                        COALESCE(SUM(provider_total_tokens), 0) AS tokens
                    FROM transcriptions
                    WHERE {where_day_all_models}
                    GROUP BY model_name
                    ORDER BY calls DESC
                    """,  # nosec B608
                    tuple(params_day_all_models),
                )
                model_rows = _safe_rows(cur.fetchall())

                # Fixed WHERE fragment; model/date values stay in params_day.
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
                    """,  # nosec B608
                    tuple(params_day),
                )
                recent = _safe_rows(cur.fetchall())

                # Fixed WHERE fragment; model value stays in params_all.
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_calls,
                        COALESCE(SUM(estimated_cost_usd), 0) AS total_cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS total_cost_lkr,
                        COALESCE(SUM(provider_total_tokens), 0) AS total_tokens
                    FROM transcriptions
                    WHERE {where_all}
                    """,  # nosec B608
                    tuple(params_all),
                )
                totals = _safe_dict(cur.fetchone())

                # Fixed WHERE fragment; model/date values stay in params_month.
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
                    """,  # nosec B608
                    tuple(params_month),
                )
                month = _month_summary(month_start, cur.fetchone())

                # Fixed WHERE fragment; model value stays in params_all.
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
                    """,  # nosec B608
                    tuple(params_all),
                )
                monthly = _monthly_history_rows(cur.fetchall())

                # Fixed WHERE fragment; model value stays in params_all.
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
                    """,  # nosec B608
                    tuple(params_all),
                )
                daily = _safe_rows(cur.fetchall())

                # Fixed WHERE fragment; model/date values stay in params_range.
                cur.execute(
                    f"""
                    SELECT
                        DATE(transcribed_at) AS day,
                        COUNT(*) AS calls,
                        COALESCE(SUM(duration_seconds), 0) AS audio_seconds,
                        COALESCE(SUM(provider_total_tokens), 0) AS tokens,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COALESCE(SUM(estimated_cost_lkr), 0) AS cost_lkr
                    FROM transcriptions
                    WHERE {where_range}
                    GROUP BY DATE(transcribed_at)
                    ORDER BY day ASC
                    """,  # nosec B608
                    tuple(params_range),
                )
                daily_by_month, range_total = _daily_by_month(cur.fetchall())
                rolling_costs = _rolling_costs(cur, model_filter)
                projected_month_end = _projected_month_end(
                    cur,
                    model_filter,
                    range_start_dt.date().replace(day=1),
                )

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

                try:
                    daily_cost_safety = build_daily_cost_safety_status(
                        _daily_cost_usage_from_cursor(cur)
                    )
                except Exception as safety_exc:
                    daily_cost_safety = build_daily_cost_safety_status(
                        error=(
                            f"{safety_exc.__class__.__name__}: "
                            "daily cost safety query failed."
                        )
                    )

        return {
            "today": today_row or empty["today"],
            "languages": lang_counts,
            "recent": recent,
            "totals": totals or empty["totals"],
            "month": month,
            "monthly": monthly,
            "daily": daily,
            "cost_period": cost_period,
            "range_total": range_total,
            "rolling_costs": rolling_costs,
            "projected_month_end": projected_month_end,
            "daily_by_month": daily_by_month,
            "silence_saved_s": silence_row.get("total_silence_s", 0),
            "model_breakdown": model_rows,
            "all_models": [row["model"] for row in all_models],
            "all_dates": [row["day"] for row in all_dates],
            "active_filter": model_filter,
            "selected_date": selected_date,
            "is_today": empty["is_today"],
            "database_enabled": True,
            "daily_cost_safety": daily_cost_safety,
        }
    except Exception as exc:
        log.warning("Optional MySQL dashboard query failed: %s", exc.__class__.__name__)
        return _empty_dashboard_data(
            model_filter=model_filter,
            date_value=date,
            start_date=start_date,
            end_date=end_date,
            error=f"{exc.__class__.__name__}: dashboard database query failed.",
        )
