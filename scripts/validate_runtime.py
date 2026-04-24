"""
Validate Linux/runtime readiness without sending audio to Gemini.

Usage:
    python scripts/validate_runtime.py
"""
from __future__ import annotations

import importlib
import shutil
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check(condition: bool, label: str, detail: str = "") -> bool:
    status = "OK" if condition else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    return condition


def _check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return _check(False, f"import {module_name}", str(exc))
    return _check(True, f"import {module_name}")


def _check_binary(binary: str) -> bool:
    path = shutil.which(binary)
    return _check(path is not None, f"{binary} on PATH", path or "not found")


def _check_writable_dir(path: Path, label: str) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        return _check(False, label, f"{path} ({exc})")
    return _check(True, label, str(path))


def main() -> int:
    ok = True

    ok &= _check(
        sys.version_info >= (3, 10),
        "Python version >= 3.10",
        sys.version.split()[0],
    )

    for module_name in (
        "google.genai",
        "watchdog",
        "flask",
        "google.cloud.storage",
        "google.cloud.aiplatform",
        "gunicorn",
    ):
        ok &= _check_import(module_name)

    ok &= _check_binary("ffmpeg")
    ok &= _check_binary("ffprobe")

    try:
        import config
    except Exception as exc:
        _check(False, "config loads", str(exc))
        return 1

    ok &= _check(True, "config loads")
    ok &= _check_writable_dir(config.DATA_DIR, "data directory writable")
    ok &= _check_writable_dir(config.INPUT_DIR, "input directory writable")
    ok &= _check_writable_dir(config.OUTPUT_DIR, "output directory writable")
    ok &= _check_writable_dir(config.LOG_DIR, "log directory writable")
    ok &= _check_writable_dir(config.DB_PATH.parent, "database directory writable")

    try:
        from database import get_job_summary, init_db

        init_db()
        summary = get_job_summary()
        ok &= _check(True, "SQLite init", str(config.DB_PATH))
        ok &= _check(isinstance(summary, dict), "SQLite job summary")
    except sqlite3.Error as exc:
        ok &= _check(False, "SQLite init", str(exc))
    except Exception as exc:
        ok &= _check(False, "SQLite init", str(exc))

    try:
        credentials_path, project_id, location = config.validate_google_runtime()
        ok &= _check(True, "Google project visible", project_id)
        ok &= _check(True, "Google location visible", location)
        if credentials_path:
            ok &= _check(Path(credentials_path).exists(), "credentials file exists", credentials_path)
        else:
            ok &= _check(
                True,
                "Google credentials",
                "GOOGLE_APPLICATION_CREDENTIALS unset; relying on ADC",
            )
    except Exception as exc:
        ok &= _check(False, "Google runtime config", str(exc))

    print("\nRuntime validation " + ("passed." if ok else "failed."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

