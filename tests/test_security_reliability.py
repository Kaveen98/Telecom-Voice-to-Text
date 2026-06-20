from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


os.environ["DASHBOARD_SECRET_KEY"] = "test-only-dashboard-secret"

import dashboard_server
import database
import gemini_flash_stt
import watcher


class DashboardAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        dashboard_server.app.config.update(TESTING=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.users_path = Path(self.temp_dir.name) / "users.json"
        self.users_patch = mock.patch.object(
            dashboard_server,
            "_USERS_FILE",
            self.users_path,
        )
        self.users_patch.start()
        self.addCleanup(self.users_patch.stop)
        self.client = dashboard_server.app.test_client()

    def _write_users(self, users: list[dict]) -> None:
        self.users_path.write_text(json.dumps(users), encoding="utf-8")

    def test_empty_object_cannot_authenticate(self) -> None:
        self._write_users(
            [
                {},
                {"password": "local-test-password"},
                {"username": "operator"},
                {"username": "", "password": "local-test-password"},
                {"username": "operator", "password": ""},
            ]
        )
        response = self.client.post(
            "/login",
            data={"username": "", "password": ""},
        )
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as session:
            self.assertNotIn("user", session)

    def test_empty_submitted_credentials_cannot_authenticate(self) -> None:
        self._write_users(
            [{"username": "operator", "password": "local-test-password"}]
        )
        for username, password in (("", "local-test-password"), ("operator", "")):
            response = self.client.post(
                "/login",
                data={"username": username, "password": password},
            )
            self.assertEqual(response.status_code, 200)
            with self.client.session_transaction() as session:
                self.assertNotIn("user", session)

    def test_valid_user_authenticates_case_insensitively(self) -> None:
        self._write_users(
            [
                {
                    "username": "Operator",
                    "password": "local-test-password",
                    "role": "viewer",
                }
            ]
        )
        wrong_password = self.client.post(
            "/login",
            data={"username": "operator", "password": "LOCAL-TEST-PASSWORD"},
        )
        self.assertEqual(wrong_password.status_code, 200)
        with self.client.session_transaction() as session:
            self.assertNotIn("user", session)

        response = self.client.post(
            "/login",
            data={"username": "operator", "password": "local-test-password"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")

    def test_protocol_relative_next_target_is_rejected(self) -> None:
        for target in (
            "//example.invalid",
            "https://example.invalid",
            "http://example.invalid",
            "/\\evil",
            "\\\\example.invalid",
            "/%5Cevil",
            "/%0d%0aLocation:%20https://example.invalid",
        ):
            self.assertEqual(dashboard_server._safe_next_target(target), "/")
        for target in ("/", "/api/data", "/api/data?model=all"):
            self.assertEqual(dashboard_server._safe_next_target(target), target)

        self._write_users(
            [{"username": "operator", "password": "local-test-password"}]
        )
        response = self.client.post(
            "/login?next=//example.invalid",
            data={"username": "operator", "password": "local-test-password"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")

        encoded_response = self.client.get("/logout")
        self.assertEqual(encoded_response.status_code, 302)
        encoded_response = self.client.post(
            "/login?next=/%2Fexample.invalid",
            data={"username": "operator", "password": "local-test-password"},
        )
        self.assertEqual(encoded_response.headers["Location"], "/")

        self.client.get("/logout")
        local_response = self.client.post(
            "/login?next=/api/data",
            data={"username": "operator", "password": "local-test-password"},
        )
        self.assertEqual(local_response.headers["Location"], "/api/data")

    def test_cookie_security_defaults(self) -> None:
        self.assertTrue(dashboard_server.app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(
            dashboard_server.app.config["SESSION_COOKIE_SAMESITE"],
            "Lax",
        )
        self.assertFalse(dashboard_server.app.config["SESSION_COOKIE_SECURE"])
        with mock.patch.dict(os.environ, {"DASHBOARD_COOKIE_SECURE": "true"}):
            self.assertTrue(dashboard_server._env_bool("DASHBOARD_COOKIE_SECURE"))


class DashboardEscapingTests(unittest.TestCase):
    def test_escape_html_blocks_script_markup(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is unavailable for the JavaScript helper test")

        match = re.search(
            r"function escapeHtml\(value\) \{.*?^\}",
            dashboard_server._HTML,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        payloads = {
            "<script>alert(1)</script>": "&lt;script&gt;alert(1)&lt;/script&gt;",
            "<img src=x onerror=alert(1)>": (
                "&lt;img src=x onerror=alert(1)&gt;"
            ),
            '\"><svg/onload=alert(1)>': (
                "&quot;&gt;&lt;svg/onload=alert(1)&gt;"
            ),
        }
        for payload, expected in payloads.items():
            script = (
                match.group(0)
                + "\nprocess.stdout.write(escapeHtml("
                + json.dumps(payload)
                + "));"
            )
            result = subprocess.run(
                [node, "-e", script],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout, expected)

    def test_untrusted_dashboard_fields_use_escaping_or_dom_text(self) -> None:
        html = dashboard_server._HTML
        self.assertIn("${escapeHtml(c.filename)}", html)
        self.assertIn("${escapeHtml(m.model)}", html)
        self.assertIn("${escapeHtml(safety.error)}", html)
        self.assertIn("${escapeHtml(l)}", html)
        self.assertIn("${escapeHtml((c.processed_at||'')", html)
        self.assertIn("encodeURIComponent(c.id)", html)
        self.assertIn("hasOwnProperty.call", html)
        self.assertIn("button.textContent =", html)
        self.assertNotIn("document.getElementById('filter-bar').innerHTML", html)


class WatcherReliabilityTests(unittest.TestCase):
    @staticmethod
    def _paths() -> watcher.RuntimePaths:
        root = Path("synthetic-test-root")
        return watcher.RuntimePaths(
            incoming=root / "incoming",
            processing=root / "processing",
            completed=root / "completed",
            failed=root / "failed",
            deferred=root / "deferred",
            transcriptions=root / "transcriptions",
            logs=root / "logs",
        )

    def test_worker_continues_after_per_file_exception(self) -> None:
        work_queue: queue.Queue[Path | None] = queue.Queue()
        work_queue.put(Path("first.wav"))
        work_queue.put(Path("second.wav"))
        work_queue.put(None)
        processed: list[str] = []

        def process(path: Path, paths: watcher.RuntimePaths) -> None:
            _ = paths
            processed.append(path.name)
            if path.name == "first.wav":
                raise RuntimeError("synthetic processing failure")

        with (
            mock.patch.object(watcher, "_file_queue", work_queue),
            mock.patch.object(watcher, "_shutdown", threading.Event()),
            mock.patch.object(watcher, "_process_candidate", side_effect=process),
            mock.patch.object(watcher, "_mark_dequeued"),
        ):
            watcher._worker(self._paths())

        self.assertEqual(processed, ["first.wav", "second.wav"])

    def test_guarded_reservation_and_finalization_order(self) -> None:
        events: list[str] = []
        finalized_ids: list[int | None] = []
        processing_path = Path("synthetic-processing.wav")
        completed_path = Path("synthetic-completed.wav")

        def move(source: Path, destination: Path) -> Path:
            _ = source
            if destination == processing_path:
                events.append("claim")
            else:
                events.append("completed_move")
            return destination

        def save_database(record: dict, existing_record_id: int | None = None):
            _ = record, existing_record_id
            events.append("database")
            finalized_ids.append(existing_record_id)
            return watcher.DatabaseStatus(status="saved", record_id=1)

        def reserve(record: dict) -> watcher.DatabaseStatus:
            _ = record
            events.append("reservation")
            return watcher.DatabaseStatus(status="reserved", record_id=41)

        def transcribe(path: str) -> dict:
            _ = path
            events.append("transcribe")
            return result

        result = {
            "transcript": "synthetic",
            "duration_seconds": 1,
            "languages_detected": ["English"],
        }
        saved = {
            "transcript_txt_path": "synthetic.txt",
            "metadata_json_path": "synthetic.json",
            "transcript_saved_at": "2026-01-01T00:00:00+05:30",
        }

        with (
            mock.patch.object(watcher, "_wait_for_stable_file"),
            mock.patch.object(watcher, "_processing_path", return_value=processing_path),
            mock.patch.object(watcher, "_safe_move", side_effect=move),
            mock.patch.object(watcher, "_compute_file_hash", return_value="hash"),
            mock.patch.object(watcher, "_is_duplicate_file", return_value=False),
            mock.patch.object(watcher, "_daily_cost_limit_configured", return_value=True),
            mock.patch.object(
                watcher,
                "_daily_cost_limit_decision",
                return_value={"allowed": True, "blocked": False, "warning": False},
            ),
            mock.patch.object(watcher, "_estimate_next_file_cost_lkr", return_value=1.0),
            mock.patch.object(watcher, "_build_cost_reservation_record", return_value={}),
            mock.patch.object(
                watcher,
                "_reserve_cost_accounting_record",
                side_effect=reserve,
            ),
            mock.patch.object(watcher, "_audio_archive_path", return_value=completed_path),
            mock.patch.object(watcher, "_save_optional_database_record", side_effect=save_database),
            mock.patch.object(watcher, "_update_metadata_json"),
            mock.patch.object(gemini_flash_stt, "transcribe_audio_file", side_effect=transcribe),
            mock.patch.object(gemini_flash_stt, "save_transcription_outputs", return_value=saved),
        ):
            watcher._process_candidate(Path("synthetic.wav"), self._paths())

        self.assertLess(events.index("reservation"), events.index("transcribe"))
        self.assertLess(events.index("completed_move"), events.index("database"))
        self.assertEqual(finalized_ids, [41])

    def test_move_to_failed_contains_secondary_move_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            current_path = Path(temp_dir) / "synthetic-processing.wav"
            current_path.write_bytes(b"synthetic")
            with (
                mock.patch.object(
                    watcher,
                    "_audio_archive_path",
                    return_value=Path(temp_dir) / "failed.wav",
                ),
                mock.patch.object(
                    watcher,
                    "_safe_move",
                    side_effect=OSError("synthetic secondary move failure"),
                ),
                self.assertLogs(watcher.log, level="ERROR"),
            ):
                failed_path = watcher._move_to_failed(
                    self._paths(),
                    current_path,
                    current_path.name,
                    RuntimeError("synthetic primary failure"),
                )

        self.assertIsNone(failed_path)

    def test_final_move_failure_preserves_outputs_without_retrying_gemini(self) -> None:
        processing_path = Path("synthetic-processing.wav")
        completed_path = Path("synthetic-completed.wav")

        def move(source: Path, destination: Path) -> Path:
            _ = source
            if destination == processing_path:
                return processing_path
            raise FileNotFoundError("synthetic completed move failure")

        result = {
            "transcript": "synthetic",
            "duration_seconds": 1,
            "languages_detected": ["English"],
        }
        saved = {
            "transcript_txt_path": "synthetic.txt",
            "metadata_json_path": "synthetic.json",
            "transcript_saved_at": "2026-01-01T00:00:00+05:30",
        }

        with (
            mock.patch.object(watcher, "_wait_for_stable_file"),
            mock.patch.object(watcher, "_processing_path", return_value=processing_path),
            mock.patch.object(watcher, "_safe_move", side_effect=move),
            mock.patch.object(watcher, "_compute_file_hash", return_value="hash"),
            mock.patch.object(watcher, "_is_duplicate_file", return_value=False),
            mock.patch.object(watcher, "_daily_cost_limit_configured", return_value=False),
            mock.patch.object(watcher, "_audio_archive_path", return_value=completed_path),
            mock.patch.object(watcher, "_move_to_failed", return_value=Path("failed.wav")),
            mock.patch.object(watcher, "_save_optional_failure_record") as save_failure,
            mock.patch.object(watcher, "_save_optional_database_record") as save_database,
            mock.patch.object(watcher, "_update_metadata_json") as update_metadata,
            mock.patch.object(gemini_flash_stt, "transcribe_audio_file", return_value=result) as transcribe,
            mock.patch.object(gemini_flash_stt, "save_transcription_outputs", return_value=saved) as save_outputs,
        ):
            watcher._process_candidate(Path("synthetic.wav"), self._paths())

        transcribe.assert_called_once()
        save_outputs.assert_called_once()
        save_database.assert_not_called()
        save_failure.assert_called_once()
        self.assertEqual(
            update_metadata.call_args.args[1]["status"],
            "failed_finalization",
        )


class DailyCostSafetyTests(unittest.TestCase):
    def test_database_accounting_error_is_always_fail_closed(self) -> None:
        with (
            mock.patch.object(database, "DAILY_COST_LIMIT_ENABLED", True),
            mock.patch.object(database, "DAILY_COST_LIMIT_LKR", 100.0),
            mock.patch.object(database, "COST_LIMIT_DB_FAILURE_POLICY", "allow"),
        ):
            status = database.build_daily_cost_safety_status(
                error="synthetic database failure"
            )
        self.assertTrue(status["blocked"])
        self.assertFalse(status["allowed"])
        self.assertEqual(status["status"], "db_unavailable")

    def test_watcher_blocks_when_daily_usage_cannot_be_read(self) -> None:
        with (
            mock.patch.object(watcher, "DAILY_COST_LIMIT_ENABLED", True),
            mock.patch.object(watcher, "DAILY_COST_LIMIT_LKR", 100.0),
            mock.patch.object(database, "DAILY_COST_LIMIT_ENABLED", True),
            mock.patch.object(database, "DAILY_COST_LIMIT_LKR", 100.0),
            mock.patch.object(
                watcher,
                "get_daily_cost_usage",
                side_effect=RuntimeError("synthetic database failure"),
            ),
        ):
            status = watcher._daily_cost_limit_decision(estimated_next_cost_lkr=1.0)
        self.assertTrue(status["blocked"])
        self.assertFalse(status["allowed"])

    def test_reservation_failure_prevents_gemini_call(self) -> None:
        paths = WatcherReliabilityTests._paths()
        processing_path = Path("synthetic-processing.wav")
        allowed = {"allowed": True, "blocked": False, "warning": False}

        with (
            mock.patch.object(watcher, "_wait_for_stable_file"),
            mock.patch.object(watcher, "_processing_path", return_value=processing_path),
            mock.patch.object(watcher, "_safe_move", return_value=processing_path),
            mock.patch.object(watcher, "_compute_file_hash", return_value="hash"),
            mock.patch.object(watcher, "_is_duplicate_file", return_value=False),
            mock.patch.object(watcher, "_daily_cost_limit_configured", return_value=True),
            mock.patch.object(watcher, "_daily_cost_limit_decision", return_value=allowed),
            mock.patch.object(watcher, "_estimate_next_file_cost_lkr", return_value=1.0),
            mock.patch.object(watcher, "_build_cost_reservation_record", return_value={}),
            mock.patch.object(
                watcher,
                "_reserve_cost_accounting_record",
                return_value=watcher.DatabaseStatus(
                    status="failed",
                    error="synthetic reservation failure",
                ),
            ),
            mock.patch.object(watcher, "_move_to_deferred") as move_to_deferred,
            mock.patch.object(gemini_flash_stt, "transcribe_audio_file") as transcribe,
        ):
            watcher._process_candidate(Path("synthetic.wav"), paths)

        move_to_deferred.assert_called_once()
        transcribe.assert_not_called()

    def test_reserved_row_is_finalized_as_failed(self) -> None:
        reservation = {"status": "reserved", "estimated_cost_lkr": 1.0}
        with (
            mock.patch.object(watcher, "is_database_enabled", return_value=True),
            mock.patch.object(
                watcher,
                "_save_optional_database_record",
                return_value=watcher.DatabaseStatus(status="saved", record_id=41),
            ) as save_record,
        ):
            watcher._save_optional_failure_record(
                "synthetic.wav",
                Path("synthetic-failed.wav"),
                "synthetic-hash",
                RuntimeError("synthetic processing failure"),
                reservation_id=41,
                reservation_record=reservation,
            )

        failure_record, existing_id = save_record.call_args.args
        self.assertEqual(existing_id, 41)
        self.assertEqual(failure_record["status"], "failed")
        self.assertEqual(failure_record["estimated_cost_lkr"], 1.0)


if __name__ == "__main__":
    unittest.main()
