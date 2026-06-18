# Telecom Voice-to-Text

Telecom Voice-to-Text is a Windows-oriented realtime transcription pipeline for
telecom call audio. It watches an incoming audio folder, transcribes supported
files with Gemini on Vertex AI, saves transcript TXT files and sidecar JSON
metadata, and optionally writes searchable metadata to MySQL for the local
dashboard.

The primary output is always file-based:

- TXT transcripts under `transcriptions/YYYY-MM-DD/`
- JSON metadata files beside each TXT transcript

MySQL is optional metadata storage. The watcher is designed to save TXT/JSON
outputs even when MySQL is disabled or unavailable. The dashboard depends on
MySQL metadata, so it may be empty in file-only mode.

## Main Features

- Realtime folder watcher for `input_audio/incoming/`
- Supported audio extensions: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.opus`
- Gemini transcription through Vertex AI using service-account credentials
- FFmpeg-based audio conversion and silence stripping
- TXT transcript output plus JSON operational metadata
- Optional MySQL metadata table with idempotent schema creation/migration
- Duplicate check when MySQL is enabled, using file hash and original filename
- Optional daily estimated API cost safety limit in LKR
- Local Flask dashboard at `http://127.0.0.1:5050`
- Dashboard login backed by a local plaintext `users.json` file
- Dashboard transcript view/download and daily/monthly cost CSV downloads

Archived batch, Linux service, and legacy materials are under `docs/archive/`.
They are not part of the active realtime workflow.

## Repository Structure

```text
Telecom-Voice-to-Text/
  watcher.py                 Realtime folder watcher
  gemini_flash_stt.py         Gemini/Vertex transcription and TXT/JSON saving
  database.py                 Optional MySQL metadata and dashboard queries
  config.py                   Shared environment and path configuration
  dashboard_server.py         Optional local Flask dashboard
  transcript_storage.py       Compatibility wrapper for transcript paths
  requirements.txt            Python dependencies
  .env.example                Safe local configuration template
  users.example.json          Safe dashboard user template
  users.json                  Local dashboard users file, ignored by Git

  static/
    app-logo.png              Dashboard logo asset

  input_audio/
    incoming/                 Drop new audio files here
    processing/               Files currently being processed
    completed/                Successful audio archive, grouped by date
    failed/                   Failed audio archive and error notes
    deferred/                 Files blocked by cost safety rules

  transcriptions/             TXT transcripts and JSON metadata
  logs/                       Watcher logs
  credentials/                Local Google service-account JSON files

  docs/archive/               Inactive batch, Linux, and legacy materials
```

Runtime data, local secrets, local databases, audio files, logs, transcripts,
and `users.json` are ignored by Git. The committed `.gitkeep` files preserve the
empty runtime folders.

## Environment Setup

Run commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install FFmpeg separately and make sure both `ffmpeg` and `ffprobe` are on the
Windows `PATH`.

```powershell
ffmpeg -version
ffprobe -version
```

Required Python packages are listed in `requirements.txt`:

- `google-genai`
- `watchdog`
- `flask`
- `mysql-connector-python`
- `python-dotenv`
- `pydub`

## .env Configuration

Create a local `.env` from the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` for the local machine. Existing process environment variables take
precedence over values in `.env`.

```text
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
STT_GEMINI_LOCATION=us-central1

DB_ENABLED=true
DB_BACKEND=mysql
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=telecom_voice_to_text
MYSQL_USER=telecom_app
MYSQL_PASSWORD=change_this_password
MYSQL_CONNECT_TIMEOUT=10

DAILY_COST_LIMIT_ENABLED=false
DAILY_COST_LIMIT_LKR=1000
DAILY_COST_WARNING_PERCENT=80
COST_LIMIT_PREFLIGHT_ENABLED=true
COST_LIMIT_DB_FAILURE_POLICY=block

INPUT_INCOMING_DIR=input_audio/incoming
INPUT_PROCESSING_DIR=input_audio/processing
INPUT_COMPLETED_DIR=input_audio/completed
INPUT_FAILED_DIR=input_audio/failed
INPUT_DEFERRED_DIR=input_audio/deferred
TRANSCRIPTIONS_DIR=transcriptions
LOG_DIR=logs

APP_TIMEZONE=Asia/Colombo
TRANSCRIPT_DATE_FORMAT=%Y.%m.%d
```

Notes:

- `DB_BACKEND=mysql` is the supported database backend.
- `DB_ENABLED=false` runs file-only mode.
- `TRANSCRIPTIONS_DIR` controls where TXT/JSON outputs are written.
- Current transcript and audio archive date folders are created as `YYYY-MM-DD`.
- `APP_TIMEZONE` affects app-local dates for dashboard and daily cost logic.
- `TRANSCRIPT_DATE_FORMAT` exists in configuration for compatibility, but the
  active saver currently uses `YYYY-MM-DD` output folders.

Optional watcher stability settings can be supplied in `.env` or the process
environment:

```text
WATCHER_STABLE_CHECK_SECONDS=5
WATCHER_STABLE_CHECK_INTERVAL=1
WATCHER_STABLE_MAX_WAIT_SECONDS=300
```

## Google, Gemini, and Vertex AI Credentials

The active transcription path uses Vertex AI service-account authentication.
Configure:

- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `STT_GEMINI_LOCATION`

Setup outline:

1. Create or select a Google Cloud project.
2. Enable the Vertex AI API for that project.
3. Create a dedicated service account for this application.
4. Grant the minimum Vertex AI permissions required by your organization.
5. Create a JSON key for that service account.
6. Store the JSON file locally, for example:

```text
credentials/google-credentials.json
```

Then point `.env` to it:

```text
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
STT_GEMINI_LOCATION=us-central1
```

The current code uses `google-genai` with `vertexai=True`. Gemini API-key
authentication is not the active path in this repository.

Treat the service-account JSON like a password. Do not commit it. If it is
exposed, revoke the key and create a new one.

## Database Setup Options

### File-Only Mode

Set:

```text
DB_ENABLED=false
```

In file-only mode:

- TXT transcripts are still saved.
- JSON metadata files are still saved.
- MySQL connection failures are avoided.
- The dashboard may show empty metrics because it reads MySQL metadata.

### MySQL Metadata Mode

Use MySQL when you need dashboard metrics, transcript history, duplicate checks,
cost history, and CSV reporting.

Create a database and dedicated application user. Log in as a MySQL
administrator and run:

```sql
CREATE DATABASE telecom_voice_to_text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER 'telecom_app'@'localhost' IDENTIFIED BY 'change_this_password';
CREATE USER 'telecom_app'@'127.0.0.1' IDENTIFIED BY 'change_this_password';

GRANT SELECT, INSERT, UPDATE, CREATE, ALTER, INDEX
ON telecom_voice_to_text.*
TO 'telecom_app'@'localhost';

GRANT SELECT, INSERT, UPDATE, CREATE, ALTER, INDEX
ON telecom_voice_to_text.*
TO 'telecom_app'@'127.0.0.1';

FLUSH PRIVILEGES;
```

Then configure `.env`:

```text
DB_ENABLED=true
DB_BACKEND=mysql
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=telecom_voice_to_text
MYSQL_USER=telecom_app
MYSQL_PASSWORD=change_this_password
MYSQL_CONNECT_TIMEOUT=10
```

The application creates the `transcriptions` metadata table when database
storage is used. It also applies non-destructive schema additions for known
metadata columns.

Validate database initialization after MySQL is running and `.env` is set:

```powershell
python -c "import database; print('enabled:', database.is_database_enabled()); database.init_database(); print('database init ok')"
```

SQLite and PostgreSQL are not active runtime backends in the current code.
Local files such as `calls.db` are ignored runtime artifacts.

## Audio Folder Workflow

The realtime workflow is:

```text
input_audio/incoming/
  -> input_audio/processing/
  -> transcriptions/YYYY-MM-DD/*.txt and *.json
  -> optional MySQL metadata
  -> input_audio/completed/YYYY-MM-DD/
```

Failure and safety paths:

```text
input_audio/failed/YYYY-MM-DD/
input_audio/deferred/YYYY-MM-DD/
```

Folder behavior:

- Put new audio only in `input_audio/incoming/`.
- The watcher waits until file size and modification time stop changing.
- Stable supported files move to `input_audio/processing/`.
- Successful files move to `input_audio/completed/YYYY-MM-DD/`.
- Failed files move to `input_audio/failed/YYYY-MM-DD/` with a `.error.txt` note.
- Files blocked by the daily cost safety limit move to
  `input_audio/deferred/YYYY-MM-DD/` with a `.deferred.txt` note.
- Unsupported file extensions are ignored.

TXT/JSON output is saved before optional MySQL metadata. A MySQL outage should
not cause another Gemini call for the same successful transcription.

## Realtime Watcher Usage

Start the watcher:

```powershell
python watcher.py
```

Use a custom incoming folder:

```powershell
python watcher.py --input E:\Path\To\IncomingAudio
```

Validate startup without processing files or calling Gemini:

```powershell
python watcher.py --dry-run
```

Logs are written to the console and to:

```text
logs/watcher_YYYY-MM-DD.log
```

Stop the watcher with `Ctrl+C`. The watcher stops the filesystem observer and
waits for the active worker to finish.

## Dashboard Usage

The dashboard is optional and implemented in `dashboard_server.py`. It is a
local Flask app with login-protected pages and API routes.

Start it:

```powershell
python dashboard_server.py
```

Open:

```text
http://127.0.0.1:5050
```

Optional host and port:

```powershell
python dashboard_server.py --host 127.0.0.1 --port 5050
python dashboard_server.py --port 8080
```

Dashboard routes include:

- `/login`
- `/logout`
- `/`
- `/api/data`
- `/api/transcripts/<call_id>`
- `/api/transcripts/<call_id>/download`
- `/api/daily-cost.csv`
- `/api/monthly-cost.csv`

The dashboard session secret is loaded from `DASHBOARD_SECRET_KEY` when set.
Otherwise, the app creates a local `.dashboard_secret` file, which is ignored by
Git.

Dashboard data depends on MySQL. If `DB_ENABLED=false` or MySQL is unavailable,
the dashboard returns an empty dashboard-shaped payload instead of failing.

## Dashboard Username and Password Setup

Dashboard users are loaded from a local plaintext JSON file:

```text
users.json
```

Create it from the committed safe template:

```powershell
Copy-Item users.example.json users.json
```

Then edit `users.json` and replace all example passwords. The file must be a
JSON array:

```json
[
  {
    "username": "admin",
    "password": "change-this-password",
    "role": "admin"
  },
  {
    "username": "viewer",
    "password": "change-this-password",
    "role": "viewer"
  }
]
```

Password checks compare the submitted password with the plaintext `password`
value in `users.json`. Restrict filesystem access to this file.

The `role` value is stored in the Flask session. Current dashboard routes are
login-protected, not separated into admin-only and viewer-only route behavior.

## Changing Dashboard Usernames and Passwords

Edit `users.json` directly:

- Add a user by adding another object to the JSON array.
- Change a username by editing `"username"`.
- Change a password by editing `"password"`.
- Remove a user by deleting that object from the array.
- Use `"role": "admin"` or `"role": "viewer"`.

Keep the JSON valid. If the file is missing, malformed, or not a JSON array, the
dashboard loads no users and login fails.

Do not commit `users.json`. Commit only `users.example.json`.

## Reports, CSV, and Downloads

Implemented dashboard reporting features include:

- Calls, duration, token, language, model, and estimated cost summary tiles
- Estimated cost in USD and LKR
- Daily cost safety status
- Cost by model for the selected day
- 14-day cost trend
- Rolling cost totals for the last 7, 30, and 90 days
- Current month summary and projected month-end cost
- Monthly cost history
- Daily cost history grouped by month
- Recent calls table
- Transcript text view
- Transcript TXT download

CSV download routes:

```text
/api/daily-cost.csv
/api/monthly-cost.csv
```

Both routes accept dashboard filters such as:

```text
?model=gemini-2.5-flash
?start_date=2026-06-01&end_date=2026-06-30
```

The daily CSV includes day rows, month subtotal rows, and a grand total row. The
monthly CSV includes calls, audio minutes, tokens, estimated USD cost,
estimated LKR cost, and average LKR cost per call.

Cost values are estimates based on token usage, configured pricing constants,
and the USD/LKR rate fetched by the transcription code, with a fallback rate if
the exchange-rate request fails. They are not guaranteed to match the final
Google Cloud invoice exactly.

## Basic Validation Commands

Run these before processing production audio:

```powershell
python -m compileall watcher.py gemini_flash_stt.py database.py config.py transcript_storage.py dashboard_server.py
python watcher.py --help
python watcher.py --dry-run
python dashboard_server.py --help
python gemini_flash_stt.py --help
python -c "import database; print('db enabled:', database.is_database_enabled())"
ffmpeg -version
ffprobe -version
```

After MySQL is configured and running:

```powershell
python -c "import database; print('enabled:', database.is_database_enabled()); database.init_database(); print('database init ok')"
```

For a direct Gemini test with an approved non-production audio sample:

```powershell
python gemini_flash_stt.py input_audio\incoming\sample.mp3 --save
```

That command makes a real Vertex AI request and may incur cost.

## Common Troubleshooting

### `ffmpeg` or `ffprobe` is not recognized

Install FFmpeg, add the FFmpeg `bin` folder to `PATH`, then open a new
PowerShell window and run:

```powershell
ffmpeg -version
ffprobe -version
```

### Google credentials are not found

Check `.env`:

```text
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
```

Confirm the JSON file exists at that path. Do not print or share the JSON
contents.

### Vertex AI permission denied

Common causes:

- Vertex AI API is not enabled.
- `GOOGLE_CLOUD_PROJECT` is the wrong project ID.
- The service account lacks the required Vertex AI permissions.
- Billing or organization policy blocks Vertex AI usage.
- `STT_GEMINI_LOCATION` points to a region where the selected model is not usable.

### MySQL connection failed

Check that MySQL is running, `.env` has the correct connection values, and the
dedicated MySQL user has privileges on `telecom_voice_to_text`.

The watcher should still save TXT/JSON files when optional MySQL metadata fails.

### Database appears disabled unexpectedly

Process environment variables override `.env`. In the current PowerShell
session, clear non-secret DB overrides and retry:

```powershell
Remove-Item Env:DB_ENABLED -ErrorAction SilentlyContinue
Remove-Item Env:DB_BACKEND -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_HOST -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_PORT -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_DATABASE -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_USER -ErrorAction SilentlyContinue
python -c "import database; print('enabled:', database.is_database_enabled())"
```

Do not print `MYSQL_PASSWORD`.

### Audio file stays in `incoming`

The watcher waits for the file to become stable before moving it. Large files or
slow network copies can remain in `incoming` until the size and modification
time stop changing.

Tune the stable-file settings if needed:

```text
WATCHER_STABLE_CHECK_SECONDS=5
WATCHER_STABLE_CHECK_INTERVAL=1
WATCHER_STABLE_MAX_WAIT_SECONDS=300
```

### Audio file moved to `failed`

Open the matching `.error.txt` file under `input_audio/failed/YYYY-MM-DD/`.
The error note contains a short sanitized failure message. The watcher continues
running for later files.

### Audio file moved to `deferred`

The daily estimated API cost safety rule blocked the request before Gemini was
called. Requeue the audio later by moving it back to the incoming folder, or
raise `DAILY_COST_LIMIT_LKR` after reviewing expected cost.

### Dashboard is empty

The dashboard reads MySQL metadata. If `DB_ENABLED=false`, MySQL is down, or no
successful rows exist yet, dashboard metrics may be empty while TXT/JSON
transcripts still exist on disk.

### Dashboard login fails

Check that `users.json` exists, contains valid JSON, and includes the username
and plaintext password being used. Invalid JSON causes the dashboard to load no
users.

### Full transcript is not printed in the terminal

Full transcript text is intentionally not printed by default. Read the TXT file
under `transcriptions/YYYY-MM-DD/`, or use `--print-transcript` with
`gemini_flash_stt.py` only when it is appropriate to display the call text.

## Security Notes

- Do not commit `.env`.
- Do not commit `users.json`.
- Do not commit `.dashboard_secret`.
- Do not commit Google service-account JSON files.
- Do not commit real audio, transcripts, logs, local databases, private keys, or archives containing secrets.
- Change every password from `users.example.json` before using the dashboard.
- Use strong local credentials for dashboard users and MySQL users.
- Restrict filesystem permissions on `users.json`, `.env`, `credentials/`, and `transcriptions/`.
- Use a dedicated non-root MySQL user for the application.
- Keep the dashboard bound to `127.0.0.1` unless it is protected by a reverse proxy, VPN, firewall rules, and HTTPS.
- Treat transcripts as sensitive customer or call data.
- Rotate Google service-account keys immediately if exposed.

