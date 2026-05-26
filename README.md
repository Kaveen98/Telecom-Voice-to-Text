# Telecom Voice-to-Text

SLT / telecom call-center speech-to-text pipeline using Google Vertex AI Gemini, PostgreSQL, FFmpeg, Watchdog, and a Flask dashboard.

This README documents the repository as it currently exists. Older README text referenced SQLite, but the active database code in `database.py` uses PostgreSQL through `DATABASE_URL`.

## Project Overview

This project transcribes Sinhala, English, and Tamil telecom call audio with Gemini on Vertex AI, records token/cost/metadata in PostgreSQL, and shows operational metrics in a local dashboard.

Typical realtime flow:

1. Audio files are placed in an input folder.
2. `watcher.py` detects supported new or pre-existing files.
3. `gemini_flash_stt.py` converts the audio with FFmpeg, optionally strips silence, and sends the audio to Gemini.
4. The transcript and usage metadata are returned.
5. `database.py` stores the call record in PostgreSQL.
6. The transcript TXT is saved under `outputs/transcriptions/YYYY.MM.DD/` using the completion/import date.
7. `dashboard_server.py` displays costs, tokens, languages, recent calls, realtime-vs-batch counts, and transcript view/download buttons.

## Key Features

- Folder-based realtime transcription with Watchdog.
- Manual single-file transcription CLI.
- Sinhala, English, and Tamil transcription prompt with `[SI]`, `[EN]`, and `[TA]` language tags.
- FFmpeg conversion to 16 kHz mono WAV before Gemini requests.
- Optional silence stripping controlled by the `STRIP_SILENCE` constant in `gemini_flash_stt.py`.
- Per-call PostgreSQL logging for transcript, model, duration, silence removed, token usage, estimated cost, languages, and processed timestamp.
- Dated transcript TXT output folders under `outputs/transcriptions/YYYY.MM.DD/`.
- Flask dashboard with auto-refresh, date filtering, model filtering, recent-call table, daily totals, 14-day cost trend, language breakdown, and batch/realtime split.
- Dashboard View Transcript and Download TXT actions for completed calls.
- Standalone Vertex AI Batch Prediction CLI in `batch_processor.py` for bulk jobs through Google Cloud Storage.
- Basic systemd service files for Linux watcher and dashboard processes.

## System Architecture / Processing Flow

```text
                 realtime path

  input folder
      |
      v
  watcher.py  ----->  gemini_flash_stt.py  ----->  Vertex AI Gemini
      |                    |
      |                    v
      |              outputs/transcriptions/YYYY.MM.DD/*.txt
      v
  database.py  ----->  PostgreSQL calls table
      |
      v
  dashboard_server.py  ----->  http://localhost:5050


                 batch path

  local audio files
      |
      v
  batch_processor.py
      |
      +--> upload audio to GCS: batch-audio/
      +--> build/upload JSONL: batch-jobs/
      +--> create Vertex AI BatchPredictionJob
      +--> poll until terminal state, unless --no-wait is used
      +--> download JSONL results from batch-output/<job_id>/
      +--> save transcripts and PostgreSQL call rows with batch_mode=1
```

## Repository Structure

```text
Telecom-Voice-to-Text/
  gemini_flash_stt.py              Core realtime Gemini transcription engine
  config.py                        Shared .env, timezone, and transcript output config
  transcript_storage.py            Dated transcript TXT storage helper
  watcher.py                       Folder watcher and realtime processing worker
  database.py                      PostgreSQL connection, schema, writes, dashboard queries
  dashboard_server.py              Flask dashboard with embedded HTML/CSS/JS
  batch_processor.py               Standalone Vertex AI Batch Prediction CLI
  how_it_works.html                Static explanatory/reference page
  requirements.txt                 Python packages currently listed by the repo
  .env.example                     Safe environment variable template
  .gitignore                       Ignore rules for env, credentials, output, input audio
  slt-watcher.service              Example Linux systemd unit for watcher.py
  slt-dashboard.service            Example Linux systemd unit for dashboard_server.py
  calls.db                         Legacy SQLite artifact; not used by current code
  examples/
    use_from_another_system.py     Minimal import example for transcribe_audio_file()
  input_audio/
    .gitkeep                       Default input folder placeholder
  outputs/transcriptions/          Generated dated transcripts, ignored by git
  output/                          Legacy generated transcript folder, ignored by git
  scripts/                         No active scripts currently; only __pycache__ was present
  experiments/                     Ignored experiment outputs, not runtime code
  credentials/
    .gitkeep                       Placeholder for service-account JSON
```

There is no `pyproject.toml`, `tests/`, `templates/`, `static/`, or `deploy/` directory in the current repository. The dashboard HTML is embedded directly in `dashboard_server.py`.

## Core Modules and Responsibilities

| File | Responsibility |
|---|---|
| `config.py` | Loads `.env` without overriding process env vars and exposes app timezone and transcript output settings. |
| `transcript_storage.py` | Saves transcript TXT files under dated output folders and returns DB-ready file metadata. |
| `gemini_flash_stt.py` | Loads `.env`, validates Google credentials/project/location, converts audio to WAV, strips silence when enabled, calls `client.models.generate_content()`, parses language footer/tags, estimates token costs, and exposes `transcribe_audio_file()`. |
| `watcher.py` | Watches an input folder, queues supported audio files, retries failed transcriptions, saves call metadata, writes dated transcript files, and logs to `watcher.log`. |
| `database.py` | Uses `psycopg2` and `DATABASE_URL`; creates/migrates the `calls`, `batch_jobs`, and `batch_items` tables on first use; provides call and durable batch helper functions. |
| `dashboard_server.py` | Runs a Flask dashboard, JSON data/reset routes, and transcript view/download routes. |
| `batch_processor.py` | Uploads audio and JSONL requests to GCS, creates a Vertex AI `BatchPredictionJob`, optionally polls, downloads JSONL results, parses Gemini output, writes dated transcripts, and saves `batch_mode=1` call rows. |
| `examples/use_from_another_system.py` | Example of importing and calling `transcribe_audio_file()` from another Python script. |
| `how_it_works.html` | Static visual reference. Treat runtime code as the source of truth if this page drifts. |

## Requirements / Prerequisites

- Python 3.10 or newer.
- PostgreSQL server.
- FFmpeg and ffprobe available on `PATH`.
- Google Cloud project with billing enabled.
- Vertex AI API enabled.
- Service account JSON credentials with permission to call Vertex AI.
- For batch processing: a Google Cloud Storage bucket and permissions to upload/list/download objects.

Python packages listed in `requirements.txt`:

```text
google-genai
watchdog
flask
google-cloud-storage
google-cloud-aiplatform
psycopg2-binary
```

`psycopg2-binary` is required because `database.py` connects to PostgreSQL with `psycopg2`.

## Environment Configuration

Create `.env` in the project root by copying `.env.example` and replacing placeholders. Do not commit `.env`.

Recommended `.env` template:

```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
STT_GEMINI_LOCATION=us-central1
DATABASE_URL=postgresql://slt:your-password@127.0.0.1:5432/slt_calls
APP_TIMEZONE=Asia/Colombo
TRANSCRIPT_OUTPUT_DIR=outputs/transcriptions
TRANSCRIPT_DATE_FORMAT=%Y.%m.%d
```

Verified environment variables used by code:

| Variable | Used by | Required | Notes |
|---|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | `gemini_flash_stt.py`, batch via `validate_setup()` | Yes | Path to service-account JSON. Relative paths are resolved from the repo root when possible. |
| `GOOGLE_CLOUD_PROJECT` | `gemini_flash_stt.py`, `batch_processor.py` | Yes | Google Cloud project ID. |
| `STT_GEMINI_LOCATION` | `gemini_flash_stt.py`, batch via `validate_setup()` | Optional but recommended | Defaults to `us-central1` if unset. Gemini preview/global models may require `global`. |
| `DATABASE_URL` | `database.py` | Yes for dashboard/watcher/database writes | PostgreSQL DSN. |
| `APP_TIMEZONE` | `config.py`, `database.py`, `transcript_storage.py` | Optional | Defaults to `Asia/Colombo`. Used for processed timestamps and transcript output dates. |
| `TRANSCRIPT_OUTPUT_DIR` | `config.py`, `transcript_storage.py` | Optional | Defaults to `outputs/transcriptions`. Relative paths are resolved from the repository root. |
| `TRANSCRIPT_DATE_FORMAT` | `config.py`, `transcript_storage.py` | Optional | Defaults to `%Y.%m.%d`, producing folders like `2026.05.26`. |
| `BATCH_INPUT_DIR` | `config.py`, `batch_processor.py` | Optional | Defaults to `input_audio/batch`; server-side folder for Phase 1 durable batch preparation. |
| `BATCH_GCS_BUCKET` | `config.py`, `batch_processor.py` | Required for real batch work | GCS bucket used for audio, JSONL input, and batch output. Validated only when preparing/submitting batch work. |
| `BATCH_MODEL` | `config.py`, `batch_processor.py` | Optional | Defaults to `gemini-2.5-flash`. |
| `BATCH_LOCATION` | `config.py`, `batch_processor.py` | Optional | Defaults to `global`. |
| `BATCH_AUDIO_PREFIX`, `BATCH_JOBS_PREFIX`, `BATCH_OUTPUT_PREFIX` | `config.py`, `batch_processor.py` | Optional | GCS prefixes for uploaded audio, JSONL inputs, and Vertex output. |
| `BATCH_LOCAL_WORK_DIR` | `config.py`, `batch_processor.py` | Optional | Defaults to `outputs/batch_work`; local JSONL staging, ignored by git. |

Other important constants are currently code-level settings, not environment variables:

| Constant | File | Current value / behavior |
|---|---|---|
| `MODEL_NAME` | `gemini_flash_stt.py` | `gemini-2.5-flash`; used by realtime/manual transcription. |
| `DEFAULT_LOCATION` | `gemini_flash_stt.py` | `us-central1`; overridden by `STT_GEMINI_LOCATION`. |
| `STRIP_SILENCE` | `gemini_flash_stt.py` | `True`; enables FFmpeg silence removal. |
| `LKR_RATE_FALLBACK` | `gemini_flash_stt.py` | `316.0`; used if live exchange-rate fetch fails. |
| `POLL_INTERVAL_S` | `batch_processor.py` | `60`; seconds between batch status checks. |

Batch pricing is explicit in `batch_processor.py` for Gemini 2.5 Flash and Gemini 3 Flash Preview, with separate standard, priority, and batch rates.

## Google Cloud Vertex AI Setup

The realtime transcription path uses Google Vertex AI Gemini through `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, and `STT_GEMINI_LOCATION`. The optional batch path in `batch_processor.py` also uses Google Cloud Storage because it uploads audio and JSONL request files to a GCS bucket and downloads JSONL result files.

### 1. Create or access a Google Cloud project

You need access to the Google Cloud project that will own billing, Vertex AI requests, service accounts, and any GCS buckets used for batch processing.

Use this placeholder project ID in local examples:

```text
your-gcp-project-id
```

Do not put a real private project ID into shared documentation unless your team explicitly wants it public.

### 2. Enable required APIs

Enable these APIs in the Google Cloud project:

- Vertex AI API: required for realtime Gemini transcription and Vertex AI Batch Prediction.
- Cloud Storage API: required only if you use `batch_processor.py`, because batch processing uploads audio/JSONL inputs to GCS and reads JSONL outputs from GCS.

Batch processing also requires a GCS bucket that already exists before running `batch_processor.py --bucket your-gcs-bucket`.

### 3. Service account setup

Create a service account for this application rather than using a personal user credential.

The service account needs permission to:

- Use Vertex AI / Gemini in the configured project and location.
- Create and inspect Vertex AI Batch Prediction jobs if you use `batch_processor.py`.
- Upload, list, and download objects in the chosen Cloud Storage bucket if you use batch processing.

The exact IAM roles may depend on your organization policy. Ask a project administrator to grant the minimum permissions needed for Vertex AI and, if batch is enabled, the specific GCS bucket.

### 4. Download credentials JSON

Download a JSON key for the service account and store it locally, for example:

```text
credentials/google-credentials.json
```

Warning: this JSON file is a secret. Never commit it to GitHub, paste it into tickets, or share it in chat. The repository keeps `credentials/.gitkeep` only so the folder exists; real credential files should remain local and ignored by git.

### 5. Configure `.env`

Create `.env` from `.env.example` and fill in local values:

```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
STT_GEMINI_LOCATION=us-central1
DATABASE_URL=postgresql://slt:your-password@127.0.0.1:5432/slt_calls
APP_TIMEZONE=Asia/Colombo
TRANSCRIPT_OUTPUT_DIR=outputs/transcriptions
TRANSCRIPT_DATE_FORMAT=%Y.%m.%d
```

`STT_GEMINI_LOCATION` must be a Vertex AI location where the selected `MODEL_NAME` is available. Some preview/global Gemini models may require `STT_GEMINI_LOCATION=global`.

Do not commit `.env`.

The code uses the Google Gen AI SDK with `vertexai=True` and API version `v1` for realtime transcription. The batch CLI uses `google-cloud-aiplatform` `JobServiceClient` and `BatchPredictionJob`.

## PostgreSQL Setup

The active database backend is PostgreSQL. `calls.db` exists in the repository as a legacy SQLite artifact, but current code does not use it and it should not be pushed to GitHub.

`database.py` reads `DATABASE_URL` from the process environment. If it is not already set, it loads `.env` from the repository root, copies the `DATABASE_URL` value into `os.environ`, and creates a shared `psycopg2.pool.ThreadedConnectionPool` from that DSN.

PostgreSQL must be running before starting `watcher.py`, `dashboard_server.py`, or any command that calls `database.init_db()`.

Example Windows/PostgreSQL setup varies by installer. Run these commands in PowerShell where `psql` is available, and replace `your-password` with a strong local password:

```powershell
psql -U postgres -c "CREATE USER slt WITH PASSWORD 'your-password';"
psql -U postgres -c "CREATE DATABASE slt_calls OWNER slt;"
```

Equivalent SQL:

```sql
CREATE USER slt WITH PASSWORD 'your-password';
CREATE DATABASE slt_calls OWNER slt;
```

Example `.env` line:

```env
DATABASE_URL=postgresql://slt:your-password@127.0.0.1:5432/slt_calls
```

Initialize the schema after dependencies and `.env` are ready:

```powershell
python -c "from database import init_db; init_db(); print('database ok')"
```

`database.init_db()` creates the `calls` table if needed and adds the optional `billed_output_tokens` column if missing.

## FFmpeg Setup

`gemini_flash_stt.py` requires both `ffmpeg` and `ffprobe` on `PATH`.

Windows:

1. Download FFmpeg from https://ffmpeg.org/download.html or install it with your preferred package manager.
2. Add the FFmpeg `bin` folder to `PATH`.
3. Verify:

```powershell
ffmpeg -version
ffprobe -version
```

Linux:

```bash
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
ffprobe -version
```

## Installation Guide for Windows PowerShell

```powershell
git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git
cd Telecom-Voice-to-Text

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create credentials and environment file:

```powershell
New-Item -ItemType Directory -Force .\credentials | Out-Null
Copy-Item C:\path\to\your-service-account.json .\credentials\google-credentials.json
Copy-Item .\.env.example .\.env
notepad .\.env
```

Edit `.env` so it includes all four required values:

```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
STT_GEMINI_LOCATION=us-central1
DATABASE_URL=postgresql://slt:your-password@127.0.0.1:5432/slt_calls
```

Initialize PostgreSQL schema:

```powershell
python -c "from database import init_db; init_db(); print('database ok')"
```

## Optional Linux Deployment Guide

The repo includes two systemd unit examples:

- `slt-watcher.service`
- `slt-dashboard.service`

They are hardcoded for:

- Linux user: `slt`
- Working directory: `/opt/slt/Telecom-Voice-to-Text`
- Python path: `/opt/slt/Telecom-Voice-to-Text/.venv/bin/python`

Adjust those values before installing them on a server.

Example:

```bash
sudo cp slt-watcher.service /etc/systemd/system/slt-watcher.service
sudo cp slt-dashboard.service /etc/systemd/system/slt-dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable --now slt-watcher.service
sudo systemctl enable --now slt-dashboard.service
sudo systemctl status slt-watcher.service
sudo systemctl status slt-dashboard.service
```

The service files use the Python scripts directly. There is no Gunicorn, reverse proxy, or logrotate configuration in the repository. For production, put the dashboard behind a properly secured reverse proxy or VPN and restrict access.

Make sure the Linux service user can create and write to the transcript output directory. With the default service examples this usually means:

```bash
sudo mkdir -p /opt/slt/Telecom-Voice-to-Text/outputs/transcriptions
sudo chown -R slt:slt /opt/slt/Telecom-Voice-to-Text/outputs
```

## How to Run

### Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Initialize the database

Run this once after PostgreSQL and `.env` are configured:

```powershell
python -c "from database import init_db; init_db(); print('database ok')"
```

### Start the watcher

Recommended: use a dedicated empty incoming folder so the watcher does not process old files already in `input_audio`.

Make sure PostgreSQL is running first; the watcher writes every completed transcription to the PostgreSQL `calls` table.

```powershell
python watcher.py --input .\input_audio\incoming
```

Default behavior if no input folder is passed:

```powershell
python watcher.py
```

This watches `input_audio/`.

Warning: on startup, `watcher.py` scans the watched folder and queues every existing supported audio file. It also processes new files created afterward.

Optional flag:

```powershell
python watcher.py --input .\input_audio\incoming --batch
```

`--batch` only marks saved call rows as `batch_mode=1`. It does not submit Vertex AI Batch Prediction jobs. Use `batch_processor.py` for true Vertex batch processing.

### Start the dashboard

In a second terminal:

Make sure PostgreSQL is running first; the dashboard queries the PostgreSQL `calls` table on load and refresh.

```powershell
.\.venv\Scripts\Activate.ps1
python dashboard_server.py
```

Open:

```text
http://localhost:5050
```

Custom host/port:

```powershell
python dashboard_server.py --host 127.0.0.1 --port 5050
python dashboard_server.py --port 8080
```

### Process a single audio file

This makes a real Vertex AI Gemini request:

```powershell
python gemini_flash_stt.py .\input_audio\sample.mp3 --save
```

Without `--save`, the transcript is printed but not written to disk:

```powershell
python gemini_flash_stt.py .\input_audio\sample.mp3
```

### Process batch jobs

This makes a real Vertex AI Batch Prediction job and uses GCS:

```powershell
python batch_processor.py .\input_audio\call1.mp3 .\input_audio\call2.mp3 --bucket your-gcs-bucket
```

Submit without waiting:

```powershell
python batch_processor.py .\input_audio\call1.mp3 --bucket your-gcs-bucket --no-wait
```

If `--no-wait` is used, the script persists the prepared/submitted job in PostgreSQL, prints the local batch job id and Vertex job resource name, and exits. Phase 1 exposes reusable Python helpers for status checks and imports, but it does not add separate check/import CLI commands or dashboard buttons yet.

## Dashboard Usage

`dashboard_server.py` serves a single embedded page and JSON endpoints.

Routes:

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Dashboard HTML. |
| `GET` | `/api/data?model=all&date=YYYY-MM-DD` | Dashboard data for the selected model/date. `model` defaults to `all`; `date` defaults to today. |
| `GET` | `/api/transcripts/<call_id>` | Plain-text transcript view for a completed call. |
| `GET` | `/api/transcripts/<call_id>/download` | TXT download for a completed call. Uses the saved transcript file when available, otherwise falls back to the transcript stored in PostgreSQL. |
| `POST` | `/api/reset` | Destructive reset. Calls `TRUNCATE TABLE calls RESTART IDENTITY`. |

Dashboard features verified in code:

- Auto-refresh every 10 seconds.
- Current clock.
- Date picker.
- Model filter buttons.
- Calls for selected day.
- Cost in USD and LKR.
- Token totals.
- Silence stripped summary.
- Audio processed summary.
- Sinhala/English/Tamil language breakdown.
- 14-day cost trend.
- All-time totals.
- Cost by model.
- Recent 20 calls for the selected date.
- View Transcript and Download TXT buttons for completed calls.
- Realtime vs batch badges from `calls.batch_mode`.

Warning: the Reset Data button permanently deletes all rows from the PostgreSQL `calls` table. It does not ask PostgreSQL for a backup and cannot be undone from this app.

Older completed calls may not have `transcript_file_path` populated. The dashboard transcript routes fall back to the `calls.transcript` text for those rows until the audio is reprocessed.

The dashboard does not currently provide Prepare Batch, Submit Batch, Check Status, Import Results, or batch job table controls.

## Batch Processing Workflow

`batch_processor.py` is the only batch-processing script in the current repo. No separate `prepare_gemini_batch.py`, `submit_gemini_batch.py`, `check_gemini_batch.py`, or `import_gemini_batch_results.py` scripts were found.

### Phase 1 durable batch backend

The backend now supports durable PostgreSQL tracking for Vertex AI Gemini batch jobs. Audio files can be staged in the Linux-friendly server-side folder configured by `BATCH_INPUT_DIR` (default `input_audio/batch`). `batch_processor.py` exposes reusable functions for prepare, submit, status check, and import while preserving the CLI entrypoint.

New batch rows are stored in `batch_jobs` and `batch_items`. Prepared jobs upload audio to GCS using unique per-item object names, create a JSONL request file under `BATCH_LOCAL_WORK_DIR`, upload that JSONL to GCS, and remember the Vertex job name after submit. Imported results save transcripts through the dated transcript helper and create normal `calls` rows with `batch_mode=True`.

Configure `BATCH_GCS_BUCKET`, `BATCH_MODEL`, `BATCH_LOCATION`, `BATCH_AUDIO_PREFIX`, `BATCH_JOBS_PREFIX`, `BATCH_OUTPUT_PREFIX`, and `BATCH_LOCAL_WORK_DIR` in `.env`. Pricing uses explicit batch rates for `gemini-2.5-flash` and `gemini-3-flash-preview`; it no longer relies on a generic percentage discount.

Dashboard UI controls are not added in Phase 1. The current dashboard still has no Prepare Batch, Submit Batch, Check Status, or Import Results buttons.

Actual batch behavior:

1. Loads `.env` through `config.py` and validates Google credentials/project only when real batch work starts.
2. Uses `BATCH_MODEL` and `BATCH_LOCATION`.
3. Uploads each provided local audio file to a job-specific GCS audio prefix using `batch_items.id` in the object name to avoid basename collisions.
4. Builds a local JSONL file under `BATCH_LOCAL_WORK_DIR`.
5. Uploads that JSONL to a job-specific GCS jobs prefix.
6. Persists job and item state in PostgreSQL, including unique GCS URIs and item ids for later matching.
7. Creates a Vertex AI `BatchPredictionJob` with:
   - `instances_format="jsonl"`
   - `predictions_format="jsonl"`
   - model resource `publishers/google/models/{BATCH_MODEL}`
   - output prefix `gs://<bucket>/<BATCH_OUTPUT_PREFIX>/<job_key>`
8. If waiting, polls every 60 seconds until a terminal status or timeout.
9. If succeeded, downloads `.jsonl` result files from the job output prefix.
10. Parses Gemini candidates, `usageMetadata`, and failed-row status/error details.
11. Writes transcripts to `outputs/transcriptions/YYYY.MM.DD/`.
12. Saves call rows with `batch_mode=True` and marks imported batch items.

Batch limitations in current code:

- The GCS bucket must already exist.
- Duration is measured with `ffprobe` using a timeout; rows fall back to `0.0` with an item warning if duration cannot be measured.
- There is no background auto-import scheduler.
- There are no dashboard batch buttons.

## Database Schema / Stored Metadata

`database.py` creates the PostgreSQL `calls` table:

```sql
CREATE TABLE IF NOT EXISTS calls (
    id                       SERIAL PRIMARY KEY,
    filename                 TEXT NOT NULL,
    audio_path               TEXT,
    duration_seconds         DOUBLE PRECISION DEFAULT 0,
    silence_removed_seconds  DOUBLE PRECISION DEFAULT 0,
    model                    TEXT,
    input_tokens             INTEGER DEFAULT 0,
    audio_tokens             INTEGER DEFAULT 0,
    text_input_tokens        INTEGER DEFAULT 0,
    output_tokens            INTEGER DEFAULT 0,
    thoughts_tokens          INTEGER DEFAULT 0,
    billed_output_tokens     INTEGER DEFAULT 0,
    total_tokens             INTEGER DEFAULT 0,
    audio_input_cost_usd     DOUBLE PRECISION DEFAULT 0,
    text_input_cost_usd      DOUBLE PRECISION DEFAULT 0,
    output_cost_usd          DOUBLE PRECISION DEFAULT 0,
    total_cost_usd           DOUBLE PRECISION DEFAULT 0,
    total_cost_lkr           DOUBLE PRECISION DEFAULT 0,
    lkr_rate                 DOUBLE PRECISION DEFAULT 316,
    languages_detected       TEXT DEFAULT '',
    transcript               TEXT DEFAULT '',
    transcript_file_path     TEXT DEFAULT '',
    transcript_saved_at      TEXT DEFAULT '',
    transcript_output_date   TEXT DEFAULT '',
    batch_mode               INTEGER DEFAULT 0,
    processed_at             TEXT NOT NULL
);
```

Index:

```sql
CREATE INDEX IF NOT EXISTS idx_calls_date ON calls (processed_at);
```

There are no migration files or Alembic setup. `database.py` creates and idempotently extends the `calls`, `batch_jobs`, and `batch_items` PostgreSQL tables on startup.

## Cost and Token Tracking

Realtime transcription uses usage metadata returned by Gemini:

- `prompt_token_count`
- `candidates_token_count`
- `thoughts_token_count`
- `total_token_count`

The code estimates audio input tokens from duration using `_AUDIO_TOKENS_PER_SECOND = 26`, then treats the remaining input tokens as text prompt tokens. Thinking tokens are included in billed output tokens.

Model pricing is defined in `_MODEL_PRICING` inside `gemini_flash_stt.py`. Unknown model names produce zero cost until pricing is added.

Realtime LKR conversion:

- `fetch_lkr_rate()` fetches USD to LKR from `https://open.er-api.com/v6/latest/USD`.
- The result is cached in-process for one hour.
- If the fetch fails, `LKR_RATE_FALLBACK = 316.0` is used.

Batch cost tracking:

- `batch_processor.py` uses explicit standard, priority, and batch rates for Gemini 2.5 Flash and Gemini 3 Flash Preview.
- Batch audio duration is measured with `ffprobe`; if duration cannot be measured, the item is kept with `duration_seconds = 0.0` and an error/warning message.
- Batch imports use the measured duration, Gemini output token metadata, and hardcoded `LKR_RATE = 305.0` when saving batch result cost.

## Input and Output Folders

Default folders:

| Path | Purpose |
|---|---|
| `input_audio/` | Default watcher input folder. |
| `input_audio/incoming/` | Recommended operational folder; the watcher creates it when passed via `--input`. |
| `outputs/transcriptions/YYYY.MM.DD/` | Generated transcript TXT files grouped by completion/import date. |
| `output/` | Legacy generated transcript folder from older versions; ignored by git. |
| `credentials/` | Local service-account JSON location. |

Supported watcher extensions:

```text
.mp3, .wav, .m4a, .ogg, .flac, .aac, .opus
```

Single-file CLI uses FFmpeg and may handle any file FFmpeg can decode, but the watcher only accepts the extensions listed above.

Transcript filenames:

- Realtime watcher: `outputs/transcriptions/YYYY.MM.DD/<audio-stem>__realtime__<model>__YYYYMMDD-HHMMSS.txt`
- Manual `--save`: `outputs/transcriptions/YYYY.MM.DD/<audio-stem>__manual__<model>__YYYYMMDD-HHMMSS.txt`
- Batch importer: `outputs/transcriptions/YYYY.MM.DD/<audio-stem>__batch__<model>__YYYYMMDD-HHMMSS.txt`

The date folder is based on transcription completion/import time in `APP_TIMEZONE`, not on dates embedded in the audio filename.

## Security / Files Not to Commit

Never commit:

- `.env`
- real service account JSON files
- `.venv/`
- generated transcripts in `outputs/` or legacy `output/`
- large or private audio files in `input_audio/`
- logs such as `watcher.log`
- local database artifacts such as `calls.db`

Use strong PostgreSQL passwords and restrict database network access.

Do not expose the Flask development server directly to the public internet. In production, bind to localhost or place it behind a properly secured reverse proxy/VPN with authentication.

The current `.gitignore` ignores `.env`, `credentials/*`, `outputs/`, legacy `output/`, and most `input_audio/*`, but it does not ignore every generated artifact already present in the repository, such as `calls.db` or `watcher.log`.

## Troubleshooting

### `psycopg2 is not installed`

Install dependencies from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

### `DATABASE_URL is not set`

Add PostgreSQL connection settings to `.env`:

```env
DATABASE_URL=postgresql://slt:your-password@127.0.0.1:5432/slt_calls
```

### `GOOGLE_APPLICATION_CREDENTIALS is not set`

Set it in `.env`:

```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
```

### `Credentials file not found`

Confirm the JSON file exists at the configured path. If the path is relative, the code resolves it relative to the repository root when possible.

### `GOOGLE_CLOUD_PROJECT is not set`

Set it in `.env`:

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
```

### FFmpeg or ffprobe errors

Install FFmpeg and verify both binaries:

```powershell
ffmpeg -version
ffprobe -version
```

### Dashboard shows no calls

The dashboard reads PostgreSQL. Start the watcher or run a manual transcription so rows are inserted into `calls`. Also verify `DATABASE_URL` points to the database you expect.

### Watcher starts processing many files immediately

This is expected if supported audio files already exist in the watched folder. Use a dedicated empty folder:

```powershell
python watcher.py --input .\input_audio\incoming
```

### Gemini model not found or wrong location

For realtime/manual transcription, check `MODEL_NAME` in `gemini_flash_stt.py` and `STT_GEMINI_LOCATION` in `.env`. For durable batch work, check `BATCH_MODEL` and `BATCH_LOCATION`. Some Gemini models may require `global` or project access approval.

## Known Issues / Limitations

- `README.md` previously referenced SQLite; current code uses PostgreSQL.
- `calls.db` exists as a legacy SQLite artifact and is not used by current code.
- There are no active test files and no `scripts/validate_runtime.py` in the current repository.
- Batch processing is CLI-only and not dashboard-controlled.
- No batch auto-import scheduler exists.
- Phase 1 durable batch helpers are callable from Python, but there are no separate check-status/import-results CLI commands yet.
- The dashboard JavaScript estimates silence savings using a hardcoded `258` tokens/sec, while realtime cost code uses `26` tokens/sec for standard uploaded audio.
- `dashboard_server.py` runs Flask's built-in server, not a production WSGI stack.
- Systemd service files use hardcoded Linux paths and user names.
- The repo has no Gunicorn, reverse-proxy, or logrotate configuration.

## Recommended Daily Operational Procedure

1. Confirm PostgreSQL is running before starting the watcher or dashboard.
2. Confirm `.env` points to the correct Google project and PostgreSQL database.
3. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Start the dashboard in one terminal:

```powershell
python dashboard_server.py --host 127.0.0.1 --port 5050
```

5. Start the watcher in another terminal with a clean incoming folder:

```powershell
python watcher.py --input .\input_audio\incoming
```

6. Drop new audio files into `input_audio\incoming`.
7. Monitor `http://localhost:5050`.
8. Review `watcher.log` if a file fails.
9. Archive or move processed source audio according to your operational policy. The current code does not move or delete processed input files.
10. Avoid using Reset Data unless you intend to permanently truncate the `calls` table.

## Development / Validation Commands

Syntax-check the Python files without calling Google APIs:

```powershell
python -m py_compile config.py transcript_storage.py gemini_flash_stt.py watcher.py database.py dashboard_server.py batch_processor.py examples\use_from_another_system.py
```

Inspect available CLI arguments:

```powershell
python gemini_flash_stt.py --help
python watcher.py --help
python dashboard_server.py --help
python batch_processor.py --help
```

Be aware that some scripts perform imports at startup. If optional dependencies like `psycopg2-binary` are missing, help commands for modules that import `database.py` may fail until dependencies are installed.

There is no pytest suite in the current repository.

## Maintainer Notes

- Prefer keeping runtime configuration in `.env` and secrets outside git.
- Consider adding migration files if the schema grows beyond the inline `CREATE TABLE` block.
- Consider removing or explicitly ignoring legacy/generated artifacts such as `calls.db` and `watcher.log`.
- Consider adding dashboard batch controls after the durable backend has enough operational mileage.
- Consider separating dashboard HTML into templates/static files if the UI grows.
- Treat `how_it_works.html` as explanatory documentation, not the runtime source of truth.
