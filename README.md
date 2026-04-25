# SLT Call Center — Gemini Transcription Pipeline

Automated Sinhala / English / Tamil call transcription system built for **SLT Telecom** using **Google Gemini on Vertex AI**.

Drops audio into `input_audio/incoming/` → transcribes automatically → shows real-time cost in LKR on a web dashboard.

---

## What this project does

- Watches a folder and auto-transcribes any audio file dropped into it
- Detects languages — Sinhala `[SI]`, English `[EN]`, Tamil `[TA]`
- Strips silence from audio before sending to reduce token cost
- Calculates exact cost in **USD and LKR** using real token counts from Google
- Logs every call (tokens, cost, languages, duration) to a SQLite database
- Shows a live web dashboard at `http://localhost:5050`
- Supports switching Gemini models — pricing updates automatically
- Supports Vertex AI Batch Prediction for 50% cost reduction on bulk jobs

---

## Project structure

```
Telecom-Voice-to-Text/
│
├── gemini_flash_stt.py       ← core transcription engine
├── config.py                 ← env-driven runtime configuration
├── watcher.py                ← auto-transcription folder watcher
├── database.py               ← SQLite call logger
├── dashboard_server.py       ← web dashboard (Flask)
├── batch_processor.py        ← Vertex AI Batch Prediction (bulk/overnight)
├── how_it_works.html         ← visual diagram of the system
│
├── requirements.txt
├── .env                      ← your credentials (never commit this)
├── .gitignore
├── README.md
├── deploy/                   ← systemd + logrotate examples
├── scripts/
│   └── validate_runtime.py   ← Linux/runtime smoke checks
│
├── credentials/
│   └── google-credentials.json   ← GCP service account key (never commit)
│
├── input_audio/incoming/     ← drop audio files here
├── output/                   ← transcripts saved here automatically
└── examples/
    └── use_from_another_system.py
```

---

## Prerequisites

- Python 3.10 or higher
- `ffmpeg` and `ffprobe` installed and available on PATH
- A Google Cloud project with:
  - Billing enabled
  - Vertex AI API enabled
  - Authentication through Application Default Credentials, an attached VM service account, or a `GOOGLE_APPLICATION_CREDENTIALS` JSON key

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git
cd Telecom-Voice-to-Text
```

---

## Step 2 — Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Used by |
|---|---|
| `google-genai` | gemini_flash_stt.py — Vertex AI API |
| `watchdog` | watcher.py — folder monitoring |
| `flask` | dashboard_server.py — web dashboard |
| `gunicorn` | production dashboard WSGI server |
| `google-cloud-storage` | batch_processor.py — GCS uploads |
| `google-cloud-aiplatform` | batch_processor.py — batch jobs |

---

## Step 4 — Install ffmpeg

ffmpeg converts audio to the format Gemini expects and strips silence to reduce token cost.

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html, extract, add the `bin` folder to PATH.

Verify both tools work:
```bash
ffmpeg -version
ffprobe -version
```

---

## Step 5 — Set up Google Cloud credentials

For local development, you can place a service account key file here:
```
credentials/google-credentials.json
```

This JSON key workflow is optional. On a Linux VM, prefer Application Default
Credentials from the attached service account and leave
`GOOGLE_APPLICATION_CREDENTIALS` unset.

Create a `.env` file in the project root:
```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
STT_MODEL_NAME=gemini-2.5-flash
STT_LKR_RATE=316.0
STT_ENABLE_RESET=false
# GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
```

---

## Step 6 — Choose your Gemini model

Set the model in `.env`:

```env
STT_MODEL_NAME=gemini-2.5-flash
```

Available models (verified from GCP billing — April 2026):

| Model | Audio Input | Text Input | Output | Notes |
|---|---|---|---|---|
| `gemini-2.5-flash` | $1.00/1M | $0.30/1M | $2.50/1M | Recommended for call centers |
| `gemini-2.5-pro` | $1.25/1M | $1.25/1M | $10.00/1M | Higher quality |
| `gemini-3-flash-preview` | $1.00/1M | $0.50/1M | $3.00/1M | Preview — needs global endpoint |
| `gemini-3.0-pro` | $2.00/1M | $2.00/1M | $12.00/1M | Preview |

> **Note:** Gemini 3.x models require `GOOGLE_CLOUD_LOCATION=global` in `.env` and may need preview access approval from Google.

Also update the LKR exchange rate if needed:
```env
STT_LKR_RATE=316.0
```

---

## Running the system

### Option A — Manual (single file)

```bash
python gemini_flash_stt.py input_audio/call.mp3
python gemini_flash_stt.py input_audio/call.mp3 --save
```

### Option B — Auto-transcription pipeline (recommended)

**Terminal 1 — Start the watcher:**
```bash
python watcher.py
```
Drop any `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, or `.opus` file into `input_audio/incoming/`. It transcribes automatically. Transcripts are saved to `output/` with the model name in the filename so nothing is ever overwritten.

**Terminal 2 — Start the dashboard:**
```bash
python dashboard_server.py
```
Open `http://localhost:5050` in your browser. Updates every 10 seconds. Shows calls, cost in LKR, language breakdown, model comparison, and per-call detail.

### Option C — Batch processing (50% cheaper, overnight)

Use this for bulk historical recordings. Requires a GCS bucket.

```bash
python batch_processor.py call1.mp3 call2.mp3 call3.mp3 --bucket your-gcs-bucket
```

Results are written back from GCS and saved to the database automatically.

---

## Dashboard features

- Calls transcribed today
- Cost today in USD and LKR
- Tokens used (audio vs output)
- Silence stripped (time saved + LKR saved)
- Language breakdown — Sinhala / English / Tamil
- Per-model cost comparison tabs
- 14-day cost trend
- Last 20 calls with full detail
- Optional reset button to clear all records (`STT_ENABLE_RESET=true`)

---

## How pricing works

Token counts come directly from Google's API response (`usage_metadata`). The code multiplies them by the rates from your GCP billing account:

```
Audio input tokens  × rate  = audio cost
Output tokens       × rate  = output cost  (includes thinking tokens)
Total cost (USD)    × 305   = cost in LKR
```

Thinking tokens are billed by Google as output tokens and are included in the cost calculation.

To verify prices match your actual GCP bill: download your billing CSV from GCP Console → Billing → Pricing, and compare the SKU rates against `_MODEL_PRICING` in `gemini_flash_stt.py`.

---

## Cost and silence metadata

For backward compatibility, `duration_seconds` means the submitted audio length
after preprocessing. New records also store:

- `original_duration_seconds`: audio length before preprocessing
- `submitted_duration_seconds`: audio length sent to Gemini
- `silence_removed_seconds`: estimated seconds removed by preprocessing
- `silence_removed_ratio`: removed seconds divided by original duration
- provider token metadata when returned by Gemini

Estimated cost is calculated from stored token usage, model pricing, and the
configured LKR rate. It is useful for operations and trend analysis, but it is
not the final cloud invoice.

---

## Silence trimming experiment

Use the Phase 3B experiment script to compare the current FFmpeg silence trim
against a no-trim baseline before changing production defaults.

```bash
python scripts/run_silence_experiment.py --input input_audio/samples --limit 25
```

The script runs each selected audio file twice by default:

- `no_trim`: sends the audio without silence stripping
- `current_trim`: uses the current configured FFmpeg `silenceremove` behavior

It writes:

- CSV results to `experiments/silence_experiment_results.csv`
- transcripts for each run to `experiments/transcripts/`

This experiment makes real Gemini provider calls and may incur cost. Start with
10-25 representative calls, then compare `no_trim` vs `current_trim` cost,
token counts, silence removed, and transcript quality before changing defaults.
Silence trimming can cut low-volume Sinhala/Tamil speech, so manually review the
transcripts and fill the manual review columns in the CSV.

Preview the planned runs without provider calls:

```bash
python scripts/run_silence_experiment.py --input input_audio/samples --limit 10 --dry-run
```

---

## Recommended `.gitignore`

```gitignore
.venv/
__pycache__/
_cleanup_backup/
*.pyc
.env
*.log
credentials/
google-credentials.json
calls.db
calls.db-shm
calls.db-wal
watcher.log
output/
tmp/
input_audio/*
!input_audio/.gitkeep
```

---

## Linux service deployment

Recommended first production deployment: Linux VM + systemd + Gunicorn. This
matches the current durable local-folder workflow and keeps SQLite, audio files,
and transcripts on the host filesystem.

### 1. Create a service user

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin slttranscribe
```

### 2. Install OS packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg sqlite3
ffmpeg -version
ffprobe -version
```

### 3. Install the app under `/opt`

```bash
sudo git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git /opt/slt-transcription
sudo chown -R slttranscribe:slttranscribe /opt/slt-transcription
cd /opt/slt-transcription
sudo -u slttranscribe python3 -m venv .venv
sudo -u slttranscribe .venv/bin/python -m pip install --upgrade pip
sudo -u slttranscribe .venv/bin/pip install -r requirements.txt
```

### 4. Create runtime folders

```bash
sudo mkdir -p /var/lib/slt-transcription/input_audio
sudo mkdir -p /var/lib/slt-transcription/output
sudo mkdir -p /var/log/slt-transcription
sudo mkdir -p /etc/slt-transcription
sudo chown -R slttranscribe:slttranscribe /var/lib/slt-transcription /var/log/slt-transcription
sudo chown root:slttranscribe /etc/slt-transcription
sudo chmod 750 /var/lib/slt-transcription /var/log/slt-transcription /etc/slt-transcription
```

The watcher will create these lifecycle folders under `STT_INPUT_DIR`:

```text
incoming/
processing/
completed/
failed/
archive/
```

Drop new audio into:

```text
/var/lib/slt-transcription/input_audio/incoming/
```

### 5. Create `/etc/slt-transcription/stt.env`

For a VM with an attached Google service account, omit
`GOOGLE_APPLICATION_CREDENTIALS` and rely on Application Default Credentials.

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

STT_MODEL_NAME=gemini-2.5-flash
STT_LKR_RATE=316.0
STT_DATA_DIR=/var/lib/slt-transcription
STT_INPUT_DIR=/var/lib/slt-transcription/input_audio
STT_OUTPUT_DIR=/var/lib/slt-transcription/output
STT_DB_PATH=/var/lib/slt-transcription/calls.db
STT_LOG_DIR=/var/log/slt-transcription
STT_LOG_FILE=/var/log/slt-transcription/watcher.log

STT_DASHBOARD_HOST=127.0.0.1
STT_DASHBOARD_PORT=5050
STT_ENABLE_RESET=false
```

Secure the environment file so systemd can read it through the service group:

```bash
sudo chown root:slttranscribe /etc/slt-transcription/stt.env
sudo chmod 640 /etc/slt-transcription/stt.env
```

If using a JSON key locally or on a restricted host:

```bash
sudo install -o root -g slttranscribe -m 640 google-credentials.json /etc/slt-transcription/google-credentials.json
```

Then add this to `stt.env`:

```env
GOOGLE_APPLICATION_CREDENTIALS=/etc/slt-transcription/google-credentials.json
```

### 6. Validate runtime setup

```bash
cd /opt/slt-transcription
sudo -u slttranscribe bash -lc \
  'set -a; source /etc/slt-transcription/stt.env; set +a; cd /opt/slt-transcription; .venv/bin/python scripts/validate_runtime.py'
```

The validation script checks Python, imports, `ffmpeg`, `ffprobe`, writable
runtime folders, SQLite initialization, and Google project/location visibility.
It does not submit audio to Gemini.

### 7. Install systemd services

Review and edit the placeholders in:

```text
deploy/systemd/slt-watcher.service
deploy/systemd/slt-dashboard.service
```

Then install:

```bash
sudo cp deploy/systemd/slt-watcher.service /etc/systemd/system/
sudo cp deploy/systemd/slt-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now slt-watcher
sudo systemctl enable --now slt-dashboard
```

Check status and logs:

```bash
systemctl status slt-watcher
systemctl status slt-dashboard
journalctl -u slt-watcher -n 100 --no-pager
journalctl -u slt-dashboard -n 100 --no-pager
tail -f /var/log/slt-transcription/watcher.log
```

The dashboard service runs Gunicorn with `dashboard_server:app`. If you expose it
outside the VM, put Nginx or another reverse proxy in front of `127.0.0.1:5050`.

### 8. Optional log rotation

```bash
sudo cp deploy/logrotate/slt-transcription /etc/logrotate.d/slt-transcription
sudo logrotate -d /etc/logrotate.d/slt-transcription
```

### 9. SQLite backup

Use SQLite's online backup command so WAL data is handled safely:

```bash
sudo -u slttranscribe sqlite3 /var/lib/slt-transcription/calls.db \
  ".backup '/var/lib/slt-transcription/backups/calls-$(date +%F).db'"
```

Create the backup directory first:

```bash
sudo mkdir -p /var/lib/slt-transcription/backups
sudo chown slttranscribe:slttranscribe /var/lib/slt-transcription/backups
```

Verify database integrity:

```bash
sudo -u slttranscribe sqlite3 /var/lib/slt-transcription/calls.db "PRAGMA integrity_check;"
```

---

## Common problems

**`watchdog` or `flask` not found**
```bash
pip install -r requirements.txt
```

**`ffmpeg is not installed or not available on PATH`**
Install ffmpeg and verify with `ffmpeg -version`.

**`404 NOT_FOUND` for Gemini 3.x models**
Change `GOOGLE_CLOUD_LOCATION=global` in `.env`. Gemini 3.x is preview only and requires the global endpoint. Your project may also need preview access approval from Google.

**`GOOGLE_CLOUD_PROJECT is not set`**
Check your `.env` file has `GOOGLE_CLOUD_PROJECT=your-project-id`.

**`Credentials file not found`**
Confirm `credentials/google-credentials.json` exists and `.env` points to it correctly.

**Dashboard shows nothing after reset**
Run `watcher.py` first so calls get logged to the database, then open the dashboard.

---

## First-time setup checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` completed
- [ ] `ffmpeg -version` and `ffprobe -version` work
- [ ] Google auth is available through ADC/attached service account or an optional JSON key
- [ ] `.env` file created with correct project ID and location
- [ ] `STT_MODEL_NAME` set to your chosen model in `.env`
- [ ] Audio file in `input_audio/incoming/` to test with
