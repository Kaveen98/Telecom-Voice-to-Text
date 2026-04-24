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
├── gemini_flash_stt.py       ← core transcription engine (edit MODEL_NAME here)
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
- `ffmpeg` installed and available on PATH
- A Google Cloud project with:
  - Billing enabled
  - Vertex AI API enabled
  - A service account key JSON file

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

Verify it works:
```bash
ffmpeg -version
```

---

## Step 5 — Set up Google Cloud credentials

Place your service account key file here:
```
credentials/google-credentials.json
```

Create a `.env` file in the project root:
```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
STT_GEMINI_LOCATION=us-central1
```

---

## Step 6 — Choose your Gemini model

Open `gemini_flash_stt.py` and change line 27:

```python
MODEL_NAME = "gemini-2.5-flash"   # ← change this
```

Available models (verified from GCP billing — April 2026):

| Model | Audio Input | Text Input | Output | Notes |
|---|---|---|---|---|
| `gemini-2.5-flash` | $1.00/1M | $0.30/1M | $2.50/1M | Recommended for call centers |
| `gemini-2.5-pro` | $1.25/1M | $1.25/1M | $10.00/1M | Higher quality |
| `gemini-3-flash-preview` | $1.00/1M | $0.50/1M | $3.00/1M | Preview — needs global endpoint |
| `gemini-3.0-pro` | $2.00/1M | $2.00/1M | $12.00/1M | Preview |

> **Note:** Gemini 3.x models require `STT_GEMINI_LOCATION=global` in `.env` and may need preview access approval from Google.

Also update the LKR exchange rate if needed:
```python
LKR_RATE = 305.0   # ← update when rate changes
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
- Reset button to clear all records

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

## Recommended `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
.env
credentials/
google-credentials.json
calls.db
calls.db-shm
calls.db-wal
watcher.log
output/
input_audio/*
!input_audio/.gitkeep
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
Change `STT_GEMINI_LOCATION=global` in `.env`. Gemini 3.x is preview only and requires the global endpoint. Your project may also need preview access approval from Google.

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
- [ ] `ffmpeg -version` works
- [ ] `credentials/google-credentials.json` exists
- [ ] `.env` file created with correct project ID and credentials path
- [ ] `MODEL_NAME` set to your chosen model in `gemini_flash_stt.py`
- [ ] Audio file in `input_audio/incoming/` to test with
