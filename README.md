# Telecom Voice-to-Text — Windows Realtime Transcription

Telecom Voice-to-Text watches a folder for telecom call audio clips, sends each
audio file to Gemini on Vertex AI for transcription, and saves the results as
dated transcript files.

The primary outputs are always files:

- TXT transcript files under `transcriptions/YYYY-MM-DD/`
- JSON metadata files beside each TXT transcript

MySQL is optional. When enabled, it stores searchable metadata for dashboards,
history, and reporting. If MySQL is disabled, unavailable, or misconfigured, the
watcher is still designed to save TXT/JSON outputs.

The optional Flask dashboard runs locally at `http://127.0.0.1:5050`. It uses
dashboard login sessions, local users from `users.json`, and MySQL metadata for
operational cost, language, token, audio, and transcript views.

## 1. Project Overview

This project is for Windows Server or local Windows realtime transcription.

It is intended for a simple operations workflow:

1. A user or another system copies audio files into `input_audio/incoming/`.
2. `watcher.py` detects stable audio files.
3. Gemini on Vertex AI transcribes the audio.
4. TXT and JSON output files are saved first.
5. MySQL metadata is saved afterward if MySQL is enabled.
6. The original audio file is moved to completed or failed storage.

The main runtime command is:

```powershell
python watcher.py
```

## 2. Current Supported Workflow

Active in this branch:

- Windows Server or local Windows deployment
- Realtime folder watcher through `watcher.py`
- File-based TXT transcript outputs
- File-based JSON metadata outputs
- Optional MySQL metadata storage
- Optional local Flask dashboard through `dashboard_server.py`
- Dashboard login/authentication backed by `users.json`
- Dashboard logo from `static/app-logo.png` on the login page and header
- Dashboard user management through `manage_users.py`
- Dashboard metrics for calls, costs, tokens, silence stripped, audio processed, language breakdown, model costs, cost history, recent calls, and transcript view/download
- CSV downloads for daily and monthly cost history

Not active in this branch:

- Vertex AI batch processing
- Linux systemd deployment
- Dashboard buttons for starting batch jobs
- SQLite or PostgreSQL deployment

Archived batch, Linux, and legacy files are preserved under `docs/archive/`.

## 3. How The Runtime Workflow Works

```text
input_audio/incoming/
  -> input_audio/processing/
  -> Gemini transcription on Vertex AI
  -> transcriptions/YYYY-MM-DD/*.txt and *.json
  -> optional MySQL metadata
  -> input_audio/completed/YYYY-MM-DD/
  or input_audio/failed/YYYY-MM-DD/
```

Folder meaning:

- `input_audio/incoming/`: place new audio files here.
- `input_audio/processing/`: watcher moves stable files here before processing so they are not picked up twice.
- `transcriptions/`: main TXT transcript and JSON metadata output folder.
- `input_audio/completed/YYYY-MM-DD/`: successfully processed original audio files.
- `input_audio/failed/YYYY-MM-DD/`: failed audio files and `.error.txt` failure notes.
- `logs/`: watcher log files such as `watcher_YYYY-MM-DD.log`.

TXT/JSON outputs are saved before MySQL metadata. A MySQL failure should not
cause a second Gemini call.

## Detailed Transcription Workflow Explained

This section explains what happens from the moment a user adds an audio file
until the transcript is saved.

```text
User drops audio
  ↓
watcher.py detects stable file
  ↓
Move to processing
  ↓
Gemini/Vertex transcription
  ↓
Save TXT transcript
  ↓
Save JSON metadata
  ↓
Try optional MySQL metadata
  ↓
Move audio to completed
  ↓
Optional dashboard reads MySQL metadata
```

### 1. User Adds Audio File

The user copies an audio file into:

```text
input_audio/incoming/
```

Supported formats:

```text
.mp3, .wav, .m4a, .flac, .ogg, .aac, .opus
```

`incoming` is the only folder the user normally needs to touch. The other
runtime folders are managed by the watcher.

### 2. Watcher Detects The File

`watcher.py` continuously monitors `input_audio/incoming/`.

It detects supported audio files and ignores unsupported files. Unsupported
files are not sent to Gemini and are not moved through the transcription flow.

### 3. Watcher Waits Until File Copy Is Complete

Before processing starts, the watcher checks that the file is stable.

Stable means the file size and modification time have stopped changing for a
short period. This matters because large files or files copied across a network
may appear in `incoming` before the copy has fully finished.

### 4. File Moves To Processing

After the file is stable, it moves from:

```text
input_audio/incoming/
```

to:

```text
input_audio/processing/
```

This prevents duplicate pickup and makes it clear that the file is currently
being processed.

### 5. Gemini Transcription Starts

`watcher.py` calls `gemini_flash_stt.py`.

`gemini_flash_stt.py` prepares the audio and sends it to Gemini on Vertex AI.
This step requires:

- internet access
- Google Cloud credentials
- Vertex AI API access
- a Google Cloud project that is allowed to use the selected Gemini model

This step may incur Google Cloud cost.

### 6. Transcript Result Is Returned

Gemini returns the transcript and related metadata, such as:

- model name
- detected language information
- audio duration
- token and cost estimates when available

For privacy, the transcript body is not printed by default. The transcript text
is saved to a TXT file.

### 7. TXT Transcript Is Saved First

The TXT transcript is saved under:

```text
transcriptions/YYYY-MM-DD/
```

This TXT file is the main output most users need.

Filename format:

```text
original-file-name__transcribed_YYYY-MM-DD_HH-MM-SS.txt
```

### 8. JSON Metadata Is Saved Beside TXT

A JSON metadata file is saved beside the TXT file.

The JSON file stores useful processing information such as:

- original filename
- transcript path
- metadata path
- model
- language information
- duration
- token and cost fields
- status

The TXT file contains the transcript body. The JSON file is for metadata.

### 9. Optional MySQL Metadata Save

After TXT and JSON files are saved, `watcher.py` attempts to save metadata to
MySQL.

MySQL is secondary. It is useful for:

- dashboard metrics
- searching
- reporting
- history and reconciliation

Important behavior:

- If MySQL works, a metadata row is saved.
- If MySQL fails, TXT/JSON files remain saved.
- If MySQL fails, the audio can still move to completed.
- A MySQL failure must not trigger another Gemini call.

### 10. Audio Moves To Completed

If transcription and TXT/JSON saving succeed, the original audio moves to:

```text
input_audio/completed/YYYY-MM-DD/
```

Filename format:

```text
original-file-name__transcribed_YYYY-MM-DD_HH-MM-SS.ext
```

### 11. Failed Processing

If Gemini transcription fails or TXT/JSON saving fails, the audio moves to:

```text
input_audio/failed/YYYY-MM-DD/
```

A matching `.error.txt` file is saved beside the failed audio. The error file
contains a short safe failure message.

One failed file does not stop the watcher. The watcher keeps running for future
files.

### 12. Dashboard Reads MySQL Metadata

The dashboard is optional. It reads metadata from MySQL and shows authenticated
operational views such as calls today, cost today in USD/LKR, tokens, silence
stripped, audio processed, language breakdown, cost history, recent calls, and
transcript links.

The dashboard does not control the main transcription process. `watcher.py` is
the main runtime.

If MySQL is disabled or unavailable, the dashboard may be empty even though TXT
and JSON transcript files exist.

Dashboard access is protected by login. Users are loaded from `users.json`, and
passwords are checked against stored password hashes.

### Success And Failure Summary

| Scenario | Result |
| --- | --- |
| Gemini succeeds + TXT/JSON saves + MySQL works | Completed, files saved, DB row saved |
| Gemini succeeds + TXT/JSON saves + MySQL fails | Completed, files saved, DB warning logged |
| Gemini fails | Audio moved to failed, error file created |
| TXT/JSON saving fails | Audio moved to failed, error file created |
| Unsupported file | Ignored |
| File still copying | Waits until stable |

## 4. Final Folder Structure

```text
Telecom-Voice-to-Text/
  watcher.py                    # Main realtime runtime
  gemini_flash_stt.py            # Gemini transcription and TXT/JSON output saving
  database.py                    # Optional MySQL metadata storage
  config.py                      # Shared environment and path configuration
  dashboard_server.py            # Optional local dashboard
  manage_users.py                # Dashboard user management
  transcript_storage.py          # Compatibility wrapper
  requirements.txt
  .env.example
  users.json                     # Local dashboard user store

  static/
    app-logo.png                 # Dashboard login/header logo

  input_audio/
    incoming/
    processing/
    completed/
    failed/

  transcriptions/
  logs/
  credentials/

  docs/
    archive/
      batch/
      linux/
      legacy/
```

Real audio, transcripts, logs, `.env`, `.dashboard_secret`, credentials, local
databases, and archive files such as `.zip`, `.rar`, `.pem`, and `.key` are
ignored by Git. Placeholder `.gitkeep` files keep required empty folders
present in the repository. Treat production `users.json` content as
deployment-specific sensitive local state.

## 5. Requirements

Use these requirements for the active Windows realtime deployment:

- Windows 10, Windows 11, or Windows Server
- Python 3.10 or newer
- FFmpeg and ffprobe on Windows PATH
- Google Cloud project with Vertex AI access
- Gemini/Vertex service-account credentials
- MySQL Server if `DB_ENABLED=true`
- Internet connection

Why each requirement matters:

- Python runs the application.
- FFmpeg decodes and converts audio before it is sent to Gemini.
- ffprobe helps inspect audio metadata and confirms the FFmpeg tools are installed correctly.
- Google Cloud Vertex AI provides Gemini transcription.
- MySQL stores searchable metadata for dashboard, search, and reporting, but it is not the primary transcript output.
- Internet access is required because Gemini/Vertex is cloud-based.

## 6. Setup Step 1 - Clone And Enter Project

If you are cloning the project for the first time:

```powershell
git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git
cd Telecom-Voice-to-Text
```

If you need this feature branch before it is merged:

```powershell
git switch feature/add-dashboard-logo
```

If the repository is already cloned on this machine:

```powershell
cd E:\Work\Telecom-Voice-to-Text
```

This puts PowerShell in the project folder so the remaining commands can find
the application files.

## 7. Setup Step 2 - Create And Activate Python Virtual Environment

Run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for your user account, then try
activation again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Why this step is needed:

- The virtual environment isolates this project's Python packages.
- `requirements.txt` installs the packages used by the realtime deployment.
- `google-genai` calls Gemini/Vertex.
- `watchdog` watches the incoming folder for new files.
- `mysql-connector-python` connects to MySQL.
- `python-dotenv` helps load local `.env` configuration.
- `pydub` helps with audio handling and works with FFmpeg.
- `flask` is used only for the optional dashboard.

## 8. Setup Step 3 - Install FFmpeg

FFmpeg must be installed separately from Python.

Windows installation steps:

1. Download a Windows FFmpeg build from the official FFmpeg website or a trusted Windows build provider.
2. Extract the downloaded archive to a permanent folder, for example `C:\ffmpeg`.
3. Add the FFmpeg `bin` folder to the Windows PATH, for example `C:\ffmpeg\bin`.
4. Close and reopen PowerShell.
5. Verify both tools:

```powershell
ffmpeg -version
ffprobe -version
```

Why this step is needed:

- `ffmpeg` decodes and converts audio formats such as MP3, M4A, WAV, FLAC, OGG, AAC, and OPUS.
- `ffprobe` helps inspect audio metadata.
- If FFmpeg or ffprobe is missing, audio preprocessing can fail before Gemini is called.

## 9. Setup Step 4 - Google Cloud / Gemini / Vertex AI Credentials

This repository uses Vertex AI service-account JSON authentication through:

```text
GOOGLE_APPLICATION_CREDENTIALS
GOOGLE_CLOUD_PROJECT
STT_GEMINI_LOCATION
```

Gemini API-key authentication is a different deployment path and is not the
main path used by this branch.

### A. Create Or Select A Google Cloud Project

1. Open Google Cloud Console.
2. Create a new project or select an existing project.
3. Record the project ID, not only the project name.
4. Make sure billing is enabled if Vertex AI usage requires it.

Use a placeholder like this in documentation and examples:

```text
your-gcp-project-id
```

### B. Enable The Required API

Enable the Vertex AI API for the selected Google Cloud project.

Gemini transcription through Vertex AI will fail if this API is not enabled.

### C. Create A Service Account

1. Go to IAM & Admin -> Service Accounts.
2. Create a service account, for example `telecom-voice-to-text`.
3. Grant the minimum role needed for Vertex AI usage, such as Vertex AI User, or follow your organization's least-privilege IAM policy.
4. Do not use Owner or Administrator roles for production unless a temporary test explicitly requires it.

Use a dedicated service account for this application. Do not use a personal
admin account for production.

### D. Create A JSON Key

1. Open the service account.
2. Go to Keys.
3. Choose Add key -> Create new key.
4. Select JSON.
5. Download the JSON file.

Treat this JSON file like a password.

### E. Place The Credentials File

Create or use this path:

```text
Telecom-Voice-to-Text/
  credentials/
    google-credentials.json
```

The file name can be different, but the path in `.env` must match it.

### F. Configure `.env`

Copy the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set safe local values:

```text
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
STT_GEMINI_LOCATION=us-central1
```

What these values mean:

- `GOOGLE_APPLICATION_CREDENTIALS` tells Google libraries where the service account JSON is.
- `GOOGLE_CLOUD_PROJECT` tells the app which Google Cloud project to use and bill.
- `STT_GEMINI_LOCATION` tells Vertex AI which region to use.

Security notes:

- Never commit `.env`.
- Never commit `google-credentials.json`.
- Treat the JSON key as a secret.
- If the JSON key is exposed, revoke or delete the key and create a new one.
- Use a dedicated service account.
- Follow least-privilege IAM.

## 10. Setup Step 5 - MySQL Setup

MySQL is optional metadata and index storage.

The watcher should still save TXT and JSON transcript outputs when MySQL is
disabled, unavailable, or misconfigured. Use MySQL when you need dashboard
metrics, searchable history, reporting, or reconciliation.

### Starting MySQL On Windows

MySQL must be running before metadata can be saved. If MySQL is stopped, the
watcher should still save TXT/JSON transcript files, but MySQL dashboard/history
data will not update.

MySQL may be installed through MySQL Installer, WAMP, XAMPP, Laragon, Docker, or
another package. The Windows service name and `mysql.exe` path can differ by
installation.

Find installed MySQL-related services:

```powershell
Get-Service *mysql*
```

Common service names may include `MySQL80`, `MySQL`, `wampmysqld64`, or similar.

Start the service you found:

```powershell
Start-Service <service-name>
Get-Service <service-name>
```

Example for MySQL Installer:

```powershell
Start-Service MySQL80
Get-Service MySQL80
```

Example for WAMP:

```powershell
Start-Service wampmysqld64
Get-Service wampmysqld64
```

To make MySQL start automatically when Windows starts, run PowerShell as
Administrator and use:

```powershell
Set-Service <service-name> -StartupType Automatic
```

Example for MySQL Installer:

```powershell
Set-Service MySQL80 -StartupType Automatic
```

Example for WAMP:

```powershell
Set-Service wampmysqld64 -StartupType Automatic
```

### Open The MySQL Command Line

Use `root` or another MySQL administrator account only for setup.
Do not use `root` in the application `.env`.

If `mysql` is available on PATH:

```powershell
mysql -u root -p
```

If `mysql` is not recognized, use the full path to `mysql.exe`.

Generic MySQL Installer example:

```powershell
& "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p
```

WAMP example:

```powershell
& "C:\wamp64\bin\mysql\mysql8.2.0\bin\mysql.exe" -u root -p
```

Your exact path may differ, especially with WAMP, XAMPP, Laragon, or Docker.

### Create A Database And Dedicated User

Log in to MySQL as an administrator, then run:

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

Important:

- Do not use the MySQL `root` user in `.env`.
- Use a dedicated MySQL user for the app.
- Replace `change_this_password` with a strong password.
- Grant both `localhost` and `127.0.0.1` because Windows/MySQL clients may resolve localhost differently.
- Do not grant `DELETE` unless future maintenance workflows explicitly require it.
- The app creates or updates its metadata table when database storage is used.
- Back up MySQL regularly in production.
- Do not print or share `MYSQL_PASSWORD`.

### Enable MySQL In `.env`

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

### Test The App MySQL User

After creating the user, test that the app user can log in.

General command when `mysql` is on PATH:

```powershell
mysql -u telecom_app -p -h 127.0.0.1 -P 3306 telecom_voice_to_text -e "SELECT DATABASE(), CURRENT_USER();"
```

Full-path alternative:

```powershell
& "C:\Path\To\mysql.exe" -u telecom_app -p -h 127.0.0.1 -P 3306 telecom_voice_to_text -e "SELECT DATABASE(), CURRENT_USER();"
```

WAMP example:

```powershell
& "C:\wamp64\bin\mysql\mysql8.2.0\bin\mysql.exe" -u telecom_app -p -h 127.0.0.1 -P 3306 telecom_voice_to_text -e "SELECT DATABASE(), CURRENT_USER();"
```

Enter the app user's password when prompted. Do not print or share the password.

### Validate Database Initialization

Run this after `.env` is configured and MySQL is running:

```powershell
python -c "import database; print('enabled:', database.is_database_enabled()); database.init_database(); print('database init ok')"
```

This command:

- Checks whether the app can read `.env`.
- Verifies MySQL metadata storage is enabled.
- Creates the `transcriptions` metadata table if needed.
- Does not call Gemini.
- Does not process audio.

### Verify The Metadata Table

General command when `mysql` is on PATH:

```powershell
mysql -u telecom_app -p -h 127.0.0.1 -P 3306 telecom_voice_to_text -e "SHOW TABLES;"
```

Full-path alternative:

```powershell
& "C:\Path\To\mysql.exe" -u telecom_app -p -h 127.0.0.1 -P 3306 telecom_voice_to_text -e "SHOW TABLES;"
```

Expected table:

```text
transcriptions
```

### Verify Recent Metadata Rows

Run this after processing an approved test audio file:

```powershell
mysql -u telecom_app -p -h 127.0.0.1 -P 3306 telecom_voice_to_text -e "SELECT id, original_file_name, status, transcript_txt_path, metadata_json_path, transcribed_at FROM transcriptions ORDER BY id DESC LIMIT 5;"
```

Expected result:

- Recent rows should appear after successful processing.
- `status` should usually be `completed`.
- `transcript_txt_path` and `metadata_json_path` should point to files under `transcriptions/YYYY-MM-DD/`.

### File-Only Mode

If you do not want MySQL, use:

```text
DB_ENABLED=false
```

In file-only mode:

- TXT transcripts are still saved.
- JSON metadata files are still saved.
- Dashboard metrics may be empty because there is no MySQL metadata.
- MySQL connection errors do not block transcript file output.

## 11. Setup Step 6 - Configure Folders

These folder settings are in `.env.example`:

```text
INPUT_INCOMING_DIR=input_audio/incoming
INPUT_PROCESSING_DIR=input_audio/processing
INPUT_COMPLETED_DIR=input_audio/completed
INPUT_FAILED_DIR=input_audio/failed
TRANSCRIPTIONS_DIR=transcriptions
LOG_DIR=logs
```

Folder purpose:

- `INPUT_INCOMING_DIR`: users or systems place new audio here.
- `INPUT_PROCESSING_DIR`: watcher moves stable files here while working.
- `INPUT_COMPLETED_DIR`: successfully processed original audio is archived by date.
- `INPUT_FAILED_DIR`: failed audio is archived by date with an `.error.txt` file.
- `TRANSCRIPTIONS_DIR`: primary TXT/JSON transcript output folder.
- `LOG_DIR`: daily watcher logs.

The watcher creates these folders if they are missing.

## 12. Setup Step 7 - Create Dashboard Users

The dashboard requires a login. Users are stored in the project-local
`users.json` file and are managed by `manage_users.py`.

Passwords are stored as secure Werkzeug password hashes in `users.json`, not as
plain text. The dashboard checks those hashes when a user signs in.

Create the first dashboard user:

```powershell
python manage_users.py add admin
```

The command prompts for:

- role: `admin` or `viewer`, defaulting to `viewer`
- password and confirmation

Passwords must be at least 6 characters.

Add another dashboard user:

```powershell
python manage_users.py add supervisor
```

List configured dashboard users:

```powershell
python manage_users.py list
```

Change or reset a dashboard user's password:

```powershell
python manage_users.py reset admin
```

Remove a dashboard user:

```powershell
python manage_users.py remove supervisor
```

`manage_users.py` also has a seed command:

```powershell
python manage_users.py seed
```

`seed` creates the users listed in the `DEFAULT_USERS` list at the top of
`manage_users.py` and skips names that already exist. Change those defaults
before using `seed` in any real deployment, or use `add` instead.

Important dashboard user-store notes:

- `users.json` is the local dashboard user store read by `dashboard_server.py`.
- Roles are stored as `admin` or `viewer`; this branch stores the role but does not apply route-level admin-only behavior.
- Treat deployment `users.json` files as sensitive and environment-specific.
- Do not commit production usernames, password hashes, or deployment-specific `users.json` content.

## 13. Validate Setup Without Calling Gemini

Run these safe commands before processing any real audio:

```powershell
python -m compileall watcher.py gemini_flash_stt.py database.py config.py transcript_storage.py dashboard_server.py manage_users.py
python watcher.py --help
python watcher.py --dry-run
ffmpeg -version
ffprobe -version
```

These checks:

- Verify Python files compile.
- Verify watcher command-line help.
- Create/check runtime folders with `--dry-run`.
- Confirm FFmpeg and ffprobe are available.

They do not process audio and do not make a paid Gemini/Vertex call.

## 14. Run Realtime Transcription

Start the watcher:

```powershell
python watcher.py
```

Then copy one supported test audio file into:

```text
input_audio/incoming/
```

Supported audio formats:

```text
.mp3, .wav, .m4a, .flac, .ogg, .aac, .opus
```

Expected behavior:

1. Watcher waits until the file is stable.
2. The file moves to `input_audio/processing/`.
3. Gemini/Vertex transcription starts.
4. TXT and JSON outputs appear under `transcriptions/YYYY-MM-DD/`.
5. MySQL metadata is saved if enabled and available.
6. The audio file moves to `input_audio/completed/YYYY-MM-DD/`.

If processing fails:

1. The audio file moves to `input_audio/failed/YYYY-MM-DD/`.
2. A matching `.error.txt` file is written beside the failed audio.
3. The watcher keeps running for future files.

## 15. Stop The Watcher Safely

Keep the terminal open while the watcher should run.

To stop it:

```text
Ctrl+C
```

The watcher handles Ctrl+C and stops after current work as cleanly as possible.
If the terminal is closed, the watcher stops.

For production Windows Server operation, consider Task Scheduler, NSSM, or a
Windows Service wrapper later. This repository currently focuses on the direct
runtime command:

```powershell
python watcher.py
```

## 16. Output File Naming

Original audio:

```text
20260103-201824_0755583408-all.mp3
```

Transcript TXT:

```text
transcriptions/2026-05-27/20260103-201824_0755583408-all__transcribed_2026-05-27_14-35-20.txt
```

Metadata JSON:

```text
transcriptions/2026-05-27/20260103-201824_0755583408-all__transcribed_2026-05-27_14-35-20.json
```

Completed audio:

```text
input_audio/completed/2026-05-27/20260103-201824_0755583408-all__transcribed_2026-05-27_14-35-20.mp3
```

Failed audio:

```text
input_audio/failed/2026-05-27/20260103-201824_0755583408-all__failed_2026-05-27_14-35-20.mp3
```

The original filename comes first. A timestamp is appended so repeated or
similar filenames do not overwrite each other.

## 17. Optional Dashboard

The dashboard is optional. It is not required for realtime transcription, but it
is useful when MySQL metadata is enabled.

The dashboard is implemented in `dashboard_server.py` as a Flask app with inline
HTML/CSS/JS templates. There is no React or Vite frontend in the active
dashboard code.

Before starting the dashboard, create at least one dashboard user:

```powershell
python manage_users.py add admin
```

Start it with:

```powershell
python dashboard_server.py
```

Open:

```text
http://127.0.0.1:5050
```

Optional startup settings:

```powershell
python dashboard_server.py --port 8080
python dashboard_server.py --host 127.0.0.1 --port 5050
```

The dashboard uses `/login` and `/logout`. Dashboard pages and API routes are
protected by the login session. Users are loaded from `users.json`, and
passwords are verified with stored password hashes.

The Flask session secret is loaded from `DASHBOARD_SECRET_KEY` if that
environment variable is set. Otherwise, `dashboard_server.py` creates a local
`.dashboard_secret` file. `.dashboard_secret` is ignored by Git so sessions can
survive local restarts without committing the secret.

Dashboard logo:

- Login page: `static/app-logo.png`
- Dashboard header: `static/app-logo.png`

Dashboard tiles and views currently include:

- Calls today
- Cost today in USD and LKR
- Tokens, including audio and output token counts
- Silence stripped
- Audio processed
- Language breakdown for Sinhala, English, and Tamil
- 14-day cost trend
- Cost by model
- Total Cost This Month tile
- Monthly Cost History table
- Daily Cost History table
- Recent calls
- Transcript view and TXT download links

Dashboard API and CSV routes:

- `/api/data`
- `/api/transcripts/<call_id>`
- `/api/transcripts/<call_id>/download`
- `/api/daily-cost.csv`
- `/api/monthly-cost.csv`
- `/api/monthly-cost.csv?model=gemini-2.5-flash`

The model filter on `/api/monthly-cost.csv` uses the same model names stored in
MySQL, for example `gemini-2.5-flash`.

Security notes:

- Dashboard defaults to `127.0.0.1`.
- Do not expose it publicly without additional access controls such as a reverse proxy, VPN, firewall rules, and HTTPS.
- The destructive reset route is removed from the active dashboard.
- Dashboard metrics depend on MySQL metadata. If MySQL is disabled or unavailable, the dashboard may show empty data.

## 18. Security Checklist

Before production use:

- Never commit `.env`.
- Never commit `.dashboard_secret`.
- Never commit real service-account JSON credentials.
- Never commit real audio files.
- Never commit transcript TXT/JSON outputs.
- Never commit logs.
- Never commit deployment-specific `users.json` content.
- Restrict file permissions on `credentials/`.
- Restrict file permissions on `transcriptions/`.
- Restrict file permissions on `users.json`.
- Use a dedicated non-root MySQL user.
- Do not use the MySQL `root` user in `.env`.
- Keep the dashboard bound to localhost by default.
- Do not expose the dashboard publicly without access controls.
- Protect the Windows Server account that runs the watcher.
- Rotate the service-account key immediately if it is exposed.
- Back up transcripts and MySQL securely.
- Treat transcripts as sensitive customer/private call data.

## 19. Troubleshooting

### `ffmpeg` is not recognized

FFmpeg is not installed or its `bin` folder is not on PATH.

Fix:

```powershell
ffmpeg -version
```

If the command fails, install FFmpeg, add `ffmpeg\bin` to PATH, and open a new
PowerShell window.

### `ffprobe` is not recognized

ffprobe usually comes with FFmpeg. Confirm the same `bin` folder is on PATH:

```powershell
ffprobe -version
```

### `GOOGLE_APPLICATION_CREDENTIALS` path is wrong

Check `.env`:

```text
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
```

Make sure the JSON file exists at that path. Do not print or share the file
contents.

### Vertex AI permission denied

Common causes:

- Vertex AI API is not enabled.
- The wrong Google Cloud project ID is configured.
- The service account does not have enough Vertex AI permissions.
- Billing or organization policy blocks Vertex AI usage.

Fix IAM using least privilege. Avoid broad owner/admin roles for production.

### MySQL connection failed

Check:

- MySQL Server is running.
- `.env` has the correct host, port, database, user, and password.
- The dedicated MySQL user has privileges on `telecom_voice_to_text`.
- Firewall or network rules allow the connection.

The watcher should still save TXT/JSON outputs even if MySQL fails.

### `database.is_database_enabled()` returns False even though `.env` says true

PowerShell environment variables can override values in `.env`.

Clear safe non-secret overrides from the current PowerShell session:

```powershell
Remove-Item Env:DB_ENABLED -ErrorAction SilentlyContinue
Remove-Item Env:DB_BACKEND -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_HOST -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_PORT -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_DATABASE -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_USER -ErrorAction SilentlyContinue
```

Then retry:

```powershell
python -c "import database; print('enabled:', database.is_database_enabled())"
```

Do not print `MYSQL_PASSWORD`. After fixing environment variables, start the
dashboard or watcher from a clean terminal.

### DB is disabled but transcripts are still saving

This is expected in file-only mode:

```text
DB_ENABLED=false
```

TXT and JSON files are the primary outputs.

### File stays in incoming

The watcher waits for file size and modification time to stop changing. Large
files or slow network copies may remain in `incoming` until stable.

Optional tuning:

```text
WATCHER_STABLE_CHECK_SECONDS=5
WATCHER_STABLE_CHECK_INTERVAL=1
WATCHER_STABLE_MAX_WAIT_SECONDS=300
```

### File moved to failed

Open the matching `.error.txt` file in `input_audio/failed/YYYY-MM-DD/`. It
contains a short safe error message.

### Dashboard is empty

The dashboard uses MySQL metadata. If `DB_ENABLED=false` or MySQL is unavailable,
the dashboard may show empty metrics while TXT/JSON transcripts still exist.

### Dashboard login fails

Check that `users.json` exists and contains the user you are trying to use:

```powershell
python manage_users.py list
```

If the user exists but the password is unknown, reset it:

```powershell
python manage_users.py reset <username>
```

If there are no users yet, create the first one:

```powershell
python manage_users.py add admin
```

### No transcript is printed in the terminal

Full transcript text is intentionally not printed by default. Transcripts may
contain customer/private information. Read the TXT file under `transcriptions/`
instead.

## 20. Hardware And Runtime Expectations

The watcher is designed to run continuously until stopped.

Expected resource behavior:

- Idle CPU usage is low.
- Idle RAM usage is modest.
- FFmpeg/audio preprocessing uses CPU.
- Gemini/Vertex transcription is mostly network/API-bound.
- Longer audio files take longer to preprocess and transcribe.
- Disk usage grows as audio archives, transcripts, metadata JSON, and logs accumulate.

Recommended minimums:

- Small pilot: 2 CPU cores, 4 GB RAM, stable internet, enough disk for retained audio/transcripts.
- Production Windows Server: 4 CPU cores, 8 GB RAM, stable internet, planned disk retention.
- Higher volume: 8+ CPU cores, 16 GB RAM, retention policy, backups, monitoring, and disk capacity planning.

Before production, test with representative audio sizes and call volumes.

## 21. Limitations

- The active watcher processes files sequentially.
- Running multiple watcher processes against the same folders is not recommended.
- Transcription depends on Google Cloud/Vertex availability and internet access.
- MySQL is metadata only. TXT/JSON files are the primary outputs.
- Batch processing is archived and not active in this branch.
- Linux service deployment is archived and not active in this branch.
- Dashboard authentication is local file-based authentication through `users.json`; use external access controls before any remote exposure.
- Dashboard role values are stored but this branch does not enforce separate admin-only dashboard routes.

## 22. Archived / Future Features

Archived material is preserved for future review:

- `docs/archive/batch/`: older batch processing code.
- `docs/archive/linux/`: older Linux/systemd service files.
- `docs/archive/legacy/`: older examples and explanatory material.

These files are not part of the active Windows Server realtime deployment flow.
Review dependencies, security, credentials, database assumptions, and runtime
behavior before reusing them.

## 23. Quick Start Summary

1. Clone the repository and enter the project folder.
2. Create and activate `.venv`.
3. Install requirements.
4. Install FFmpeg and confirm `ffmpeg` / `ffprobe`.
5. Create a Google Cloud service account JSON key.
6. Place the JSON file at `credentials/google-credentials.json`.
7. Copy `.env.example` to `.env`.
8. Configure Google Cloud values in `.env`.
9. Configure input/output folders in `.env` if the defaults are not right.
10. Configure MySQL and set `DB_ENABLED=true` and `DB_BACKEND=mysql` if you want dashboard metadata.
11. Run database initialization if MySQL is enabled.
12. Create the first dashboard user with `python manage_users.py add admin`.
13. Run `python watcher.py --dry-run`.
14. Run `python watcher.py`.
15. Drop supported audio into `input_audio/incoming/`.
16. Start the dashboard with `python dashboard_server.py` and open `http://127.0.0.1:5050`.

Safe validation commands:

```powershell
python -m compileall watcher.py gemini_flash_stt.py database.py config.py transcript_storage.py dashboard_server.py manage_users.py
python watcher.py --help
python watcher.py --dry-run
python dashboard_server.py --help
ffmpeg -version
ffprobe -version
```

Do not use real production audio for first validation. Start with a small test
clip that is approved for transcription.
