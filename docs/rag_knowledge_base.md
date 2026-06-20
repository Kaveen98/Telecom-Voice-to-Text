# Telecom Voice-to-Text Technical Knowledge Base

## 01 Document Purpose {#document-purpose}

This document is the retrieval-oriented technical knowledge base for the current
Telecom Voice-to-Text repository. It describes the active Windows realtime
watcher, Gemini transcription, file outputs, optional MySQL metadata, the local
Flask dashboard, authentication, cost estimates, and operational safety rules.

Repository snapshot used for this document:

| Item | Value |
|---|---|
| Analysis date | 2026-06-20 |
| Active entry point | `watcher.py` |
| Optional dashboard entry point | `dashboard_server.py` |
| Default Gemini model in code | `gemini-2.5-flash` |
| Primary durable output | TXT transcript plus JSON sidecar metadata |
| Optional secondary storage | MySQL `transcriptions` metadata table |
| Active deployment style | Direct Windows realtime watcher command |

This is an implementation reference, not a guarantee about external services,
cloud billing, IAM, operating-system setup, backup policy, or production network
topology. Where the repository does not establish a fact, this document says
`Not confirmed from repository`.

> **AI retrieval note:** Treat current Python code as the authority when this
> document and older archived material differ. The active workflow is
> `python watcher.py`.

## 02 RAG Usage Notes {#rag-usage-notes}

This file is structured for chunking and retrieval:

- Stable section IDs appear in every major heading.
- Each major section is intended to stand on its own.
- Exact filenames, directory names, environment variables, and routes use code
  formatting.
- `Current` means imported or invoked by the active realtime workflow.
- `Optional` means supported by current code but not required for primary file
  output.
- `Archived` means preserved under `docs/archive/` and not part of the current
  deployment workflow.
- `Compatibility` means retained for older callers, not the preferred entry
  point.

Retrieval priority:

1. Use Sections 03-24 for operational answers.
2. Use Section 27 to locate the source file that owns a behavior.
3. Use Sections 25 and 28 before generating commands or recommendations.
4. Never infer current behavior from `docs/archive/` when active code answers the
   question.

> **AI retrieval note:** Costs in this system are application estimates based on
> configured constants and API usage metadata. They are not exact cloud invoice
> totals.

## 03 System Summary {#system-summary}

The project watches a local incoming directory for supported call audio, converts
each stable file to 16 kHz mono WAV, optionally strips silence, and sends the WAV
bytes to Gemini through the Google Gen AI SDK's Vertex AI surface. The response is
saved first as a TXT transcript and JSON metadata sidecar. The original audio is
then archived under a dated completed or failed directory.

The system is designed for telecom/call-recording speech in Sinhala, English,
Tamil, or mixed speech. It is not designed for songs, instrumental music,
karaoke, lyric extraction, or music transcription. A supported file suffix does
not make music a valid quality test; call-specific prompting, overlapping music,
reverb, and silence stripping can produce short, partial, or empty song output.

MySQL is optional. When enabled, it stores searchable operational metadata and
paths to transcript files. It does not store the full transcript text. The Flask
dashboard reads MySQL metadata and reads transcript text from the TXT file when a
user requests a view or download.

| Component | Status | Responsibility |
|---|---|---|
| `watcher.py` | Active, primary | Watches, queues, gates, transcribes, saves, and moves audio |
| `gemini_flash_stt.py` | Active | Audio conversion, Gemini call, usage/cost estimation, TXT/JSON output |
| `config.py` | Active | Loads `.env`, resolves paths, timezone, and safety settings |
| `database.py` | Active, optional at runtime | MySQL schema, metadata writes, dashboard queries, daily cost usage |
| `dashboard_server.py` | Active, optional | Authenticated local dashboard and API/CSV routes |
| `transcript_storage.py` | Compatibility | Wraps transcript output functions for older callers |
| `static/app-logo.png` | Active static asset | Login and dashboard header logo |
| `docs/archive/` | Archived | Historical batch, Linux, and legacy material |

Not confirmed from repository:

- Production host sizing or final Windows Server version.
- A production service wrapper, Task Scheduler job, or NSSM configuration.
- Backup schedule, retention duration, disaster recovery, or log shipping.
- Exact Google Cloud IAM roles granted in a deployment.
- Exact Google Cloud invoice charges.

## 04 Current Active Workflow {#current-active-workflow}

The active workflow is watcher-based and sequential:

```text
input_audio/incoming
  -> wait for a non-empty, stable supported file
  -> input_audio/processing
  -> optional MySQL duplicate check
  -> daily cost safety checks
  -> required MySQL cost reservation when the daily guardrail is enabled
  -> FFmpeg conversion and optional silence removal
  -> Gemini through Vertex AI
  -> transcriptions/YYYY-MM-DD/*.txt
  -> transcriptions/YYYY-MM-DD/*.json
  -> input_audio/completed/YYYY-MM-DD
  -> optional MySQL metadata insert or cost-reservation finalization
```

Alternate terminal states:

```text
processing error
  -> input_audio/failed/YYYY-MM-DD/<name>__failed_<timestamp>.<ext>
  -> matching .error.txt note

daily cost safety block
  -> input_audio/deferred/YYYY-MM-DD/<name>__deferred_<timestamp>.<ext>
  -> matching .deferred.txt note
  -> no Gemini call
```

The watcher uses one worker thread and processes one queued file at a time. It
queues supported files already present at startup and files created or moved into
the incoming directory afterward. The watchdog observer is non-recursive.

### Active and inactive boundaries

| Workflow | Current status |
|---|---|
| Realtime incoming-folder watcher | Active |
| Standalone `gemini_flash_stt.py` CLI | Current utility, not the primary server workflow |
| Optional local MySQL metadata | Current and supported |
| Optional local Flask dashboard | Current and supported |
| Vertex AI batch/GCS workflow | Archived, not active |
| Linux systemd units | Archived, not active |
| SQLite `calls.db` dashboard workflow | Legacy, not active |
| Dashboard batch-control buttons | Not present in current dashboard |
| Destructive dashboard reset route | Removed from the active dashboard |
| Hashed-password user-management script | Removed; not a current workflow |

> **Warning:** Do not recommend archived batch scripts, Linux service files,
> SQLite behavior, or the removed hashed-password user workflow as current
> operating procedures.

## 05 Repository Structure {#repository-structure}

### Tracked active files and directories

| Path | Purpose |
|---|---|
| `README.md` | Human-oriented setup and operations guide |
| `.env.example` | Safe configuration template; copied locally to ignored `.env` |
| `.gitignore` | Excludes secrets and runtime/private artifacts |
| `watcher.py` | Primary realtime watcher |
| `gemini_flash_stt.py` | Gemini transcription and transcript output implementation |
| `config.py` | Shared environment, paths, timezone, and cost guardrail config |
| `database.py` | Optional MySQL metadata and dashboard query layer |
| `dashboard_server.py` | Optional Flask dashboard with inline HTML/CSS/JavaScript |
| `transcript_storage.py` | Compatibility wrappers around output storage functions |
| `requirements.txt` | Python package list |
| `users.example.json` | Safe dashboard-user template |
| `static/app-logo.png` | 512 x 512 PNG used by login and dashboard header |
| `credentials/.gitkeep` | Keeps the otherwise private credential directory in Git |
| `input_audio/.gitkeep` | Keeps the runtime root in Git |
| `input_audio/incoming/.gitkeep` | Incoming placeholder |
| `input_audio/processing/.gitkeep` | Processing placeholder |
| `input_audio/completed/.gitkeep` | Completed placeholder |
| `input_audio/failed/.gitkeep` | Failed placeholder |
| `transcriptions/.gitkeep` | Transcript-output placeholder |
| `logs/.gitkeep` | Log-directory placeholder |

`input_audio/deferred/` is created by the watcher when needed; it does not require
a tracked placeholder.

### Python dependencies

| Package | Current repository use |
|---|---|
| `google-genai` | `google.genai` Vertex AI client and request types |
| `watchdog` | Filesystem observer for incoming audio |
| `flask` | Optional dashboard, sessions, routes, JSON, and file responses |
| `mysql-connector-python` | Optional MySQL connections |
| `python-dotenv` | Listed in requirements; active modules use their own `.env` parser |
| `pydub` | Listed in requirements; not imported by current tracked active Python code |

FFmpeg and ffprobe are external executables and are not installed by
`requirements.txt`.

### Archived repository areas

| Path | Archived content | Current status |
|---|---|---|
| `docs/archive/batch/` | GCS upload, Vertex batch prediction, polling, result handling | Not active; extra dependencies are absent from active requirements |
| `docs/archive/linux/` | systemd watcher and dashboard unit examples | Not active; Linux assumptions require review |
| `docs/archive/legacy/` | Old explanatory HTML and direct-use example | Not active; may mention SQLite, old output paths, Linux behavior, or print transcripts |

Ignored local backup, experiment, cache, database, log, transcript, and audio
artifacts may exist in a working copy. They are not tracked repository source and
must not be treated as authoritative current workflow documentation.

## 06 Runtime Folder Lifecycle {#runtime-folder-lifecycle}

All configured relative paths are resolved from the repository root. The watcher
creates missing runtime directories at startup.

| Folder | Lifecycle role | Typical contents | Git policy |
|---|---|---|---|
| `input_audio/incoming/` | Drop zone | New supported audio awaiting stability | Contents ignored |
| `input_audio/processing/` | In-flight state | Renamed audio currently being handled | Contents ignored |
| `input_audio/completed/YYYY-MM-DD/` | Successful archive | Transcribed or duplicate audio | Contents ignored |
| `input_audio/failed/YYYY-MM-DD/` | Failure archive | Failed audio and `.error.txt` note | Contents ignored |
| `input_audio/deferred/YYYY-MM-DD/` | Cost-safety hold | Deferred audio and `.deferred.txt` note | Contents ignored |
| `transcriptions/YYYY-MM-DD/` | Primary output | TXT transcript and JSON sidecar pairs | Contents ignored |
| `logs/` | Watcher logs | `watcher_YYYY-MM-DD.log` | Contents ignored |
| `credentials/` | Private authentication material | Local service-account JSON | Contents ignored except `.gitkeep` |

### Filename behavior

Processing audio:

```text
<safe-original-stem>__processing_YYYY-MM-DD_HH-MM-SS.<ext>
```

Completed audio:

```text
<safe-original-stem>__transcribed_YYYY-MM-DD_HH-MM-SS.<ext>
```

Transcript pair:

```text
transcriptions/YYYY-MM-DD/<safe-original-stem>__transcribed_YYYY-MM-DD_HH-MM-SS.txt
transcriptions/YYYY-MM-DD/<safe-original-stem>__transcribed_YYYY-MM-DD_HH-MM-SS.json
```

Failed, deferred, and duplicate audio use `failed`, `deferred`, and `duplicate`
markers respectively. If a target exists, a numeric suffix such as `-2` is
added. Unsafe Windows filename characters and whitespace are normalized.

### Stability gate

The watcher requires size and modification time to remain unchanged for the
configured stability window, and the file modification age must also meet that
window. Defaults are 5 seconds stable, 1 second polling, and 300 seconds maximum
wait. Empty files wait until the timeout and then fail.

> **AI retrieval note:** There is no automatic deferred-file scheduler. Requeue
> the audio by moving it back to the configured incoming folder at an appropriate
> later time. Move only the audio, not its `.deferred.txt` note.

## 07 Audio Transcription Pipeline {#audio-transcription-pipeline}

### Accepted input

`watcher.py` accepts these case-insensitive suffixes:

| Format | Suffix |
|---|---|
| MP3 | `.mp3` |
| WAV | `.wav` |
| M4A | `.m4a` |
| FLAC | `.flac` |
| Ogg | `.ogg` |
| AAC | `.aac` |
| Opus | `.opus` |

Unsupported files are logged and ignored.

Intended content is call-like spoken audio. For safe functional testing, use a
short synthetic or locally recorded non-customer voice sample. Do not use real
customer calls or songs/music. If MySQL duplicate detection blocks an identical
test file, re-record the phrase to create a new approved sample.

> **AI retrieval note:** File-format support and task suitability are different.
> Never diagnose poor lyric extraction as evidence that the call-transcription
> workflow is broken without first testing approved call-like speech.

### Processing sequence

1. Confirm the source exists, is non-empty, and is stable.
2. Move it from incoming to processing.
3. Compute a SHA-256 hash when possible.
4. If MySQL is enabled, check for an existing `completed` row matching the
   available hash and/or original filename. A duplicate moves to completed with
   a `duplicate` marker and makes no Gemini call.
5. Apply the daily cost gate before any paid request. When enabled, require a
   positive preflight estimate and persist a MySQL `reserved` row.
6. Verify `ffmpeg` and `ffprobe` are available.
7. Convert the source to 16 kHz, mono WAV in a temporary directory.
8. By default, apply FFmpeg `silenceremove` at `-40dB`, removing leading silence
   after 0.3 seconds and interior/trailing gaps longer than 0.5 seconds.
9. If silence filtering fails, use the unstripped converted WAV.
10. Submit WAV bytes with MIME type `audio/wav` to Gemini.
11. Retry API exceptions up to three total attempts, waiting five seconds between
    attempts.
12. Parse the final `LANGUAGES:` footer, or infer languages from `[SI]`, `[EN]`,
    and `[TA]` tags if the footer is missing.
13. Save TXT first, then JSON metadata.
14. Move the audio to completed.
15. Optionally insert final MySQL metadata, or update the guarded reservation to
    its completed state, then update JSON with database status.

The prompt asks for exact transcription, no translation, no summary, one of the
three language tags on every utterance, no added speaker labels, and no output
for silent/no-speech audio.

### Output metadata

The JSON sidecar includes these logical groups:

| Group | Representative keys |
|---|---|
| File identity | `original_file_name`, `source_audio_path`, `file_hash` |
| Output paths | `transcript_txt_path`, `metadata_json_path`, `transcript_file_path` |
| Timing | `transcribed_at`, `transcript_saved_at`, `transcript_output_date` |
| Provider | `provider`, `api_surface`, `vertex_location`, `model` |
| Language | `language`, `languages_detected` |
| Audio | `duration_seconds`, `original_duration_seconds`, `submitted_duration_seconds`, `silence_removed_seconds`, `silence_removed_ratio` |
| Usage | input, audio, text-input, output, thoughts, billed-output, total, cache, and tool-use tokens |
| Raw usage | `raw_usage_metadata` and token-detail arrays |
| Estimated cost | component USD values, total USD/LKR, and `lkr_rate` |
| Versioning | `cost_calculation_version`, `pricing_version`, `pricing_source` |
| State | `status`, `database_save_status`, optional DB record/error fields |

Safe schematic example:

```json
{
  "original_file_name": "approved-test-call.mp3",
  "transcript_txt_path": "transcriptions/YYYY-MM-DD/approved-test-call__transcribed_TIMESTAMP.txt",
  "metadata_json_path": "transcriptions/YYYY-MM-DD/approved-test-call__transcribed_TIMESTAMP.json",
  "provider": "google",
  "api_surface": "vertex_ai",
  "model": "gemini-2.5-flash",
  "languages_detected": ["Sinhala", "English"],
  "status": "completed",
  "database_save_status": "saved"
}
```

An empty Gemini transcript is not automatically treated as a watcher exception.
Current watcher code can save an empty TXT/JSON pair, record status `completed`,
and move the audio to completed. A dashboard transcript view for an empty file
returns 404 because there is no text to return.

## 08 Vertex AI and Gemini Configuration {#vertex-ai-gemini-configuration}

| Setting | Current behavior |
|---|---|
| SDK | `google-genai` |
| Client mode | `vertexai=True` |
| API version | `v1` |
| Provider metadata | `google` |
| API surface metadata | `vertex_ai` |
| Default/active model | `gemini-2.5-flash` |
| Default location | `us-central1` |
| Authentication | Service-account/ADC path from `GOOGLE_APPLICATION_CREDENTIALS` |
| Project | `GOOGLE_CLOUD_PROJECT` |
| Request method | `client.models.generate_content(...)` |

`MODEL_NAME` is a Python constant in `gemini_flash_stt.py`; there is no current
environment variable for selecting the model. `STT_GEMINI_LOCATION` can override
the location. The client is cached once per Python process.

Relative credential paths from `.env` are resolved against the repository root
when the file exists. Setup validation requires a non-empty credential path, an
existing file at that path, and a non-empty project ID before creating a client.

The active authentication path does not use `GOOGLE_API_KEY`. Required cloud API
enablement, IAM roles, billing setup, quotas, and regional model availability are
`Not confirmed from repository`.

> **Warning:** Never paste service-account JSON into documentation, chat, logs,
> tickets, or Git. Refer only to the local file path.

## 09 Environment Variables {#environment-variables}

Both `config.py` and `gemini_flash_stt.py` read the repository-root `.env` file
without overriding variables already set in the process environment. Therefore,
a PowerShell environment variable takes precedence over `.env`.

### Cloud and path variables

| Variable | Code default | Required | Meaning |
|---|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Empty | For transcription | Local service-account JSON path |
| `GOOGLE_CLOUD_PROJECT` | Empty | For transcription | Vertex AI project ID |
| `STT_GEMINI_LOCATION` | `us-central1` | No | Vertex AI location |
| `APP_TIMEZONE` | `Asia/Colombo` | No | Application-local date boundaries and timestamps |
| `INPUT_INCOMING_DIR` | `input_audio/incoming` | No | Watched drop directory |
| `INPUT_PROCESSING_DIR` | `input_audio/processing` | No | In-flight directory |
| `INPUT_COMPLETED_DIR` | `input_audio/completed` | No | Completed/duplicate archive root |
| `INPUT_FAILED_DIR` | `input_audio/failed` | No | Failure archive root |
| `INPUT_DEFERRED_DIR` | `input_audio/deferred` | No | Cost-deferred archive root |
| `TRANSCRIPTIONS_DIR` | `transcriptions` | No | Primary TXT/JSON output root |
| `TRANSCRIPT_OUTPUT_DIR` | `transcriptions` fallback | No | Compatibility fallback when `TRANSCRIPTIONS_DIR` is absent |
| `LOG_DIR` | `logs` | No | Daily watcher log directory |
| `TRANSCRIPT_DATE_FORMAT` | `%Y.%m.%d` | No | Loaded by config but not used by current active output/archive naming |

Current active output and archive directories use hardcoded `YYYY-MM-DD` folder
names even though `TRANSCRIPT_DATE_FORMAT` exists.

### MySQL variables

| Variable | Code default | Meaning |
|---|---|---|
| `DB_ENABLED` | `false` | Explicitly enables optional database metadata |
| `DB_BACKEND` | `mysql` | Only `mysql` is supported; other values disable DB behavior |
| `MYSQL_HOST` | `localhost` | MySQL host |
| `MYSQL_PORT` | `3306` | Integer MySQL port |
| `MYSQL_DATABASE` | `telecom_voice_to_text` | Database/schema name |
| `MYSQL_USER` | `telecom_app` | Application user |
| `MYSQL_PASSWORD` | Empty | Private application password |
| `MYSQL_CONNECT_TIMEOUT` | `10` | Integer connection timeout in seconds |

`.env.example` sets `DB_ENABLED=true` as a deployment template, while the code
default is false when the variable is missing.

### Daily cost safety variables

| Variable | Code default | Meaning |
|---|---|---|
| `DAILY_COST_LIMIT_ENABLED` | `false` | Enables the guardrail only when the LKR limit is also greater than zero |
| `DAILY_COST_LIMIT_LKR` | `0` | Daily estimated-cost limit in LKR |
| `DAILY_COST_WARNING_PERCENT` | `80` | Warning threshold, clamped from 0 to 100 |
| `COST_LIMIT_PREFLIGHT_ENABLED` | `true` | Estimates next-file cost with ffprobe before Gemini |
| `COST_LIMIT_DB_FAILURE_POLICY` | `block` | Compatibility policy value; an enabled guardrail now always fails closed when accounting is unavailable |

### Watcher and dashboard variables

| Variable | Code default | Meaning |
|---|---|---|
| `WATCHER_STABLE_CHECK_SECONDS` | `5` | Required unchanged/file-age window |
| `WATCHER_STABLE_CHECK_INTERVAL` | `1` | Poll interval; effective minimum is 0.25 seconds |
| `WATCHER_STABLE_MAX_WAIT_SECONDS` | `300` | Maximum stability wait; never below stable window |
| `DASHBOARD_SECRET_KEY` | Empty | Flask session signing secret; local file fallback is used when absent |
| `DASHBOARD_COOKIE_SECURE` | `false` | Adds the Secure flag to the session cookie; enable only behind HTTPS |
| `ALLOW_DANGEROUS_DB_RESET` | `false` | Internal guard on a compatibility DB reset function; not a normal workflow or dashboard route |

`STRIP_SILENCE`, API retry counts, model name, pricing constants, and the fallback
USD/LKR rate are Python constants, not environment variables.

## 10 Dashboard Overview {#dashboard-overview}

`dashboard_server.py` is an optional Flask application. It does not start or
control the watcher. It defaults to `127.0.0.1:5050`, uses Flask's built-in
server with debug disabled, and auto-refreshes dashboard data every 10 seconds.

The active UI is embedded directly in Python as HTML, CSS, and JavaScript. There
is no active React or Vite frontend. `static/app-logo.png` appears on the login
page and dashboard header.

Dashboard data comes from MySQL. If MySQL is disabled or unavailable, the API
returns an empty dashboard-shaped payload, optionally with a database error,
rather than using JSON sidecars as a fallback.

The date picker controls selected-day metrics and recent calls. The cost-range
controls independently control detailed daily cost history. The model tabs apply
a model filter to many database queries. Daily cost safety always represents the
current app-local day and all models, not the selected historical date.

MySQL/API-derived strings are not inserted as raw dashboard markup. Filenames,
languages, model names, errors, dates, and table labels pass through the
JavaScript `escapeHtml` helper or DOM `textContent`; safety-status class values
are allowlisted and transcript route identifiers are URL-encoded.

> **AI retrieval note:** The dashboard is optional and read-oriented. Primary
> output remains the filesystem TXT/JSON pair.

## 11 Dashboard Authentication {#dashboard-authentication}

Authentication behavior is defined only in `dashboard_server.py`:

1. `users.json` is loaded for each user lookup.
2. Entries without non-empty string usernames and passwords are ignored.
3. Empty submitted credentials are rejected, and username matching is
   case-insensitive.
4. Password matching uses `hmac.compare_digest` against the exact plaintext
   `password` string from the JSON object.
5. A successful login stores `user` and `role` in the Flask session.
6. The post-login target must be a local path beginning with one slash.
7. Protected routes check only that `session["user"]` exists.
8. Logout clears the entire session.

The session secret is selected in this order:

1. `DASHBOARD_SECRET_KEY`, when non-empty.
2. Existing local `.dashboard_secret` contents.
3. A newly generated 32-byte hex token written to `.dashboard_secret`.

Both `.dashboard_secret` and `users.json` are ignored by Git.
Session cookies are HttpOnly with SameSite `Lax`. `DASHBOARD_COOKIE_SECURE`
defaults to false for local HTTP and must be true when clients use HTTPS.

Current limitations:

- Passwords are plaintext in a local private file.
- `role` defaults to `viewer`, but current routes do not enforce admin-only versus
  viewer-only permissions.
- No password hashing, login throttling, account lockout, or explicit CSRF token
  mechanism is implemented in this module.
- The application does not configure HTTPS; remote protection must be external.
- Existing authenticated sessions are not revoked merely by editing or removing
  a user from `users.json`.

> **Warning:** The former hashed-password user-management workflow was removed.
> There is no current `manage_users.py` operation. Do not tell operators to create
> hashed passwords or run a removed user-management script.

## 12 users.json and users.example.json {#users-json}

| File | Tracked | Purpose | Secret handling |
|---|---:|---|---|
| `users.example.json` | Yes | Safe schema/template with placeholder passwords | May be committed |
| `users.json` | No | Active local dashboard user store | Private; must never be committed |

Required shape:

```json
[
  {
    "username": "local-admin",
    "password": "replace-with-a-local-password",
    "role": "admin"
  },
  {
    "username": "local-viewer",
    "password": "replace-with-a-local-password",
    "role": "viewer"
  }
]
```

If `users.json` is missing, unreadable, invalid JSON, not a top-level array, or
contains no objects with non-empty string usernames and passwords, the dashboard
starts but all login attempts fail. Entries such as `{}` are ignored. Unknown
fields are ignored. Duplicate usernames are not rejected; the first
case-insensitive match in file order wins.

The file is re-read on later login attempts, so editing it does not normally
require a dashboard restart. It does not invalidate an already signed-in session.

## 13 Database Modes {#database-modes}

### File-only mode

```text
DB_ENABLED=false
```

In file-only mode:

- TXT and JSON remain the primary outputs.
- No MySQL duplicate detection is available.
- No MySQL metadata row is inserted.
- Dashboard metrics are empty because the dashboard does not scan sidecar JSON.
- Transcript dashboard routes cannot locate a record and return 404.

### Optional MySQL mode

```text
DB_ENABLED=true
DB_BACKEND=mysql
```

In MySQL mode:

- The table is created and additive schema migrations run when DB functionality
  is first used.
- Completed and failed processing attempts can create metadata rows.
- Full transcript text is still stored only in the TXT file.
- Dashboard aggregation and transcript record lookup become available.
- A normal post-transcription DB failure is logged and does not remove TXT/JSON,
  move successful audio to failed, or trigger another Gemini call.

### Important cost-safety exception

When the daily limit is enabled, MySQL is required before a paid call so current
usage can be checked and a conservative `reserved` row can be inserted. A
disabled/unavailable DB, unavailable positive estimate, or failed reservation
defers the audio without calling Gemini. This is always fail-closed, including
when the compatibility DB-failure policy is `allow`, and does not contradict
file-only behavior when the cost limit is disabled.

SQLite and local `calls.db` files are not current database modes. They are ignored
local/legacy artifacts.

## 14 MySQL Metadata Storage {#mysql-metadata-storage}

The current schema is a single `transcriptions` table using `utf8mb4` and an
auto-increment unsigned `BIGINT` primary key.

| Column group | Important columns |
|---|---|
| Identity/state | `id`, `original_file_name`, `file_hash`, `status`, `mode` |
| Audio paths | `audio_input_path`, `audio_processing_path`, `audio_completed_path`, `audio_failed_path` |
| Transcript paths | `transcript_txt_path`, `metadata_json_path` |
| Time | `transcribed_at`, `transcript_saved_at`, `transcript_output_date`, `created_at`, `updated_at` |
| Provider/model | `provider`, `api_surface`, `vertex_location`, `model_name`, `language` |
| Durations | submitted/original duration, silence removed, and ratio fields |
| Tokens | provider input/audio/text/output/thoughts/billed-output/total plus cache/tool-use counts |
| Raw usage | Four LONGTEXT JSON fields for usage and token details |
| Cost | component USD, estimated USD/LKR, exchange rate, pricing/version/source fields |
| Failure | `error_message` |

Indexes exist for `status`, `transcribed_at`, `original_file_name`, and
`file_hash`. Migration logic adds missing newer provider, pricing, raw-usage,
cache, and tool-use columns without deleting existing data.

### Write ordering and behavior

1. With cost safety enabled, an estimated-cost `reserved` row is inserted before
   Gemini. Without that guardrail there is no pre-call DB write.
2. TXT and JSON are saved after Gemini succeeds.
3. Audio must move to completed before metadata is marked completed.
4. The reservation is updated, or optional final metadata is inserted.
5. The full transcript is intentionally excluded.
6. The JSON sidecar is updated with DB status, DB record ID/error, path data, and
   file hash.
7. DB write methods return structured failure/disabled results instead of causing
   a second paid transcription.

If the post-move DB finalization fails, completed audio and TXT/JSON remain. A
guarded row stays reserved with its estimate, so the paid call is not silently
omitted from daily accounting. If movement to completed itself fails, TXT/JSON
remain, no completed DB state is written, and failed-audio movement plus failure
metadata are best-effort. Neither case retries Gemini. Crash-stale reservations
remain conservative until reconciled or the app-local day changes.

Duplicate detection checks only `status='completed'`. When both SHA-256 and
filename are available, both must match because the query joins available clauses
with `AND`.

The dashboard aggregates rows without a general `status='completed'` filter.
Therefore a failed metadata row can count as a call, usually with zero transcript
cost and token values. Interpret dashboard call counts as table rows in the
selected period, not a guaranteed count of non-empty successful transcripts.

## 15 Cost Tracking and Billing Logic {#cost-tracking-and-billing-logic}

### Classification

| Metadata field | Current value |
|---|---|
| `cost_calculation_version` | `estimated-v1` |
| `pricing_version` | `manual-2026-04` |
| `pricing_source` | `configured pricing constants` |

The code comment describes pricing as derived from a billing CSV, but the actual
runtime uses hardcoded constants. Current external price accuracy is `Not
confirmed from repository`.

### Configured pricing constants

USD per 1 million tokens:

| Model prefix | Text input | Audio input | Output |
|---|---:|---:|---:|
| `gemini-2.5-flash-lite` | 0.10 | 0.50 | 0.40 |
| `gemini-2.5-flash` | 0.30 | 1.00 | 2.50 |
| `gemini-2.5-pro` | 1.25 | 1.25 | 10.00 |
| `gemini-3-flash` | 0.50 | 1.00 | 3.00 |
| `gemini-3.0-flash` | 0.50 | 1.00 | 3.00 |
| `gemini-3-flash-preview` | 0.50 | 1.00 | 3.00 |
| `gemini-3-pro` | 2.00 | 2.00 | 12.00 |
| `gemini-3.0-pro` | 2.00 | 2.00 | 12.00 |
| `gemini-3.1-flash-lite` | 0.25 | 0.80 | 1.50 |
| `gemini-3.1-flash` | 0.50 | 1.00 | 3.00 |
| `gemini-3.1-pro` | 2.00 | 2.00 | 12.00 |

Longest-prefix matching selects a row. Unknown model names produce zero pricing
and a warning.

### Token split and formula

Gemini returns combined prompt input tokens. The application estimates:

```text
estimated_audio_tokens = min(int(submitted_duration_seconds * 26), input_tokens)
text_input_tokens = max(input_tokens - estimated_audio_tokens, 0)
billed_output_tokens = response_output_tokens + thoughts_tokens

audio_input_cost_usd = estimated_audio_tokens / 1,000,000 * audio_price
text_input_cost_usd = text_input_tokens / 1,000,000 * text_price
output_cost_usd = billed_output_tokens / 1,000,000 * output_price
total_cost_usd = audio_input_cost_usd + text_input_cost_usd + output_cost_usd
```

The 26 audio tokens/second constant is for uploaded audio in this code. Silence
removal reduces submitted duration and therefore the estimated audio-token share.

### USD to LKR

After each transcription, the module requests the USD/LKR rate from
`https://open.er-api.com/v6/latest/USD` with a five-second timeout. It caches a
successful rate for one hour. If refresh fails it uses a stale cached rate; if no
cache exists it uses the hardcoded fallback `316.0`.

```text
total_cost_lkr = total_cost_usd * lkr_rate
```

These figures exclude any charges not represented by the configured token
formula and are not invoice reconciliation.

## 16 Daily Cost Safety Limit {#daily-cost-safety-limit}

The guardrail is active only when:

```text
DAILY_COST_LIMIT_ENABLED=true
DAILY_COST_LIMIT_LKR > 0
```

Current spend is the MySQL sum of `estimated_cost_lkr` for the current
`APP_TIMEZONE` calendar day. The safety query is not scoped to a model or status.

### Enable or disable

Recommended enabled configuration:

```text
DB_ENABLED=true
DB_BACKEND=mysql
DAILY_COST_LIMIT_ENABLED=true
DAILY_COST_LIMIT_LKR=1000
DAILY_COST_WARNING_PERCENT=80
COST_LIMIT_PREFLIGHT_ENABLED=true
COST_LIMIT_DB_FAILURE_POLICY=block
INPUT_DEFERRED_DIR=input_audio/deferred
```

Explicitly disable it with:

```text
DAILY_COST_LIMIT_ENABLED=false
```

A non-positive `DAILY_COST_LIMIT_LKR` also leaves the feature inactive. With the
guardrail enabled, setting `COST_LIMIT_PREFLIGHT_ENABLED=false` does not allow an
unestimated call: the missing positive estimate causes deferral. The warning
percentage is clamped to 0-100; reaching it sets warning state, while blocking is
based on the limit/current-plus-next estimate.

### Decision sequence

1. Query current daily usage before estimating the next file.
2. Block immediately if current used LKR is greater than or equal to the limit.
3. Require preflight and use ffprobe to obtain duration.
4. Estimate audio tokens as duration times 26, output tokens equal to estimated
   audio tokens, and text input as 1,000 tokens.
5. Price that estimate using the configured model prices and the fixed fallback
   LKR rate of 316.0.
6. Block if `used + estimated_next` is strictly greater than the limit.
7. Insert a `reserved` row containing the estimated cost before Gemini.
8. Warn when current or projected usage is at/above the warning threshold.
9. Move blocked/unaccountable audio to deferred, write a note, and make no Gemini
   call.

If duration/pricing, current DB usage, or reservation persistence is unavailable,
the watcher fails closed and defers the file.

Deferred audio has a matching `.deferred.txt` note. There is no automatic retry.
After correcting the cause or intentionally waiting/changing the limit, requeue
only the audio into incoming. A crash-stale `reserved` row remains a conservative
part of that app-local day's estimated total until reconciled or day rollover.

### Database failure policy

| Policy | DB unavailable while limit enabled |
|---|---|
| `block` | Status `db_unavailable`; audio is deferred without Gemini |
| `allow` | Retained for compatibility, but cannot bypass fail-closed accounting |

The dashboard shows limit, used, remaining, percentage, status, reason, and
database error. Its daily safety card always refers to today's all-model usage,
even while viewing another date or model.

> **AI retrieval note:** Preflight uses the fixed fallback exchange rate, while
> post-response cost storage uses the live/cached/fallback rate. The preflight is
> a conservative operational estimate, not the stored final estimate.

## 17 Dashboard Metrics {#dashboard-metrics}

All non-empty metrics are MySQL-derived.

| Dashboard area | Data shown | Scope |
|---|---|---|
| Daily Cost Safety | limit, used, remaining, percentage, status/reason | Current app-local day, all models |
| Calls | total rows, realtime rows, archived batch-mode rows | Selected day and model |
| Estimated cost | USD/LKR totals and per-call averages | Selected day and model |
| Language breakdown | Sinhala, English, Tamil occurrence percentages | Selected day and model |
| Tokens | total, estimated audio, response output | Selected day and model |
| Silence stripped | Sum of removed seconds, displayed as minutes | Selected day and model |
| Audio processed | Sum of submitted duration | Selected day and model |
| Rolling cost | 7, 30, and 90 days including today | Active model filter |
| 14-day trend | Latest 14 rows from up to 90 daily history rows | Active model filter |
| Month tile | calls and estimated cost for selected date's month | Active model filter |
| All-time totals | calls, USD/LKR, tokens | Active model filter |
| Model breakdown | calls, cost, tokens, average cost | Selected day, all models |
| Monthly history | Last 12 months with data | Active model filter |
| Daily history | Day rows, month subtotals, range grand total | Selected cost range and model |
| Range summary | calls/audio/tokens/cost and averages | Selected cost range and model |
| Month projection | month-to-date average extrapolated across month days | Month containing range start |
| Recent calls | Last 20 rows, newest ID first | Selected day and model |

Language counts count a row once for each recognized language name present in its
comma-separated `language` field. The percentage denominator is the sum of those
language occurrences, so a multilingual call contributes to multiple counts.

Default cost range behavior:

- No start/end: entire current calendar month.
- Start only: that single day.
- End only: first day of end month through end date.
- Both: inclusive range; reversed inputs are swapped.

Month-end projection is:

```text
month_to_date_cost / elapsed_days * days_in_month
```

For a completed past month, elapsed days equals all days in that month. For a
future month, projection is zero.

## 18 CSV and API Export Routes {#csv-api-export-routes}

All listed data/transcript routes require a valid dashboard login session.

| Route | Method | Query/path inputs | Response |
|---|---|---|---|
| `/login` | GET, POST | form `username`, `password`, `next` | Login form or redirect |
| `/logout` | GET | None | Clears session and redirects to login |
| `/` | GET | None | Dashboard HTML |
| `/api/data` | GET | `model`, `date`, `start_date`, `end_date` | Dashboard JSON payload |
| `/api/transcripts/<int:call_id>` | GET | MySQL row ID | UTF-8 plain transcript text |
| `/api/transcripts/<int:call_id>/download` | GET | MySQL row ID | TXT attachment |
| `/api/daily-cost.csv` | GET | `model`, `start_date`, `end_date` | `slt_daily_cost.csv` |
| `/api/monthly-cost.csv` | GET | `model`, optional `start_date`, `end_date` | `slt_monthly_cost.csv` |

The daily CSV columns are:

```text
Row Type, Month, Date, Calls, Audio (min), Tokens,
Estimated Cost (USD), Estimated Cost (LKR),
Avg / Call (LKR), Avg / Audio Min (LKR)
```

It emits `day` rows, a `month_subtotal` after each month, and one `grand_total`.

The monthly CSV columns are:

```text
Month, Calls, Audio (min), Tokens, Estimated Cost (USD),
Estimated Cost (LKR), Avg per Call (LKR)
```

Without range parameters, monthly CSV uses up to 12 months of history. With
either range parameter, it uses monthly subtotals derived from the selected daily
range. The current UI's monthly download link sends only the active model; the
route itself supports range parameters.

Transcript route safety:

- The row is first looked up in MySQL.
- The stored path must resolve inside configured `TRANSCRIPTIONS_DIR`.
- The target must be an existing file.
- View returns 404 for a missing row, unsafe/missing path, or empty transcript.
- Download sends the actual TXT filename when the file exists.

## 19 How to Run the Watcher {#run-watcher}

### One-time local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally without printing or committing credentials. Install FFmpeg
separately and confirm both executables:

```powershell
ffmpeg -version
ffprobe -version
```

### No-call validation

```powershell
python -m compileall -q watcher.py dashboard_server.py config.py database.py gemini_flash_stt.py transcript_storage.py
python -c "import config, database, gemini_flash_stt, transcript_storage, watcher, dashboard_server; print('ACTIVE_IMPORTS=PASS')"
python watcher.py --help
python watcher.py --dry-run
python -m pytest
python -m pip check
pip-audit -r requirements.txt
bandit -r watcher.py dashboard_server.py config.py database.py gemini_flash_stt.py transcript_storage.py
git diff --check
git status -sb
```

`--dry-run` creates/checks folders, configures logs, reports startup settings,
and exits without processing audio or calling Gemini.
Compilation, imports, the current mocked/synthetic test suite, dependency checks,
Bandit, and Git diff checks also make no Gemini request. `pytest`, `pip-audit`,
and `bandit` are validation tools and are not runtime dependencies in
`requirements.txt`; they must exist in the validation environment.

A passing run means no known issues in the tested scope, not that the system is
error-free or vulnerability-free. Starting the watcher with queued audio,
dropping supported audio into incoming, or invoking the standalone transcription
CLI can make paid requests and must not be used as routine validation without
explicit approval.

Secret scanning should inspect tracked plus untracked non-ignored files for
private-key markers and known API-token signatures while reporting filenames or
counts only. It must not print matching secret values or inspect ignored private
runtime data merely to display it.

If MySQL is enabled, validate it without a Gemini call:

```powershell
python -c "import database; print('enabled:', database.is_database_enabled()); database.init_database(); print('database init ok')"
```

Do not print `MYSQL_PASSWORD`.

### Safe functional audio

Use short, approved call-like speech in Sinhala, English, Tamil, or a mixture.
Synthetic or locally recorded non-customer speech is preferred. Songs, music,
karaoke, and real/private calls are unsuitable validation inputs. If duplicate
prevention blocks an identical test recording, create a new recording rather
than repeatedly requeueing the same file.

### Start and stop

```powershell
python watcher.py
```

Optional incoming-folder override:

```powershell
python watcher.py --input input_audio/incoming
```

Drop an approved test audio file into `input_audio/incoming/`. Stop with
`Ctrl+C`. The signal handler stops observing, lets active work finish, and then
joins the worker. Queued but unstarted files remain in incoming.

Running multiple watcher processes against the same folders is not recommended
because the repository implements no cross-process claim/lock protocol.
The worker catches unexpected per-file exceptions and continues with later queue
entries. Failed-audio moves and error-note writes are best-effort secondary steps.

## 20 How to Run the Dashboard {#run-dashboard}

Create the private user file first:

```powershell
Copy-Item users.example.json users.json
```

Edit `users.json` with local credentials, then run:

```powershell
python dashboard_server.py
```

Open:

```text
http://127.0.0.1:5050
```

Custom local port:

```powershell
python dashboard_server.py --host 127.0.0.1 --port 8080
```

The dashboard can start without MySQL, but metrics will be empty. It can also
start without `users.json`, but no user can log in.

Do not bind to a non-loopback interface unless external HTTPS, reverse-proxy or
VPN access control, firewall policy, and operational hardening have been provided.
Those controls are `Not confirmed from repository`.

## 21 How to Add or Change Dashboard Users {#manage-dashboard-users}

There is no current user-management command. Edit the local JSON file directly.

### Create the local file

```powershell
Copy-Item users.example.json users.json
```

### Supported operations

| Operation | Action in `users.json` |
|---|---|
| Add user | Add another object to the top-level array |
| Change password | Replace that user's plaintext `password` value |
| Change role | Edit `role`; current code stores it but does not enforce route differences |
| Remove user | Remove the object from the array |

Validate JSON shape without displaying passwords:

```powershell
python -c "import json; from pathlib import Path; data=json.loads(Path('users.json').read_text(encoding='utf-8')); print('valid user objects:', sum(isinstance(x, dict) for x in data) if isinstance(data, list) else 0)"
```

After changing/removing a user, clear existing browser sessions with `/logout`.
Editing the file alone does not revoke already authenticated sessions.

> **Warning:** `users.json` is local/private and plaintext. Never commit it. Do
> not use a removed hashing script or put hashes into `password`; current code
> compares the submitted password to the stored string exactly.

## 22 How to Use Local Files Safely {#local-files-safely}

| Local artifact | Safe handling rule |
|---|---|
| `.env` | Keep local; never commit or paste wholesale |
| `credentials/*.json` | Restrict access; never print, copy into docs, or commit |
| `users.json` | Restrict access; contains plaintext passwords; never commit |
| `.dashboard_secret` | Treat as a session-signing secret; never commit |
| `input_audio/**` | Treat as private call data; use approved test audio only |
| `transcriptions/**` | Treat TXT and JSON as private customer/operational data |
| `logs/**` | Review before sharing; logs include filenames, paths, timing, and errors |
| `*.db`, dumps, backups | Treat as private operational data |
| experiment/backup folders | Do not treat ignored local copies as current source |

Use file paths in commands rather than showing sensitive contents. For example,
test existence without reading a credential:

```powershell
Test-Path credentials/google-credentials.json
```

Check ignore rules before creating local data:

```powershell
git check-ignore -v .env users.json .dashboard_secret credentials/google-credentials.json
```

Metadata JSON can include source and runtime paths. Do not publish a sidecar
without reviewing and redacting operational path information and call metadata.

## 23 Security and Git Hygiene {#security-git-hygiene}

The current `.gitignore` excludes:

- `.env` and `.env.*`, while allowing `.env.example`.
- `.dashboard_secret` and `users.json`.
- Runtime logs and `*.log`.
- Transcript/output contents and generated transcript JSON/TXT patterns.
- Audio formats under runtime folders and repository-wide audio suffixes.
- Credential directory contents and common credential/private-key patterns.
- Archives such as ZIP/RAR.
- Local databases, SQLite files, dumps, and backups.
- Local backup and experiment directories.
- Virtual environments and Python caches.

Operational rules:

1. Keep only placeholder `.gitkeep` files in tracked runtime/private folders.
2. Use a dedicated non-root MySQL account.
3. Use least-privilege Google Cloud IAM and rotate exposed keys.
4. Keep the dashboard on loopback unless hardened externally.
5. Treat transcripts and audio as sensitive customer data.
6. Review `git diff --cached` before every commit.
7. Never commit generated cost reports if they contain customer or operational
   metadata unless a separate approved policy allows it.

Safe checks:

```powershell
git status --short
git diff --cached --name-only
git ls-files .env users.json .dashboard_secret credentials
```

The watcher redacts configured MySQL passwords, credential paths, API keys, and
project IDs from handled exception messages where possible. This is defense in
depth, not a guarantee that every external library message or local path is safe
to publish.

### Tested scope and accepted limitations

- Passing static, dependency, and mocked tests means no known issues in the
  tested scope; it is not an error-free or vulnerability-free guarantee.
- Live browser, MySQL server, FFmpeg conversion, and Gemini integration are not
  confirmed unless explicitly exercised in the current environment.
- `users.json` is plaintext by design. There is no built-in HTTPS, login
  throttling, account lockout, explicit CSRF token, or role-based route
  authorization.
- Editing/removing a user does not revoke an existing signed session.
- Multiple watchers sharing folders have no cross-process claim/lock protocol.
- A production Windows service wrapper, reverse proxy, network topology,
  backup/retention policy, and disaster recovery process are `Not confirmed from
  repository`.
- Cost/pricing/rate values are application estimates, not invoice guarantees.
- Crash-stale reservations can conservatively count until reconciliation or the
  next app-local day.

## 24 Troubleshooting Guide {#troubleshooting}

### Missing credentials or project

Symptoms include `GOOGLE_APPLICATION_CREDENTIALS is not set`, credentials file
not found, or `GOOGLE_CLOUD_PROJECT is not set`.

```powershell
Test-Path .env
Test-Path credentials/google-credentials.json
python watcher.py --dry-run
```

Confirm `.env` points to the local credential filename and project ID. Do not
print the JSON. Note that `--dry-run` intentionally avoids Gemini setup validation;
the missing credential may appear only when a real transcription begins.

### Vertex AI permission, quota, region, or API error

Check project ID, `STT_GEMINI_LOCATION`, API enablement, IAM, billing, quotas,
organization policy, and network access. Exact required deployment IAM and quota
are `Not confirmed from repository`. The module retries transient API exceptions
three total times, then the watcher moves audio to failed.

### FFmpeg or ffprobe missing

```powershell
ffmpeg -version
ffprobe -version
```

Install FFmpeg and ensure its `bin` directory is on the process `PATH`. FFmpeg is
required for transcription conversion; ffprobe is also used by cost preflight.

### File remains in incoming

The copy may still be changing, may be empty, or may not use a supported suffix.
Review the current daily watcher log. Increase stability settings only when slow
copy behavior requires it. The observer does not recurse into subdirectories.

### Reused test file is treated as duplicate

When MySQL is enabled, completed rows are checked against the available SHA-256
hash and original filename. A detected duplicate moves to completed without
Gemini. Re-record the approved phrase to create a new test sample instead of
reusing the identical recording.

### Missing TXT/JSON output files

Check in this order:

1. Confirm the audio left incoming and entered processing.
2. Review `logs/watcher_YYYY-MM-DD.log`.
3. Check `input_audio/failed/YYYY-MM-DD/` for audio plus `.error.txt`.
4. Check `input_audio/deferred/YYYY-MM-DD/` for audio plus `.deferred.txt`.
5. Confirm `TRANSCRIPTIONS_DIR` and process-environment overrides.
6. Confirm the service account, project, FFmpeg, disk space, and directory write
   permissions.

MySQL is not required for TXT/JSON. A DB outage alone should not explain missing
primary outputs after a successful Gemini response.

### Failed transcription

Open the matching `.error.txt` note and watcher log. Common repository-visible
causes are missing input, unstable/empty input, FFmpeg conversion failure,
credential validation failure, or Gemini/API failure after retries. Requeue only
after correcting the cause; avoid repeated paid requests for the same audio.

### Empty transcript with completed audio

Silent/no-speech audio or an empty model response can produce an empty TXT while
the watcher still marks the pipeline completed. Inspect duration, silence-removal
metadata, audio content, and the JSON usage fields before deciding whether an
approved retry is appropriate.

### Song or music gives a short transcript

This is outside the intended task. The prompt targets telecom conversations,
not lyrics; music, overlapping vocals, reverb, and silence filtering can yield
partial or empty text. Retest with a short approved call-like voice recording.

### MySQL connection or initialization issue

```powershell
python -c "import database; print('enabled:', database.is_database_enabled()); database.init_database(); print('database init ok')"
```

Check server availability, host, integer port, database, dedicated user grants,
password, timeout, and firewall. Existing PowerShell environment variables can
override `.env`. Never print the password. With daily safety disabled, primary
file outputs continue even if MySQL metadata fails.

### Dashboard empty

Verify `DB_ENABLED=true`, `DB_BACKEND=mysql`, MySQL availability, and that the
`transcriptions` table contains rows. The dashboard does not scan filesystem JSON
when DB data is absent.

### Missing users.json

```powershell
Copy-Item users.example.json users.json
```

Edit the copy locally. Keep it ignored and private.

### Dashboard login failure

Check that `users.json` is valid UTF-8 JSON with a top-level array, that the entry
has non-empty string `username` and `password` fields, that username
case-insensitively matches, and that the submitted password exactly matches the
plaintext value. Invalid entries such as `{}` are ignored. A role value does not
grant login by itself.

### Transcript view/download returns 404

Confirm MySQL has the requested row, `transcript_txt_path` is populated, the TXT
exists, and the resolved path remains under `TRANSCRIPTIONS_DIR`. An empty TXT has
no viewable text and the view route returns 404.

### Cost-limit deferral

Read the matching `.deferred.txt` note. Check current MySQL daily estimated LKR,
the configured limit, warning level, preflight estimate, and accounting
reservation availability.
Move the audio back to incoming on a later day or after an intentional limit/policy
change. The deferred file was not sent to Gemini.

### Cost safety says DB unavailable

This is expected fail-closed behavior regardless of the compatibility policy.
Restore MySQL before requeueing. File-only mode and a fail-closed daily DB-backed
budget cannot both provide verified daily usage.

### Wrong date boundary or folder date

Check `APP_TIMEZONE`. Invalid non-Colombo zones fall back to UTC with a warning;
`Asia/Colombo` has a fixed UTC+05:30 fallback when zone data is unavailable.

## 25 AI Assistant Operating Rules {#ai-assistant-operating-rules}

When assisting an operator or developer:

1. Identify whether the question concerns active realtime code, optional current
   components, compatibility wrappers, or archives.
2. Prefer `watcher.py`, `gemini_flash_stt.py`, `config.py`, `database.py`, and
   `dashboard_server.py` over archived prose.
3. State that intended input is call-like speech, not songs, music, karaoke, or
   lyric extraction.
4. State that TXT/JSON are primary and MySQL is optional metadata, except that an
   enabled fail-closed daily budget depends on MySQL before paid calls.
5. Never claim dashboard metrics come from JSON sidecars; they come from MySQL.
6. Never claim full transcripts are stored in MySQL.
7. Never call cost estimates exact billing.
8. Never expose or request credential JSON, `.env`, plaintext user passwords,
   `.dashboard_secret`, database passwords, real audio, or transcript contents.
9. Never recommend the removed hashed-password workflow.
10. Never recommend archived batch or Linux commands as active deployment steps.
11. Use `Not confirmed from repository` instead of guessing cloud IAM, production
    infrastructure, retention, or external pricing.
12. During validation, do not suggest or run a paid Gemini call unless the user
    explicitly requests it. `--help`,
    `watcher.py --dry-run`, syntax compilation, and DB initialization do not call
    Gemini.
13. Describe a clean validation as `no known issues in the tested scope`, never
    as proof that the project has no errors or vulnerabilities.
14. Preserve local/private files and avoid commands that print secrets.

## 26 Glossary {#glossary}

| Term | Meaning in this repository |
|---|---|
| Active workflow | Sequential realtime folder processing started by `python watcher.py` |
| ADC | Google application credentials selected through the local credential path |
| API surface | `vertex_ai` metadata value for the Google Gen AI client configuration |
| Archived mode | Dashboard label for rows whose DB `mode` is `batch`; batch execution itself is archived |
| Billed output tokens | Response output tokens plus thoughts tokens |
| Call-like audio | Spoken telecom/conversation input; the intended transcription content |
| Completed | Watcher terminal archive for successfully handled or duplicate audio |
| Deferred | Audio blocked before Gemini by daily cost safety |
| Estimated cost | Application calculation from configured token prices and exchange rate |
| File-only mode | `DB_ENABLED=false`; TXT/JSON work without MySQL dashboard metadata |
| Incoming | Watched directory for new audio |
| Metadata sidecar | JSON file beside a TXT transcript |
| Processing | Directory used after a file is claimed by the watcher |
| Reserved row | Conservative pre-Gemini MySQL accounting row used by the daily guardrail |
| Realtime | Active synchronous per-file Gemini request mode |
| RAG | Retrieval-augmented generation; this document is formatted for chunk retrieval |
| Stable file | Non-empty file whose size/mtime and age satisfy the stability window |
| Thoughts tokens | Gemini reasoning tokens counted by this code as billed output |
| Transcript path | Filesystem path stored in JSON/MySQL; not transcript text in MySQL |

## 27 Source File Index {#source-file-index}

| Source | Authority for |
|---|---|
| `watcher.py` | Active file lifecycle, queueing, stability, duplicates, deferral, DB ordering, logs, CLI |
| `gemini_flash_stt.py` | Audio conversion, prompt, API retries, token/cost formula, exchange rate, TXT/JSON schema |
| `config.py` | `.env` precedence, path resolution, timezone, cost-safety defaults |
| `database.py` | MySQL enablement, schema, inserts, duplicate checks, daily budget status, dashboard metrics |
| `dashboard_server.py` | Login/session behavior, UI metrics, API/CSV/transcript routes, host/port CLI |
| `transcript_storage.py` | Compatibility output wrappers |
| `.env.example` | Safe example deployment variables; not code-default authority where values differ |
| `.gitignore` | Secret and runtime artifact exclusions |
| `users.example.json` | Current dashboard-user JSON template |
| `requirements.txt` | Installed Python package set |
| `tests/test_security_reliability.py` | No-call authentication, rendering, watcher resilience, and cost-safety regression tests |
| `static/app-logo.png` | Current login/header static image |
| `README.md` | Human setup narrative and active/archive statements |
| `docs/archive/batch/README.md` | Explicit batch archive warning |
| `docs/archive/batch/batch_processor.py` | Historical GCS/Vertex batch implementation; not active |
| `docs/archive/linux/README.md` | Explicit Linux archive warning |
| `docs/archive/linux/slt-watcher.service` | Historical systemd watcher unit; not active |
| `docs/archive/linux/slt-dashboard.service` | Historical systemd dashboard unit; not active |
| `docs/archive/legacy/README.md` | Explicit legacy archive warning |
| `docs/archive/legacy/how_it_works.html` | Historical SQLite/old-output explanation; not current |
| `docs/archive/legacy/use_from_another_system.py` | Historical direct-call example that prints transcript text |
| `credentials/.gitkeep` | Credential directory placeholder only |
| `input_audio/**/.gitkeep` | Runtime folder placeholders only |
| `logs/.gitkeep` | Log folder placeholder only |
| `transcriptions/.gitkeep` | Transcript folder placeholder only |

## 28 AI Assistant Rules {#ai-assistant-rules}

This is the final answer policy for a RAG assistant using this knowledge base:

- Say the active command is `python watcher.py`.
- Say intended input is call-like Sinhala/English/Tamil speech; songs, music,
  karaoke, and lyric extraction are outside scope and are poor quality tests.
- Describe the active lifecycle as incoming -> processing -> Gemini/Vertex ->
  TXT/JSON -> optional MySQL -> completed, with failed and deferred branches.
- Say `users.example.json` is the committed template and `users.json` is the
  local private plaintext store that must not be committed.
- Do not describe any hashed-password manager as current.
- Say dashboard data and transcript record IDs depend on MySQL; transcript text
  itself is read from TXT.
- Say MySQL failure after output saving must not cause another Gemini call.
- Explain the fail-closed daily-cost exception when MySQL usage cannot be checked.
- Label cost as estimated, identify hardcoded pricing and exchange-rate behavior,
  and do not equate it with an invoice.
- Mark batch, Linux systemd, SQLite, old output paths, and legacy examples as
  archived/non-active.
- Do not suggest a paid Gemini call during validation unless the user explicitly
  asks for one; prefer compile, import, help, dry-run, mocked tests, dependency,
  Bandit, diff, and secret checks.
- Never reveal secrets or private call content, and never ask the user to paste
  them.
- Say successful validation found `no known issues in the tested scope`; do not
  claim the project is error-free or vulnerability-free.
- When evidence is absent or conflicting, answer `Not confirmed from repository`
  and identify the source file that would need to change or be inspected.
