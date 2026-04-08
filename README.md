# Telecom Voice to Text

Minimal standalone voice-to-text project for telecom call transcription using the **Google Gen AI SDK with Gemini 2.5 Flash on Vertex AI**.

This project is designed to:

- run as a simple CLI tool.
- be imported into another Python system as a function.
- keep the setup small and Windows-friendly.

---

## Project structure

```text
Telecom-Voice-to-Text/
│
├─ gemini_flash_stt.py
├─ .env
├─ .env.example
├─ .gitignore
├─ requirements.txt
├─ README.md
│
├─ credentials/
│  └─ google-credentials.json
│
├─ input_audio/
│  ├─ .gitkeep
│  └─ sample_call.mp3
│
├─ output/
│
└─ examples/
   └─ use_from_another_system.py
```

---

## What this project does

- Converts input audio to 16 kHz mono WAV using ffmpeg.
- Uses the Google Gen AI SDK on Vertex AI.
- Sends the audio to Gemini 2.5 Flash.
- Returns the transcription text.
- Can save the transcript to a text file.

## Prerequisites

You need:

- Python installed on Windows.
- A Google Cloud project.
- Billing enabled for that Google Cloud project.
- The Vertex AI API enabled.
- A Google service account key JSON file.
- ffmpeg installed and available on PATH.

---

### Step 1 — Clone the repository

Open PowerShell:

```powershell
cd E:\Work
git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git
cd Telecom-Voice-to-Text
```

### Step 2 — Create and activate a virtual environment

In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**If PowerShell blocks activation, run:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 3 — Install Python dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Install FFmpeg on Windows

Recommended method: manual install

1. Go to the official FFmpeg download page.
2. Open the Windows builds link from there.
3. Download a release build ZIP.
4. Extract it to a stable folder, for example:

```text
C:\ffmpeg
```

5. Find the bin folder inside it. It will look something like:

```text
C:\ffmpeg\bin
```

6. Add that bin folder to your User PATH:

- Open Edit the system environment variables
- Click Environment Variables
- Under User variables, select Path
- Click Edit
- Click New
- Add:

```text
C:\ffmpeg\bin
```

7. Close and reopen PowerShell.

Verify FFmpeg

```powershell
ffmpeg -version
```

If it prints version information, FFmpeg is installed correctly.

### Step 5 — Set up Google Cloud / Vertex AI

Make sure your Google Cloud project is ready:

- create or select a Google Cloud project.
- enable billing.
- enable the Vertex AI API.

Then create or use a service account key JSON file and place it here:

```text
credentials/google-credentials.json
```

### Step 6 — Create the .env file

Create a file named ```.env``` in the project root:

```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
STT_GEMINI_LOCATION=us-central1
```

Runtime uses `.env`.
`.env.example` is only a template/reference file and is not loaded at runtime.
Evidence: `load_env()` in `gemini_flash_stt.py` resolves `base_dir / ".env"` and opens only that file.

### Step 7 — Put a test audio file in input_audio

Example:

```text
input_audio/sample_call.mp3
```

### Step 8 — Run the transcription script

Basic run:

```powershell
python .\gemini_flash_stt.py .\input_audio\sample_call.mp3
```

or

Run and save transcript to output/:

```powershell
python .\gemini_flash_stt.py .\input_audio\sample_call.mp3 --save
```

### Step 9 — Use the code from another Python system

Example file: ```examples/use_from_another_system.py```

```python
from gemini_flash_stt import transcribe_audio_file

result = transcribe_audio_file("input_audio/sample_call.mp3")

print("Success:", result["success"])
print("Model:", result["model"])
print("Duration:", result["duration_seconds"])
print("Elapsed:", result["elapsed_seconds"])
print("Transcript:")
print(result["transcript"])
```

Run it with:

```powershell
python .\examples\use_from_another_system.py
```

#### Expected return value

The reusable function returns a dictionary like this:

```python
{
    "transcript": "...",
    "model": "gemini-2.5-flash",
    "audio_path": "E:\\Work\\Telecom-Voice-to-Text\\input_audio\\sample_call.mp3",
    "duration_seconds": 120.5,
    "elapsed_seconds": 8.3,
    "success": True,
}
```
