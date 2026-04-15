# Telecom Voice to Text

Minimal standalone voice-to-text project for telecom call transcription using the **Google Gen AI SDK with Gemini on Vertex AI**.

This project is designed to:

- run as a simple CLI tool
- be imported into another Python system as a function
- keep the setup small and Windows-friendly

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

- converts input audio to 16 kHz mono WAV using `ffmpeg`
- uses the Google Gen AI SDK with Vertex AI
- sends the audio to a Gemini model
- returns the transcription text
- can save the transcript to a text file

---

## Prerequisites

You need:

- Python installed on Windows
- a Google Cloud project
- billing enabled for that Google Cloud project
- the Vertex AI API enabled
- `ffmpeg` installed and available on PATH

For this repository’s current implementation, also use a **Google service account key JSON file**, because the local `.env` configuration points `GOOGLE_APPLICATION_CREDENTIALS` to that file.

A virtual environment is **recommended**, but **not strictly required**. This repository can run without a virtual environment as long as the required packages are already installed in your current Python environment. A virtual environment is safer because it keeps this project’s dependencies isolated from other Python projects on your machine.

---

## Step 1 — Clone the repository

Open PowerShell:

```powershell
cd E:\Work
git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git
cd Telecom-Voice-to-Text
```

---

## Step 2 — Create and activate a virtual environment (recommended)

In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Note

Using a virtual environment is recommended, but not strictly required.

You can run this project without a virtual environment if the required packages are already installed in your current Python environment. However, using a virtual environment is safer because it keeps this project’s dependencies isolated from other Python projects on your machine.

---

## Step 3 — Install Python dependencies

Your `requirements.txt` should contain:

```text
google-genai
```

Then install:

```powershell
pip install -r requirements.txt
```

### Optional — Run without a virtual environment

If you do not want to use a virtual environment, you can install the dependency directly into your current Python environment:

```powershell
pip install -r requirements.txt
```

Then run the script normally:

```powershell
python .\gemini_flash_stt.py .\input_audio\sample_call.mp3 --save
```

Note:
Using a virtual environment is still recommended to avoid package conflicts with other Python projects.

---

## Step 4 — Install FFmpeg on Windows

This project needs `ffmpeg` to be callable from PowerShell.

### Option A — Manual install

1. Go to the official FFmpeg download page (<https://www.ffmpeg.org/download.html>).
2. Open one of the Windows builds links.
3. Download a Windows build.
4. Extract it to a stable folder, for example:

```text
C:\ffmpeg
```

5. Find the `bin` folder inside it. It will usually look like:

```text
C:\ffmpeg\bin
```

6. Add that `bin` folder to your **User PATH**:
   - open **Edit the system environment variables**
   - click **Environment Variables**
   - under **User variables**, select `Path`
   - click **Edit**
   - click **New**
   - add:

```text
C:\ffmpeg\bin
```

7. Close and reopen PowerShell.

### Option B — Installer or package manager

If the Windows build provider you choose offers an installer executable, you can use that instead of extracting a ZIP manually.

No matter which method you use, the important thing is that this command works:

```powershell
ffmpeg -version
```

If it prints version information, FFmpeg is installed correctly.

---

## Step 5 — Set up Google Cloud / Vertex AI

Make sure your Google Cloud project is ready:

- create or select a Google Cloud project
- enable billing
- enable the Vertex AI API

For this repository’s current setup, place your service account key JSON file here:

```text
credentials/google-credentials.json
```

### Note about authentication

Google Cloud also supports other authentication methods such as Application Default Credentials (ADC). This repository currently uses the **service-account-file approach** through `GOOGLE_APPLICATION_CREDENTIALS`.

---

## Step 6 — Create the `.env` file

Create a file named `.env` in the project root:

```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
STT_GEMINI_LOCATION=us-central1
```

### Important notes

- runtime uses `.env`
- `.env.example` is only a template/reference file and is not loaded at runtime
- `STT_GEMINI_LOCATION` is this project’s own config variable used by the Python script
- `STT_GEMINI_LOCATION` is not the official Google-standard environment variable name shown in Google Gen AI SDK quickstarts

If your selected region does not support the model you want, or your project does not have access or quota there, change the location to a supported Vertex AI region.

---

## Step 7 — Put a test audio file in `input_audio`

Example:

```text
input_audio/sample_call.mp3
```

---

## Step 8 — Run the transcription script

Basic run:

```powershell
python .\gemini_flash_stt.py .\input_audio\sample_call.mp3
```

Run and save transcript to `output\`:

```powershell
python .\gemini_flash_stt.py .\input_audio\sample_call.mp3 --save
```

---

## Changing the model

By default, this project uses the model defined in `gemini_flash_stt.py`.

Look for this line:

```python
MODEL_NAME = "gemini-2.5-flash"
```

If you want to use a different supported Gemini model, change that value.

For example:

```python
MODEL_NAME = "gemini-2.5-pro"
```

### Important notes when changing the model

- make sure the model you choose is supported on Vertex AI
- make sure the selected model supports your intended input type
- make sure your selected region in `.env` has access to that model
- if the model is not available in your current region, change `STT_GEMINI_LOCATION`

Your `.env` file still stays the same unless you also want to change the region:

```env
GOOGLE_APPLICATION_CREDENTIALS=credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
STT_GEMINI_LOCATION=us-central1
```

If you change the model and it does not work, check:

- model availability in your chosen Vertex AI region
- project access or quota
- authentication and permissions

---

## Step 9 — Use the code from another Python system

Example file: `examples/use_from_another_system.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemini_flash_stt import transcribe_audio_file


result = transcribe_audio_file("input_audio/sample_call.mp3")

print("Success:", result["success"])
print("Model:", result["model"])
print("Audio path:", result["audio_path"])
print("Duration:", result["duration_seconds"])
print("Elapsed:", result["elapsed_seconds"])
print("Transcript:")
print(result["transcript"])
```

Run it with:

```powershell
python .\examples\use_from_another_system.py
```

---

## Expected return value

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

---

## Recommended `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
.env
credentials/
output/
input_audio/*
!input_audio/.gitkeep
```

Then create the placeholder file once:

```powershell
New-Item .\input_audio\.gitkeep -ItemType File
```

---

## First-time setup checklist

Before running the script, confirm:

- [ ] virtual environment created, or required packages installed in the current Python environment
- [ ] virtual environment activated, if using one
- [ ] `pip install -r requirements.txt` completed
- [ ] `ffmpeg -version` works
- [ ] `credentials/google-credentials.json` exists
- [ ] `.env` exists
- [ ] `GOOGLE_CLOUD_PROJECT` is correct
- [ ] test audio file exists in `input_audio\`

---

## Common problems

### `ffmpeg is not installed or not available on PATH`

Fix:

- install FFmpeg
- add its `bin` folder to PATH
- close and reopen PowerShell
- test again with:

```powershell
ffmpeg -version
```

### `Credentials file not found`

Fix:

- confirm this file exists:

```text
credentials/google-credentials.json
```

- confirm `.env` points to the correct relative path

### `GOOGLE_CLOUD_PROJECT is not set`

Fix:

- open `.env`
- make sure this line exists:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
```

### Authentication or permission errors

Fix:

- verify the service account key belongs to the correct Google Cloud project
- verify the Vertex AI API is enabled
- verify billing is enabled
- verify the service account has permission to use Vertex AI

### Region or model access issues

Fix:

- verify the selected region supports your intended model
- verify your project has access and quota in that region
- change `STT_GEMINI_LOCATION` if needed

### PowerShell execution policy blocks venv activation

Fix:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Example setup flow on Windows

### With virtual environment

```powershell
cd E:\Work
git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git
cd Telecom-Voice-to-Text

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

ffmpeg -version

python .\gemini_flash_stt.py .\input_audio\sample_call.mp3 --save
```

### Without virtual environment

```powershell
cd E:\Work
git clone https://github.com/Kaveen98/Telecom-Voice-to-Text.git
cd Telecom-Voice-to-Text

python -m pip install --upgrade pip
pip install -r requirements.txt

ffmpeg -version

python .\gemini_flash_stt.py .\input_audio\sample_call.mp3 --save
```

---

## Notes

- this project currently uses a Gemini model on Vertex AI.
- the code is intentionally minimal.
- `ffmpeg` is a system dependency, not a Python package dependency.
- the project can run with or without a virtual environment, but using one is recommended.
- the project is designed so the transcription module can later be wrapped by FastAPI or another backend.

---
