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
