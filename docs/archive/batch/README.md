# Archived Batch Processing

Batch processing is preserved here for future review, but it is not part of
the current Windows Server realtime deployment branch.

The active workflow is `python watcher.py`, which watches
`input_audio/incoming/`, performs realtime Gemini transcription, saves TXT/JSON
outputs first, and then optionally writes MySQL metadata.

Files in this archive may need dependency, configuration, credential, storage,
and database review before reuse. The active branch now focuses on realtime
MySQL metadata, and batch-specific Google Cloud Storage / Vertex AI Batch
dependencies are not part of the primary deployment flow.
