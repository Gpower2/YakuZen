# YakuZen

YakuZen is a desktop subtitle-generation app for Japanese anime video files. It takes local episode files, isolates the vocal track, transcribes Japanese dialogue, generates phonetic romaji, translates the lines to English with a local Ollama model, and exports multiple `.srt` files beside the source video.

## What the app does

- Lets you pick a folder of video files (`.mkv`, `.mp4`, `.avi`, `.mov`) in a CustomTkinter UI
- Queues selected files and shows per-file plus total progress
- Produces Japanese, romaji, and English subtitle artifacts
- Supports skipping the current file or cancelling the entire queue

## How it works

1. `src\app.py` is the GUI and orchestrator. It scans the chosen folder, queues selected videos, and runs the worker scripts in background subprocesses. It parses `tqdm` output from those subprocesses to drive the GUI progress bars.
2. `src\process_audio.py` handles the audio and transcription stage:
   - Reuses an existing `<base>.json` cache if present
   - Separates the vocal stem with `audio_separator`
   - Normalizes the stem with FFmpeg loudness normalization and resamples it to 16 kHz
   - Transcribes Japanese speech with `faster-whisper` `large-v3` on CUDA
   - Runs a second-pass timing refinement with `stable-ts align_words()` against a cached mono 16 kHz extract of the original episode mix to tighten subtitle boundaries without changing the subtitle count
   - Converts the Japanese transcript to phonetic romaji with `cutlet`
   - Writes `<base>.json`, `<base>.kanji.srt`, `<base>.romaji.srt`, and `<base>_debug_raw.json`
3. `src\translate_subs.py` handles translation and final export:
   - Reuses `<base>_translated.json` if present
   - Sends subtitle batches to Ollama at `http://localhost:11434/api/generate`
   - Expects a 1:1 JSON array of English strings and falls back to single-line translation when batch output is malformed
   - Writes `<base>_translated.json`, `<base>.jp.srt`, `<base>.romaji.srt`, `<base>.en.raw.srt`, and `<base>.en.srt`
4. `src\check_gpu.py` is a diagnostics script that checks PyTorch CUDA support, ONNX Runtime providers, and `faster-whisper` GPU loading.

The English export has two forms: `.en.raw.srt` keeps the original segment timing 1:1, while `.en.srt` wraps long lines and proportionally splits oversized subtitles into multiple timed chunks.

## Dependencies

The repository currently does not include `requirements.txt` or `pyproject.toml`, so the dependency list below is inferred from the source code.

### Python packages

- `customtkinter`
- `requests`
- `tqdm`
- `cutlet`
- `faster-whisper`
- `stable-ts`
- `audio-separator`
- `torch`
- `onnxruntime-gpu`
- `unidic-lite`

### External tools and services

- `ffmpeg` available on `PATH`
- Ollama running locally on `http://localhost:11434`
- A local Ollama model downloaded in advance; `translate_subs.py` currently defaults to `qwen3:14b`
- `audio_separator` model assets cached/downloaded under `.\temp`

### Runtime assumptions

- A CUDA-capable GPU is effectively part of the current pipeline because `process_audio.py` hard-codes `device="cuda"` and `compute_type="float16"`
- `tkinter` is used for the native file dialogs and message boxes
- The GUI should be launched from `src\` because it starts `process_audio.py` and `translate_subs.py` by bare filename
- Existing subtitle JSON caches without the current timing-refinement version are treated as upgradeable transcripts: the app can reuse the cached text and rerun only the timing-refinement stage

## Running the app

```powershell
Set-Location src
python app.py
```

To check the local GPU stack before running the full pipeline:

```powershell
Set-Location src
python check_gpu.py
```
