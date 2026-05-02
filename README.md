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
   - Runs a second-pass timing refinement with `stable-ts align_words()` against a cached mono 16 kHz extract of the original episode mix
   - Runs an extra rescue pass for suspiciously long sparse cues with torchaudio MMS forced alignment plus `pykakasi` romanization to recover late-starting dialogue that stable-ts still anchors too early
   - Converts the Japanese transcript to phonetic romaji with `pykakasi`, including cache-only romaji refreshes for older outputs
   - Writes `<base>.json`, `<base>.kanji.srt`, `<base>.romaji.srt`, and `<base>_debug_raw.json`
3. `src\translate_subs.py` handles translation and final export:
   - Reuses `<base>_translated.json` if present
   - Infers the anime series title from the input filename and feeds it to the translator so character, organization, and place names have immediate context
   - Uses Ollama at `http://localhost:11434/api/generate`; the default model is `qwen3:14b`, while `translategemma:12b` remains available as an optional translation-specialist alternative
   - Uses context-aware batch prompting for general chat models, but switches to direct subtitle-style prompts for translation-specialist models like TranslateGemma
   - Repairs fragmentary multi-cue translations by retranslating the combined Japanese window once, splitting that natural English sentence back across the original cues, and storing a merged viewer-facing line for export
   - Post-processes the final `.en.srt` so oversized display cues are split back into timed subtitle chunks of at most two lines, preferring punctuation boundaries such as `!`, `?`, and sentence breaks when possible
   - Writes `<base>_translated.json`, `<base>.jp.srt`, `<base>.romaji.srt`, `<base>.en.raw.srt`, and `<base>.en.srt`
4. `src\check_gpu.py` is a diagnostics script that checks PyTorch CUDA support, ONNX Runtime providers, and `faster-whisper` GPU loading.

The English export has two forms: `.en.raw.srt` keeps the original segment timing 1:1 for debugging, while `.en.srt` merges repaired fragment windows into viewer-facing display cues and then splits oversized English subtitles back into timed chunks capped at two on-screen lines.

## Dependencies

Python dependencies are declared in `pyproject.toml`.

### Python packages

- `audio-separator`
- `customtkinter`
- `faster-whisper`
- `onnxruntime-gpu` on Windows/Linux, `onnxruntime` on macOS
- `pykakasi`
- `requests`
- `stable-ts[fw]`
- `torch`
- `torchaudio`
- `tqdm`

### External tools and services

- `ffmpeg` and `ffprobe` available on `PATH`
- Ollama running locally on `http://localhost:11434`
- A local Ollama model downloaded in advance; `translate_subs.py` currently defaults to `qwen3:14b`
- `translategemma:12b` is an optional alternative if you want to compare a translation-specialist model
- `audio_separator` model assets cached/downloaded under `.\temp`

## Platform support

| OS | Status | Notes |
|---|---|---|
| Windows | Best-supported path | This is the environment the project has been exercised in most heavily. NVIDIA CUDA is strongly recommended. |
| Linux | Supported | NVIDIA CUDA is strongly recommended. CPU-only runs are possible but slow. |
| macOS | Supported with CPU fallback | `process_audio.py` now falls back to CPU `int8` Whisper when CUDA is unavailable. Expect much slower transcription/alignment than on an NVIDIA machine. |

Across all operating systems, launch the GUI from the `src` directory because `app.py` starts sibling scripts by bare filename.

Existing subtitle JSON caches without the current timing-refinement version are treated as upgradeable transcripts: the app can reuse the cached text and rerun only the timing-refinement stage. Translated caches are stricter: if the inferred series title, default translation model, or translation prompt version changes, `translate_subs.py` will regenerate the English cache instead of silently reusing stale wording.

## Installing dependencies

### Windows

1. Install system dependencies:
   ```powershell
   winget install Gyan.FFmpeg.Essentials
   winget install Ollama.Ollama
   ```
2. Create and activate a virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -e .
   ```
3. If you want GPU acceleration for Whisper and `torch.cuda.is_available()` is still `False`, reinstall PyTorch with a CUDA wheel that matches your system. Example for CUDA 12.8:
   ```powershell
   pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
4. Start Ollama and pull the translation model:
   ```powershell
   ollama pull qwen3:14b
   ollama pull translategemma:12b
   ```

### Linux

1. Install system dependencies. Example for Debian/Ubuntu:
   ```bash
   sudo apt update
   sudo apt install -y ffmpeg python3 python3-venv python3-tk curl
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .
   ```
3. If you want NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build that matches your driver/CUDA stack. Example for CUDA 12.8:
   ```bash
   pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
4. Start Ollama if it is not already running, then pull the model:
   ```bash
   ollama serve
   ollama pull qwen3:14b
   ollama pull translategemma:12b
   ```

### macOS

1. Install system dependencies:
   ```bash
   brew install python@3.12 ffmpeg ollama
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .
   ```
3. Start Ollama and pull the model:
   ```bash
   ollama serve
   ollama pull qwen3:14b
   ollama pull translategemma:12b
   ```
4. Expect the audio pipeline to run on CPU unless you adapt the project to a different backend; the committed code now handles that fallback automatically.

## Running the app

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location src
python app.py
```

### Linux

```bash
source .venv/bin/activate
cd src
python app.py
```

### macOS

```bash
source .venv/bin/activate
cd src
python app.py
```

## Running the worker scripts directly

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location src
python process_audio.py ..\sample\Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv
python translate_subs.py ..\sample\Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.json
python check_gpu.py
```

### Linux / macOS

```bash
source .venv/bin/activate
cd src
python process_audio.py ../sample/Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv
python translate_subs.py ../sample/Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.json
python check_gpu.py
```
