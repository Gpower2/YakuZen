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
   - Separates the vocal stem with `audio_separator`; the separator checkpoint is configurable for advanced users
   - Extracts a mono 16 kHz copy of the original episode mix and now uses that **mixed track** as the default ASR source
   - Keeps the raw separated vocal stem available as an advanced alternative, while the old loudness-normalized stem remains available as a legacy option
   - Transcribes Japanese speech with `faster-whisper` `large-v3` by default on CUDA, with `kotoba-whisper-v1.1` exposed as an advanced alternative and a `hybrid` Whisper+Kotoba rescue mode exposed as an experimental option
   - Runs a second-pass timing refinement with `stable-ts align_words()` against a cached mono 16 kHz extract of the original episode mix
   - Runs an extra rescue pass for suspiciously long sparse cues with torchaudio MMS forced alignment plus `pykakasi` romanization to recover late-starting dialogue that stable-ts still anchors too early
   - Converts the Japanese transcript to phonetic romaji with `pykakasi`, including cache-only romaji refreshes for older outputs
   - Writes `<base>.json`, `<base>.kanji.srt`, `<base>.romaji.srt`, and `<base>_debug_raw.json`
3. `src\translate_subs.py` handles translation and final export:
   - Reuses `<base>_translated.json` if present
   - Infers the anime series title from the input filename and feeds it to the translator so character, organization, and place names have immediate context
   - Uses Ollama at `http://localhost:11434/api/generate`; the default model is `qwen3:14b`, and the translation model is now configurable from both the CLI and desktop app
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
- `accelerate`
- `customtkinter`
- `faster-whisper`
- `onnxruntime-gpu` on Windows/Linux, `onnxruntime` on macOS
- `pykakasi`
- `requests`
- `safetensors`
- `sentencepiece`
- `stable-ts[fw]`
- `torch`
- `torchaudio`
- `transformers`
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

Existing subtitle JSON caches without the current timing-refinement version are treated as upgradeable transcripts: the app can reuse the cached text and rerun only the timing-refinement stage. Full transcript caches are also keyed by the selected transcription source, ASR model, and relevant separator checkpoint, so changing those options intentionally regenerates the Japanese transcript. Translated caches are stricter: if the inferred series title, translation model, or translation prompt version changes, `translate_subs.py` will regenerate the English cache instead of silently reusing stale wording.

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

When you launch the desktop app, the **Advanced Settings** panel lets you override:

- **Transcription source**: `mix` (default), `raw_vocals`, or `normalized_vocals`
- **ASR model**: `large-v3` (default) or `kotoba-whisper-v1.1`
- **Experimental ASR model**: `hybrid`, which runs Whisper first and then lets Kotoba retry suspicious windows
- **Separator model**: defaults to `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
- **Translation model**: defaults to `qwen3:14b`

The shipped defaults are intentionally conservative:

- **ASR source**: `mix`
- **ASR model**: `large-v3`
- **Separator model**: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
- **Translation model**: `qwen3:14b`

That combination gave the best balance of subtitle coverage, timing stability, and low-friction behavior in the bundled sample tests. `kotoba-whisper-v1.1` remains available as an advanced option when you want to experiment with better Japanese wording/proper-noun recovery, but it currently merges cues more aggressively and is therefore not the default.

The new **`hybrid`** mode is available for experimentation when you want Whisper to keep the baseline segmentation while Kotoba tries to improve selected suspicious windows. It is intentionally **not** the default yet: on the bundled sample it improved some proper-noun phrasing, but it can still produce longer merged Japanese cues or partial noisy phrases in rescue windows.

### How hybrid suspicious-window collection works

The hybrid mode does **not** run Kotoba on the entire episode. Instead it:

1. runs the normal `mix + large-v3` baseline first,
2. refines that baseline with the existing timing pass,
3. scans the refined Japanese transcript for windows that look worth retrying,
4. reruns Kotoba only on those windows,
5. and keeps Kotoba only when the replacement passes the merge heuristics.

The suspicious-window detector currently uses two kinds of triggers:

- **Suspicious blocks**: subtitle regions expanded around a seed cue that looks unreliable, such as:
  - obvious ASCII-heavy output inside Japanese text,
  - repeated-noise style text,
  - or the existing long sparse cue heuristic used elsewhere in the pipeline.
- **Long gap windows**: unusually large subtitle gaps that are long enough to suggest a dropped line rather than a natural conversational pause.

Once a window is selected, Kotoba is transcribed only for that slice of audio. The candidate is then filtered before it can replace Whisper:

- it must contain enough Japanese text to look real,
- it must beat the baseline on a simple quality score,
- it cannot collapse too aggressively into an obviously giant segment,
- and close duplicate/subsequence fragments are pruned back out after merging.

In practice, this means the hybrid mode is trying to recover **proper nouns** and **missed windows** without giving Kotoba permission to rewrite the whole episode. It is useful for A/B testing and edge-case recovery, but it still needs more refinement before it is safe as the default.

## Running the worker scripts directly

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location src
python process_audio.py ..\sample\Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --asr-source mix --asr-model large-v3
python process_audio.py ..\sample\Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --asr-source mix --asr-model hybrid
python process_audio.py ..\sample\Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --asr-source mix --asr-model kotoba-whisper-v1.1
python translate_subs.py ..\sample\Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.json --translation-model qwen3:14b
python check_gpu.py
```

### Linux / macOS

```bash
source .venv/bin/activate
cd src
python process_audio.py ../sample/Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --asr-source mix --asr-model large-v3
python process_audio.py ../sample/Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --asr-source mix --asr-model hybrid
python process_audio.py ../sample/Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --asr-source mix --asr-model kotoba-whisper-v1.1
python translate_subs.py ../sample/Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.json --translation-model qwen3:14b
python check_gpu.py
```
