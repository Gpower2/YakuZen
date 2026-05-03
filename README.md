# YakuZen

YakuZen is a desktop subtitle-generation app for Japanese anime video files. It takes local episode files, isolates the vocal track, transcribes Japanese dialogue, generates phonetic romaji, translates the lines to English with local Ollama models, and exports multiple `.srt` files beside the source video. It now supports reusable **series context** plus three processing profiles: **Current**, **Balanced**, and **Max Accuracy**.

## What the app does

- Lets you pick a folder of video files (`.mkv`, `.mp4`, `.avi`, `.mov`) in a CustomTkinter UI
- Shows how many files are selected and includes quick **Select All** / **Deselect All** shortcuts
- Queues selected files and shows per-file plus total progress
- Streams worker-script status and output into the UI console while processing runs
- Lets you choose between **Current**, **Balanced**, and **Max Accuracy** processing profiles
- Includes a **Check Settings** button that validates the current UI settings, required tools, Ollama models, separator checkpoint, and optional series-context override before you launch a full run
- Includes a **Check GPU** button that runs `check_gpu.py` and prints CUDA / ONNX / faster-whisper diagnostics into the UI console
- Produces Japanese, romaji, and English subtitle artifacts
- Reuses a global **series context cache** so titles, names, and glossary hints can carry across future runs
- Supports skipping the current file or cancelling the entire queue

## How it works

1. `src\app.py` is the GUI and orchestrator. It scans the chosen folder, queues selected videos, and runs the worker scripts in background subprocesses. It parses `tqdm` output from those subprocesses to drive the GUI progress bars.
2. `src\process_audio.py` handles the audio and transcription stage:
   - Reuses an existing `<base>.json` cache if present
   - Separates the vocal stem with `audio_separator`; the separator checkpoint is configurable for advanced users
   - Extracts a mono 16 kHz copy of the original episode mix and now uses that **mixed track** as the default ASR source
   - Keeps the raw separated vocal stem available as an advanced alternative, while the old loudness-normalized stem remains available as a legacy option
   - In **Current** mode, runs the existing single-path transcription flow (`mix` + configurable ASR model)
   - In **Balanced** and **Max Accuracy** modes, reuses a global `series_context.json` cache, runs `large-v3` on both the mixed track and the raw vocals stem, and resolves suspicious/disagreement windows with targeted rescue logic before the final timing pass
   - Keeps `kotoba-whisper-v1.1` exposed as an advanced alternative in Current mode, while Balanced/Max use targeted rescue/adjudication instead of a single full-episode Kotoba pass
   - Feeds the inferred/cached series context into the default Whisper path as a front-loaded prompt hint plus compact hotwords so recurring titles and proper nouns drift less often on fresh reruns
   - Runs a second-pass timing refinement with `stable-ts align_words()` against a cached mono 16 kHz extract of the original episode mix
   - Runs an extra rescue pass for suspiciously long sparse cues with torchaudio MMS forced alignment plus `pykakasi` romanization to recover late-starting dialogue that stable-ts still anchors too early
   - Converts the Japanese transcript to phonetic romaji with `pykakasi`, including cache-only romaji refreshes for older outputs
   - Writes `<base>.json`, `<base>.kanji.srt`, `<base>.romaji.srt`, and `<base>_debug_raw.json`
3. `src\translate_subs.py` handles translation and final export:
    - Reuses `<base>_translated.json` if present
   - Loads the same inferred/cached series context used by the JP stage so character, organization, place, and series-title references have immediate context
   - Uses Ollama at `http://localhost:11434/api/generate`; the default primary model is `qwen3:14b`, while judge/context/secondary-model roles are configurable from both the CLI and desktop app
   - Uses deterministic Ollama decoding by default so fresh reruns stop drifting between equally valid English phrasings when the Japanese input is unchanged
   - In **Current** mode, runs one deterministic translation model
   - In **Balanced** mode, keeps the primary translator for all lines but uses a judge/normalizer model on flagged lines and glossary-sensitive windows
   - In **Max Accuracy** mode, can generate a second English candidate and use a judge model to pick between primary/secondary outputs before the final consistency pass
   - Uses context-aware batch prompting for general chat models, but switches to direct subtitle-style prompts for translation-specialist models like TranslateGemma
   - Repairs fragmentary multi-cue translations by retranslating the combined Japanese window once, splitting that natural English sentence back across the original cues, and storing a merged viewer-facing line for export
   - Post-processes the final `.en.srt` so oversized display cues are split back into timed subtitle chunks of at most two lines, preferring punctuation boundaries such as `!`, `?`, and sentence breaks when possible
   - Writes `<base>_translated.json`, `<base>.jp.srt`, `<base>.romaji.srt`, `<base>.en.raw.srt`, and `<base>.en.srt`
4. `src\series_context_tool.py` fetches anime metadata, optionally normalizes it with a local Ollama model, and stores a reusable `series_context.json` entry in the global app cache for future runs.
5. `src\check_settings.py` validates the active UI configuration without processing a video: it checks the local Python/runtime dependencies, confirms Ollama is reachable, verifies the selected local model names exist, validates the optional `series_context.json` override, and tries to resolve the selected separator checkpoint.
6. `src\check_gpu.py` is a diagnostics script that checks PyTorch CUDA support, ONNX Runtime providers, and `faster-whisper` GPU loading.

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

- `ffmpeg` and `ffprobe` available on `PATH` (or installed in the standard Windows Winget location that YakuZen now auto-detects)
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

Existing subtitle JSON caches without the current timing-refinement version are treated as upgradeable transcripts: the app can reuse the cached text and rerun only the timing-refinement stage. Full transcript caches are also keyed by the selected transcription source, ASR model, processing profile, judge/context settings, inferred series title, series-context hash, transcription prompt version, and relevant separator checkpoint, so changing those options intentionally regenerates the Japanese transcript. Translated caches are stricter: if the inferred series title, series-context hash, profile, translation-role models, or translation prompt version change, `translate_subs.py` will regenerate the English cache instead of silently reusing stale wording.

## Installing dependencies

> [!IMPORTANT]
> The repository does **not** include a ready-made `.venv` directory. After cloning the repo, create `.venv` locally in the repository root, activate it, and only then install dependencies or run the app. The command examples below assume you start in the repository root.

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
   A fresh Windows `venv` created this way does include `.\.venv\Scripts\Activate.ps1` for **PowerShell**. If you are using **Command Prompt** instead, use `.\.venv\Scripts\activate.bat`.
3. `pyproject.toml` already declares `onnxruntime-gpu` on Windows/Linux, so `pip install -e .` installs the GPU ONNX build automatically. If **Check Settings** still reports both `onnxruntime` and `onnxruntime-gpu`, prefer the GPU build explicitly:
   ```powershell
   pip uninstall -y onnxruntime
   pip install --force-reinstall onnxruntime-gpu
   ```
4. If you want GPU acceleration for Whisper and `torch.cuda.is_available()` is still `False`, reinstall PyTorch with a CUDA wheel that matches your system. Example for CUDA 12.8:
   ```powershell
   pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
5. Start Ollama and pull the translation model:
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
3. `pyproject.toml` already declares `onnxruntime-gpu` on Linux, so `pip install -e .` installs the GPU ONNX build automatically. If **Check Settings** still reports both `onnxruntime` and `onnxruntime-gpu`, prefer the GPU build explicitly:
   ```bash
   pip uninstall -y onnxruntime
   pip install --force-reinstall onnxruntime-gpu
   ```
4. If you want NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build that matches your driver/CUDA stack. Example for CUDA 12.8:
   ```bash
   pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
5. Start Ollama if it is not already running, then pull the model:
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
py -3.12 -m venv .venv   # first time only
.\.venv\Scripts\Activate.ps1
Set-Location src
python app.py
```

The activation line above is the **PowerShell** script created by the Windows `venv` command shown above (`py -3.12 -m venv .venv`). Run it from the repository root. If you use **Command Prompt** instead of PowerShell, use `.\.venv\Scripts\activate.bat`.

### Linux

```bash
python3 -m venv .venv    # first time only
source .venv/bin/activate
cd src
python app.py
```

### macOS

```bash
python3 -m venv .venv    # first time only
source .venv/bin/activate
cd src
python app.py
```

When you launch the desktop app, the **Advanced Settings** panel lets you override:

- **Processing profile**: `current` (fast/compatible), `balanced` (recommended automation), or `max` (slowest/highest accuracy)
- **Transcription source**: `mix` (default), `raw_vocals`, or `normalized_vocals`
- **ASR model**: `large-v3` (default) or `kotoba-whisper-v1.1`
- **Experimental ASR model**: `hybrid`, which runs Whisper first and then lets Kotoba retry suspicious windows
- **Separator model preset**: currently ships with `BS-RoFormer (Default)`, which resolves to `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
- **Custom separator model override**: optional; if filled in, the app passes that checkpoint filename straight through to `process_audio.py`
- **Primary translation model preset**: `qwen3:14b` (default), `qwen3.5:9b`, or `translategemma:12b`
- **Secondary translation model preset**: used mainly by `max` mode for a second candidate
- **Judge / normalizer model**: defaults to `qwen3.5:9b`
- **Series context model**: defaults to `qwen3.5:9b` for cache enrichment when balanced/max need to fetch metadata
- **Series context file**: optional local `series_context.json` override
- **Custom model overrides**: each model-role field can be overridden with any other local Ollama model name

Use **Check Settings** after changing these fields if you want a fast preflight pass before starting a long episode run. It validates the current selections and prints any missing local models, missing tools, invalid paths, separator-load issues, or GPU-provider warnings into the same console area as the main pipeline.

The shipped defaults are intentionally conservative:

- **Processing profile**: `current`
- **ASR source**: `mix`
- **ASR model**: `large-v3`
- **Separator model**: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
- **Primary translation model**: `qwen3:14b`
- **Judge / context model**: `qwen3.5:9b`
- **Secondary translation model**: `translategemma:12b`

That combination keeps the fast path conservative while still exposing the stronger multi-stage modes for users who want more accuracy. `Balanced` is the recommended unattended automation profile; `Max Accuracy` is the slowest offline batch mode.

The default `large-v3` path is also title-aware now: it infers the series title from the filename and uses that as a light Whisper bias (front-loaded prompt + hotwords). In fresh isolated reruns this substantially reduced late-episode title drift on the bundled Blue Noah sample without turning `condition_on_previous_text` back on.

### Global series context cache

YakuZen now maintains a global cache of reusable structured series metadata:

- **Windows**: `%LOCALAPPDATA%\\YakuZen\\series-cache\\`
- **Linux/macOS**: `~/.cache/yakuzen/series-cache/`

Balanced and Max modes load that cache automatically. If an entry is missing, they can fetch public anime metadata, normalize it into `series_context.json`, and save it for future runs. You can also supply your own local `series_context.json` file to override the cached entry for a specific run.

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
py -3.12 -m venv .venv   # first time only
.\.venv\Scripts\Activate.ps1
Set-Location src
python series_context_tool.py ..\sample\Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --context-model qwen3.5:9b
python process_audio.py ..\sample\sample_video.mkv --asr-source mix --asr-model large-v3
python process_audio.py ..\sample\sample_video.mkv --pipeline-mode balanced --judge-model qwen3.5:9b --context-model qwen3.5:9b
python process_audio.py ..\sample\sample_video.mkv --pipeline-mode max --judge-model qwen3.5:9b --context-model qwen3.5:9b
python process_audio.py ..\sample\sample_video.mkv --asr-source mix --asr-model hybrid
python process_audio.py ..\sample\sample_video.mkv --asr-source mix --asr-model kotoba-whisper-v1.1
python translate_subs.py ..\sample\sample_video.json --pipeline-mode current --translation-model qwen3:14b
python translate_subs.py ..\sample\sample_video.json --pipeline-mode balanced --translation-model qwen3:14b --judge-model qwen3.5:9b
python translate_subs.py ..\sample\sample_video.json --pipeline-mode max --translation-model qwen3:14b --translation-secondary-model translategemma:12b --judge-model qwen3.5:9b
python check_settings.py --pipeline-mode balanced --asr-source mix --asr-model large-v3 --separator-model model_bs_roformer_ep_317_sdr_12.9755.ckpt --translation-model qwen3:14b --translation-secondary-model translategemma:12b --judge-model qwen3.5:9b --context-model qwen3.5:9b
python check_gpu.py
```

### Linux / macOS

```bash
python3 -m venv .venv    # first time only
source .venv/bin/activate
cd src
python series_context_tool.py ../sample/Blue.Noah.1979.S01E01.AMZN.WEBRip.BK.mkv --context-model qwen3.5:9b
python process_audio.py ../sample/sample_video.mkv --asr-source mix --asr-model large-v3
python process_audio.py ../sample/sample_video.mkv --pipeline-mode balanced --judge-model qwen3.5:9b --context-model qwen3.5:9b
python process_audio.py ../sample/sample_video.mkv --pipeline-mode max --judge-model qwen3.5:9b --context-model qwen3.5:9b
python process_audio.py ../sample/sample_video.mkv --asr-source mix --asr-model hybrid
python process_audio.py ../sample/sample_video.mkv --asr-source mix --asr-model kotoba-whisper-v1.1
python translate_subs.py ../sample/sample_video.json --pipeline-mode current --translation-model qwen3:14b
python translate_subs.py ../sample/sample_video.json --pipeline-mode balanced --translation-model qwen3:14b --judge-model qwen3.5:9b
python translate_subs.py ../sample/sample_video.json --pipeline-mode max --translation-model qwen3:14b --translation-secondary-model translategemma:12b --judge-model qwen3.5:9b
python check_settings.py --pipeline-mode balanced --asr-source mix --asr-model large-v3 --separator-model model_bs_roformer_ep_317_sdr_12.9755.ckpt --translation-model qwen3:14b --translation-secondary-model translategemma:12b --judge-model qwen3.5:9b --context-model qwen3.5:9b
python check_gpu.py
```
