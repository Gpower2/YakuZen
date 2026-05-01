# Copilot Instructions

## Commands
This repository does not commit build, test, or lint tooling, and it does not include a dependency manifest. Work with the Python entry points directly, and launch them from `src\` because `app.py` starts sibling scripts by bare filename.

```powershell
Set-Location src
python app.py
python process_audio.py <video-file>
python translate_subs.py <json-file>
python check_gpu.py
```

`process_audio.py` assumes `ffmpeg`/`ffprobe` are on `PATH`, CUDA-backed `faster-whisper` is available, `stable-ts` is installed for timestamp refinement, and the `audio_separator` model assets can be cached under `.\temp`. `translate_subs.py` assumes a local Ollama server at `http://localhost:11434` with the configured model available; the current default is `qwen3:14b`.

## Architecture
- `app.py` is the CustomTkinter desktop shell. It owns folder selection, queueing, skip/cancel controls, and the progress UI; the actual AI work happens in child processes.
- For each selected video, `app.py` runs `process_audio.py` first and then `translate_subs.py` against the generated `<base>.json`. It reads the child process output byte-by-byte so `tqdm` carriage-return updates can drive the GUI progress bar.
- `process_audio.py` is the audio/transcription stage. It separates vocals with `audio_separator`, normalizes the stem with FFmpeg loudnorm to 16 kHz audio, transcribes with `faster-whisper` `large-v3` on CUDA, then extracts a mono 16 kHz copy of the original episode mix and runs `stable-ts align_words()` as a second pass to tighten segment start/end times against that mixed audio before converting Japanese text to phonetic romaji with `cutlet` and writing `<base>.json`, `<base>.kanji.srt`, `<base>.romaji.srt`, and `<base>_debug_raw.json`.
- `translate_subs.py` is the LLM translation/export stage. It reads the subtitle JSON, calls the local Ollama API in batches with rolling context, caches the translated JSON, and writes `.jp.srt`, `.romaji.srt`, `.en.raw.srt`, and a reformatted `.en.srt`.
- `check_gpu.py` is a diagnostics-only script for CUDA/provider readiness across PyTorch, ONNX Runtime, and `faster-whisper`.

## Key conventions
- Cache files control reruns. If `<base>.json` already exists and its `meta.timing_refinement_version` matches the current code, `process_audio.py` skips work and just re-exports the SRTs. Older JSON caches are treated as upgradeable transcripts: `process_audio.py` reuses the cached text and reruns only the timing-refinement stage. If `<base>_translated.json` exists, `translate_subs.py` skips Ollama calls, but it will sync fresh timings from the source JSON when the Japanese transcript text still matches. Delete those files to force regeneration.
- Output files are written next to the source video or JSON, but intermediate vocal stems and normalized WAV files live under `.\temp` relative to the current working directory.
- `app.py` filters out lines containing `{` when copying worker output into the GUI console, so the workers rely on human-readable progress plus JSON-ish status messages on stderr. If you change logging, keep percentage-bearing `tqdm` output intact or the GUI progress bar will stop updating.
- The transcription settings are intentionally tuned for Japanese anime, not general speech: `language="ja"`, `condition_on_previous_text=False`, VAD enabled with tighter silence/padding thresholds, `word_timestamps=False`, and `cutlet.use_foreign_spelling = False`. Treat these as product decisions unless you are deliberately re-tuning subtitle behavior.
- The translation step expects a 1:1 JSON list of English strings from Ollama. `translate_subs.py` strips code fences and filler, unwraps list-or-dict variants, and falls back to per-line translation when batch output length does not match the input batch.
- English subtitle export has two modes: `.en.raw.srt` preserves the original segment timing 1:1, while `.en.srt` wraps to 42 columns and proportionally splits long English lines into multiple timed subtitle chunks.
