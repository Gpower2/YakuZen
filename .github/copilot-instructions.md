# Copilot Instructions

## Commands
This repository does not commit build, test, or lint tooling. Python dependencies are declared in `pyproject.toml`, and the entry points should still be launched from `src\` because `app.py` starts sibling scripts by bare filename.

```powershell
Set-Location src
python app.py
python process_audio.py <video-file> --asr-source mix --asr-model large-v3
python process_audio.py <video-file> --asr-source mix --asr-model hybrid
python process_audio.py <video-file> --asr-source mix --asr-model kotoba-whisper-v1.1
python translate_subs.py <json-file> --translation-model qwen3:14b
python check_gpu.py
```

`process_audio.py` assumes `ffmpeg`/`ffprobe` are on `PATH`, CUDA-backed `faster-whisper` is available, `stable-ts` is installed for timestamp refinement, and the `audio_separator` model assets can be cached under `.\temp`. The advanced Kotoba path also depends on the Transformers stack declared in `pyproject.toml`. `translate_subs.py` assumes a local Ollama server at `http://localhost:11434` with the configured model available; the current default is `qwen3:14b`, while other Ollama models can now be passed in explicitly.

## Architecture
- `app.py` is the CustomTkinter desktop shell. It owns folder selection, queueing, skip/cancel controls, and the progress UI. It now also owns the advanced processing settings that are passed to the worker scripts as explicit CLI arguments.
- For each selected video, `app.py` runs `process_audio.py` first and then `translate_subs.py` against the generated `<base>.json`. It reads the child process output byte-by-byte so `tqdm` carriage-return updates can drive the GUI progress bar.
- `process_audio.py` is the audio/transcription stage. It still separates vocals with `audio_separator`, but the current default transcription path is now the **original mixed track** (`--asr-source mix`) rather than the loudness-normalized stem. The stable default ASR backend remains `faster-whisper` `large-v3`; `kotoba-whisper-v1.1` is available as an advanced alternative, and `hybrid` now runs Whisper first and then lets Kotoba retry suspicious windows. Regardless of the chosen ASR backend, timing is still refined against the mono 16 kHz episode mix with `stable-ts align_words()` and the torchaudio MMS rescue pass.
- `translate_subs.py` is the LLM translation/export stage. It reads the subtitle JSON, infers the anime series title from the filename, calls the local Ollama API, caches the translated JSON, repairs fragmentary multi-cue windows by retranslating the combined Japanese text once, and writes `.jp.srt`, `.romaji.srt`, `.en.raw.srt`, and a merged/display-formatted `.en.srt`. The translation model is now selected by CLI/UI argument instead of being fixed in code.
- `check_gpu.py` is a diagnostics-only script for CUDA/provider readiness across PyTorch, ONNX Runtime, and `faster-whisper`.

## Key conventions
- Cache files control reruns. If `<base>.json` already exists and its `meta.timing_refinement_version` matches the current code, `process_audio.py` skips work and just re-exports the SRTs. Older JSON caches are treated as upgradeable transcripts only when the selected transcription source, ASR model, and relevant separator checkpoint still match; otherwise the script intentionally regenerates the Japanese transcript. If `<base>_translated.json` exists, `translate_subs.py` skips Ollama calls only when the Japanese transcript text, inferred series title, selected translation model, and translation prompt version still match; otherwise it regenerates the translated cache on purpose so English wording does not drift against the current configuration. Delete those files to force regeneration.
- Output files are written next to the source video or JSON, but intermediate vocal stems and normalized WAV files live under `.\temp` relative to the current working directory.
- `.gitattributes` enforces LF line endings for text files in the repository. Preserve LF when editing or adding tracked text files, even when working from Windows.
- `app.py` filters out lines containing `{` when copying worker output into the GUI console, so the workers rely on human-readable progress plus JSON-ish status messages on stderr. If you change logging, keep percentage-bearing `tqdm` output intact or the GUI progress bar will stop updating.
- The transcription defaults are intentionally tuned for Japanese anime, not general speech: `--asr-source mix`, `--asr-model large-v3`, `language="ja"`, `condition_on_previous_text=False`, VAD enabled with tighter silence/padding thresholds, and `word_timestamps=False`. `normalized_vocals` remains a legacy option, not the default, because recent A/B work showed it can hurt subtitle coverage. `kotoba-whisper-v1.1` is available as an advanced option but currently produces more aggressive cue merging than the default path. `hybrid` is intentionally **experimental**: it can improve some proper-noun/problem windows, but it may still over-merge or add rough rescue phrases, so do not promote it to the default without re-validating on the sample.
- Hybrid suspicious-window collection is deliberately narrow: it starts from the refined Whisper baseline, expands around suspicious seed cues (ASCII-heavy output, repeated-noise style text, or the existing long sparse cue heuristic), adds selected long-gap rescue windows, and only accepts Kotoba replacements that pass the current score/count/duration safeguards. Keep that behavior targeted; do not silently turn hybrid into a full second-pass rewrite of the whole episode.
- The translation step has two prompt modes. General chat models still use the older 1:1 JSON-list batch prompt with previous/next subtitle context. Translation-specialist models like `translategemma:12b` instead use direct per-line subtitle prompts plus inferred filename context so they do not accidentally translate the surrounding context blocks.
- English subtitle export has two modes: `.en.raw.srt` preserves the original segment timing 1:1, while `.en.srt` can merge repaired fragment windows into viewer-facing cues and then split oversized English subtitles back into timed chunks capped at two on-screen lines.
