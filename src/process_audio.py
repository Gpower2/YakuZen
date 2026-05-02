import sys
import json
import os
import time
import logging
import glob
import subprocess
import contextlib
import wave
import warnings
import re
import argparse
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import stable_whisper
import torch
import torchaudio
from audio_separator.separator import Separator
from pykakasi import kakasi

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
TEMP_DIR = os.path.abspath("./temp")
DEFAULT_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_ALIGNMENT_MODEL = "large-v3"
DEFAULT_ASR_SOURCE = "mix"
DEFAULT_ASR_MODEL = "large-v3"
SUPPORTED_ASR_SOURCES = ("mix", "raw_vocals", "normalized_vocals")
SUPPORTED_ASR_MODELS = ("large-v3", "kotoba-whisper-v1.1", "hybrid")
LEGACY_ASR_SOURCE = "normalized_vocals"
KOTOBA_MODEL_ID = "kotoba-tech/kotoba-whisper-v1.1"
KOTOBA_CHUNK_LENGTH_S = 20
KOTOBA_BATCH_SIZE = 8
HYBRID_BLOCK_MAX_GAP = 2.0
HYBRID_GAP_RESCUE_THRESHOLD = 12.0
HYBRID_GAP_RESCUE_MAX = 25.0
HYBRID_BLOCK_CONTEXT = 0.75
HYBRID_GAP_CONTEXT = 0.5
HYBRID_SUSPICIOUS_EXPANSION_GAP = 5.0
HYBRID_SUSPICIOUS_MAX_DURATION = 45.0
HYBRID_MIN_CANDIDATE_JAPANESE_CHARS = 6
HYBRID_MIN_SCORE_DELTA = 3
HYBRID_MAX_SEGMENT_COUNT_DROP = 2
HYBRID_MAX_CANDIDATE_SEGMENT_DURATION = 18.0
TIMING_REFINEMENT_VAD_THRESHOLD = 0.35
TIMING_REFINEMENT_VERSION = 6
ROMAJI_VERSION = 2
SUSPICIOUS_SEGMENT_MIN_DURATION = 8.0
SUSPICIOUS_SEGMENT_MAX_CHARS_PER_SECOND = 2.5
RESCUE_ACCEPT_MIN_SHIFT = 0.75
CTC_ALIGNMENT_SAMPLE_RATE = 16000
HONORIFIC_ROMAJI_PREFIXES = {"お", "ご", "御"}

_ctc_alignment_resources = None
_kotoba_pipeline = None

def get_audio_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / rate
    except Exception:
        return 0

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def normalize_audio(input_path, output_path, duration):
    print(json.dumps({"status": "normalizing_audio"}), file=sys.stderr)

    # FFmpeg Loudness Normalization (loudnorm)
    # Default: N/A (FFmpeg passes audio as-is)
    # Description: Analyzes the audio and mathematically flattens the dynamic range.
    # Reasoning: Kept as an optional legacy path only. Recent A/B testing showed that using
    # loudnorm-normalized vocals as the primary ASR source can hurt subtitle coverage on
    # difficult scenes, so this is no longer the default transcription path.
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000", output_path
    ]

    process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)

    with tqdm(total=duration, unit='s', desc="Normalizing Audio", file=sys.stderr) as pbar:
        for line in process.stderr:
            time_match = re.search(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)", line)
            if time_match:
                h, m, s = float(time_match.group(1)), float(time_match.group(2)), float(time_match.group(3))
                current_time = (h * 3600) + (m * 60) + s
                pbar.n = min(current_time, duration)
                pbar.refresh()

    process.wait()

def extract_alignment_audio(input_path, output_path):
    print(json.dumps({"status": "extracting_alignment_audio"}), file=sys.stderr)

    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000", output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def save_srt(subtitles, output_path, text_key):
    srt_output = []
    for idx, sub in enumerate(subtitles, 1):
        start = format_timestamp(sub['start'])
        end = format_timestamp(sub['end'])
        text = sub[text_key]
        srt_output.append(f"{idx}\n{start} --> {end}\n{text}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_output))

def clamp_timestamp(value, duration):
    return max(0.0, min(float(value), duration))

def get_whisper_runtime():
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"

def normalize_separator_model_name(separator_model):
    return os.path.basename(separator_model).strip() or DEFAULT_SEPARATOR_MODEL

def separator_output_tag(separator_model):
    stem = os.path.splitext(normalize_separator_model_name(separator_model))[0]
    return re.sub(r"\.\d+$", "", stem)

def find_cached_raw_vocals(base_name, separator_model):
    search_pattern = os.path.join(
        TEMP_DIR,
        f"*{base_name}*(Vocals)*{separator_output_tag(separator_model)}*.wav",
    )
    found_files = glob.glob(search_pattern)
    raw_files = [f for f in found_files if not os.path.basename(f).startswith("normalized_")]
    if not raw_files:
        return None
    return max(raw_files, key=os.path.getctime)

def normalize_cached_asr_model(cached_meta):
    raw_value = str(
        cached_meta.get("transcription_model")
        or cached_meta.get("model")
        or ""
    ).lower()
    if raw_value.startswith("hybrid"):
        return "hybrid"
    if raw_value.startswith("kotoba-whisper"):
        return "kotoba-whisper-v1.1"
    return "large-v3"

def transcription_settings_match(cached_meta, asr_source, asr_model, separator_model):
    cached_source = cached_meta.get("transcription_source", LEGACY_ASR_SOURCE)
    cached_model = normalize_cached_asr_model(cached_meta)
    cached_separator = cached_meta.get("separator_model", DEFAULT_SEPARATOR_MODEL)

    if cached_source != asr_source or cached_model != asr_model:
        return False

    if asr_source == "mix":
        return True

    return cached_separator == normalize_separator_model_name(separator_model)

def select_transcription_audio_path(asr_source, raw_vocals_path, normalized_vocals_path, alignment_audio_path):
    if asr_source == "mix":
        return alignment_audio_path
    if asr_source == "raw_vocals":
        return raw_vocals_path
    return normalized_vocals_path

def normalize_alignment_text(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z']", "", text)
    return text.strip()

def romanize_text(text, romanizer):
    converted = romanizer.convert(text)
    parts = []
    index = 0

    while index < len(converted):
        current = converted[index]
        original = str(current.get("orig", "")).strip()
        hepburn = str(current.get("hepburn", "")).strip()

        if not hepburn:
            index += 1
            continue

        if original in HONORIFIC_ROMAJI_PREFIXES and index + 1 < len(converted):
            next_hepburn = str(converted[index + 1].get("hepburn", "")).strip()
            if next_hepburn:
                parts.append(f"{hepburn}{next_hepburn}")
                index += 2
                continue

        parts.append(hepburn)
        index += 1

    romaji = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if not romaji:
        return ""
    return romaji[:1].upper() + romaji[1:]

def rebuild_subtitles_romaji(subtitles, romanizer):
    updated_subtitles = []
    for subtitle in subtitles:
        updated_subtitle = dict(subtitle)
        updated_subtitle["text_romaji"] = romanize_text(subtitle.get("text_jp", ""), romanizer)
        updated_subtitles.append(updated_subtitle)
    return updated_subtitles

def get_ctc_alignment_resources():
    global _ctc_alignment_resources

    if _ctc_alignment_resources is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = torchaudio.pipelines.MMS_FA
        model = bundle.get_model(with_star=False).to(device)
        model.eval()
        _ctc_alignment_resources = {
            "device": device,
            "model": model,
            "tokenizer": bundle.get_tokenizer(),
            "aligner": bundle.get_aligner(),
            "kakasi": kakasi(),
        }

    return _ctc_alignment_resources

def read_wav_clip(audio_path, start_sec, end_sec):
    audio, _sample_rate = read_wav_clip_with_sample_rate(audio_path, start_sec, end_sec)
    return audio

def read_wav_clip_with_sample_rate(audio_path, start_sec, end_sec):
    with contextlib.closing(wave.open(audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()

        if sample_width != 2:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        start_frame = max(0, min(total_frames, int(start_sec * sample_rate)))
        end_frame = max(start_frame, min(total_frames, int(end_sec * sample_rate)))
        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate

def is_suspicious_segment(subtitle):
    duration = subtitle["end"] - subtitle["start"]
    chars_per_second = len(subtitle["text_jp"]) / max(duration, 0.001)
    return (
        duration >= SUSPICIOUS_SEGMENT_MIN_DURATION
        and chars_per_second <= SUSPICIOUS_SEGMENT_MAX_CHARS_PER_SECOND
    )

def count_japanese_chars(text):
    return len(re.findall(r"[一-龯ぁ-ゟ゠-ヿー]", str(text or "")))

def count_ascii_chars(text):
    return len(re.findall(r"[A-Za-z]", str(text or "")))

def count_repeated_noise_runs(text):
    return len(re.findall(r"(.)\1{4,}", str(text or "")))

def text_quality_score(text):
    clean_text = str(text or "")
    japanese_chars = count_japanese_chars(clean_text)
    ascii_chars = count_ascii_chars(clean_text)
    repeated_runs = count_repeated_noise_runs(clean_text)
    score = japanese_chars - (ascii_chars * 3) - (repeated_runs * 4)
    if japanese_chars == 0:
        score -= 8
    return score

def segment_midpoint(segment):
    return (float(segment["start"]) + float(segment["end"])) / 2.0

def normalize_raw_segments(raw_segments):
    normalized_segments = []
    for segment in raw_segments:
        text = str(segment.get("text", "")).replace(" ", "").strip()
        if not text:
            continue
        start = round(float(segment.get("start", 0.0)), 3)
        end = round(max(start, float(segment.get("end", start))), 3)
        normalized_segments.append({
            "start": start,
            "end": end,
            "text": text,
        })

    normalized_segments.sort(key=lambda item: (item["start"], item["end"], item["text"]))

    subtitles = []
    previous_end = 0.0
    for segment in normalized_segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if start < previous_end:
            if end <= previous_end:
                continue
            start = previous_end
        if end <= start:
            continue
        normalized_raw_segments = {
            "start": round(start, 3),
            "end": round(end, 3),
            "text": segment["text"],
        }
        subtitles.append(normalized_raw_segments)
        previous_end = end
    return subtitles

def build_subtitles_from_raw_segments(raw_segments, romanizer):
    normalized_segments = normalize_raw_segments(raw_segments)

    subtitles = []
    for segment in normalized_segments:
        text_jp = segment["text"]
        if text_jp:
            subtitles.append({
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "text_jp": text_jp,
                "text_romaji": romanize_text(text_jp, romanizer),
            })
    return subtitles

def build_dialogue_blocks(subtitles, max_gap=HYBRID_BLOCK_MAX_GAP):
    if not subtitles:
        return []

    blocks = []
    start_idx = 0
    for idx in range(1, len(subtitles)):
        if subtitles[idx]["start"] - subtitles[idx - 1]["end"] > max_gap:
            blocks.append({
                "start": subtitles[start_idx]["start"],
                "end": subtitles[idx - 1]["end"],
                "subtitles": subtitles[start_idx:idx],
            })
            start_idx = idx

    blocks.append({
        "start": subtitles[start_idx]["start"],
        "end": subtitles[-1]["end"],
        "subtitles": subtitles[start_idx:],
    })
    return blocks

def is_hybrid_suspicious_subtitle(subtitle):
    text = subtitle["text_jp"]
    return (
        is_suspicious_segment(subtitle)
        or count_ascii_chars(text) > 0
        or count_repeated_noise_runs(text) > 0
    )

def expand_hybrid_suspicious_window(subtitles, seed_index):
    start_idx = seed_index
    end_idx = seed_index

    while start_idx > 0:
        previous = subtitles[start_idx - 1]
        current = subtitles[start_idx]
        if current["start"] - previous["end"] > HYBRID_SUSPICIOUS_EXPANSION_GAP:
            break
        if subtitles[end_idx]["end"] - previous["start"] > HYBRID_SUSPICIOUS_MAX_DURATION:
            break
        start_idx -= 1

    while end_idx + 1 < len(subtitles):
        current = subtitles[end_idx]
        following = subtitles[end_idx + 1]
        if following["start"] - current["end"] > HYBRID_SUSPICIOUS_EXPANSION_GAP:
            break
        if following["end"] - subtitles[start_idx]["start"] > HYBRID_SUSPICIOUS_MAX_DURATION:
            break
        end_idx += 1

    return start_idx, end_idx

def collect_hybrid_rescue_windows(base_subtitles, duration):
    rescue_windows = []

    for idx, subtitle in enumerate(base_subtitles):
        if is_hybrid_suspicious_subtitle(subtitle):
            start_idx, end_idx = expand_hybrid_suspicious_window(base_subtitles, idx)
            block = {
                "start": base_subtitles[start_idx]["start"],
                "end": base_subtitles[end_idx]["end"],
            }
            rescue_windows.append({
                "window_start": max(0.0, block["start"] - HYBRID_BLOCK_CONTEXT),
                "window_end": min(duration, block["end"] + HYBRID_BLOCK_CONTEXT),
                "target_start": block["start"],
                "target_end": block["end"],
                "reasons": {"suspicious_block"},
            })

    for previous, current in zip(base_subtitles, base_subtitles[1:]):
        gap = current["start"] - previous["end"]
        if HYBRID_GAP_RESCUE_THRESHOLD <= gap <= HYBRID_GAP_RESCUE_MAX:
            rescue_windows.append({
                "window_start": max(0.0, previous["end"] - HYBRID_GAP_CONTEXT),
                "window_end": min(duration, current["start"] + HYBRID_GAP_CONTEXT),
                "target_start": previous["end"],
                "target_end": current["start"],
                "reasons": {"gap"},
            })

    rescue_windows.sort(key=lambda item: (item["window_start"], item["window_end"]))
    merged_windows = []
    for window in rescue_windows:
        if not merged_windows:
            merged_windows.append(window)
            continue

        current = merged_windows[-1]
        overlapping = (
            window["window_start"] <= current["window_end"] + HYBRID_BLOCK_CONTEXT
            and window["target_start"] <= current["target_end"] + HYBRID_BLOCK_MAX_GAP
        )
        if overlapping:
            current["window_start"] = min(current["window_start"], window["window_start"])
            current["window_end"] = max(current["window_end"], window["window_end"])
            current["target_start"] = min(current["target_start"], window["target_start"])
            current["target_end"] = max(current["target_end"], window["target_end"])
            current["reasons"].update(window["reasons"])
        else:
            merged_windows.append(window)

    for idx, window in enumerate(merged_windows, start=1):
        window["id"] = idx
        window["reasons"] = sorted(window["reasons"])

    return merged_windows

def extract_segments_for_target(raw_segments, start, end):
    return [
        segment for segment in raw_segments
        if start <= segment_midpoint(segment) <= end
    ]

def clamp_segments_to_target(raw_segments, start, end):
    clamped = []
    for segment in raw_segments:
        clipped_start = max(float(start), float(segment["start"]))
        clipped_end = min(float(end), float(segment["end"]))
        if clipped_end <= clipped_start:
            continue
        clamped.append({
            "start": round(clipped_start, 3),
            "end": round(clipped_end, 3),
            "text": segment["text"],
        })
    return normalize_raw_segments(clamped)

def prune_redundant_segments(raw_segments):
    cleaned_segments = []
    for segment in normalize_raw_segments(raw_segments):
        if cleaned_segments:
            previous = cleaned_segments[-1]
            close_gap = segment["start"] - previous["end"] <= 0.5
            if close_gap:
                if segment["text"] == previous["text"]:
                    previous_duration = previous["end"] - previous["start"]
                    current_duration = segment["end"] - segment["start"]
                    if current_duration > previous_duration:
                        cleaned_segments[-1] = segment
                    continue
                if len(segment["text"]) <= 8 and segment["text"] in previous["text"]:
                    continue
        cleaned_segments.append(segment)
    return cleaned_segments

def should_accept_hybrid_candidate(base_segments, candidate_segments, reasons):
    if not candidate_segments:
        return False

    candidate_text = "".join(segment["text"] for segment in candidate_segments)
    candidate_score = text_quality_score(candidate_text)
    if count_japanese_chars(candidate_text) < HYBRID_MIN_CANDIDATE_JAPANESE_CHARS:
        return False
    if any((segment["end"] - segment["start"]) > HYBRID_MAX_CANDIDATE_SEGMENT_DURATION for segment in candidate_segments):
        return False

    if "gap" in reasons and not base_segments:
        return candidate_score >= HYBRID_MIN_CANDIDATE_JAPANESE_CHARS

    base_text = "".join(segment["text"] for segment in base_segments)
    base_score = text_quality_score(base_text)
    base_segment_count = len(base_segments)
    candidate_segment_count = len(candidate_segments)

    if candidate_text == base_text:
        return False

    if candidate_segment_count < max(1, base_segment_count - HYBRID_MAX_SEGMENT_COUNT_DROP):
        return False

    if count_ascii_chars(base_text) > count_ascii_chars(candidate_text) and candidate_score >= base_score - 1:
        return True

    if candidate_score >= base_score + HYBRID_MIN_SCORE_DELTA:
        return True

    if base_segment_count == 1 and candidate_segment_count >= 1 and candidate_score > base_score:
        return True

    return False

def transcribe_with_hybrid(alignment_model, transcription_audio_path, transcription_duration, alignment_audio_path, alignment_duration, romanizer):
    print("5% Running faster-whisper baseline...", file=sys.stderr)
    base_segments, info = transcribe_with_faster_whisper(
        alignment_model,
        transcription_audio_path,
        transcription_duration,
    )

    print("45% Refining baseline timings for hybrid comparison...", file=sys.stderr)
    base_subtitles = build_subtitles_from_raw_segments(base_segments, romanizer)
    base_subtitles = refine_subtitle_timings(
        alignment_model,
        alignment_audio_path,
        base_subtitles,
        alignment_duration,
    )
    merged_segments = normalize_raw_segments([
        {
            "start": subtitle["start"],
            "end": subtitle["end"],
            "text": subtitle["text_jp"],
        }
        for subtitle in base_subtitles
    ])
    rescue_windows = collect_hybrid_rescue_windows(base_subtitles, alignment_duration)
    if not rescue_windows:
        return base_subtitles, merged_segments, info, {"windows_total": 0, "windows_accepted": 0}

    print(f"60% Running Kotoba rescue on {len(rescue_windows)} suspicious windows...", file=sys.stderr)
    accepted_windows = 0
    for window in rescue_windows:
        clip_audio, clip_sample_rate = read_wav_clip_with_sample_rate(
            transcription_audio_path,
            window["window_start"],
            window["window_end"],
        )
        if clip_audio.size == 0:
            continue

        candidate_segments, _candidate_info = transcribe_with_kotoba(
            (clip_audio, clip_sample_rate),
            window["window_end"] - window["window_start"],
            announce=False,
        )
        candidate_segments = normalize_raw_segments([
            {
                "start": segment["start"] + window["window_start"],
                "end": segment["end"] + window["window_start"],
                "text": segment["text"],
            }
            for segment in candidate_segments
        ])
        candidate_segments = extract_segments_for_target(
            candidate_segments,
            window["target_start"],
            window["target_end"],
        )
        candidate_segments = clamp_segments_to_target(
            candidate_segments,
            window["target_start"],
            window["target_end"],
        )

        base_target_segments = extract_segments_for_target(
            merged_segments,
            window["target_start"],
            window["target_end"],
        )
        if not should_accept_hybrid_candidate(base_target_segments, candidate_segments, window["reasons"]):
            continue

        merged_segments = [
            segment for segment in merged_segments
            if not (window["target_start"] <= segment_midpoint(segment) <= window["target_end"])
        ]
        merged_segments.extend(candidate_segments)
        merged_segments = normalize_raw_segments(merged_segments)
        accepted_windows += 1

    print("88% Finalizing hybrid transcript...", file=sys.stderr)
    merged_segments = prune_redundant_segments(merged_segments)
    merged_subtitles = build_subtitles_from_raw_segments(merged_segments, romanizer)
    return merged_subtitles, merged_segments, info, {
        "windows_total": len(rescue_windows),
        "windows_accepted": accepted_windows,
    }

def rescue_suspicious_timings(audio_path, source_subtitles, refined_subtitles, duration):
    rescued_subtitles = []

    for source_subtitle, refined_subtitle in zip(source_subtitles, refined_subtitles):
        if not is_suspicious_segment(source_subtitle):
            rescued_subtitles.append(refined_subtitle)
            continue

        clip_audio = read_wav_clip(audio_path, source_subtitle["start"], source_subtitle["end"])
        if clip_audio.size == 0:
            rescued_subtitles.append(refined_subtitle)
            continue

        resources = get_ctc_alignment_resources()
        romanized_tokens = [
            normalize_alignment_text(item["hepburn"])
            for item in resources["kakasi"].convert(source_subtitle["text_jp"])
        ]
        romanized_tokens = [token for token in romanized_tokens if token]
        if not romanized_tokens:
            rescued_subtitles.append(refined_subtitle)
            continue

        waveform = torch.from_numpy(clip_audio).unsqueeze(0).to(resources["device"])
        with torch.inference_mode():
            emission, _ = resources["model"](waveform)

        token_ids = resources["tokenizer"](romanized_tokens)
        token_spans = resources["aligner"](emission[0].cpu(), token_ids)
        token_spans = [spans for spans in token_spans if spans]
        if not token_spans:
            rescued_subtitles.append(refined_subtitle)
            continue

        ratio = waveform.size(1) / emission.size(1) / CTC_ALIGNMENT_SAMPLE_RATE
        rescued_start = round(
            clamp_timestamp(source_subtitle["start"] + token_spans[0][0].start * ratio, duration),
            3,
        )
        rescued_end = round(
            max(
                rescued_start,
                clamp_timestamp(source_subtitle["start"] + token_spans[-1][-1].end * ratio, duration),
            ),
            3,
        )

        start_shift = abs(rescued_start - refined_subtitle["start"])
        end_shift = abs(rescued_end - refined_subtitle["end"])
        if start_shift < RESCUE_ACCEPT_MIN_SHIFT and end_shift < RESCUE_ACCEPT_MIN_SHIFT:
            rescued_subtitles.append(refined_subtitle)
            continue

        rescued_subtitles.append({
            **refined_subtitle,
            "start": rescued_start,
            "end": rescued_end,
        })

    return rescued_subtitles

def refine_subtitle_timings(model, audio_path, subtitles, duration):
    print(json.dumps({"status": "refining_timestamps"}), file=sys.stderr)

    alignment_segments = [
        {
            "start": sub["start"],
            "end": sub["end"],
            "text": sub["text_jp"]
        }
        for sub in subtitles
    ]

    refined_result = model.align_words(
        audio_path,
        alignment_segments,
        language="ja",
        vad=True,
        vad_threshold=TIMING_REFINEMENT_VAD_THRESHOLD,
        suppress_silence=True,
        suppress_word_ts=True,
        regroup=False,
        verbose=False,
        inplace=False,
    )

    if len(refined_result.segments) != len(subtitles):
        raise ValueError(
            f"Timestamp refinement changed segment count: {len(subtitles)} -> {len(refined_result.segments)}"
        )

    refined_subtitles = []
    for original, refined in zip(subtitles, refined_result.segments):
        start = round(clamp_timestamp(refined.start, duration), 3)
        end = round(max(start, clamp_timestamp(refined.end, duration)), 3)

        refined_subtitles.append({
            **original,
            "start": start,
            "end": end,
        })

    return rescue_suspicious_timings(audio_path, subtitles, refined_subtitles, duration)

def transcribe_with_faster_whisper(model, audio_path, duration):
    transcription_result = model.transcribe(
        audio_path,

        # language
        # Default: None (Auto-detect)
        # Description: The language of the audio.
        # Reasoning: Forced to "ja" to skip the detection phase and prevent the AI from arbitrarily switching languages mid-episode.
        language="ja",

        # beam_size
        # Default: 5
        # Description: Number of parallel paths the AI evaluates to find the most accurate translation.
        # Reasoning: Kept at 5. Offers the best balance between high accuracy and VRAM usage.
        beam_size=5,

        # patience
        # Default: 1.0
        # Description: How long the beam search waits to finalize a complex sentence path.
        # Reasoning: Kept at 1.0. Ensures high logical accuracy on long anime monologues.
        patience=1.0,

        # condition_on_previous_text
        # Default: True
        # Description: Feeds the previously transcribed text into the current window to provide context.
        # Reasoning: Set to False. Anime has long musical/silent breaks. Context windowing causes the AI to hallucinate and repeat the previous line endlessly during silences.
        condition_on_previous_text=False,

        # initial_prompt
        # Default: None
        # Description: A string of text fed to the AI before it starts transcribing.
        # Reasoning: Subliminally forces the AI to use proper Japanese punctuation (。、) and permits it to output English characters (Hello) for loan words.
        initial_prompt="こんにちは。Hello。今日はいい天気ですね。",

        # vad_filter
        # Default: False
        # Description: Enables Silero Voice Activity Detection to skip silent parts of the audio.
        # Reasoning: Set to True. Starves the AI of static and tape hiss, drastically reducing hallucinations.
        vad_filter=True,

        vad_parameters=dict(
            # threshold
            # Default: 0.5
            # Description: Confidence threshold (0.0 to 1.0) for detecting speech.
            # Reasoning: Kept at 0.5. Relaxing this globally recovered coverage but also introduced heavy junk and hallucinations.
            threshold=0.5,

            # min_speech_duration_ms
            # Default: 250
            # Description: Minimum length of a sound to be considered speech.
            # Reasoning: Kept at 250ms. Prevents random micro-noises (door clicks, footsteps) from triggering transcription.
            min_speech_duration_ms=250,

            # min_silence_duration_ms
            # Default: 2000
            # Description: How much silence must occur before the VAD cuts a segment.
            # Reasoning: Tightened to 500ms. Longer values stretched subtitles across long pauses.
            min_silence_duration_ms=500,

            # speech_pad_ms
            # Default: 400
            # Description: Extra audio buffer added before and after detected speech.
            # Reasoning: Tightened to 150ms. Larger pads inflated subtitle durations.
            speech_pad_ms=150,
        ),

        # word_timestamps
        # Default: False
        # Description: Uses Dynamic Time Warping (DTW) to calculate millisecond timings for individual words.
        # Reasoning: STRICTLY FALSE. DTW fundamentally misunderstands Japanese trailing vowels and shrinks boundaries or snaps to static. We are relying entirely on the VAD segment bounds.
        word_timestamps=False,

        # hallucination_silence_threshold
        # Default: None
        # Description: Drops transcriptions if they are stretched across massive silences.
        # Reasoning: Set to 2.0s. Failsafe to prevent the "Early Anchor" bug where a word is artificially snapped 10 seconds early to background noise.
        hallucination_silence_threshold=2.0,
    )

    if isinstance(transcription_result, tuple):
        segments_source, info = transcription_result
        info_dict = None
        if info is not None:
            info_dict = {
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
            }
    else:
        segments_source = getattr(transcription_result, "segments", [])
        info_dict = {
            "language": getattr(transcription_result, "language", None),
            "language_probability": getattr(transcription_result, "language_probability", None),
        }

    raw_segments = []
    with tqdm(total=duration, unit='s', desc="Transcribing", file=sys.stderr) as pbar:
        for segment in segments_source:
            raw_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
            pbar.update(segment.end - pbar.n)

    return raw_segments, info_dict

def load_kotoba_pipeline():
    global _kotoba_pipeline

    if _kotoba_pipeline is not None:
        return _kotoba_pipeline

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Kotoba-Whisper requires the Transformers stack. Install transformers, accelerate, sentencepiece, and safetensors."
        ) from exc

    _kotoba_pipeline = pipeline(
        "automatic-speech-recognition",
        model=KOTOBA_MODEL_ID,
        device=0 if torch.cuda.is_available() else -1,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ignore_warning=True,
    )
    return _kotoba_pipeline

def transcribe_with_kotoba(audio_input, duration, announce=True):
    if announce:
        print("10% Loading Kotoba-Whisper model...", file=sys.stderr)
    asr_pipeline = load_kotoba_pipeline()
    if announce:
        print("35% Running Kotoba transcription...", file=sys.stderr)
    pipeline_input = audio_input
    if isinstance(audio_input, tuple):
        pipeline_input = {
            "array": np.asarray(audio_input[0], dtype=np.float32),
            "sampling_rate": int(audio_input[1]),
        }
    result = asr_pipeline(
        pipeline_input,
        chunk_length_s=KOTOBA_CHUNK_LENGTH_S,
        batch_size=KOTOBA_BATCH_SIZE,
        return_timestamps=True,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )

    raw_segments = []
    for chunk in result.get("chunks") or []:
        timestamp = chunk.get("timestamp")
        text = str(chunk.get("text", "")).strip()
        if not timestamp or timestamp[0] is None or timestamp[1] is None or not text:
            continue
        raw_segments.append({
            "start": float(timestamp[0]),
            "end": float(timestamp[1]),
            "text": text,
        })

    if not raw_segments and result.get("text"):
        raw_segments.append({
            "start": 0.0,
            "end": duration,
            "text": str(result["text"]).strip(),
        })

    if announce:
        print("85% Finalizing Kotoba transcript...", file=sys.stderr)
    return raw_segments, {"language": "ja", "language_probability": None}

def main(input_file, asr_source=DEFAULT_ASR_SOURCE, asr_model=DEFAULT_ASR_MODEL, separator_model=DEFAULT_SEPARATOR_MODEL):
    start_time = time.time()

    if not os.path.exists(input_file):
        print(json.dumps({"error": f"Input file not found: {input_file}"}), file=sys.stderr)
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    dir_name = os.path.dirname(input_file)
    output_json_path = os.path.join(dir_name, f"{base_name}.json")
    output_srt_romaji = os.path.join(dir_name, f"{base_name}.romaji.srt")
    output_srt_kanji = os.path.join(dir_name, f"{base_name}.kanji.srt")
    debug_json_path = os.path.join(dir_name, f"{base_name}_debug_raw.json")
    cached_data = None
    cached_meta = {}
    results = None
    info = None
    hybrid_stats = None
    romanizer = kakasi()
    selected_separator_model = normalize_separator_model_name(separator_model)

    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        cached_meta = cached_data.get("meta", {})
        config_matches = transcription_settings_match(cached_meta, asr_source, asr_model, selected_separator_model)

        if not config_matches:
            print(
                "\n[!] CACHE INVALIDATED: Transcription source, ASR model, or separator model changed. Regenerating transcript.\n",
                file=sys.stderr,
            )
            cached_data = None
            cached_meta = {}
        else:
            timing_current = cached_meta.get("timing_refinement_version", 0) >= TIMING_REFINEMENT_VERSION
            romaji_current = cached_meta.get("romaji_version", 0) >= ROMAJI_VERSION

            if timing_current and romaji_current:
                print("\n[!] CACHE LOADED: Refined JSON found. Skipping transcription.\n", file=sys.stderr)
                save_srt(cached_data["subtitles"], output_srt_romaji, "text_romaji")
                save_srt(cached_data["subtitles"], output_srt_kanji, "text_jp")
                return

            if timing_current:
                print("\n[!] CACHE LOADED: Refined JSON found. Refreshing romaji output.\n", file=sys.stderr)
                refreshed_subtitles = rebuild_subtitles_romaji(cached_data["subtitles"], romanizer)
                refreshed_meta = dict(cached_meta)
                refreshed_meta["processing_time"] = round(time.time() - start_time, 2)
                refreshed_meta["romaji_version"] = ROMAJI_VERSION
                refreshed_meta["romaji_converter"] = "pykakasi-hepburn"

                refreshed_output = {
                    "meta": refreshed_meta,
                    "subtitles": refreshed_subtitles,
                }

                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(refreshed_output, f, ensure_ascii=False, indent=2)

                save_srt(refreshed_subtitles, output_srt_romaji, "text_romaji")
                save_srt(refreshed_subtitles, output_srt_kanji, "text_jp")
                return

            if cached_meta.get("timing_refined"):
                print("\n[!] CACHE LOADED: Older refined JSON found. Reusing transcript and upgrading timings.\n", file=sys.stderr)
            else:
                print("\n[!] CACHE LOADED: Existing JSON found. Reusing transcript and upgrading timings.\n", file=sys.stderr)
            results = cached_data["subtitles"]

    os.makedirs(TEMP_DIR, exist_ok=True)

    # 1. SEPARATE VOCALS
    raw_vocals_path = find_cached_raw_vocals(base_name, selected_separator_model)
    if raw_vocals_path:
        print(json.dumps({"status": "skipping_separation", "cached_file": raw_vocals_path}), file=sys.stderr)
    else:
        print(json.dumps({"status": "separating_vocals"}), file=sys.stderr)
        separator = Separator(log_level=logging.ERROR, model_file_dir=TEMP_DIR, output_dir=TEMP_DIR, output_single_stem="Vocals")
        separator.load_model(model_filename=selected_separator_model)
        output_files = separator.separate(input_file)
        raw_vocals_path = os.path.join(TEMP_DIR, output_files[0])

    normalized_vocals_path = os.path.join(TEMP_DIR, f"normalized_{os.path.basename(raw_vocals_path)}")
    source_duration = get_audio_duration(raw_vocals_path)

    # 2. NORMALIZE AUDIO (legacy / optional source)
    if asr_source == "normalized_vocals" and not os.path.exists(normalized_vocals_path):
        normalize_audio(raw_vocals_path, normalized_vocals_path, source_duration)

    alignment_audio_path = os.path.join(TEMP_DIR, f"alignment_{base_name}.wav")
    if not os.path.exists(alignment_audio_path):
        extract_alignment_audio(input_file, alignment_audio_path)
    alignment_duration = get_audio_duration(alignment_audio_path)
    transcription_audio_path = select_transcription_audio_path(
        asr_source,
        raw_vocals_path,
        normalized_vocals_path,
        alignment_audio_path,
    )
    transcription_duration = get_audio_duration(transcription_audio_path) or source_duration or alignment_duration

    whisper_device, whisper_compute_type = get_whisper_runtime()
    alignment_model = stable_whisper.load_faster_whisper(
        DEFAULT_ALIGNMENT_MODEL,
        device=whisper_device,
        compute_type=whisper_compute_type,
    )

    raw_segments_debug = []
    if results is not None and os.path.exists(debug_json_path):
        with open(debug_json_path, "r", encoding="utf-8") as f:
            raw_segments_debug = json.load(f)
        results = build_subtitles_from_raw_segments(raw_segments_debug, romanizer)

    if results is None:
        # 3. TRANSCRIBE
        print(json.dumps({"status": "transcribing"}), file=sys.stderr)
        if asr_model == "large-v3":
            raw_segments_debug, info = transcribe_with_faster_whisper(
                alignment_model,
                transcription_audio_path,
                transcription_duration,
            )
        elif asr_model == "kotoba-whisper-v1.1":
            raw_segments_debug, info = transcribe_with_kotoba(
                transcription_audio_path,
                transcription_duration,
            )
        elif asr_model == "hybrid":
            results, raw_segments_debug, info, hybrid_stats = transcribe_with_hybrid(
                alignment_model,
                transcription_audio_path,
                transcription_duration,
                alignment_audio_path,
                alignment_duration,
                romanizer,
            )
        else:
            raise ValueError(f"Unsupported ASR model: {asr_model}")

        if asr_model != "hybrid":
            results = build_subtitles_from_raw_segments(raw_segments_debug, romanizer)

        print(json.dumps({"status": "saving_debug", "file": debug_json_path}), file=sys.stderr)
        with open(debug_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_segments_debug, f, ensure_ascii=False, indent=2)

    results = refine_subtitle_timings(alignment_model, alignment_audio_path, results, alignment_duration)

    total_time = time.time() - start_time
    output_meta = dict(cached_meta)
    output_meta["processing_time"] = round(total_time, 2)
    output_meta["model"] = f"{asr_model}-transcribe+{DEFAULT_ALIGNMENT_MODEL}-align+mms-rescue"
    output_meta["timing_refined"] = True
    output_meta["timing_refiner"] = "stable-ts-align_words+mms-forced-align-rescue"
    output_meta["timing_refinement_version"] = TIMING_REFINEMENT_VERSION
    output_meta["timing_alignment_audio"] = "mixed_track_mono_16k"
    output_meta["timing_alignment_model"] = DEFAULT_ALIGNMENT_MODEL
    output_meta["whisper_device"] = whisper_device
    output_meta["whisper_compute_type"] = whisper_compute_type
    output_meta["romaji_version"] = ROMAJI_VERSION
    output_meta["romaji_converter"] = "pykakasi-hepburn"
    output_meta["transcription_model"] = asr_model
    if asr_model == "large-v3":
        output_meta["transcription_backend"] = "faster-whisper"
    elif asr_model == "hybrid":
        output_meta["transcription_backend"] = "hybrid(faster-whisper+kotoba)"
    else:
        output_meta["transcription_backend"] = "transformers"
    output_meta["transcription_source"] = asr_source
    output_meta["separator_model"] = selected_separator_model
    if hybrid_stats:
        output_meta["hybrid_rescue_windows"] = hybrid_stats.get("windows_total", 0)
        output_meta["hybrid_rescue_accepted"] = hybrid_stats.get("windows_accepted", 0)
    if info:
        output_meta["language"] = info.get("language", "ja")
        if info.get("language_probability") is not None:
            output_meta["probability"] = round(info["language_probability"], 2)
    else:
        output_meta.setdefault("language", "ja")

    final_output = {
        "meta": output_meta,
        "subtitles": results
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "saved_to_disk", "file": output_json_path}), file=sys.stderr)

    save_srt(results, output_srt_romaji, "text_romaji")
    save_srt(results, output_srt_kanji, "text_jp")

    print(json.dumps({"status": "done"}), file=sys.stderr)

def parse_args():
    parser = argparse.ArgumentParser(description="Separate vocals, transcribe Japanese audio, and refine subtitle timings.")
    parser.add_argument("video_file", help="Path to the input video file.")
    parser.add_argument(
        "--asr-source",
        choices=SUPPORTED_ASR_SOURCES,
        default=DEFAULT_ASR_SOURCE,
        help="Audio source used for transcription. 'mix' is the current recommended default.",
    )
    parser.add_argument(
        "--asr-model",
        choices=SUPPORTED_ASR_MODELS,
        default=DEFAULT_ASR_MODEL,
        help="ASR model used for transcription. 'large-v3' is the stable default; 'hybrid' runs Whisper first and then lets Kotoba rescue suspicious windows.",
    )
    parser.add_argument(
        "--separator-model",
        default=DEFAULT_SEPARATOR_MODEL,
        help="Separator checkpoint filename to use when generating the vocals stem.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        args.video_file,
        asr_source=args.asr_source,
        asr_model=args.asr_model,
        separator_model=args.separator_model,
    )
