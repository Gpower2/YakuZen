import sys
import json
import os
import itertools
import textwrap
import re
import argparse
from functools import lru_cache
from datetime import timedelta
from tqdm import tqdm

from ollama_utils import call_ollama_model
from pipeline_profiles import (
    BALANCED_PIPELINE_MODE,
    CURRENT_PIPELINE_MODE,
    DEFAULT_CONTEXT_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PRIMARY_TRANSLATION_MODEL,
    DEFAULT_SECONDARY_TRANSLATION_MODEL,
    MAX_PIPELINE_MODE,
    SUPPORTED_PIPELINE_MODES,
    mode_uses_secondary_translation,
    mode_uses_translation_review,
)
from series_context import (
    build_series_context_prompt_block,
    canonical_series_title,
    infer_series_title,
    resolve_series_context,
    series_context_hash,
)

# ==========================================
# --- CONFIGURATION & LLM PARAMETERS ---
# ==========================================

DEFAULT_TRANSLATION_MODEL = DEFAULT_PRIMARY_TRANSLATION_MODEL

# MODEL_NAME
# Default: "llama3" (varies by user)
# Description: The LLM model to use for translation.
# Reasoning: Qwen3 14B remains the best same-tier local default here. Newer Qwen families
# on Ollama currently jump to much larger 27B/30B/35B classes, while TranslateGemma can
# still be useful as an optional translation-specialist fallback.
#MODEL_NAME = "sakura-anime"
#MODEL_NAME = "qwen2.5:14b"
#MODEL_NAME = "qwen2.5:32b"
MODEL_NAME = DEFAULT_TRANSLATION_MODEL
#MODEL_NAME = "translategemma:12b"
#MODEL_NAME = "qwen3:32b"

# BATCH_SIZE
# Default: 1
# Description: How many lines to send to the LLM in a single request.
# Reasoning: Set to 5. Sending too many lines confuses the LLM's JSON output structure. Sending 1 is too slow. 5 is the sweet spot for contextual accuracy and speed.
BATCH_SIZE = 5

# CONTEXT_SIZE
# Default: 0
# Description: How many previously translated lines to include as context for the next batch.
# Reasoning: Set to 3. Helps the LLM maintain conversational flow and pronoun consistency without overflowing the prompt memory.
CONTEXT_SIZE = 3

# LOOKAHEAD_SIZE
# Default: 0
# Description: How many upcoming Japanese lines to include as non-translated context.
# Reasoning: Set to 3. This gives the LLM visibility into sentence continuations that fall just outside the current batch, which
# helps avoid dangling English fragments at batch boundaries without breaking 1-to-1 output alignment.
LOOKAHEAD_SIZE = 3

TRANSLATION_PROMPT_VERSION = 8
TRANSLATION_REPAIR_VERSION = 5
FRAGMENT_WINDOW_SIZE = 3
FRAGMENT_WINDOW_MAX_GAP = 1.0
FRAGMENTARY_ENDINGS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "in", "into", "is", "of", "on", "or", "that", "the", "to",
    "was", "were", "which", "with",
}
SOURCE_LANGUAGE_NAME = "Japanese"
SOURCE_LANGUAGE_CODE = "ja"
TARGET_LANGUAGE_NAME = "English"
TARGET_LANGUAGE_CODE = "en"
MAX_DISPLAY_LINES = 2
MIN_SPLIT_CHUNK_DURATION = 0.75
DISPLAY_CHUNK_PUNCTUATION_SCORES = {
    "!": -8.0,
    "?": -6.0,
    ".": -5.0,
    ":": -3.0,
    ";": -2.0,
    ",": -1.0,
}
# ==========================================
# --- SUBTITLE FORMATTING PARAMETERS ---
# ==========================================

# LINE_WRAP_WIDTH
# Default: N/A
# Description: The maximum character width of a single text line on screen.
# Reasoning: Set to 46. Slightly wider lines reduce the need for artificial timing splits
# while still keeping English subtitles comfortably within a two-line layout.
LINE_WRAP_WIDTH = 46

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a professional anime localizer.
Translate the input array of Japanese lines into a matching array of natural, conversational English strings.
RULES:
1. Output ONLY a JSON List of strings: ["string1", "string2"]
2. NO Objects. NO Keys. NO Dictionaries.
3. Maintain exact 1-to-1 alignment with the input.
4. Translate the full meaning without omitting details.
5. Input lines may break in the middle of a sentence. Keep one English string per input line, but you may redistribute phrasing across adjacent outputs so the sequence reads naturally.
6. Avoid leaving dangling auxiliaries, articles, or conjunctions at the end of a line when a nearby rephrasing can avoid it."""

DISPLAY_SYSTEM_PROMPT = "You are a professional anime localizer. Output only the final English subtitle line."

# --- HELPER FUNCTIONS ---

def emit_status(status, **payload):
    message = {"status": status}
    message.update(payload)
    print(json.dumps(message), file=sys.stderr)

def format_context_list(items):
    clean_items = [str(item).strip() for item in items if str(item).strip()]
    if not clean_items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in clean_items)

def uses_direct_translation_model(model_name):
    return str(model_name or "").lower().startswith("translategemma")

def build_translation_context_note(previous_context=None, next_context=None, series_context=None, fallback_title=None):
    notes = []

    context_block = build_series_context_prompt_block(series_context, fallback_title=fallback_title)
    if context_block.strip():
        notes.append(context_block.strip())

    previous_items = [str(item).strip() for item in (previous_context or []) if str(item).strip()]
    next_items = [str(item).strip() for item in (next_context or []) if str(item).strip()]

    if previous_items:
        notes.append(
            "Previous English subtitle context for consistency only: "
            + " | ".join(previous_items[-CONTEXT_SIZE:])
        )

    if next_items:
        notes.append(
            "Upcoming Japanese subtitle context for consistency only: "
            + " | ".join(next_items[:LOOKAHEAD_SIZE])
        )

    return " ".join(notes).strip()

def build_direct_translation_prompt(text_jp, previous_context=None, next_context=None, series_context=None, fallback_title=None):
    context_sentence = ""
    context_block = build_translation_context_note(
        previous_context=previous_context,
        next_context=next_context,
        series_context=series_context,
        fallback_title=fallback_title,
    )
    if context_block:
        context_sentence = f" Context:\n{context_block}\n"
    return (
        f"You are a professional {SOURCE_LANGUAGE_NAME} ({SOURCE_LANGUAGE_CODE}) to {TARGET_LANGUAGE_NAME} ({TARGET_LANGUAGE_CODE}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANGUAGE_NAME} text while adhering to {TARGET_LANGUAGE_NAME} grammar, vocabulary, and cultural sensitivities.{context_sentence} "
        " Prefer concise subtitle phrasing, and do not translate or summarize any context other than the main Japanese text below. "
        f"Produce only the {TARGET_LANGUAGE_NAME} translation, without any additional explanations or commentary. "
        f"Please translate the following {SOURCE_LANGUAGE_NAME} text into {TARGET_LANGUAGE_NAME}:\n\n\n"
        f"{text_jp}"
    )

def build_direct_display_prompt(text_jp, series_context=None, fallback_title=None):
    context_note = build_translation_context_note(series_context=series_context, fallback_title=fallback_title)
    context_sentence = f" {context_note}" if context_note else ""
    return (
        f"You are a professional {SOURCE_LANGUAGE_NAME} ({SOURCE_LANGUAGE_CODE}) to {TARGET_LANGUAGE_NAME} ({TARGET_LANGUAGE_CODE}) subtitle translator. "
        f"Your goal is to translate the following {SOURCE_LANGUAGE_NAME} text into one concise, natural {TARGET_LANGUAGE_NAME} subtitle line while preserving its meaning and nuance.{context_sentence} "
        f"Produce only the {TARGET_LANGUAGE_NAME} subtitle line, without any additional explanations or commentary.\n\n\n"
        f"{text_jp}"
    )

def wrap_display_lines(text):
    return textwrap.wrap(
        text,
        width=LINE_WRAP_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [text]

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def iterate_display_cues(subtitles):
    idx = 0

    while idx < len(subtitles):
        subtitle = subtitles[idx]
        group_id = subtitle.get("display_group_id")

        if group_id and subtitle.get("display_text_en"):
            end_idx = idx
            while (
                end_idx + 1 < len(subtitles)
                and subtitles[end_idx + 1].get("display_group_id") == group_id
            ):
                end_idx += 1

            yield {
                "start": subtitles[idx]["start"],
                "end": subtitles[end_idx]["end"],
                "text": subtitle.get("display_text_en", ""),
                "subtitles": subtitles[idx:end_idx + 1],
            }
            idx = end_idx + 1
            continue

        yield {
            "start": subtitle["start"],
            "end": subtitle["end"],
            "text": subtitle.get("text_en", ""),
            "subtitles": [subtitle],
        }
        idx += 1

def score_display_chunk(text, is_last_chunk):
    wrapped_lines = wrap_display_lines(text)
    if len(wrapped_lines) > MAX_DISPLAY_LINES:
        return None

    score = len(wrapped_lines) * 0.5
    words = text.split()

    if len(words) < 2:
        score += 20.0
    elif len(words) < 4:
        score += 12.0
    elif len(words) < 6:
        score += 4.0

    if len(wrapped_lines) == 2:
        score += abs(len(wrapped_lines[0]) - len(wrapped_lines[1])) / LINE_WRAP_WIDTH

    for line in wrapped_lines:
        if len(line.split()) <= 1 and len(wrapped_lines) > 1:
            score += 6.0
        if len(line) < 10 and len(wrapped_lines) > 1:
            score += 2.5

    if not is_last_chunk and ends_with_dangling_word(text):
        score += 10.0

    if not is_last_chunk:
        for punctuation, punctuation_score in DISPLAY_CHUNK_PUNCTUATION_SCORES.items():
            if text.endswith(punctuation):
                score += punctuation_score
                break

    return score

def split_display_text_into_chunks(text):
    normalized = normalize_display_text(text)
    if not normalized:
        return []

    if len(wrap_display_lines(normalized)) <= MAX_DISPLAY_LINES:
        return [normalized]

    words = normalized.split()

    @lru_cache(maxsize=None)
    def best_split(start_index):
        if start_index >= len(words):
            return 0.0, []

        best_result = None
        for end_index in range(start_index + 1, len(words) + 1):
            chunk_text = " ".join(words[start_index:end_index]).strip()
            if len(wrap_display_lines(chunk_text)) > MAX_DISPLAY_LINES:
                break

            remaining_words = len(words) - end_index
            if remaining_words == 1:
                continue

            chunk_score = score_display_chunk(chunk_text, end_index == len(words))
            if chunk_score is None:
                continue

            rest_result = best_split(end_index)
            if rest_result is None:
                continue

            rest_score, rest_chunks = rest_result
            total_score = chunk_score + rest_score + (2.5 if end_index < len(words) else 0.0)

            if best_result is None or total_score < best_result[0]:
                best_result = (total_score, [chunk_text] + rest_chunks)

        return best_result

    result = best_split(0)
    if result:
        return result[1]

    wrapped_lines = wrap_display_lines(normalized)
    fallback_chunks = []
    for index in range(0, len(wrapped_lines), MAX_DISPLAY_LINES):
        fallback_chunks.append(" ".join(wrapped_lines[index:index + MAX_DISPLAY_LINES]).strip())
    return fallback_chunks

def allocate_chunk_timings(start_time, end_time, chunks):
    if not chunks:
        return []

    if len(chunks) == 1:
        return [(round(start_time, 3), round(end_time, 3))]

    total_duration = max(0.0, float(end_time) - float(start_time))
    if total_duration == 0:
        return [(round(start_time, 3), round(end_time, 3)) for _ in chunks]

    weights = [max(1, len(re.sub(r"\s+", "", chunk))) for chunk in chunks]
    total_weight = sum(weights)
    timings = []
    elapsed_weight = 0
    current_start = float(start_time)

    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            current_end = float(end_time)
        else:
            elapsed_weight += weight
            current_end = float(start_time) + total_duration * elapsed_weight / total_weight

        timings.append((round(current_start, 3), round(max(current_start, current_end), 3)))
        current_start = current_end

    return timings

def split_display_cue(start_time, end_time, text):
    chunks = split_display_text_into_chunks(text)
    normalized_text = normalize_display_text(text)

    if len(chunks) > 1:
        total_duration = float(end_time) - float(start_time)
        if total_duration / len(chunks) < MIN_SPLIT_CHUNK_DURATION:
            return [{
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "text": normalized_text,
            }]

    timings = allocate_chunk_timings(start_time, end_time, chunks)

    if any((chunk_end - chunk_start) < MIN_SPLIT_CHUNK_DURATION for chunk_start, chunk_end in timings):
        return [{
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "text": normalized_text,
        }]

    split_cues = []

    for (chunk_start, chunk_end), chunk_text in zip(timings, chunks):
        split_cues.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text,
        })

    return split_cues

def create_srt_content(subtitles, lang_key):
    """Generates standard 1-to-1 SRTs."""
    srt_output = []
    for idx, sub in enumerate(subtitles, 1):
        start = format_timestamp(sub['start'])
        end = format_timestamp(sub['end'])
        text = sub.get(lang_key, "")
        if not isinstance(text, str): text = str(text)
        srt_output.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_output)

def generate_english_srt(subtitles):
    """
    Auto-formats English subtitles.
    Merges repaired fragment windows and splits oversized display cues back into
    timed subtitle chunks of at most two lines.
    """
    srt_output = []
    cue_index = 1

    for display_cue in iterate_display_cues(subtitles):
        split_cues = split_display_cue(display_cue["start"], display_cue["end"], display_cue["text"])
        if any(len(wrap_display_lines(split_cue["text"])) > MAX_DISPLAY_LINES for split_cue in split_cues):
            split_cues = []
            for source_subtitle in display_cue.get("subtitles", []):
                split_cues.extend(
                    split_display_cue(
                        source_subtitle["start"],
                        source_subtitle["end"],
                        source_subtitle.get("text_en", ""),
                    )
                )

        for split_cue in split_cues:
            formatted_text = "\n".join(wrap_display_lines(split_cue["text"]))
            s = format_timestamp(split_cue["start"])
            e = format_timestamp(split_cue["end"])
            srt_output.append(f"{cue_index}\n{s} --> {e}\n{formatted_text}\n")
            cue_index += 1

    return "\n".join(srt_output)

def unwrap_text(data):
    if isinstance(data, str): return data
    if isinstance(data, list):
        return unwrap_text(data[0]) if len(data) > 0 else ""
    if isinstance(data, dict):
        return unwrap_text(list(data.values())[0]) if len(data) > 0 else ""
    return str(data)

def extract_json_array(raw_text):
    """
    Strips conversational filler, markdown, and <think> blocks.
    It scientifically extracts only the actual array brackets.
    """
    text = raw_text.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def call_ollama(model_name, prompt, system_prompt=SYSTEM_PROMPT, options=None):
    try:
        return call_ollama_model(
            model_name,
            prompt,
            system_prompt=system_prompt,
            options=options,
            timeout=600,
        )
    except Exception as e:
        tqdm.write(f"\n[!] API Error: {str(e)}")
    return None

def translate_single_line(text_jp, previous_context, next_context=None, series_context=None, fallback_title=None, model_name=None):
    model_name = model_name or MODEL_NAME
    if uses_direct_translation_model(model_name):
        prompt = build_direct_translation_prompt(
            text_jp,
            previous_context=previous_context,
            next_context=next_context,
            series_context=series_context,
            fallback_title=fallback_title,
        )
        raw_response = call_ollama(model_name, prompt, system_prompt="", options={"temperature": 0.0})
        return normalize_display_text(raw_response) if raw_response else "[Translation Failed]"

    context_text = format_context_list(previous_context)
    next_context_text = format_context_list(next_context or [])
    series_context_block = build_series_context_prompt_block(series_context, fallback_title=fallback_title)
    user_prompt = (
        f"{series_context_block}"
        f"PREVIOUS ENGLISH CONTEXT:\n{context_text}\n\n"
        "UPCOMING JAPANESE CONTEXT (for coherence only, do not translate unless it is in the main array):\n"
        f"{next_context_text}\n\n"
        f"TRANSLATE (Return string only):\n{json.dumps([text_jp], ensure_ascii=False)}"
    )
    raw_response = call_ollama(model_name, user_prompt)
    if raw_response:
        try:
            clean_json = extract_json_array(raw_response)
            parsed = json.loads(clean_json)
            return unwrap_text(parsed)
        except:
            pass
    return normalize_display_text(raw_response) if raw_response else "[Translation Failed]"

def translate_batch_robust(current_lines, previous_context_lines, next_context_lines, series_context=None, fallback_title=None, model_name=None):
    model_name = model_name or MODEL_NAME
    if uses_direct_translation_model(model_name):
        cleaned_results = []
        temp_context = [x.get('text_en', '') for x in previous_context_lines if x.get('text_en', '').strip()]

        for idx, item in enumerate(current_lines):
            remaining_context = [x['text_jp'] for x in current_lines[idx + 1:idx + 1 + LOOKAHEAD_SIZE]]
            if len(remaining_context) < LOOKAHEAD_SIZE:
                remaining_context.extend(x['text_jp'] for x in next_context_lines[:LOOKAHEAD_SIZE - len(remaining_context)])

            result = translate_single_line(
                item['text_jp'],
                temp_context[-CONTEXT_SIZE:],
                remaining_context,
                series_context=series_context,
                fallback_title=fallback_title,
                model_name=model_name,
            )
            cleaned_results.append(result)
            temp_context.append(result)

        return [{"en": txt} for txt in cleaned_results]

    context_text = format_context_list([x.get('text_en', '') for x in previous_context_lines])
    next_context_text = format_context_list([x['text_jp'] for x in next_context_lines])
    to_translate = [x['text_jp'] for x in current_lines]
    series_context_block = build_series_context_prompt_block(series_context, fallback_title=fallback_title)

    user_prompt = (
        f"{series_context_block}"
        f"PREVIOUS ENGLISH CONTEXT:\n{context_text}\n\n"
        f"UPCOMING JAPANESE CONTEXT (for coherence only, do not translate unless it is in the main array):\n{next_context_text}\n\n"
        f"TRANSLATE THESE {len(to_translate)} LINES:\n{json.dumps(to_translate, ensure_ascii=False)}"
    )

    raw_response = call_ollama(model_name, user_prompt)
    batch_success = False
    cleaned_results = []

    if raw_response:
        try:
            clean_json = extract_json_array(raw_response)
            parsed = json.loads(clean_json)

            if isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        parsed = val
                        break
                if isinstance(parsed, dict):
                     parsed = list(parsed.values())

            if isinstance(parsed, list) and len(parsed) == len(current_lines):
                cleaned_results = [unwrap_text(x) for x in parsed]
                batch_success = True
        except:
            pass

    if not batch_success:
        safe_raw = (raw_response[:50] + "...") if raw_response else "No Response (Service Unavailable)"
        tqdm.write(f"\n[!] Batch Failed. Raw snippet: {safe_raw}")
        tqdm.write("    -> activating self-healing...")

        cleaned_results = []
        temp_context = [x.get('text_en', '') for x in previous_context_lines]
        for idx, item in enumerate(current_lines):
            remaining_context = [x['text_jp'] for x in current_lines[idx + 1:idx + 1 + LOOKAHEAD_SIZE]]
            if len(remaining_context) < LOOKAHEAD_SIZE:
                remaining_context.extend(x['text_jp'] for x in next_context_lines[:LOOKAHEAD_SIZE - len(remaining_context)])

            res = translate_single_line(
                item['text_jp'],
                temp_context[-3:],
                remaining_context,
                series_context=series_context,
                fallback_title=fallback_title,
                model_name=model_name,
            )
            cleaned_results.append(res)
            temp_context.append(res)

    return [{"en": txt} for txt in cleaned_results]

def normalize_english_word(word):
    return re.sub(r"[^A-Za-z']+", "", word).lower()

def ends_with_dangling_word(text):
    stripped = text.strip()
    if not stripped:
        return False

    last_word = normalize_english_word(stripped.split()[-1])
    return last_word in FRAGMENTARY_ENDINGS

def looks_fragmentary(text):
    stripped = text.strip()
    if not stripped:
        return False

    if stripped.endswith((",", ":", ";")):
        return True

    return ends_with_dangling_word(stripped)

def starts_like_continuation(text):
    stripped = text.lstrip(" \t\r\n\"'“”‘’([{")
    if not stripped:
        return False

    if stripped[0].islower():
        return True

    first_word = normalize_english_word(stripped.split()[0])
    return first_word in FRAGMENTARY_ENDINGS

def can_extend_fragment_window(subtitles, start_idx, end_idx):
    next_idx = end_idx + 1
    if next_idx >= len(subtitles):
        return False

    if next_idx - start_idx + 1 > FRAGMENT_WINDOW_SIZE:
        return False

    if subtitles[next_idx]["start"] - subtitles[end_idx]["end"] > FRAGMENT_WINDOW_MAX_GAP:
        return False

    current_text = subtitles[end_idx].get("text_en", "")
    next_text = subtitles[next_idx].get("text_en", "")
    if not current_text.strip() or not next_text.strip():
        return False

    return looks_fragmentary(current_text) or starts_like_continuation(next_text)

def build_fragment_window(subtitles, start_idx):
    end_idx = start_idx
    while can_extend_fragment_window(subtitles, start_idx, end_idx):
        end_idx += 1
    return subtitles[start_idx:end_idx + 1]

def collect_existing_display_windows(subtitles):
    windows = {}
    idx = 0

    while idx < len(subtitles):
        subtitle = subtitles[idx]
        group_id = subtitle.get("display_group_id")
        if not group_id or not subtitle.get("display_text_en"):
            idx += 1
            continue

        end_idx = idx
        while (
            end_idx + 1 < len(subtitles)
            and subtitles[end_idx + 1].get("display_group_id") == group_id
        ):
            end_idx += 1

        windows[idx] = end_idx
        idx = end_idx + 1

    return windows

def normalize_display_text(text):
    clean_text = text.replace("```json", "").replace("```", "").strip()
    if not clean_text:
        return ""

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, str):
            clean_text = parsed
        elif isinstance(parsed, list) and parsed:
            clean_text = unwrap_text(parsed[0])
    except Exception:
        pass

    clean_text = " ".join(line.strip() for line in clean_text.splitlines() if line.strip())
    clean_text = clean_text.strip("\"'")
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()

def translate_combined_window(window_subtitles, series_context=None, fallback_title=None, model_name=None):
    model_name = model_name or MODEL_NAME
    jp_lines = [sub["text_jp"] for sub in window_subtitles]
    en_lines = [sub.get("text_en", "") for sub in window_subtitles]

    if uses_direct_translation_model(model_name):
        combined_text = "\n".join(jp_lines)
        prompt = build_direct_display_prompt(combined_text, series_context=series_context, fallback_title=fallback_title)
        raw_response = call_ollama(model_name, prompt, system_prompt="", options={"temperature": 0.0})
        if not raw_response:
            return None
        merged_text = normalize_display_text(raw_response)
        return merged_text or None

    series_context_block = build_series_context_prompt_block(series_context, fallback_title=fallback_title)
    prompt = (
        f"{series_context_block}"
        "Create ONE natural English subtitle line for the combined meaning of these consecutive Japanese subtitle cues. "
        "This is for the final viewer-facing subtitle, so you do NOT need to preserve one English line per cue. "
        "Preserve the meaning, keep it concise, and return ONLY the English sentence with no JSON and no commentary.\n\n"
        f"JAPANESE CUES:\n{json.dumps(jp_lines, ensure_ascii=False)}\n\n"
        f"CURRENT ENGLISH CUES:\n{json.dumps(en_lines, ensure_ascii=False)}"
    )

    raw_response = call_ollama(
        model_name,
        prompt,
        system_prompt=DISPLAY_SYSTEM_PROMPT,
        options={"temperature": 0.0},
    )
    if not raw_response:
        return None

    merged_text = normalize_display_text(raw_response)
    return merged_text or None

def split_translation_across_cues(merged_text, cue_count):
    normalized = normalize_display_text(merged_text)
    if not normalized:
        return None

    if cue_count <= 1:
        return [normalized]

    words = normalized.split()
    if len(words) < cue_count:
        return None

    target_words = len(words) / cue_count
    best_parts = None
    best_score = None

    for split_points in itertools.combinations(range(1, len(words)), cue_count - 1):
        previous = 0
        parts = []
        for split_point in split_points + (len(words),):
            parts.append(" ".join(words[previous:split_point]).strip())
            previous = split_point

        if any(not part for part in parts):
            continue

        score = 0.0
        for part_index, part in enumerate(parts):
            word_count = len(part.split())
            score += (word_count - target_words) ** 2

            if word_count < 2:
                score += 12.0

            if part_index < len(parts) - 1:
                if ends_with_dangling_word(part):
                    score += 10.0
                if part.endswith((",", ";", ":")):
                    score -= 2.0

        if best_score is None or score < best_score:
            best_score = score
            best_parts = parts

    return best_parts

def repair_translation_window(window_subtitles, series_context=None, fallback_title=None, model_name=None):
    merged_text = translate_combined_window(
        window_subtitles,
        series_context=series_context,
        fallback_title=fallback_title,
        model_name=model_name,
    )
    if not merged_text:
        return None

    split_lines = split_translation_across_cues(merged_text, len(window_subtitles))
    if not split_lines:
        return None

    return {
        "lines": split_lines,
        "display_text": merged_text,
    }

def repair_fragmented_translations(subtitles, series_context=None, fallback_title=None, model_name=None):
    repair_count = 0
    idx = 0
    existing_windows = collect_existing_display_windows(subtitles)

    for subtitle in subtitles:
        subtitle.pop("display_group_id", None)
        subtitle.pop("display_text_en", None)

    while idx < len(subtitles):
        if idx in existing_windows:
            window = subtitles[idx:existing_windows[idx] + 1]
        else:
            if not looks_fragmentary(subtitles[idx].get("text_en", "")):
                idx += 1
                continue
            window = build_fragment_window(subtitles, idx)

        repaired = repair_translation_window(
            window,
            series_context=series_context,
            fallback_title=fallback_title,
            model_name=model_name,
        )
        if repaired:
            group_id = f"{idx}-{idx + len(window) - 1}"
            for offset, repaired_text in enumerate(repaired["lines"]):
                subtitles[idx + offset]["text_en"] = repaired_text
                subtitles[idx + offset]["display_group_id"] = group_id
                if offset == 0:
                    subtitles[idx + offset]["display_text_en"] = repaired["display_text"]
            repair_count += 1
            idx += len(window)
        else:
            idx += 1

    return repair_count

def sync_translated_cache(source_data, translated_data):
    source_subs = source_data.get("subtitles", [])
    translated_subs = translated_data.get("subtitles", [])

    if len(source_subs) != len(translated_subs):
        return None, False

    synced_subs = []
    timings_changed = False

    for source_sub, translated_sub in zip(source_subs, translated_subs):
        if source_sub.get("text_jp") != translated_sub.get("text_jp"):
            return None, False

        if (
            source_sub.get("start") != translated_sub.get("start")
            or source_sub.get("end") != translated_sub.get("end")
            or source_sub.get("text_romaji") != translated_sub.get("text_romaji")
        ):
            timings_changed = True

        synced_sub = dict(translated_sub)
        synced_sub["start"] = source_sub["start"]
        synced_sub["end"] = source_sub["end"]
        synced_sub["text_jp"] = source_sub["text_jp"]
        synced_sub["text_romaji"] = source_sub.get("text_romaji", translated_sub.get("text_romaji", ""))
        synced_subs.append(synced_sub)

    synced_data = dict(translated_data)
    synced_meta = dict(source_data.get("meta", translated_data.get("meta", {})))
    synced_meta["translation_prompt_version"] = translated_data.get("meta", {}).get("translation_prompt_version", TRANSLATION_PROMPT_VERSION)
    synced_meta["translation_repair_version"] = translated_data.get("meta", {}).get("translation_repair_version", TRANSLATION_REPAIR_VERSION)
    synced_meta["series_title"] = translated_data.get("meta", {}).get("series_title")
    synced_meta["translation_model"] = translated_data.get("meta", {}).get("translation_model", MODEL_NAME)
    synced_meta["translation_pipeline_mode"] = translated_data.get("meta", {}).get("translation_pipeline_mode", CURRENT_PIPELINE_MODE)
    synced_meta["translation_primary_model"] = translated_data.get("meta", {}).get("translation_primary_model", translated_data.get("meta", {}).get("translation_model", MODEL_NAME))
    synced_meta["translation_secondary_model"] = translated_data.get("meta", {}).get("translation_secondary_model", DEFAULT_SECONDARY_TRANSLATION_MODEL)
    synced_meta["translation_judge_model"] = translated_data.get("meta", {}).get("translation_judge_model", DEFAULT_JUDGE_MODEL)
    synced_meta["series_context_hash"] = translated_data.get("meta", {}).get("series_context_hash")
    synced_meta["series_context_source"] = translated_data.get("meta", {}).get("series_context_source")
    synced_data["meta"] = synced_meta
    synced_data["subtitles"] = synced_subs
    return synced_data, timings_changed


def get_artifact_dir(base_name):
    artifact_dir = f"{base_name}_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir


def save_json_artifact(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def collect_series_context_terms(series_context, fallback_title=None):
    terms = []
    canonical_title = canonical_series_title(series_context, fallback_title=fallback_title)
    if canonical_title:
        terms.append(canonical_title)

    if series_context:
        terms.extend(series_context.get("aliases", []))
        for character in series_context.get("characters", []):
            terms.extend([character.get("name_en"), character.get("name_ja")])
            terms.extend(character.get("aliases", []))
        for glossary in series_context.get("glossary", []):
            terms.extend([glossary.get("preferred_en"), glossary.get("source_ja")])
            terms.extend(glossary.get("aliases", []))

    cleaned_terms = []
    seen = set()
    for term in terms:
        text = str(term or "").strip()
        if len(text) < 2:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned_terms.append(text)
    return cleaned_terms


def subtitle_references_series_context(subtitle, series_context, fallback_title=None):
    text_jp = str(subtitle.get("text_jp", ""))
    if not text_jp:
        return False

    for term in collect_series_context_terms(series_context, fallback_title=fallback_title):
        if term in text_jp:
            return True
    return False


def needs_translation_review(subtitle, pipeline_mode, series_context, fallback_title=None):
    if pipeline_mode == MAX_PIPELINE_MODE:
        return True

    text_en = str(subtitle.get("text_en", "")).strip()
    if not text_en or text_en == "[Translation Failed]":
        return True
    if looks_fragmentary(text_en):
        return True
    return subtitle_references_series_context(subtitle, series_context, fallback_title=fallback_title)


def judge_translation_choice(
    subtitle,
    primary_text,
    secondary_text,
    previous_context,
    next_context,
    judge_model,
    series_context=None,
    fallback_title=None,
):
    context_block = build_series_context_prompt_block(series_context, fallback_title=fallback_title)
    prompt = (
        f"{context_block}"
        "Choose the better English subtitle candidate for the Japanese line below.\n"
        "Rules:\n"
        "- Preserve the Japanese meaning.\n"
        "- Prefer canonical series terminology when the Japanese supports it.\n"
        "- Do not invent a new translation.\n"
        "- Reply ONLY JSON like {\"choice\":\"primary\"} or {\"choice\":\"secondary\"}.\n\n"
        f"JAPANESE: {subtitle.get('text_jp', '')}\n"
        f"PRIMARY: {primary_text}\n"
        f"SECONDARY: {secondary_text}\n"
        f"PREVIOUS ENGLISH CONTEXT: {' | '.join(previous_context[-CONTEXT_SIZE:]) if previous_context else '(none)'}\n"
        f"UPCOMING JAPANESE CONTEXT: {' | '.join(next_context[:LOOKAHEAD_SIZE]) if next_context else '(none)'}"
    )
    raw_response = call_ollama(
        judge_model,
        prompt,
        system_prompt="Return only valid JSON. Pick either the primary or secondary candidate.",
        options={"temperature": 0.0},
    )
    match = re.search(r"\{.*\}", raw_response or "", re.DOTALL)
    if not match:
        return "primary"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "primary"
    choice = str(parsed.get("choice", "primary")).strip().lower()
    if choice not in {"primary", "secondary"}:
        return "primary"
    return choice


def apply_translation_review(
    subtitles,
    pipeline_mode,
    primary_model,
    secondary_model,
    judge_model,
    series_context=None,
    fallback_title=None,
):
    if not mode_uses_translation_review(pipeline_mode):
        return []

    review_log = []
    for idx, subtitle in enumerate(subtitles):
        should_review = needs_translation_review(
            subtitle,
            pipeline_mode,
            series_context,
            fallback_title=fallback_title,
        )
        if not should_review:
            continue

        previous_context = [item.get("text_en", "") for item in subtitles[max(0, idx - CONTEXT_SIZE):idx]]
        next_context = [item.get("text_jp", "") for item in subtitles[idx + 1:idx + 1 + LOOKAHEAD_SIZE]]
        primary_text = subtitle.get("text_en", "")
        secondary_text = translate_single_line(
            subtitle.get("text_jp", ""),
            previous_context,
            next_context,
            series_context=series_context,
            fallback_title=fallback_title,
            model_name=secondary_model,
        )
        choice = judge_translation_choice(
            subtitle,
            primary_text,
            secondary_text,
            previous_context,
            next_context,
            judge_model,
            series_context=series_context,
            fallback_title=fallback_title,
        )
        subtitle["text_en_primary"] = primary_text
        subtitle["text_en_secondary"] = secondary_text
        subtitle["text_en"] = secondary_text if choice == "secondary" else primary_text
        review_log.append(
            {
                "index": idx,
                "choice": choice,
                "primary": primary_text,
                "secondary": secondary_text,
                "jp": subtitle.get("text_jp", ""),
            }
        )
    return review_log


def normalize_glossary_line(
    subtitle,
    previous_context,
    next_context,
    judge_model,
    series_context=None,
    fallback_title=None,
):
    prompt = (
        f"{build_series_context_prompt_block(series_context, fallback_title=fallback_title)}"
        "Revise the current English subtitle line only if needed to enforce canonical series/title/character terminology.\n"
        "Rules:\n"
        "- Keep the meaning unchanged.\n"
        "- Keep the line concise for subtitles.\n"
        "- If the current line is already correct, return it unchanged.\n"
        "- Return ONLY the final English subtitle line.\n\n"
        f"JAPANESE: {subtitle.get('text_jp', '')}\n"
        f"CURRENT ENGLISH: {subtitle.get('text_en', '')}\n"
        f"PREVIOUS ENGLISH CONTEXT: {' | '.join(previous_context[-CONTEXT_SIZE:]) if previous_context else '(none)'}\n"
        f"UPCOMING JAPANESE CONTEXT: {' | '.join(next_context[:LOOKAHEAD_SIZE]) if next_context else '(none)'}"
    )
    raw_response = call_ollama(
        judge_model,
        prompt,
        system_prompt=DISPLAY_SYSTEM_PROMPT,
        options={"temperature": 0.0},
    )
    normalized = normalize_display_text(raw_response) if raw_response else ""
    return normalized or subtitle.get("text_en", "")


def apply_consistency_normalization(subtitles, judge_model, series_context=None, fallback_title=None):
    normalization_log = []
    for idx, subtitle in enumerate(subtitles):
        if not subtitle_references_series_context(subtitle, series_context, fallback_title=fallback_title):
            continue
        previous_context = [item.get("text_en", "") for item in subtitles[max(0, idx - CONTEXT_SIZE):idx]]
        next_context = [item.get("text_jp", "") for item in subtitles[idx + 1:idx + 1 + LOOKAHEAD_SIZE]]
        original_text = subtitle.get("text_en", "")
        normalized_text = normalize_glossary_line(
            subtitle,
            previous_context,
            next_context,
            judge_model,
            series_context=series_context,
            fallback_title=fallback_title,
        )
        if normalized_text != original_text:
            subtitle["text_en"] = normalized_text
            normalization_log.append(
                {
                    "index": idx,
                    "before": original_text,
                    "after": normalized_text,
                    "jp": subtitle.get("text_jp", ""),
                }
            )
    return normalization_log

def main(
    input_json_file,
    translation_model=None,
    pipeline_mode=CURRENT_PIPELINE_MODE,
    translation_secondary_model=DEFAULT_SECONDARY_TRANSLATION_MODEL,
    judge_model=DEFAULT_JUDGE_MODEL,
    series_context_path=None,
    context_model=DEFAULT_CONTEXT_MODEL,
):
    global MODEL_NAME

    primary_model = translation_model or DEFAULT_TRANSLATION_MODEL
    MODEL_NAME = primary_model
    secondary_model = translation_secondary_model or DEFAULT_SECONDARY_TRANSLATION_MODEL

    if not os.path.exists(input_json_file):
        print(f"Error: File {input_json_file} not found.")
        return

    base_name = os.path.splitext(input_json_file)[0]
    translated_json_path = f"{base_name}_translated.json"
    artifact_dir = get_artifact_dir(base_name)
    series_title = infer_series_title(input_json_file)
    emit_status("loading_series_context", file=input_json_file)
    try:
        series_context, series_context_source, series_context_cache_path = resolve_series_context(
            input_json_file,
            override_path=series_context_path,
            allow_fetch=(pipeline_mode != CURRENT_PIPELINE_MODE),
            enrich_model=context_model,
        )
    except Exception as exc:
        tqdm.write(f"\n[!] Series context fetch failed: {exc}")
        series_context, series_context_source, series_context_cache_path = resolve_series_context(
            input_json_file,
            override_path=series_context_path,
            allow_fetch=False,
        )
    emit_status(
        "loaded_series_context",
        source=series_context_source,
        cache_path=series_context_cache_path,
        title=canonical_series_title(series_context, fallback_title=series_title),
    )
    save_json_artifact(os.path.join(artifact_dir, "series_context.runtime.json"), series_context)
    context_hash = series_context_hash(series_context, fallback_title=series_title)

    with open(input_json_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    data = source_data
    original_subs = source_data['subtitles']
    needs_translation = True

    # --- 1. CACHE CHECK ---
    if os.path.exists(translated_json_path):
        with open(translated_json_path, 'r', encoding='utf-8') as f:
            translated_data = json.load(f)

        translated_meta = translated_data.get("meta", {})
        prompt_version_matches = (
            translated_meta.get("translation_prompt_version") == TRANSLATION_PROMPT_VERSION
            and translated_meta.get("series_title") == series_title
            and translated_meta.get("translation_pipeline_mode", CURRENT_PIPELINE_MODE) == pipeline_mode
            and translated_meta.get("translation_primary_model", translated_meta.get("translation_model")) == primary_model
            and translated_meta.get("translation_secondary_model", DEFAULT_SECONDARY_TRANSLATION_MODEL) == secondary_model
            and translated_meta.get("translation_judge_model", DEFAULT_JUDGE_MODEL) == judge_model
            and translated_meta.get("series_context_hash") == context_hash
        )
        repair_version_matches = translated_meta.get("translation_repair_version") == TRANSLATION_REPAIR_VERSION

        if prompt_version_matches:
            synced_cache, timings_changed = sync_translated_cache(source_data, translated_data)
        else:
            synced_cache, timings_changed = None, False

        if synced_cache is not None:
            data = synced_cache
            original_subs = data['subtitles']
            needs_translation = False

            repaired_count = 0
            normalization_log = []
            if not repair_version_matches:
                repaired_count = repair_fragmented_translations(
                    original_subs,
                    series_context=series_context,
                    fallback_title=series_title,
                    model_name=primary_model,
                )
                if mode_uses_translation_review(pipeline_mode):
                    normalization_log = apply_consistency_normalization(
                        original_subs,
                        judge_model,
                        series_context=series_context,
                        fallback_title=series_title,
                    )
                data.setdefault("meta", {})
                data["meta"]["translation_prompt_version"] = TRANSLATION_PROMPT_VERSION
                data["meta"]["translation_repair_version"] = TRANSLATION_REPAIR_VERSION
                data["meta"]["series_title"] = series_title
                data["meta"]["translation_model"] = primary_model
                data["meta"]["translation_pipeline_mode"] = pipeline_mode
                data["meta"]["translation_primary_model"] = primary_model
                data["meta"]["translation_secondary_model"] = secondary_model
                data["meta"]["translation_judge_model"] = judge_model
                data["meta"]["series_context_hash"] = context_hash
                data["meta"]["series_context_source"] = series_context_source
                data["meta"]["series_context_model"] = context_model

            if timings_changed or repaired_count or normalization_log:
                if repaired_count or normalization_log:
                    print("\n[!] CACHE UPDATED: Reused translations, refreshed timings, repaired fragmentary windows, and refreshed consistency normalization.\n")
                else:
                    print("\n[!] CACHE UPDATED: Reused translations and pulled latest subtitle timings from source JSON.\n")
                with open(translated_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                print("\n[!] CACHE LOADED: Translated JSON found. Skipping Ollama generation.\n")
        else:
            print("\n[!] CACHE INVALIDATED: Transcript text, mode, models, or series context changed. Regenerating translations.\n")

    if needs_translation:
        # --- 2. TRANSLATION LOOP ---
        total_subs = len(original_subs)
        emit_status("translating_primary", model=primary_model)
        print(f"Translating {total_subs} lines ({pipeline_mode.title()} mode).")
        print("------------------------------------------------")

        with tqdm(total=total_subs, unit="lines") as pbar:
            for i in range(0, total_subs, BATCH_SIZE):
                batch = original_subs[i : i + BATCH_SIZE]
                context_start = max(0, i - CONTEXT_SIZE)
                prev_context = original_subs[context_start : i]
                next_context = original_subs[i + BATCH_SIZE : i + BATCH_SIZE + LOOKAHEAD_SIZE]

                translations = translate_batch_robust(
                    batch,
                    prev_context,
                    next_context,
                    series_context=series_context,
                    fallback_title=series_title,
                    model_name=primary_model,
                )

                for j, trans in enumerate(translations):
                    if j < len(batch):
                        batch[j]['text_en'] = trans['en']
                        batch[j]['text_en_primary'] = trans['en']

                if translations:
                    last_text = str(translations[-1]['en'])
                    pbar.set_description(f"Last: {last_text[:20].replace('\n', ' ')}...")

                pbar.update(len(batch))

        save_json_artifact(os.path.join(artifact_dir, "translation_primary.json"), original_subs)

        review_log = []
        if mode_uses_translation_review(pipeline_mode):
            emit_status("reviewing_translations", model=judge_model)
            review_log = apply_translation_review(
                original_subs,
                pipeline_mode,
                primary_model,
                secondary_model,
                judge_model,
                series_context=series_context,
                fallback_title=series_title,
            )
            if review_log:
                save_json_artifact(os.path.join(artifact_dir, "translation_review.json"), review_log)

        repair_fragmented_translations(
            original_subs,
            series_context=series_context,
            fallback_title=series_title,
            model_name=primary_model,
        )

        normalization_log = []
        if mode_uses_translation_review(pipeline_mode):
            emit_status("normalizing_translation_terms", model=judge_model)
            normalization_log = apply_consistency_normalization(
                original_subs,
                judge_model,
                series_context=series_context,
                fallback_title=series_title,
            )
            if normalization_log:
                save_json_artifact(os.path.join(artifact_dir, "translation_normalization.json"), normalization_log)

        save_json_artifact(os.path.join(artifact_dir, "translation_final.json"), original_subs)

        # Save the translated cache
        data.setdefault("meta", {})
        data["meta"]["translation_prompt_version"] = TRANSLATION_PROMPT_VERSION
        data["meta"]["translation_repair_version"] = TRANSLATION_REPAIR_VERSION
        data["meta"]["series_title"] = series_title
        data["meta"]["translation_model"] = primary_model
        data["meta"]["translation_pipeline_mode"] = pipeline_mode
        data["meta"]["translation_primary_model"] = primary_model
        data["meta"]["translation_secondary_model"] = secondary_model
        data["meta"]["translation_judge_model"] = judge_model
        data["meta"]["series_context_hash"] = context_hash
        data["meta"]["series_context_source"] = series_context_source
        data["meta"]["series_context_model"] = context_model
        with open(translated_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # --- 3. EXPORT SRT FILES ---
    emit_status("exporting_subtitles")
    print("\nExporting SRT files...")

    outputs_raw = [
        (f"{base_name}.jp.srt", "text_jp"),
        (f"{base_name}.romaji.srt", "text_romaji"),
        (f"{base_name}.en.raw.srt", "text_en"),
    ]
    for filename, key in outputs_raw:
        content = create_srt_content(original_subs, key)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"- {filename}")

    en_filename = f"{base_name}.en.srt"
    en_content = generate_english_srt(original_subs)
    with open(en_filename, "w", encoding="utf-8") as f:
        f.write(en_content)
    print(f"- {en_filename} (Merged + Timed 2-Line Split)")
    emit_status("done")


def parse_args():
    parser = argparse.ArgumentParser(description="Translate Japanese subtitle JSON into English subtitle files.")
    parser.add_argument("json_file", help="Path to the source subtitle JSON file.")
    parser.add_argument(
        "--translation-model",
        default=DEFAULT_TRANSLATION_MODEL,
        help="Primary Ollama model name to use for translation.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=SUPPORTED_PIPELINE_MODES,
        default=CURRENT_PIPELINE_MODE,
        help="Pipeline profile to run: current, balanced, or max.",
    )
    parser.add_argument(
        "--translation-secondary-model",
        default=DEFAULT_SECONDARY_TRANSLATION_MODEL,
        help="Secondary Ollama model used for candidate generation in review-heavy modes.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Ollama model used for review and terminology normalization.",
    )
    parser.add_argument(
        "--series-context",
        default="",
        help="Optional path to a local series_context.json override file.",
    )
    parser.add_argument(
        "--context-model",
        default=DEFAULT_CONTEXT_MODEL,
        help="Ollama model used to enrich missing global series context when needed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.json_file,
        translation_model=args.translation_model,
        pipeline_mode=args.pipeline_mode,
        translation_secondary_model=args.translation_secondary_model,
        judge_model=args.judge_model,
        series_context_path=args.series_context or None,
        context_model=args.context_model,
    )
