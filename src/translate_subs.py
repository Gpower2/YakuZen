import sys
import json
import os
import itertools
import requests
import time
import textwrap
import re
from functools import lru_cache
from datetime import timedelta
from tqdm import tqdm

# ==========================================
# --- CONFIGURATION & LLM PARAMETERS ---
# ==========================================

# OLLAMA_URL
# Default: "http://localhost:11434/api/generate"
# Description: The local API endpoint for Ollama.
# Reasoning: Assumes Ollama is running locally on the standard port.
OLLAMA_URL = "http://localhost:11434/api/generate"

# MODEL_NAME
# Default: "llama3" (varies by user)
# Description: The LLM model to use for translation.
# Reasoning: Qwen3 14B remains the best same-tier local default here. Newer Qwen families
# on Ollama currently jump to much larger 27B/30B/35B classes, while TranslateGemma can
# still be useful as an optional translation-specialist fallback.
#MODEL_NAME = "sakura-anime"
#MODEL_NAME = "qwen2.5:14b"
#MODEL_NAME = "qwen2.5:32b"
MODEL_NAME = "qwen3:14b"
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

TRANSLATION_PROMPT_VERSION = 6
TRANSLATION_REPAIR_VERSION = 4
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
TITLE_NOISE_TOKENS = {
    "AMZN", "AMZNWEBRIP", "WEB", "WEBRIP", "WEBDL", "WEB-DL", "BD", "BDRIP", "BLURAY", "BRRIP",
    "DVDRIP", "HDRIP", "REMUX", "NF", "DSNP", "CR", "ATVP", "FUNI", "HULU", "1080P", "720P",
    "480P", "2160P", "4K", "UHD", "HEVC", "X264", "X265", "H264", "H265", "AAC", "DDP", "AC3",
    "TRUEHD", "DTS", "10BIT", "8BIT", "MULTI", "SUBS", "DUB", "RAW", "BK", "JPN", "ENG",
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

def infer_series_title(input_json_file):
    stem = os.path.splitext(os.path.basename(input_json_file))[0]
    tokens = [token for token in re.split(r"[._\-\s]+", stem) if token]
    title_tokens = []
    year_token = None

    for raw_token in tokens:
        token = raw_token.strip()
        upper_token = token.upper()

        if re.match(r"^(S\d{1,2}E\d{1,3}|E\d{1,3}|EP?\d{1,3}|OVA\d*|OAD\d*|NCOP\d*|NCED\d*)$", upper_token):
            break

        if re.match(r"^(19|20)\d{2}$", token):
            year_token = token
            continue

        if upper_token in TITLE_NOISE_TOKENS:
            if title_tokens:
                break
            continue

        if re.match(r"^\d{3,4}P$", upper_token) or re.match(r"^(Q\d|REMASTER)$", upper_token):
            if title_tokens:
                break
            continue

        title_tokens.append(token)

    if not title_tokens:
        fallback = re.sub(r"[._\-]+", " ", stem).strip()
        return fallback or stem

    title = " ".join(title_tokens).strip()
    if year_token:
        return f"{title} ({year_token})"
    return title

def format_context_list(items):
    clean_items = [str(item).strip() for item in items if str(item).strip()]
    if not clean_items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in clean_items)

def build_series_context_block(series_title):
    if not series_title:
        return ""
    return (
        "ANIME SERIES CONTEXT:\n"
        f"- Inferred series title from the filename: {series_title}\n"
        "- Use this only to keep character names, organizations, places, and technical terms consistent if you recognize the series.\n\n"
    )

def uses_direct_translation_model():
    return MODEL_NAME.lower().startswith("translategemma")

def build_translation_context_note(previous_context=None, next_context=None, series_title=None):
    notes = []

    if series_title:
        notes.append(
            f'This subtitle comes from the anime series "{series_title}". Keep character names, places, organizations, and technical terms consistent if you recognize them.'
        )

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

def build_direct_translation_prompt(text_jp, previous_context=None, next_context=None, series_title=None):
    context_sentence = ""
    if series_title:
        context_sentence = (
            f' This subtitle comes from the anime series "{series_title}". '
            "Keep character names, places, organizations, and technical terms consistent if you recognize them."
        )
    return (
        f"You are a professional {SOURCE_LANGUAGE_NAME} ({SOURCE_LANGUAGE_CODE}) to {TARGET_LANGUAGE_NAME} ({TARGET_LANGUAGE_CODE}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANGUAGE_NAME} text while adhering to {TARGET_LANGUAGE_NAME} grammar, vocabulary, and cultural sensitivities.{context_sentence} "
        " Prefer concise subtitle phrasing, and do not translate or summarize any context other than the main Japanese text below. "
        f"Produce only the {TARGET_LANGUAGE_NAME} translation, without any additional explanations or commentary. "
        f"Please translate the following {SOURCE_LANGUAGE_NAME} text into {TARGET_LANGUAGE_NAME}:\n\n\n"
        f"{text_jp}"
    )

def build_direct_display_prompt(text_jp, series_title=None):
    context_note = build_translation_context_note(series_title=series_title)
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

def call_ollama(prompt, system_prompt=SYSTEM_PROMPT, options=None):
    payload_options = {
        "temperature": 0.1,
        "num_ctx": 4096,
    }
    if options:
        payload_options.update(options)

    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        # Intentionally NOT using "format": "json" here to prevent grammar panic
        "options": payload_options,
    }
    try:
        # Timeout increased to 600s (10 minutes) to allow large models (like Sakura) to load into VRAM
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        if response.status_code == 200:
            return response.json().get('response', '')
    except Exception as e:
        tqdm.write(f"\n[!] API Error: {str(e)}")
    return None

def translate_single_line(text_jp, previous_context, next_context=None, series_title=None):
    if uses_direct_translation_model():
        prompt = build_direct_translation_prompt(
            text_jp,
            previous_context=previous_context,
            next_context=next_context,
            series_title=series_title,
        )
        raw_response = call_ollama(prompt, system_prompt="", options={"temperature": 0.0})
        return normalize_display_text(raw_response) if raw_response else "[Translation Failed]"

    context_text = format_context_list(previous_context)
    next_context_text = format_context_list(next_context or [])
    series_context = build_series_context_block(series_title)
    user_prompt = (
        f"{series_context}"
        f"PREVIOUS ENGLISH CONTEXT:\n{context_text}\n\n"
        "UPCOMING JAPANESE CONTEXT (for coherence only, do not translate unless it is in the main array):\n"
        f"{next_context_text}\n\n"
        f"TRANSLATE (Return string only):\n{json.dumps([text_jp], ensure_ascii=False)}"
    )
    raw_response = call_ollama(user_prompt)
    if raw_response:
        try:
            clean_json = extract_json_array(raw_response)
            parsed = json.loads(clean_json)
            return unwrap_text(parsed)
        except:
            pass
    return normalize_display_text(raw_response) if raw_response else "[Translation Failed]"

def translate_batch_robust(current_lines, previous_context_lines, next_context_lines, series_title=None):
    if uses_direct_translation_model():
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
                series_title=series_title,
            )
            cleaned_results.append(result)
            temp_context.append(result)

        return [{"en": txt} for txt in cleaned_results]

    context_text = format_context_list([x.get('text_en', '') for x in previous_context_lines])
    next_context_text = format_context_list([x['text_jp'] for x in next_context_lines])
    to_translate = [x['text_jp'] for x in current_lines]
    series_context = build_series_context_block(series_title)

    user_prompt = (
        f"{series_context}"
        f"PREVIOUS ENGLISH CONTEXT:\n{context_text}\n\n"
        f"UPCOMING JAPANESE CONTEXT (for coherence only, do not translate unless it is in the main array):\n{next_context_text}\n\n"
        f"TRANSLATE THESE {len(to_translate)} LINES:\n{json.dumps(to_translate, ensure_ascii=False)}"
    )

    raw_response = call_ollama(user_prompt)
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

            res = translate_single_line(item['text_jp'], temp_context[-3:], remaining_context, series_title=series_title)
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

def translate_combined_window(window_subtitles, series_title=None):
    jp_lines = [sub["text_jp"] for sub in window_subtitles]
    en_lines = [sub.get("text_en", "") for sub in window_subtitles]

    if uses_direct_translation_model():
        combined_text = "\n".join(jp_lines)
        prompt = build_direct_display_prompt(combined_text, series_title=series_title)
        raw_response = call_ollama(prompt, system_prompt="", options={"temperature": 0.0})
        if not raw_response:
            return None
        merged_text = normalize_display_text(raw_response)
        return merged_text or None

    series_context = build_series_context_block(series_title)
    prompt = (
        f"{series_context}"
        "Create ONE natural English subtitle line for the combined meaning of these consecutive Japanese subtitle cues. "
        "This is for the final viewer-facing subtitle, so you do NOT need to preserve one English line per cue. "
        "Preserve the meaning, keep it concise, and return ONLY the English sentence with no JSON and no commentary.\n\n"
        f"JAPANESE CUES:\n{json.dumps(jp_lines, ensure_ascii=False)}\n\n"
        f"CURRENT ENGLISH CUES:\n{json.dumps(en_lines, ensure_ascii=False)}"
    )

    raw_response = call_ollama(
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

def repair_translation_window(window_subtitles, series_title=None):
    merged_text = translate_combined_window(window_subtitles, series_title=series_title)
    if not merged_text:
        return None

    split_lines = split_translation_across_cues(merged_text, len(window_subtitles))
    if not split_lines:
        return None

    return {
        "lines": split_lines,
        "display_text": merged_text,
    }

def repair_fragmented_translations(subtitles, series_title=None):
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

        repaired = repair_translation_window(window, series_title=series_title)
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
    synced_data["meta"] = synced_meta
    synced_data["subtitles"] = synced_subs
    return synced_data, timings_changed

def main(input_json_file):
    if not os.path.exists(input_json_file):
        print(f"Error: File {input_json_file} not found.")
        return

    base_name = os.path.splitext(input_json_file)[0]
    translated_json_path = f"{base_name}_translated.json"
    series_title = infer_series_title(input_json_file)

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
            and translated_meta.get("translation_model") == MODEL_NAME
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
            if not repair_version_matches:
                repaired_count = repair_fragmented_translations(original_subs, series_title=series_title)
                data.setdefault("meta", {})
                data["meta"]["translation_prompt_version"] = TRANSLATION_PROMPT_VERSION
                data["meta"]["translation_repair_version"] = TRANSLATION_REPAIR_VERSION
                data["meta"]["series_title"] = series_title
                data["meta"]["translation_model"] = MODEL_NAME

            if timings_changed or repaired_count:
                if repaired_count:
                    print("\n[!] CACHE UPDATED: Reused translations, refreshed timings, and repaired fragmentary translation windows.\n")
                else:
                    print("\n[!] CACHE UPDATED: Reused translations and pulled latest subtitle timings from source JSON.\n")
                with open(translated_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                print("\n[!] CACHE LOADED: Translated JSON found. Skipping Ollama generation.\n")
        else:
            print("\n[!] CACHE INVALIDATED: Transcript text, inferred series title, translation model, or translation prompt changed. Regenerating translations.\n")

    if needs_translation:
        # --- 2. TRANSLATION LOOP ---
        total_subs = len(original_subs)

        print(f"Translating {total_subs} lines (Robust V2 + Auto-Formatter).")
        print("------------------------------------------------")

        with tqdm(total=total_subs, unit="lines") as pbar:
            for i in range(0, total_subs, BATCH_SIZE):
                batch = original_subs[i : i + BATCH_SIZE]
                context_start = max(0, i - CONTEXT_SIZE)
                prev_context = original_subs[context_start : i]
                next_context = original_subs[i + BATCH_SIZE : i + BATCH_SIZE + LOOKAHEAD_SIZE]

                translations = translate_batch_robust(batch, prev_context, next_context, series_title=series_title)

                for j, trans in enumerate(translations):
                    if j < len(batch):
                        batch[j]['text_en'] = trans['en']

                if translations:
                    last_text = str(translations[-1]['en'])
                    pbar.set_description(f"Last: {last_text[:20].replace('\n', ' ')}...")

                pbar.update(len(batch))

        repair_fragmented_translations(original_subs, series_title=series_title)

        # Save the translated cache
        data.setdefault("meta", {})
        data["meta"]["translation_prompt_version"] = TRANSLATION_PROMPT_VERSION
        data["meta"]["translation_repair_version"] = TRANSLATION_REPAIR_VERSION
        data["meta"]["series_title"] = series_title
        data["meta"]["translation_model"] = MODEL_NAME
        with open(translated_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # --- 3. EXPORT SRT FILES ---
    print("\nExporting SRT files...")

    # Export 1-to-1 mapped SRTs (No mathematical splitting)
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

    # Export English using the merged display pass plus timed two-line splitting
    en_filename = f"{base_name}.en.srt"
    en_content = generate_english_srt(original_subs)
    with open(en_filename, "w", encoding="utf-8") as f:
        f.write(en_content)
    print(f"- {en_filename} (Merged + Timed 2-Line Split)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translate_subs.py <json_file>")
    else:
        main(sys.argv[1])
