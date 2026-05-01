import sys
import json
import os
import requests
import time
import textwrap
import re
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
# Reasoning: "sakura-anime" is heavily specialized for Japanese-to-English anime/VN localization.
# "qwen3:14b" and "qwen2.5:14b" are also excellent fallbacks that fit in 16GB VRAM.
#MODEL_NAME = "sakura-anime"
#MODEL_NAME = "qwen2.5:14b"
#MODEL_NAME = "qwen2.5:32b"
MODEL_NAME = "qwen3:14b"
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

# ==========================================
# --- SUBTITLE SPLITTING PARAMETERS ---
# ==========================================

# MAX_SUB_LENGTH
# Default: N/A
# Description: The maximum character count for a single English subtitle before we mathematically slice it into multiple timed segments.
# Reasoning: Set to 84. Two lines of 42 characters each is the industry maximum for readable subtitles. Anything longer gets split into a new timed screen.
MAX_SUB_LENGTH = 84

# CHUNK_WRAP_WIDTH
# Default: N/A
# Description: When splitting a massive string, how many characters should each new chunk target?
# Reasoning: Set to 75. Gives us a comfortable 9-character margin below the absolute 84 limit to account for natural word boundaries (preventing words from being sliced in half).
CHUNK_WRAP_WIDTH = 75

# LINE_WRAP_WIDTH
# Default: N/A
# Description: The maximum character width of a single text line on screen.
# Reasoning: Set to 42. Industry standard for video subtitles (Netflix, BBC, etc.) to prevent eye fatigue.
LINE_WRAP_WIDTH = 42

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a professional anime localizer.
Translate the input array of Japanese lines into a matching array of natural, conversational English strings.
RULES:
1. Output ONLY a JSON List of strings: ["string1", "string2"]
2. NO Objects. NO Keys. NO Dictionaries.
3. Maintain exact 1-to-1 alignment with the input.
4. Translate the full meaning without omitting details."""

# --- HELPER FUNCTIONS ---

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

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
    Mathematically splits massive text blocks into smaller, timed chunks,
    and automatically wraps text into standard 2-line subtitle format.
    """
    processed_subs = []
    
    for sub in subtitles:
        text = sub.get('text_en', '').strip()
        if not text:
            continue
            
        start = sub['start']
        end = sub['end']
        duration = end - start

        if len(text) > MAX_SUB_LENGTH:
            chunks = textwrap.wrap(text, width=CHUNK_WRAP_WIDTH)
            total_chars = sum(len(c) for c in chunks)
            current_start = start

            for chunk in chunks:
                chunk_duration = duration * (len(chunk) / max(total_chars, 1))
                chunk_end = current_start + chunk_duration
                formatted_chunk = "\n".join(textwrap.wrap(chunk, width=LINE_WRAP_WIDTH))

                processed_subs.append({
                    'start': current_start,
                    'end': chunk_end,
                    'text': formatted_chunk
                })
                current_start = chunk_end
        else:
            formatted_text = "\n".join(textwrap.wrap(text, width=LINE_WRAP_WIDTH))
            processed_subs.append({
                'start': start,
                'end': end,
                'text': formatted_text
            })

    srt_output = []
    for idx, sub in enumerate(processed_subs, 1):
        s = format_timestamp(sub['start'])
        e = format_timestamp(sub['end'])
        srt_output.append(f"{idx}\n{s} --> {e}\n{sub['text']}\n")
    
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

def call_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        # Intentionally NOT using "format": "json" here to prevent grammar panic
        "options": {
            "temperature": 0.1, 
            "num_ctx": 4096
        }
    }
    try:
        # Timeout increased to 600s (10 minutes) to allow large models (like Sakura) to load into VRAM
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        if response.status_code == 200:
            return response.json().get('response', '')
    except Exception as e:
        tqdm.write(f"\n[!] API Error: {str(e)}")
    return None

def translate_single_line(text_jp, previous_context):
    context_text = "\n".join([f"- {x}" for x in previous_context])
    user_prompt = f"CONTEXT:\n{context_text}\n\nTRANSLATE (Return string only):\n[\"{text_jp}\"]"
    raw_response = call_ollama(user_prompt)
    if raw_response:
        try:
            clean_json = extract_json_array(raw_response)
            parsed = json.loads(clean_json)
            return unwrap_text(parsed)
        except:
            pass
    return "[Translation Failed]"

def translate_batch_robust(current_lines, previous_context_lines):
    context_text = "\n".join([f"- {x.get('text_en', '')}" for x in previous_context_lines])
    to_translate = [x['text_jp'] for x in current_lines]
    
    user_prompt = f"CONTEXT:\n{context_text}\n\nTRANSLATE THESE {len(to_translate)} LINES:\n{json.dumps(to_translate, ensure_ascii=False)}"
    
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
        for item in current_lines:
            res = translate_single_line(item['text_jp'], temp_context[-3:])
            cleaned_results.append(res)
            temp_context.append(res)

    return [{"en": txt} for txt in cleaned_results]

def main(input_json_file):
    if not os.path.exists(input_json_file):
        print(f"Error: File {input_json_file} not found.")
        return

    base_name = os.path.splitext(input_json_file)[0]
    translated_json_path = f"{base_name}_translated.json"

    # --- 1. CACHE CHECK ---
    if os.path.exists(translated_json_path):
        print("\n[!] CACHE LOADED: Translated JSON found. Skipping Ollama generation.\n")
        with open(translated_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        original_subs = data['subtitles']
        
    else:
        # --- 2. TRANSLATION LOOP ---
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_subs = data['subtitles']
        total_subs = len(original_subs)

        print(f"Translating {total_subs} lines (Robust V2 + Auto-Formatter).")
        print("------------------------------------------------")

        with tqdm(total=total_subs, unit="lines") as pbar:
            for i in range(0, total_subs, BATCH_SIZE):
                batch = original_subs[i : i + BATCH_SIZE]
                context_start = max(0, i - CONTEXT_SIZE)
                prev_context = original_subs[context_start : i]
                
                translations = translate_batch_robust(batch, prev_context)
                
                for j, trans in enumerate(translations):
                    if j < len(batch):
                        batch[j]['text_en'] = trans['en']
                
                if translations:
                    last_text = str(translations[-1]['en']) 
                    pbar.set_description(f"Last: {last_text[:20].replace('\n', ' ')}...")
                
                pbar.update(len(batch))

        # Save the translated cache
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
        
    # Export English using the Mathematical Proportional Splitter
    en_filename = f"{base_name}.en.srt"
    en_content = generate_english_srt(original_subs)
    with open(en_filename, "w", encoding="utf-8") as f:
        f.write(en_content)
    print(f"- {en_filename} (Auto-Wrapped & Split)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translate_subs.py <json_file>")
    else:
        main(sys.argv[1])