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
from datetime import timedelta
from tqdm import tqdm
import cutlet
from faster_whisper import WhisperModel
from audio_separator.separator import Separator

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
TEMP_DIR = os.path.abspath("./temp")

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
    # Reasoning: Anime dynamic range is extreme. This ensures quiet whispers and loud screams 
    # are fed to the AI at the exact same optimal volume level, preventing dropped words.
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

def save_srt(subtitles, output_path, text_key):
    srt_output = []
    for idx, sub in enumerate(subtitles, 1):
        start = format_timestamp(sub['start'])
        end = format_timestamp(sub['end'])
        text = sub[text_key]
        srt_output.append(f"{idx}\n{start} --> {end}\n{text}\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_output))

def main(input_file):
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

    if os.path.exists(output_json_path):
        print("\n[!] CACHE LOADED: Delete the .json file to process fresh.\n", file=sys.stderr)
        with open(output_json_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            save_srt(cached_data["subtitles"], output_srt_romaji, "text_romaji")
            save_srt(cached_data["subtitles"], output_srt_kanji, "text_jp")
        return

    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 1. SEPARATE VOCALS
    search_pattern = os.path.join(TEMP_DIR, f"*{base_name}*(Vocals)*.wav")
    found_files = glob.glob(search_pattern)
    
    # Filter out previously normalized files to prevent 'normalized_normalized_' looping
    raw_files = [f for f in found_files if not os.path.basename(f).startswith("normalized_")]
    raw_vocals_path = None

    if raw_files:
        raw_vocals_path = max(raw_files, key=os.path.getctime)
        print(json.dumps({"status": "skipping_separation", "cached_file": raw_vocals_path}), file=sys.stderr)
    else:
        print(json.dumps({"status": "separating_vocals"}), file=sys.stderr)
        separator = Separator(log_level=logging.ERROR, model_file_dir=TEMP_DIR, output_dir=TEMP_DIR, output_single_stem="Vocals")
        separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
        output_files = separator.separate(input_file)
        raw_vocals_path = os.path.join(TEMP_DIR, output_files[0])

    # 2. NORMALIZE AUDIO
    normalized_vocals_path = os.path.join(TEMP_DIR, f"normalized_{os.path.basename(raw_vocals_path)}")
    duration = get_audio_duration(raw_vocals_path)
    
    if not os.path.exists(normalized_vocals_path):
        normalize_audio(raw_vocals_path, normalized_vocals_path, duration)

    # 3. TRANSCRIBE
    print(json.dumps({"status": "transcribing"}), file=sys.stderr)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    segments_generator, info = model.transcribe(
        normalized_vocals_path, 
        
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
            # Reasoning: Back to default 0.5. 0.4 was too sensitive and allowed room tone to bridge gaps between words.
            threshold=0.5,                
            
            # min_speech_duration_ms
            # Default: 250
            # Description: Minimum length of a sound to be considered speech.
            # Reasoning: Kept at 250ms. Prevents random micro-noises (door clicks, footsteps) from triggering transcription.
            min_speech_duration_ms=250,   
            
            # min_silence_duration_ms
            # Default: 2000
            # Description: How much silence must occur before the VAD cuts a segment.
            # Reasoning: TIGHTENED to 500ms. 1000ms was too long, causing the VAD to stretch subtitles across long pauses. Half a second is a natural boundary.
            min_silence_duration_ms=500, 
            
            # speech_pad_ms
            # Default: 400
            # Description: Extra audio buffer added before and after detected speech.
            # Reasoning: TIGHTENED to 150ms. 400ms artificially inflated the visual duration of the subtitle. 150ms is just enough to catch the trailing breath without lingering on screen.
            speech_pad_ms=150             
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
        hallucination_silence_threshold=2.0 
    )

    katsu = cutlet.Cutlet() 
    
    # use_foreign_spelling
    # Default: True
    # Description: Attempts to reverse-engineer katakana loan words back into English (パーティー -> party).
    # Reasoning: Set to False. We want pure phonetic Romaji (paatii) so the text perfectly matches the sounds coming out of the characters' mouths.
    katsu.use_foreign_spelling = False 

    results = []
    raw_segments_debug = []
    
    with tqdm(total=duration, unit='s', desc="Transcribing", file=sys.stderr) as pbar:
        for segment in segments_generator:
            raw_segments_debug.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

            text_jp = segment.text.replace(" ", "").strip()
            if text_jp:
                results.append({
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text_jp": text_jp,
                    "text_romaji": katsu.romaji(text_jp).strip()
                })
            pbar.update(segment.end - pbar.n)

    print(json.dumps({"status": "saving_debug", "file": debug_json_path}), file=sys.stderr)
    with open(debug_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_segments_debug, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    
    final_output = {
        "meta": {
            "processing_time": round(total_time, 2),
            "language": info.language,
            "probability": round(info.language_probability, 2),
            "model": "large-v3-faster-raw"
        },
        "subtitles": results
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "saved_to_disk", "file": output_json_path}), file=sys.stderr)

    save_srt(results, output_srt_romaji, "text_romaji")
    save_srt(results, output_srt_kanji, "text_jp")
    
    print(json.dumps({"status": "done"}), file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python process_audio.py <video_file>"}), file=sys.stderr)
        sys.exit(1)
        
    main(sys.argv[1])