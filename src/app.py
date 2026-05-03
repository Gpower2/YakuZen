import os
import sys
import threading
import subprocess
import re
import codecs
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox

from pipeline_profiles import (
    BALANCED_PIPELINE_MODE,
    CURRENT_PIPELINE_MODE,
    DEFAULT_CONTEXT_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PRIMARY_TRANSLATION_MODEL,
    DEFAULT_SECONDARY_TRANSLATION_MODEL,
    MAX_PIPELINE_MODE,
)

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

ASR_SOURCE_OPTIONS = {
    "Original mix (Recommended)": "mix",
    "Raw vocals stem": "raw_vocals",
    "Normalized vocals stem (Legacy)": "normalized_vocals",
}
ASR_MODEL_OPTIONS = {
    "Whisper large-v3 (Default)": "large-v3",
    "Hybrid (Experimental: Whisper + Kotoba)": "hybrid",
    "Kotoba-Whisper v1.1 (Advanced)": "kotoba-whisper-v1.1",
}
PIPELINE_MODE_OPTIONS = {
    "Current (Fast / Compatible)": CURRENT_PIPELINE_MODE,
    "Balanced (Recommended)": BALANCED_PIPELINE_MODE,
    "Max Accuracy (Slowest)": MAX_PIPELINE_MODE,
}
DEFAULT_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_TRANSLATION_MODEL = DEFAULT_PRIMARY_TRANSLATION_MODEL
SEPARATOR_MODEL_OPTIONS = {
    "BS-RoFormer (Default)": DEFAULT_SEPARATOR_MODEL,
}
TRANSLATION_MODEL_OPTIONS = {
    "Qwen3 14B (Default)": "qwen3:14b",
    "Qwen3.5 9B": "qwen3.5:9b",
    "TranslateGemma 12B": "translategemma:12b",
}
JUDGE_MODEL_OPTIONS = {
    "Qwen3.5 9B (Default)": DEFAULT_JUDGE_MODEL,
    "Qwen3 14B": "qwen3:14b",
}
CONTEXT_MODEL_OPTIONS = {
    "Qwen3.5 9B (Default)": DEFAULT_CONTEXT_MODEL,
    "Qwen3 14B": "qwen3:14b",
}
SECONDARY_TRANSLATION_MODEL_OPTIONS = {
    "TranslateGemma 12B (Default Secondary)": DEFAULT_SECONDARY_TRANSLATION_MODEL,
    "Qwen3.5 9B": "qwen3.5:9b",
    "Qwen3 14B": "qwen3:14b",
}
SCRIPT_STATUS_LABELS = {
    "process_audio.py": {
        "loading_series_context": "Loading series context",
        "loaded_series_context": "Series context ready",
        "extracting_alignment_audio": "Extracting alignment audio",
        "normalizing_audio": "Normalizing vocals audio",
        "separating_vocals": "Separating vocals stem",
        "skipping_separation": "Reusing cached vocals stem",
        "transcribing": "Transcribing Japanese audio",
        "transcribing_mix": "Transcribing mixed track",
        "transcribing_raw_vocals": "Transcribing raw vocals stem",
        "judging_transcript": "Resolving JP disagreement windows",
        "refining_timestamps": "Refining subtitle timings",
        "saving_debug": "Saving raw transcript debug output",
        "saved_to_disk": "Writing subtitle files to disk",
        "done": "Japanese transcription stage finished",
    },
    "translate_subs.py": {
        "loading_series_context": "Loading series context",
        "loaded_series_context": "Series context ready",
        "translating_primary": "Generating primary English translation",
        "reviewing_translations": "Reviewing flagged English lines",
        "normalizing_translation_terms": "Normalizing title and glossary terms",
        "exporting_subtitles": "Writing translated subtitle files",
        "done": "Translation stage finished",
    },
    "series_context_tool.py": {
        "resolving_series_context": "Fetching series metadata",
        "saved_series_context": "Saved global series cache entry",
        "saved_local_series_context": "Saved local series context file",
    },
    "check_settings.py": {
        "checking_environment": "Checking Python, FFmpeg, and GPU runtime",
        "checking_ollama": "Checking Ollama model availability",
        "checking_separator_model": "Validating separator checkpoint",
        "checking_series_context": "Validating series context override",
        "done": "Settings validation finished",
    },
}

class AnimePipelineApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Anime AI Subtitler")
        self.geometry("1480x900")
        
        # State variables
        self.selected_folder = ""
        self.video_files = []
        self.checkboxes = []
        self.is_running = False
        self.cancel_all_flag = False
        self.cancel_current_flag = False
        self.current_process = None

        self.setup_ui()

    def setup_ui(self):
        # --- TOP: Folder Selection ---
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(pady=10, padx=10, fill="x")
        
        self.folder_label = ctk.CTkLabel(self.top_frame, text="No folder selected", font=("Arial", 14))
        self.folder_label.pack(side="left", padx=10, pady=10)
        
        self.browse_btn = ctk.CTkButton(self.top_frame, text="Browse Folder", command=self.browse_folder)
        self.browse_btn.pack(side="right", padx=10, pady=10)

        # --- MIDDLE: Split View (Files vs Progress) ---
        self.middle_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.middle_frame.pack(pady=5, padx=10, fill="both", expand=True)

        # Left: File Selection
        self.file_panel = ctk.CTkFrame(self.middle_frame)
        self.file_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.file_header = ctk.CTkFrame(self.file_panel, fg_color="transparent")
        self.file_header.pack(fill="x", padx=10, pady=(10, 5))

        self.selected_count_lbl = ctk.CTkLabel(
            self.file_header,
            text="Selected Files: 0 / 0",
            font=("Arial", 14, "bold"),
        )
        self.selected_count_lbl.pack(side="left")

        self.file_shortcuts = ctk.CTkFrame(self.file_header, fg_color="transparent")
        self.file_shortcuts.pack(side="right")

        self.select_all_btn = ctk.CTkButton(
            self.file_shortcuts,
            text="Select All",
            width=110,
            command=self.select_all_files,
        )
        self.select_all_btn.pack(side="left", padx=(0, 5))

        self.deselect_all_btn = ctk.CTkButton(
            self.file_shortcuts,
            text="Deselect All",
            width=110,
            command=self.deselect_all_files,
        )
        self.deselect_all_btn.pack(side="left")

        self.file_frame = ctk.CTkScrollableFrame(self.file_panel, label_text="Select Video Files")
        self.file_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Right: Progress & Controls
        self.progress_frame = ctk.CTkFrame(self.middle_frame)
        self.progress_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Task Progress (Current Script)
        self.task_lbl = ctk.CTkLabel(self.progress_frame, text="Current Task Progress: 0%", font=("Arial", 14, "bold"))
        self.task_lbl.pack(pady=(20, 5))
        self.task_progress = ctk.CTkProgressBar(self.progress_frame)
        self.task_progress.pack(pady=5, padx=20, fill="x")
        self.task_progress.set(0)

        # Total Progress (Queue)
        self.total_lbl = ctk.CTkLabel(self.progress_frame, text="Total Queue Progress: 0 / 0", font=("Arial", 14, "bold"))
        self.total_lbl.pack(pady=(20, 5))
        self.total_progress = ctk.CTkProgressBar(self.progress_frame, progress_color="green")
        self.total_progress.pack(pady=5, padx=20, fill="x")
        self.total_progress.set(0)

        # Status Text
        self.status_lbl = ctk.CTkLabel(self.progress_frame, text="Idle", text_color="gray")
        self.status_lbl.pack(pady=10)

        # Buttons
        self.btn_frame = ctk.CTkFrame(self.progress_frame, fg_color="transparent")
        self.btn_frame.pack(pady=20)
        self.btn_frame.grid_columnconfigure(0, weight=1)
        self.btn_frame.grid_columnconfigure(1, weight=1)
        self.btn_frame.grid_columnconfigure(2, weight=1)

        self.run_btn = ctk.CTkButton(self.btn_frame, text="Start Processing", command=self.start_pipeline, fg_color="green", hover_color="darkgreen")
        self.run_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.cancel_curr_btn = ctk.CTkButton(self.btn_frame, text="Skip Current", command=self.cancel_current, state="disabled", fg_color="orange", hover_color="darkorange")
        self.cancel_curr_btn.grid(row=0, column=1, padx=5, sticky="ew")

        self.cancel_all_btn = ctk.CTkButton(self.btn_frame, text="Cancel All", command=self.cancel_all, state="disabled", fg_color="red", hover_color="darkred")
        self.cancel_all_btn.grid(row=0, column=2, padx=5, sticky="ew")

        self.tools_btn_frame = ctk.CTkFrame(self.btn_frame, fg_color="transparent")
        self.tools_btn_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=(10, 0), sticky="ew")
        self.tools_btn_frame.grid_columnconfigure(0, weight=1)
        self.tools_btn_frame.grid_columnconfigure(1, weight=1)

        self.settings_check_btn = ctk.CTkButton(
            self.tools_btn_frame,
            text="Check Settings",
            command=self.start_settings_check,
            fg_color="slateblue",
            hover_color="mediumpurple",
        )
        self.settings_check_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.gpu_check_btn = ctk.CTkButton(
            self.tools_btn_frame,
            text="Check GPU",
            command=self.start_gpu_check,
            fg_color="steelblue",
            hover_color="royalblue",
        )
        self.gpu_check_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # Advanced settings
        self.advanced_frame = ctk.CTkFrame(self.progress_frame)
        self.advanced_frame.pack(pady=(0, 20), padx=20, fill="x")

        self.advanced_lbl = ctk.CTkLabel(self.advanced_frame, text="Advanced Settings", font=("Arial", 14, "bold"))
        self.advanced_lbl.pack(anchor="w", padx=10, pady=(10, 6))

        self.advanced_content = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        self.advanced_content.pack(fill="x", padx=10, pady=(0, 4))

        self.advanced_left_frame = ctk.CTkFrame(self.advanced_content, fg_color="transparent")
        self.advanced_left_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.advanced_left_frame.grid_columnconfigure(1, weight=1)
        self.advanced_left_frame.grid_columnconfigure(2, weight=0)

        self.advanced_right_frame = ctk.CTkFrame(self.advanced_content, fg_color="transparent")
        self.advanced_right_frame.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.advanced_right_frame.grid_columnconfigure(1, weight=1)

        self.pipeline_mode_var = ctk.StringVar(value="Current (Fast / Compatible)")
        self.asr_source_var = ctk.StringVar(value="Original mix (Recommended)")
        self.asr_model_var = ctk.StringVar(value="Whisper large-v3 (Default)")
        self.separator_model_var = ctk.StringVar(value="BS-RoFormer (Default)")
        self.separator_model_custom_var = ctk.StringVar(value="")
        self.translation_model_var = ctk.StringVar(value="Qwen3 14B (Default)")
        self.translation_model_custom_var = ctk.StringVar(value="")
        self.secondary_translation_model_var = ctk.StringVar(value="TranslateGemma 12B (Default Secondary)")
        self.secondary_translation_model_custom_var = ctk.StringVar(value="")
        self.judge_model_var = ctk.StringVar(value="Qwen3.5 9B (Default)")
        self.judge_model_custom_var = ctk.StringVar(value="")
        self.context_model_var = ctk.StringVar(value="Qwen3.5 9B (Default)")
        self.context_model_custom_var = ctk.StringVar(value="")
        self.series_context_path_var = ctk.StringVar(value="")

        ctk.CTkLabel(self.advanced_left_frame, text="Processing profile").grid(row=0, column=0, sticky="w", padx=0, pady=4)
        self.pipeline_mode_menu = ctk.CTkComboBox(
            self.advanced_left_frame,
            values=list(PIPELINE_MODE_OPTIONS.keys()),
            variable=self.pipeline_mode_var,
            state="readonly",
        )
        self.pipeline_mode_menu.grid(row=0, column=1, columnspan=2, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_left_frame, text="Transcription source (Current mode)").grid(row=1, column=0, sticky="w", padx=0, pady=4)
        self.asr_source_menu = ctk.CTkComboBox(
            self.advanced_left_frame,
            values=list(ASR_SOURCE_OPTIONS.keys()),
            variable=self.asr_source_var,
            state="readonly",
        )
        self.asr_source_menu.grid(row=1, column=1, columnspan=2, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_left_frame, text="ASR model (Current mode)").grid(row=2, column=0, sticky="w", padx=0, pady=4)
        self.asr_model_menu = ctk.CTkComboBox(
            self.advanced_left_frame,
            values=list(ASR_MODEL_OPTIONS.keys()),
            variable=self.asr_model_var,
            state="readonly",
        )
        self.asr_model_menu.grid(row=2, column=1, columnspan=2, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_left_frame, text="Separator model preset").grid(row=3, column=0, sticky="w", padx=0, pady=4)
        self.separator_model_menu = ctk.CTkComboBox(
            self.advanced_left_frame,
            values=list(SEPARATOR_MODEL_OPTIONS.keys()),
            variable=self.separator_model_var,
            state="readonly",
        )
        self.separator_model_menu.grid(row=3, column=1, columnspan=2, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_left_frame, text="Custom separator model (Optional)").grid(row=4, column=0, sticky="w", padx=0, pady=4)
        self.separator_model_entry = ctk.CTkEntry(
            self.advanced_left_frame,
            textvariable=self.separator_model_custom_var,
            placeholder_text="Override with another separator checkpoint filename",
        )
        self.separator_model_entry.grid(row=4, column=1, columnspan=2, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_left_frame, text="Series context file (Optional)").grid(row=5, column=0, sticky="w", padx=0, pady=4)
        self.series_context_entry = ctk.CTkEntry(
            self.advanced_left_frame,
            textvariable=self.series_context_path_var,
            placeholder_text="Optional local series_context.json override",
        )
        self.series_context_entry.grid(row=5, column=1, sticky="ew", padx=(0, 5), pady=4)
        self.series_context_browse_btn = ctk.CTkButton(
            self.advanced_left_frame,
            text="Browse",
            width=90,
            command=self.browse_series_context_file,
        )
        self.series_context_browse_btn.grid(row=5, column=2, sticky="e", padx=(5, 0), pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Primary translation model").grid(row=0, column=0, sticky="w", padx=0, pady=4)
        self.translation_model_menu = ctk.CTkComboBox(
            self.advanced_right_frame,
            values=list(TRANSLATION_MODEL_OPTIONS.keys()),
            variable=self.translation_model_var,
            state="readonly",
        )
        self.translation_model_menu.grid(row=0, column=1, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Custom primary translation model (Optional)").grid(row=1, column=0, sticky="w", padx=0, pady=4)
        self.translation_model_entry = ctk.CTkEntry(
            self.advanced_right_frame,
            textvariable=self.translation_model_custom_var,
            placeholder_text="Override with another local Ollama model name",
        )
        self.translation_model_entry.grid(row=1, column=1, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Secondary translation model").grid(row=2, column=0, sticky="w", padx=0, pady=4)
        self.secondary_translation_model_menu = ctk.CTkComboBox(
            self.advanced_right_frame,
            values=list(SECONDARY_TRANSLATION_MODEL_OPTIONS.keys()),
            variable=self.secondary_translation_model_var,
            state="readonly",
        )
        self.secondary_translation_model_menu.grid(row=2, column=1, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Custom secondary model (Optional)").grid(row=3, column=0, sticky="w", padx=0, pady=4)
        self.secondary_translation_model_entry = ctk.CTkEntry(
            self.advanced_right_frame,
            textvariable=self.secondary_translation_model_custom_var,
            placeholder_text="Override the secondary candidate generator model",
        )
        self.secondary_translation_model_entry.grid(row=3, column=1, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Judge / normalizer model").grid(row=4, column=0, sticky="w", padx=0, pady=4)
        self.judge_model_menu = ctk.CTkComboBox(
            self.advanced_right_frame,
            values=list(JUDGE_MODEL_OPTIONS.keys()),
            variable=self.judge_model_var,
            state="readonly",
        )
        self.judge_model_menu.grid(row=4, column=1, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Custom judge model (Optional)").grid(row=5, column=0, sticky="w", padx=0, pady=4)
        self.judge_model_entry = ctk.CTkEntry(
            self.advanced_right_frame,
            textvariable=self.judge_model_custom_var,
            placeholder_text="Override the JP/EN review model",
        )
        self.judge_model_entry.grid(row=5, column=1, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Series context model").grid(row=6, column=0, sticky="w", padx=0, pady=4)
        self.context_model_menu = ctk.CTkComboBox(
            self.advanced_right_frame,
            values=list(CONTEXT_MODEL_OPTIONS.keys()),
            variable=self.context_model_var,
            state="readonly",
        )
        self.context_model_menu.grid(row=6, column=1, sticky="ew", padx=0, pady=4)

        ctk.CTkLabel(self.advanced_right_frame, text="Custom context model (Optional)").grid(row=7, column=0, sticky="w", padx=0, pady=4)
        self.context_model_entry = ctk.CTkEntry(
            self.advanced_right_frame,
            textvariable=self.context_model_custom_var,
            placeholder_text="Override the global series-context enrichment model",
        )
        self.context_model_entry.grid(row=7, column=1, sticky="ew", padx=0, pady=4)

        self.advanced_note = ctk.CTkLabel(
            self.advanced_frame,
            text="`Current` keeps the existing fast path. `Balanced` adds dual-source ASR review plus judge-based cleanup and is the recommended automation profile. `Max Accuracy` adds the slowest full review path with a secondary translator. The series-context file is optional; if left empty, balanced/max will use the global cache and fetch or enrich it when needed.",
            justify="left",
            wraplength=760,
            text_color="gray70",
        )
        self.advanced_note.pack(anchor="w", padx=10, pady=(4, 10))

        # --- BOTTOM: Console ---
        self.console = ctk.CTkTextbox(self, height=220, font=("Consolas", 12))
        self.console.pack(pady=10, padx=10, fill="x")
        self.console.configure(state="disabled")

    # --- UI UPDATERS (Thread Safe) ---
    def log_to_console(self, text):
        def update():
            self.console.configure(state="normal")
            self.console.insert("end", text + "\n")
            self.console.see("end")
            self.console.configure(state="disabled")
        self.after(0, update)

    def update_selected_count(self):
        selected = len(self.get_selected_files())
        total = len(self.checkboxes)
        self.selected_count_lbl.configure(text=f"Selected Files: {selected} / {total}")

    def update_task_progress(self, percent, status_text=None):
        percent = max(0, min(100, int(percent)))

        def update():
            self.task_progress.set(percent / 100.0)
            self.task_lbl.configure(text=f"Current Task Progress: {int(percent)}%")
            if status_text:
                self.status_lbl.configure(text=status_text)
        self.after(0, update)

    def update_total_progress(self, current, total):
        def update():
            fraction = current / max(total, 1)
            self.total_progress.set(fraction)
            self.total_lbl.configure(text=f"Total Queue Progress: {current} / {total}")
        self.after(0, update)

    def set_buttons_state(self, running):
        def update():
            if running:
                self.run_btn.configure(state="disabled")
                self.cancel_curr_btn.configure(state="normal")
                self.cancel_all_btn.configure(state="normal")
                self.settings_check_btn.configure(state="disabled")
                self.gpu_check_btn.configure(state="disabled")
            else:
                self.run_btn.configure(state="normal")
                self.cancel_curr_btn.configure(state="disabled")
                self.cancel_all_btn.configure(state="disabled")
                self.settings_check_btn.configure(state="normal")
                self.gpu_check_btn.configure(state="normal")
        self.after(0, update)

    # --- ACTIONS ---
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder = folder
            self.folder_label.configure(text=folder)
            self.scan_for_videos()

    def browse_series_context_file(self):
        path = filedialog.askopenfilename(
            title="Select series context file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.series_context_path_var.set(path)

    def get_selected_files(self):
        return [var.get() for _cb, var in self.checkboxes if var.get()]

    def select_all_files(self):
        for checkbox, var in self.checkboxes:
            var.set(checkbox.cget("text"))
        self.update_selected_count()

    def deselect_all_files(self):
        for _checkbox, var in self.checkboxes:
            var.set("")
        self.update_selected_count()

    def scan_for_videos(self):
        for cb, var in self.checkboxes:
            cb.destroy()
        self.checkboxes.clear()
        self.video_files.clear()
        self.update_selected_count()

        valid_exts = (".mkv", ".mp4", ".avi", ".mov")
        for f in os.listdir(self.selected_folder):
            if f.lower().endswith(valid_exts):
                self.video_files.append(f)

        if not self.video_files:
            messagebox.showinfo("Notice", "No video files found in this folder.")
            self.update_selected_count()
            return

        for file_name in self.video_files:
            var = ctk.StringVar(value=file_name) # Default checked
            cb = ctk.CTkCheckBox(
                self.file_frame,
                text=file_name,
                variable=var,
                onvalue=file_name,
                offvalue="",
                command=self.update_selected_count,
            )
            cb.pack(anchor="w", pady=5, padx=5)
            self.checkboxes.append((cb, var))

        self.update_selected_count()

    def cancel_current(self):
        if self.is_running and self.current_process:
            self.cancel_current_flag = True
            # TWEAK: Upgraded from terminate() to kill() for instant obliteration
            self.current_process.kill()
            self.log_to_console("[!] User cancelled current file. Skipping to next...")

    def cancel_all(self):
        if self.is_running:
            self.cancel_all_flag = True
            self.cancel_current_flag = True
            if self.current_process:
                # TWEAK: Upgraded from terminate() to kill()
                self.current_process.kill()
            self.log_to_console("[!!!] User cancelled the entire queue.")

    def start_pipeline(self):
        selected_files = self.get_selected_files()
        if not selected_files:
            messagebox.showwarning("Warning", "Please select at least one file to process.")
            return

        pipeline_settings = self.get_pipeline_settings()

        self.is_running = True
        self.cancel_all_flag = False
        self.set_buttons_state(running=True)
        self.log_to_console("=== PIPELINE STARTED ===")
        self.log_to_console(
            "Settings: "
            f"mode={pipeline_settings['pipeline_mode']}, "
            f"source={pipeline_settings['asr_source']}, "
            f"asr={pipeline_settings['asr_model']}, "
            f"separator={pipeline_settings['separator_model']}, "
            f"translation_primary={pipeline_settings['translation_model']}, "
            f"translation_secondary={pipeline_settings['translation_secondary_model']}, "
            f"judge={pipeline_settings['judge_model']}, "
            f"context={pipeline_settings['context_model']}"
        )
        
        threading.Thread(target=self.run_queue, args=(selected_files, pipeline_settings), daemon=True).start()

    def start_gpu_check(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Please wait for the current task to finish first.")
            return

        self.is_running = True
        self.cancel_all_flag = False
        self.cancel_current_flag = False
        self.current_process = None
        self.set_buttons_state(running=True)
        self.update_total_progress(0, 1)
        self.log_to_console("=== GPU CHECK STARTED ===")
        threading.Thread(target=self.run_gpu_check, daemon=True).start()

    def start_settings_check(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Please wait for the current task to finish first.")
            return

        pipeline_settings = self.get_pipeline_settings()

        self.is_running = True
        self.cancel_all_flag = False
        self.cancel_current_flag = False
        self.current_process = None
        self.set_buttons_state(running=True)
        self.update_total_progress(0, 1)
        self.log_to_console("=== SETTINGS CHECK STARTED ===")
        self.log_to_console(
            "Settings: "
            f"mode={pipeline_settings['pipeline_mode']}, "
            f"source={pipeline_settings['asr_source']}, "
            f"asr={pipeline_settings['asr_model']}, "
            f"separator={pipeline_settings['separator_model']}, "
            f"translation_primary={pipeline_settings['translation_model']}, "
            f"translation_secondary={pipeline_settings['translation_secondary_model']}, "
            f"judge={pipeline_settings['judge_model']}, "
            f"context={pipeline_settings['context_model']}"
        )
        threading.Thread(
            target=self.run_settings_check,
            args=(pipeline_settings,),
            daemon=True,
        ).start()

    def get_pipeline_settings(self):
        separator_model = self.separator_model_custom_var.get().strip() or SEPARATOR_MODEL_OPTIONS[self.separator_model_var.get()]
        translation_model = self.translation_model_custom_var.get().strip() or TRANSLATION_MODEL_OPTIONS[self.translation_model_var.get()]
        secondary_translation_model = self.secondary_translation_model_custom_var.get().strip() or SECONDARY_TRANSLATION_MODEL_OPTIONS[self.secondary_translation_model_var.get()]
        judge_model = self.judge_model_custom_var.get().strip() or JUDGE_MODEL_OPTIONS[self.judge_model_var.get()]
        context_model = self.context_model_custom_var.get().strip() or CONTEXT_MODEL_OPTIONS[self.context_model_var.get()]
        return {
            "pipeline_mode": PIPELINE_MODE_OPTIONS[self.pipeline_mode_var.get()],
            "asr_source": ASR_SOURCE_OPTIONS[self.asr_source_var.get()],
            "asr_model": ASR_MODEL_OPTIONS[self.asr_model_var.get()],
            "separator_model": separator_model,
            "translation_model": translation_model,
            "translation_secondary_model": secondary_translation_model,
            "judge_model": judge_model,
            "context_model": context_model,
            "series_context_path": self.series_context_path_var.get().strip(),
        }

    def get_settings_check_args(self, pipeline_settings):
        args = [
            "--pipeline-mode", pipeline_settings["pipeline_mode"],
            "--asr-source", pipeline_settings["asr_source"],
            "--asr-model", pipeline_settings["asr_model"],
            "--separator-model", pipeline_settings["separator_model"],
            "--translation-model", pipeline_settings["translation_model"],
            "--translation-secondary-model", pipeline_settings["translation_secondary_model"],
            "--judge-model", pipeline_settings["judge_model"],
            "--context-model", pipeline_settings["context_model"],
        ]
        if pipeline_settings["series_context_path"]:
            args.extend(["--series-context", pipeline_settings["series_context_path"]])
        return args

    # --- PROCESS EXECUTION ---
    def get_script_label(self, script_name):
        return os.path.splitext(os.path.basename(script_name))[0]

    def describe_script_status(self, script_name, payload):
        if "error" in payload:
            return f"Error: {payload['error']}"

        status = payload.get("status")
        if not status:
            return None

        message = SCRIPT_STATUS_LABELS.get(script_name, {}).get(status, status.replace("_", " ").title())
        file_hint = payload.get("file") or payload.get("cached_file")
        if file_hint and status in {"saving_debug", "saved_to_disk", "skipping_separation"}:
            message = f"{message}: {os.path.basename(file_hint)}"
        return message

    def extract_progress_status(self, line):
        match = re.match(r"^(.*):\s+\d+%", line)
        if match:
            return match.group(1).strip()
        return None

    def handle_output_line(self, line, script_name, progress_state):
        clean_line = line.strip()
        if not clean_line:
            return

        script_label = self.get_script_label(script_name)

        try:
            payload = json.loads(clean_line)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            status_text = self.describe_script_status(script_name, payload)
            if status_text:
                if status_text != progress_state["last_status_text"]:
                    self.log_to_console(f"[{script_label}] {status_text}")
                    progress_state["last_status_text"] = status_text
                if payload.get("status") == "done":
                    progress_state["completed"] = True
                    progress_state["last_percent"] = 100
                    self.update_task_progress(100, status_text=status_text)
                else:
                    self.update_task_progress(progress_state["last_percent"], status_text=status_text)
            return

        percent_match = re.search(r"(\d+)%", clean_line)
        if percent_match:
            percent = int(percent_match.group(1))
            if percent >= 100 and not progress_state["completed"]:
                percent = 99
            progress_state["last_percent"] = max(progress_state["last_percent"], percent)

            status_text = self.extract_progress_status(clean_line)
            if status_text:
                progress_state["last_status_text"] = status_text
                self.update_task_progress(progress_state["last_percent"], status_text=status_text)
            else:
                self.update_task_progress(progress_state["last_percent"])
            return

        self.log_to_console(f"[{script_label}] {clean_line}")
        self.update_task_progress(progress_state["last_percent"], status_text=clean_line)
        progress_state["last_status_text"] = clean_line

    def read_output_stream(self, process, script_name, progress_state):
        """Reads raw stdout byte-by-byte while preserving UTF-8 multibyte characters and tqdm carriage returns."""
        buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        while True:
            raw_byte = process.stdout.read(1)
            if not raw_byte:
                break

            decoded = decoder.decode(raw_byte)
            if not decoded:
                continue

            for char in decoded:
                if char == '\r' or char == '\n':
                    if buffer:
                        self.handle_output_line(buffer, script_name, progress_state)
                    buffer = ""
                else:
                    buffer += char

        tail = decoder.decode(b"", final=True)
        for char in tail:
            if char == '\r' or char == '\n':
                if buffer:
                    self.handle_output_line(buffer, script_name, progress_state)
                buffer = ""
            else:
                buffer += char

        if buffer:
            self.handle_output_line(buffer, script_name, progress_state)

    def run_script(self, script_name, target_file=None, status_msg="Running...", extra_args=None):
        self.cancel_current_flag = False
        self.update_task_progress(0, status_text=status_msg)
        script_label = self.get_script_label(script_name)
        progress_state = {
            "completed": False,
            "last_percent": 0,
            "last_status_text": status_msg,
        }
        self.log_to_console(f">> Running {script_name}...")

        try:
            # TWEAK: Removed text=True and universal_newlines. 
            # bufsize=0 forces true unbuffered binary streaming.
            command = [sys.executable, "-u", script_name]
            if target_file:
                command.append(target_file)
            if extra_args:
                command.extend(extra_args)
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0 
            )

            # Thread to read the unified binary stream
            reader_thread = threading.Thread(
                target=self.read_output_stream,
                args=(self.current_process, script_name, progress_state),
                daemon=True,
            )
            reader_thread.start()

            # Wait for the AI process to finish (or get killed)
            self.current_process.wait()
            reader_thread.join(timeout=1)

            if self.cancel_current_flag:
                return False

            # -9 is the standard exit code for SIGKILL on Linux/WSL.
            if self.current_process.returncode != 0 and self.current_process.returncode != -9:
                self.log_to_console(f"[ERROR] {script_name} returned non-zero exit code: {self.current_process.returncode}")
                return False

            if not progress_state["completed"]:
                self.update_task_progress(100, status_text=f"{script_label} finished")
            return True

        except Exception as e:
            self.log_to_console(f"[EXCEPTION] {str(e)}")
            return False

    def run_gpu_check(self):
        success = self.run_script(
            "check_gpu.py",
            status_msg="Checking GPU readiness",
        )
        self.current_process = None
        self.update_total_progress(1, 1)
        self.is_running = False
        self.update_task_progress(0, "Idle")
        self.set_buttons_state(running=False)
        if success:
            self.log_to_console("=== GPU CHECK FINISHED ===")
        else:
            self.log_to_console("=== GPU CHECK FAILED ===")

    def run_settings_check(self, pipeline_settings):
        success = self.run_script(
            "check_settings.py",
            status_msg="Validating current settings",
            extra_args=self.get_settings_check_args(pipeline_settings),
        )
        self.current_process = None
        self.update_total_progress(1, 1)
        self.is_running = False
        self.update_task_progress(0, "Idle")
        self.set_buttons_state(running=False)
        if success:
            self.log_to_console("=== SETTINGS CHECK FINISHED ===")
        else:
            self.log_to_console("=== SETTINGS CHECK FAILED ===")

    def run_queue(self, files_to_process, pipeline_settings):
        total_files = len(files_to_process)
        self.update_total_progress(0, total_files)

        for idx, file_name in enumerate(files_to_process, 1):
            if self.cancel_all_flag:
                break

            full_path = os.path.join(self.selected_folder, file_name)
            base_name = os.path.splitext(full_path)[0]
            json_path = f"{base_name}.json"
            
            self.log_to_console(f"\n--- Processing [{idx}/{total_files}]: {file_name} ---")

            # 1. Process Audio
            process_audio_args = [
                "--pipeline-mode", pipeline_settings["pipeline_mode"],
                "--asr-source", pipeline_settings["asr_source"],
                "--asr-model", pipeline_settings["asr_model"],
                "--separator-model", pipeline_settings["separator_model"],
                "--judge-model", pipeline_settings["judge_model"],
                "--context-model", pipeline_settings["context_model"],
            ]
            if pipeline_settings["series_context_path"]:
                process_audio_args.extend(["--series-context", pipeline_settings["series_context_path"]])
            success = self.run_script(
                "process_audio.py",
                full_path,
                f"Isolating & Transcribing: {file_name}",
                extra_args=process_audio_args,
            )
            
            if self.cancel_all_flag: break
            if not success and not self.cancel_current_flag:
                self.log_to_console("Audio processing failed. Moving to next file.")
                self.update_total_progress(idx, total_files)
                continue
            
            # 2. Translate Subtitles (only if not skipping current)
            if not self.cancel_current_flag:
                translate_args = [
                    "--pipeline-mode", pipeline_settings["pipeline_mode"],
                    "--translation-model", pipeline_settings["translation_model"],
                    "--translation-secondary-model", pipeline_settings["translation_secondary_model"],
                    "--judge-model", pipeline_settings["judge_model"],
                    "--context-model", pipeline_settings["context_model"],
                ]
                if pipeline_settings["series_context_path"]:
                    translate_args.extend(["--series-context", pipeline_settings["series_context_path"]])
                self.run_script(
                    "translate_subs.py",
                    json_path,
                    f"Translating & Formatting: {file_name}",
                    extra_args=translate_args,
                )

            self.update_total_progress(idx, total_files)

        self.is_running = False
        self.current_process = None
        self.update_task_progress(0, "Idle")
        self.set_buttons_state(running=False)
        self.log_to_console("\n=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    app = AnimePipelineApp()
    app.mainloop()
