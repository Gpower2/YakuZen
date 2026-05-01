import os
import sys
import threading
import subprocess
import re
import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AnimePipelineApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Anime AI Subtitler")
        self.geometry("900x750")
        
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
        self.file_frame = ctk.CTkScrollableFrame(self.middle_frame, label_text="Select Video Files")
        self.file_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

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

        self.run_btn = ctk.CTkButton(self.btn_frame, text="Start Processing", command=self.start_pipeline, fg_color="green", hover_color="darkgreen")
        self.run_btn.grid(row=0, column=0, padx=5)

        self.cancel_curr_btn = ctk.CTkButton(self.btn_frame, text="Skip Current", command=self.cancel_current, state="disabled", fg_color="orange", hover_color="darkorange")
        self.cancel_curr_btn.grid(row=0, column=1, padx=5)

        self.cancel_all_btn = ctk.CTkButton(self.btn_frame, text="Cancel All", command=self.cancel_all, state="disabled", fg_color="red", hover_color="darkred")
        self.cancel_all_btn.grid(row=0, column=2, padx=5)

        # --- BOTTOM: Console ---
        self.console = ctk.CTkTextbox(self, height=150, font=("Consolas", 12))
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

    def update_task_progress(self, percent, status_text=None):
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
            else:
                self.run_btn.configure(state="normal")
                self.cancel_curr_btn.configure(state="disabled")
                self.cancel_all_btn.configure(state="disabled")
        self.after(0, update)

    # --- ACTIONS ---
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder = folder
            self.folder_label.configure(text=folder)
            self.scan_for_videos()

    def scan_for_videos(self):
        for cb, var in self.checkboxes:
            cb.destroy()
        self.checkboxes.clear()
        self.video_files.clear()

        valid_exts = (".mkv", ".mp4", ".avi", ".mov")
        for f in os.listdir(self.selected_folder):
            if f.lower().endswith(valid_exts):
                self.video_files.append(f)

        if not self.video_files:
            messagebox.showinfo("Notice", "No video files found in this folder.")
            return

        for file_name in self.video_files:
            var = ctk.StringVar(value=file_name) # Default checked
            cb = ctk.CTkCheckBox(self.file_frame, text=file_name, variable=var, onvalue=file_name, offvalue="")
            cb.pack(anchor="w", pady=5, padx=5)
            self.checkboxes.append((cb, var))

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
        selected_files = [var.get() for cb, var in self.checkboxes if var.get() != ""]
        if not selected_files:
            messagebox.showwarning("Warning", "Please select at least one file to process.")
            return

        self.is_running = True
        self.cancel_all_flag = False
        self.set_buttons_state(running=True)
        self.log_to_console("=== PIPELINE STARTED ===")
        
        threading.Thread(target=self.run_queue, args=(selected_files,), daemon=True).start()

    # --- PROCESS EXECUTION ---
    def read_output_stream(self, process):
        """Reads raw binary stdout byte-by-byte to perfectly catch tqdm carriage returns."""
        buffer = ""
        while True:
            # Read exactly 1 raw byte
            raw_byte = process.stdout.read(1)
            if not raw_byte:
                break
            
            try:
                char = raw_byte.decode('utf-8')
            except UnicodeDecodeError:
                continue # Skip weird binary artifacts
            
            if char == '\r' or char == '\n':
                if buffer:
                    # Look for tqdm percentage e.g., " 45%"
                    match = re.search(r"(\d+)%", buffer)
                    if match:
                        self.update_task_progress(int(match.group(1)))
                    else:
                        # Log normal text output (ignore raw JSON dictionary spam)
                        clean_line = buffer.strip()
                        if clean_line and "{" not in clean_line:
                            self.log_to_console(clean_line)
                buffer = ""
            else:
                buffer += char

    def run_script(self, script_name, target_file, status_msg):
        self.cancel_current_flag = False
        self.update_task_progress(0, status_text=status_msg)
        self.log_to_console(f">> Running {script_name}...")

        try:
            # TWEAK: Removed text=True and universal_newlines. 
            # bufsize=0 forces true unbuffered binary streaming.
            self.current_process = subprocess.Popen(
                [sys.executable, "-u", script_name, target_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0 
            )

            # Thread to read the unified binary stream
            reader_thread = threading.Thread(target=self.read_output_stream, args=(self.current_process,), daemon=True)
            reader_thread.start()

            # Wait for the AI process to finish (or get killed)
            self.current_process.wait()

            if self.cancel_current_flag:
                return False

            # -9 is the standard exit code for SIGKILL on Linux/WSL.
            if self.current_process.returncode != 0 and self.current_process.returncode != -9:
                self.log_to_console(f"[ERROR] {script_name} returned non-zero exit code: {self.current_process.returncode}")
                return False

            self.update_task_progress(100)
            return True

        except Exception as e:
            self.log_to_console(f"[EXCEPTION] {str(e)}")
            return False

    def run_queue(self, files_to_process):
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
            success = self.run_script("process_audio.py", full_path, f"Isolating & Transcribing: {file_name}")
            
            if self.cancel_all_flag: break
            if not success and not self.cancel_current_flag:
                self.log_to_console("Audio processing failed. Moving to next file.")
                self.update_total_progress(idx, total_files)
                continue
            
            # 2. Translate Subtitles (only if not skipping current)
            if not self.cancel_current_flag:
                self.run_script("translate_subs.py", json_path, f"Translating & Formatting: {file_name}")

            self.update_total_progress(idx, total_files)

        self.is_running = False
        self.current_process = None
        self.update_task_progress(0, "Idle")
        self.set_buttons_state(running=False)
        self.log_to_console("\n=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    app = AnimePipelineApp()
    app.mainloop()