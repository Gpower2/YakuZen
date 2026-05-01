import torch
import onnxruntime as ort
from faster_whisper import WhisperModel

print("--- DIAGNOSTICS ---")

# 1. Check PyTorch CUDA
print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# 2. Check ONNX Runtime (Used by Audio Separator)
print(f"ONNX Runtime Providers: {ort.get_available_providers()}")
# You want to see 'CUDAExecutionProvider' in this list!

# 3. Check Faster-Whisper
try:
    # Just initialize, don't run. 'cuda' ensures it tries to load libs.
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    print("Faster-Whisper loaded on GPU successfully.")
except Exception as e:
    print(f"Faster-Whisper Error: {e}")

print("--- END ---")