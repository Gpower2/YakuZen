import argparse
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
from importlib import metadata
from urllib import error as urllib_error
from urllib import request as urllib_request

from ffmpeg_utils import ensure_ffmpeg_tools_available, find_ffmpeg_binary
from pipeline_profiles import (
    CURRENT_PIPELINE_MODE,
    DEFAULT_CONTEXT_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PRIMARY_TRANSLATION_MODEL,
    DEFAULT_SECONDARY_TRANSLATION_MODEL,
    SUPPORTED_PIPELINE_MODES,
    mode_uses_context_fetch,
    mode_uses_secondary_translation,
    mode_uses_translation_review,
    profile_label,
)


SUPPORTED_ASR_SOURCES = ("mix", "raw_vocals", "normalized_vocals")
SUPPORTED_ASR_MODELS = ("large-v3", "kotoba-whisper-v1.1", "hybrid")
DEFAULT_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
TEMP_DIR = os.path.abspath("./temp")


def emit_status(status, **payload):
    message = {"status": status}
    message.update(payload)
    print(json.dumps(message), file=sys.stderr)


def log(level, message):
    print(f"[{level}] {message}")


def log_info(message):
    log("INFO", message)


def log_pass(message):
    log("PASS", message)


def log_warn(message):
    log("WARN", message)


def log_fail(message):
    log("FAIL", message)


def log_progress(label, percent):
    print(f"{label}: {int(percent)}%")


def normalize_model_name(name):
    return str(name or "").strip()


def distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def module_available(module_name):
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, exc


def validate_command(command_name, errors):
    path = find_ffmpeg_binary(command_name) if command_name in {"ffmpeg", "ffprobe"} else shutil.which(command_name)
    if not path:
        message = f"{command_name} was not found."
        log_fail(message)
        errors.append(message)
        return None
    log_pass(f"{command_name} found at {path}")
    return path


def validate_python_module(module_name, friendly_name, errors):
    available, exc = module_available(module_name)
    if available:
        log_pass(f"{friendly_name} import succeeded")
        return True
    message = f"{friendly_name} import failed: {exc}"
    log_fail(message)
    errors.append(message)
    return False


def fetch_ollama_model_names():
    request = urllib_request.Request(DEFAULT_OLLAMA_TAGS_URL, method="GET")
    with urllib_request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models") or []
    discovered = []
    for model in models:
        name = normalize_model_name(model.get("name"))
        if name:
            discovered.append(name)
    return discovered


def model_exists(model_name, available_models):
    target = normalize_model_name(model_name).casefold()
    normalized_available = {normalize_model_name(model).casefold() for model in available_models}
    if target in normalized_available:
        return True
    if ":" not in target and f"{target}:latest" in normalized_available:
        return True
    if target.endswith(":latest") and target[:-7] in normalized_available:
        return True
    return False


def validate_runtime_environment(args, errors, warnings):
    emit_status("checking_environment")
    log_progress("Environment checks", 15)
    log_info(f"Processing profile: {profile_label(args.pipeline_mode)} ({args.pipeline_mode})")
    log_info(f"ASR source: {args.asr_source}")
    log_info(f"ASR model: {args.asr_model}")

    ensure_ffmpeg_tools_available()
    validate_command("ffmpeg", errors)
    validate_command("ffprobe", errors)
    validate_python_module("torch", "torch", errors)
    validate_python_module("faster_whisper", "faster-whisper", errors)
    validate_python_module("stable_whisper", "stable-ts", errors)
    validate_python_module("audio_separator", "audio-separator", errors)

    if args.asr_model in {"hybrid", "kotoba-whisper-v1.1"}:
        validate_python_module("transformers", "transformers", errors)
        validate_python_module("torchaudio", "torchaudio", errors)

    cpu_ort_version = distribution_version("onnxruntime")
    gpu_ort_version = distribution_version("onnxruntime-gpu")
    if gpu_ort_version:
        log_pass(f"onnxruntime-gpu installed ({gpu_ort_version})")
    else:
        message = "onnxruntime-gpu is not installed."
        log_fail(message)
        errors.append(message)
    if cpu_ort_version:
        message = (
            f"onnxruntime CPU package is also installed ({cpu_ort_version}). "
            "This often happens because faster-whisper pulls it in as a dependency and it can hide the CUDA provider."
        )
        log_warn(message)
        warnings.append(message)

    torch_available, torch_exc = module_available("torch")
    if torch_available:
        import torch

        log_info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log_pass(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            message = "PyTorch CUDA is not available; Whisper will fall back to CPU."
            log_warn(message)
            warnings.append(message)
    elif torch_exc:
        log_fail(f"torch import failed during CUDA check: {torch_exc}")

    ort_available, ort_exc = module_available("onnxruntime")
    if ort_available:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        log_info(f"ONNX Runtime providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            log_pass("ONNX Runtime CUDAExecutionProvider is available")
        else:
            message = (
                "ONNX Runtime CUDAExecutionProvider is missing. "
                "The separator stage will likely run on CPU even though faster-whisper can still use the GPU."
            )
            log_warn(message)
            warnings.append(message)
    else:
        message = f"onnxruntime import failed: {ort_exc}"
        log_fail(message)
        errors.append(message)


def validate_ollama_models(args, errors, warnings):
    emit_status("checking_ollama")
    log_progress("Ollama checks", 45)
    try:
        available_models = fetch_ollama_model_names()
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        message = f"Could not query Ollama models at {DEFAULT_OLLAMA_TAGS_URL}: {exc}"
        log_fail(message)
        errors.append(message)
        return

    if not available_models:
        message = "Ollama is reachable, but no local models were reported."
        log_fail(message)
        errors.append(message)
        return

    log_pass(f"Ollama is reachable and reported {len(available_models)} local model tag(s)")

    model_roles = [
        ("Primary translation", args.translation_model, True),
        ("Judge / normalizer", args.judge_model, mode_uses_translation_review(args.pipeline_mode)),
        ("Series context", args.context_model, mode_uses_context_fetch(args.pipeline_mode)),
        ("Secondary translation", args.translation_secondary_model, mode_uses_secondary_translation(args.pipeline_mode)),
    ]

    for role_label, model_name, required in model_roles:
        if model_exists(model_name, available_models):
            requirement_text = "required" if required else "optional for the current profile"
            log_pass(f"{role_label} model found: {model_name} ({requirement_text})")
            continue

        message = f"{role_label} model not found in Ollama: {model_name}"
        if required:
            log_fail(message)
            errors.append(message)
        else:
            message = f"{message} (not used by {args.pipeline_mode})"
            log_warn(message)
            warnings.append(message)


def validate_separator_model(separator_model, errors):
    emit_status("checking_separator_model")
    log_progress("Separator checks", 70)

    try:
        from audio_separator.separator import Separator
    except Exception as exc:
        message = f"audio-separator could not be imported for separator validation: {exc}"
        log_fail(message)
        errors.append(message)
        return

    model_name = os.path.basename(separator_model).strip() or DEFAULT_SEPARATOR_MODEL
    os.makedirs(TEMP_DIR, exist_ok=True)
    try:
        separator = Separator(
            log_level=logging.ERROR,
            model_file_dir=TEMP_DIR,
            output_dir=TEMP_DIR,
            output_single_stem="Vocals",
        )
        separator.load_model(model_filename=model_name)
        log_pass(f"Separator model resolved successfully: {model_name}")
    except Exception as exc:
        message = f"Separator model could not be loaded: {model_name} ({exc})"
        log_fail(message)
        errors.append(message)


def validate_series_context(series_context_path, errors, warnings):
    emit_status("checking_series_context")
    log_progress("Series context checks", 85)
    if not series_context_path:
        log_info("No local series-context override provided; cache/fetch fallback will be used when needed")
        return

    resolved_path = os.path.abspath(series_context_path)
    if not os.path.exists(resolved_path):
        message = f"Series context override does not exist: {resolved_path}"
        log_fail(message)
        errors.append(message)
        return

    try:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        message = f"Series context override is not valid JSON: {resolved_path} ({exc})"
        log_fail(message)
        errors.append(message)
        return

    if not isinstance(payload, dict):
        message = f"Series context override must be a JSON object: {resolved_path}"
        log_fail(message)
        errors.append(message)
        return

    title_hint = (
        str(payload.get("canonical_title_en") or "").strip()
        or str(payload.get("canonical_title_ja") or "").strip()
    )
    if title_hint:
        log_pass(f"Series context override loaded successfully: {resolved_path}")
    else:
        message = (
            f"Series context override loaded, but it does not declare canonical_title_en or canonical_title_ja: {resolved_path}"
        )
        log_warn(message)
        warnings.append(message)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the current YakuZen UI settings and environment.")
    parser.add_argument("--pipeline-mode", choices=SUPPORTED_PIPELINE_MODES, default=CURRENT_PIPELINE_MODE)
    parser.add_argument("--asr-source", choices=SUPPORTED_ASR_SOURCES, default="mix")
    parser.add_argument("--asr-model", choices=SUPPORTED_ASR_MODELS, default="large-v3")
    parser.add_argument("--separator-model", default=DEFAULT_SEPARATOR_MODEL)
    parser.add_argument("--translation-model", default=DEFAULT_PRIMARY_TRANSLATION_MODEL)
    parser.add_argument("--translation-secondary-model", default=DEFAULT_SECONDARY_TRANSLATION_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--context-model", default=DEFAULT_CONTEXT_MODEL)
    parser.add_argument("--series-context", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []
    warnings = []

    print("--- SETTINGS VALIDATION ---")
    print(json.dumps(
        {
            "pipeline_mode": args.pipeline_mode,
            "asr_source": args.asr_source,
            "asr_model": args.asr_model,
            "separator_model": normalize_model_name(args.separator_model),
            "translation_model": normalize_model_name(args.translation_model),
            "translation_secondary_model": normalize_model_name(args.translation_secondary_model),
            "judge_model": normalize_model_name(args.judge_model),
            "context_model": normalize_model_name(args.context_model),
            "series_context": normalize_model_name(args.series_context),
        },
        ensure_ascii=False,
    ))

    validate_runtime_environment(args, errors, warnings)
    validate_ollama_models(args, errors, warnings)
    validate_separator_model(args.separator_model, errors)
    validate_series_context(args.series_context, errors, warnings)

    log_progress("Settings validation", 100)
    if errors:
        log_fail(f"Settings validation found {len(errors)} blocking issue(s)")
    else:
        log_pass("Settings validation completed without blocking issues")
    if warnings:
        log_warn(f"Settings validation found {len(warnings)} warning(s)")

    emit_status("done", ok=not errors, errors=len(errors), warnings=len(warnings))
    print("--- SETTINGS VALIDATION COMPLETE ---")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
