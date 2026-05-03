CURRENT_PIPELINE_MODE = "current"
BALANCED_PIPELINE_MODE = "balanced"
MAX_PIPELINE_MODE = "max"

SUPPORTED_PIPELINE_MODES = (
    CURRENT_PIPELINE_MODE,
    BALANCED_PIPELINE_MODE,
    MAX_PIPELINE_MODE,
)

DEFAULT_CONTEXT_MODEL = "qwen3.5:9b"
DEFAULT_JUDGE_MODEL = "qwen3.5:9b"
DEFAULT_PRIMARY_TRANSLATION_MODEL = "qwen3:14b"
DEFAULT_SECONDARY_TRANSLATION_MODEL = "translategemma:12b"


def mode_uses_dual_source(mode):
    return mode in {BALANCED_PIPELINE_MODE, MAX_PIPELINE_MODE}


def mode_uses_secondary_translation(mode):
    return mode == MAX_PIPELINE_MODE


def mode_uses_translation_review(mode):
    return mode in {BALANCED_PIPELINE_MODE, MAX_PIPELINE_MODE}


def mode_uses_context_fetch(mode):
    return mode in {BALANCED_PIPELINE_MODE, MAX_PIPELINE_MODE}


def mode_uses_jp_judging(mode):
    return mode in {BALANCED_PIPELINE_MODE, MAX_PIPELINE_MODE}


def mode_uses_maximum_jp_review(mode):
    return mode == MAX_PIPELINE_MODE


def profile_label(mode):
    return {
        CURRENT_PIPELINE_MODE: "Current",
        BALANCED_PIPELINE_MODE: "Balanced",
        MAX_PIPELINE_MODE: "Max Accuracy",
    }.get(mode, mode)
