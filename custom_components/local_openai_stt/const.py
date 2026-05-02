"""Constants for the Local OpenAI STT integration."""

from __future__ import annotations

from enum import StrEnum

DOMAIN = "local_openai_stt"

CONF_BASE_URL = "base_url"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_TEMPERATURE = "temperature"
CONF_PROMPT = "prompt"

CONF_VAD_SENSITIVITY = "vad_sensitivity"
CONF_VAD_MIN_SPEECH_SECONDS = "vad_min_speech_seconds"
CONF_VAD_SPEECH_THRESHOLD = "vad_speech_threshold"
CONF_DEBUG_LOG = "debug_log"
CONF_DEBUG_LOG_KEEP = "debug_log_keep"

LOG_DIR_NAME = "local_openai_stt_sessions"

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT = ""

DEFAULT_VAD_MIN_SPEECH_SECONDS = 0.3
DEFAULT_VAD_SPEECH_THRESHOLD = 0.4
DEFAULT_DEBUG_LOG = False
DEFAULT_DEBUG_LOG_KEEP = 20


class VadSensitivity(StrEnum):
    """End-of-speech sensitivity. Mirrors ``homeassistant.components.assist_pipeline.vad.VadSensitivity`` so users see the same Relaxed/Default/Aggressive choices they see elsewhere in HA voice config."""

    DEFAULT = "default"
    RELAXED = "relaxed"
    AGGRESSIVE = "aggressive"

    @staticmethod
    def to_seconds(sensitivity: VadSensitivity | str) -> float:
        """Return the trailing-silence threshold in seconds for the given level."""
        sensitivity = VadSensitivity(sensitivity)
        if sensitivity is VadSensitivity.RELAXED:
            return 1.25
        if sensitivity is VadSensitivity.AGGRESSIVE:
            return 0.25
        return 0.7


DEFAULT_VAD_SENSITIVITY = VadSensitivity.DEFAULT.value
