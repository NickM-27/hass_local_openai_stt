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
CONF_VAD_SPEECH_THRESHOLD = "vad_speech_threshold"
CONF_MIC_GAIN = "mic_gain"
CONF_DEBUG_LOG = "debug_log"
CONF_DEBUG_LOG_KEEP = "debug_log_keep"

LOG_DIR_NAME = "local_openai_stt_sessions"

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT = ""

DEFAULT_VAD_SPEECH_THRESHOLD = 0.4
DEFAULT_MIC_GAIN = 1.0
DEFAULT_DEBUG_LOG = False
DEFAULT_DEBUG_LOG_KEEP = 20


class VadSensitivity(StrEnum):
    """End-of-speech sensitivity.

    ``DYNAMIC`` grows the trailing-silence threshold with accumulated
    speech; the other tiers mirror
    ``homeassistant.components.assist_pipeline.vad.VadSensitivity``
    (Relaxed/Default/Aggressive) so users see the same fixed-tier
    choices they see elsewhere in HA voice config.
    """

    DYNAMIC = "dynamic"
    DEFAULT = "default"
    RELAXED = "relaxed"
    AGGRESSIVE = "aggressive"

    @staticmethod
    def to_seconds(
        sensitivity: VadSensitivity | str, speech_seconds: float = 0.0
    ) -> float:
        """Return the trailing-silence threshold in seconds for the given level.

        ``speech_seconds`` is only consulted for :attr:`DYNAMIC`, where short
        utterances get a snappy cutoff and longer ones get more room for
        thinking pauses. Piecewise-linear between 1.5 s (floor 0.5 s) and
        4.0 s (ceiling 1.25 s, matching :attr:`RELAXED`).
        """
        sensitivity = VadSensitivity(sensitivity)
        if sensitivity is VadSensitivity.DYNAMIC:
            if speech_seconds <= 1.5:
                return 0.5
            if speech_seconds >= 4.0:
                return 1.25
            return 0.5 + (speech_seconds - 1.5) * (1.25 - 0.5) / (4.0 - 1.5)
        if sensitivity is VadSensitivity.RELAXED:
            return 1.25
        if sensitivity is VadSensitivity.AGGRESSIVE:
            return 0.25
        return 0.7


DEFAULT_VAD_SENSITIVITY = VadSensitivity.DYNAMIC.value
