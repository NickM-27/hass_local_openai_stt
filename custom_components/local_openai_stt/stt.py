"""Speech-to-text entity for the Local OpenAI STT integration."""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
import time
import wave

import numpy as np
from openai import AsyncOpenAI, OpenAIError
from pysilero_vad import SileroVoiceActivityDetector

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechAudioProcessing,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_DEBUG_LOG,
    CONF_DEBUG_LOG_KEEP,
    CONF_MIC_GAIN,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_VAD_SENSITIVITY,
    CONF_VAD_SPEECH_THRESHOLD,
    DEFAULT_API_KEY,
    DEFAULT_DEBUG_LOG,
    DEFAULT_DEBUG_LOG_KEEP,
    DEFAULT_MIC_GAIN,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_VAD_SENSITIVITY,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    DOMAIN,
    VadSensitivity,
)
from .session_log import SessionLogger, open_session_logger

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
# Silero requires exactly 512 samples per call at 16 kHz (~32 ms).
SAMPLES_PER_VAD_CHUNK = SileroVoiceActivityDetector.chunk_samples()
BYTES_PER_VAD_CHUNK = SileroVoiceActivityDetector.chunk_bytes()
VAD_CHUNK_SECONDS = SAMPLES_PER_VAD_CHUNK / SAMPLE_RATE

# How long we keep listening before giving up if VAD never declares speech.
# Once speech has started, this no longer applies — long questions are fine
# as long as voice activity continues.
NO_SPEECH_TIMEOUT_SECONDS = 5.0


SUPPORTED_LANGUAGES: list[str] = [
    "af",
    "ar",
    "az",
    "be",
    "bg",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "fi",
    "fr",
    "gl",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "kk",
    "kn",
    "ko",
    "lt",
    "lv",
    "mi",
    "mk",
    "mr",
    "ms",
    "ne",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "sw",
    "ta",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the STT entity from a config entry."""
    async_add_entities([LocalOpenAISTTEntity(config_entry)])


class LocalOpenAISTTEntity(SpeechToTextEntity):
    """OpenAI-compatible STT entity that performs its own end-of-speech detection."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialise the entity."""
        self._entry = entry
        self._attr_unique_id = entry.entry_id
        self._client: AsyncOpenAI | None = None

    @property
    def _opts(self) -> dict:
        """Return effective settings with options overriding initial data."""
        return {**self._entry.data, **self._entry.options}

    @property
    def supported_languages(self) -> list[str]:
        """Return supported language codes."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return supported audio formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return supported bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return supported sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return supported channel layouts."""
        return [AudioChannels.CHANNEL_MONO]

    @property
    def audio_processing(self) -> SpeechAudioProcessing:
        """Tell the pipeline this entity does its own end-of-speech detection."""
        return SpeechAudioProcessing(
            requires_external_vad=False,
            prefers_auto_gain_enabled=True,
            prefers_noise_reduction_enabled=True,
        )

    def _get_client(self) -> AsyncOpenAI:
        """Return a cached AsyncOpenAI client pointed at the configured server."""
        if self._client is None:
            opts = self._opts
            self._client = AsyncOpenAI(
                base_url=opts[CONF_BASE_URL],
                api_key=opts.get(CONF_API_KEY) or DEFAULT_API_KEY,
            )
        return self._client

    async def async_will_remove_from_hass(self) -> None:
        """Close the OpenAI client when the entity is removed."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Run VAD, then send the captured utterance to the STT backend."""
        opts = self._opts
        sensitivity = opts.get(CONF_VAD_SENSITIVITY, DEFAULT_VAD_SENSITIVITY)
        silence_seconds = VadSensitivity.to_seconds(sensitivity)
        threshold = float(
            opts.get(CONF_VAD_SPEECH_THRESHOLD, DEFAULT_VAD_SPEECH_THRESHOLD)
        )
        mic_gain = float(opts.get(CONF_MIC_GAIN, DEFAULT_MIC_GAIN))
        silence_prob_threshold = max(0.1, threshold * 0.4)

        session_logger = open_session_logger(
            hass=self.hass,
            enabled=bool(opts.get(CONF_DEBUG_LOG, DEFAULT_DEBUG_LOG)),
            keep=int(opts.get(CONF_DEBUG_LOG_KEEP, DEFAULT_DEBUG_LOG_KEEP)),
            metadata=metadata,
            chunk_samples=SAMPLES_PER_VAD_CHUNK,
            chunk_bytes=BYTES_PER_VAD_CHUNK,
            silence_seconds=silence_seconds,
            threshold=threshold,
            silence_prob_threshold=silence_prob_threshold,
            mic_gain=mic_gain,
        )

        pcm = await _collect_until_silence(
            stream,
            silence_seconds=silence_seconds,
            speech_threshold=threshold,
            silence_prob_threshold=silence_prob_threshold,
            mic_gain=mic_gain,
            session_logger=session_logger,
        )

        if not pcm:
            session_logger.write_event("RESULT empty_audio")
            session_logger.close()
            return SpeechResult(None, SpeechResultState.ERROR)

        wav_bytes = _pcm_to_wav(pcm)

        try:
            text = await self._transcribe(metadata, wav_bytes)
        except OpenAIError as err:
            _LOGGER.error("Transcription failed: %s", err)
            session_logger.write_event(f"RESULT transcription_error: {err!r}")
            session_logger.close()
            return SpeechResult(None, SpeechResultState.ERROR)

        if text is None:
            session_logger.write_event("RESULT no_text")
            session_logger.close()
            return SpeechResult(None, SpeechResultState.ERROR)

        session_logger.write_event(f"RESULT ok text={text!r} chars={len(text)}")
        session_logger.close()
        return SpeechResult(text, SpeechResultState.SUCCESS)

    async def _transcribe(
        self, metadata: SpeechMetadata, wav_bytes: bytes
    ) -> str | None:
        """POST the WAV to the OpenAI-compatible transcription endpoint."""
        opts = self._opts
        client = self._get_client()

        kwargs: dict = {
            "model": opts[CONF_MODEL],
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "response_format": "json",
            "temperature": float(opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
        }

        prompt = opts.get(CONF_PROMPT, DEFAULT_PROMPT)
        if prompt:
            kwargs["prompt"] = prompt

        if metadata.language:
            # OpenAI/Whisper expects ISO-639-1 (e.g. "en", not "en-US").
            kwargs["language"] = metadata.language.split("-")[0]

        result = await client.audio.transcriptions.create(**kwargs)
        return getattr(result, "text", None)


def _apply_gain(pcm: bytes, gain: float) -> bytes:
    """Multiply each int16 sample by ``gain`` and clip to int16 range."""
    if gain == 1.0 or not pcm:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * gain
    np.clip(arr, np.iinfo(np.int16).min, np.iinfo(np.int16).max, out=arr)
    return arr.astype(np.int16).tobytes()


async def _collect_until_silence(
    stream: AsyncIterable[bytes],
    *,
    silence_seconds: float,
    speech_threshold: float,
    silence_prob_threshold: float,
    mic_gain: float,
    session_logger: SessionLogger,
) -> bytes:
    """Read PCM frames from the stream until the user stops speaking.

    The HA pipeline streams 16 kHz mono int16 PCM. We feed the data through
    Silero VAD in fixed-size chunks (512 samples / ~32 ms) and stop once we
    have observed enough trailing silence after speech to consider the
    utterance complete. If the stream ends before that condition is met, we
    return whatever we have.

    ``mic_gain`` is applied to each chunk before VAD *and* before recording,
    so Whisper sees the same boosted audio Silero used for its decision.

    Hysteresis: ``speech_threshold`` decides "this is clearly speech"
    (resets trailing-silence). ``silence_prob_threshold`` (lower) decides
    "this is clearly silence" (accumulates trailing-silence). Probabilities
    between the two are "uncertain" and leave the state alone — that
    prevents sentences from being cut off when the user's voice naturally
    dips through the speech threshold mid-utterance.
    """
    vad = SileroVoiceActivityDetector()
    recorded = bytearray()
    leftover = bytearray()

    speech_started = False
    speech_seconds = 0.0
    trailing_silence = 0.0
    chunk_index = 0
    last_recv_monotonic: float | None = None
    session_start = time.monotonic()

    async for chunk in stream:
        now = time.monotonic()
        gap = (now - last_recv_monotonic) if last_recv_monotonic is not None else 0.0
        last_recv_monotonic = now

        if not chunk:
            session_logger.write_event(
                f"STREAM end_of_stream gap={gap:.3f}s "
                f"speech_started={speech_started} "
                f"speech_seconds={speech_seconds:.3f} "
                f"trailing_silence={trailing_silence:.3f}"
            )
            break

        chunk = _apply_gain(chunk, mic_gain)
        session_logger.write_event(f"RECV bytes={len(chunk)} gap_since_prev={gap:.3f}s")

        recorded.extend(chunk)
        leftover.extend(chunk)

        while len(leftover) >= BYTES_PER_VAD_CHUNK:
            frame = bytes(leftover[:BYTES_PER_VAD_CHUNK])
            del leftover[:BYTES_PER_VAD_CHUNK]

            prob = vad(frame)

            if prob >= speech_threshold:
                state = "speech"
            elif prob < silence_prob_threshold:
                state = "silence"
            else:
                state = "uncertain"

            if state == "speech":
                if not speech_started:
                    session_logger.write_event(
                        f"SPEECH_START prob={prob:.3f} chunk_index={chunk_index}"
                    )
                speech_started = True
                speech_seconds += VAD_CHUNK_SECONDS
                trailing_silence = 0.0
            elif state == "silence" and speech_started:
                trailing_silence += VAD_CHUNK_SECONDS
            # "uncertain" while in speech: hold state, neither reset nor accumulate.

            session_logger.log_chunk(
                index=chunk_index,
                prob=prob,
                state=state,
                speech_started=speech_started,
                speech_seconds=speech_seconds,
                trailing_silence=trailing_silence,
            )
            chunk_index += 1

            if not speech_started:
                elapsed = time.monotonic() - session_start
                if elapsed >= NO_SPEECH_TIMEOUT_SECONDS:
                    session_logger.write_event(
                        f"NO_SPEECH_TIMEOUT elapsed={elapsed:.3f}s "
                        f"recorded_bytes={len(recorded)}"
                    )
                    session_logger.close_collection(audio_bytes=len(recorded))
                    return bytes(recorded)

            if speech_started and trailing_silence >= silence_seconds:
                # End-of-speech: returning closes the stream from our side, which
                # is required because `requires_external_vad=False` means the
                # pipeline will not send a terminating empty chunk.
                # Whisper benefits from ~200 ms of trailing silence; drop the rest.
                extra = max(0.0, trailing_silence - 0.2)
                drop = int(extra * SAMPLE_RATE) * SAMPLE_WIDTH
                if 0 < drop < len(recorded):
                    del recorded[len(recorded) - drop :]
                _LOGGER.debug(
                    "End of speech: %.2fs speech, %.2fs trailing silence, %d bytes",
                    speech_seconds,
                    trailing_silence,
                    len(recorded),
                )
                session_logger.write_event(
                    f"END_OF_SPEECH speech_seconds={speech_seconds:.3f} "
                    f"trailing_silence={trailing_silence:.3f} "
                    f"recorded_bytes={len(recorded)} dropped_bytes={max(0, drop)}"
                )
                session_logger.close_collection(audio_bytes=len(recorded))
                return bytes(recorded)

    session_logger.write_event(
        f"STREAM_EXHAUSTED speech_started={speech_started} "
        f"speech_seconds={speech_seconds:.3f} "
        f"trailing_silence={trailing_silence:.3f} "
        f"recorded_bytes={len(recorded)}"
    )
    session_logger.close_collection(audio_bytes=len(recorded))
    return bytes(recorded)


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw 16 kHz mono int16 PCM in a WAV container."""
    buf = io.BytesIO()
    wf: wave.Wave_write = wave.open(buf, "wb")  # type: ignore[assignment]
    try:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    finally:
        wf.close()
    return buf.getvalue()


__all__ = ["DOMAIN", "LocalOpenAISTTEntity", "async_setup_entry"]
