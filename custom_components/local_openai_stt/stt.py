"""Speech-to-text entity for the Local OpenAI STT integration."""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
import wave

from openai import AsyncOpenAI, OpenAIError
from pymicro_vad import MicroVad

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
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_VAD_MIN_SPEECH_SECONDS,
    CONF_VAD_SILENCE_SECONDS,
    CONF_VAD_SPEECH_THRESHOLD,
    DEFAULT_API_KEY,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_VAD_MIN_SPEECH_SECONDS,
    DEFAULT_VAD_SILENCE_SECONDS,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
SAMPLES_PER_VAD_CHUNK = 160  # 10 ms at 16 kHz
BYTES_PER_VAD_CHUNK = SAMPLES_PER_VAD_CHUNK * SAMPLE_WIDTH
VAD_CHUNK_SECONDS = SAMPLES_PER_VAD_CHUNK / SAMPLE_RATE


SUPPORTED_LANGUAGES: list[str] = [
    "af", "ar", "az", "be", "bg", "bs", "ca", "cs", "cy", "da",
    "de", "el", "en", "es", "et", "fa", "fi", "fr", "gl", "he",
    "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "kk", "kn",
    "ko", "lt", "lv", "mi", "mk", "mr", "ms", "ne", "nl", "no",
    "pl", "pt", "ro", "ru", "sk", "sl", "sr", "sv", "sw", "ta",
    "th", "tl", "tr", "uk", "ur", "vi", "zh",
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
        silence_seconds = float(
            opts.get(CONF_VAD_SILENCE_SECONDS, DEFAULT_VAD_SILENCE_SECONDS)
        )
        min_speech_seconds = float(
            opts.get(CONF_VAD_MIN_SPEECH_SECONDS, DEFAULT_VAD_MIN_SPEECH_SECONDS)
        )
        threshold = float(
            opts.get(CONF_VAD_SPEECH_THRESHOLD, DEFAULT_VAD_SPEECH_THRESHOLD)
        )

        pcm = await _collect_until_silence(
            stream,
            silence_seconds=silence_seconds,
            min_speech_seconds=min_speech_seconds,
            speech_threshold=threshold,
        )

        if not pcm:
            return SpeechResult(None, SpeechResultState.ERROR)

        wav_bytes = _pcm_to_wav(pcm)

        try:
            text = await self._transcribe(metadata, wav_bytes)
        except OpenAIError as err:
            _LOGGER.error("Transcription failed: %s", err)
            return SpeechResult(None, SpeechResultState.ERROR)

        if text is None:
            return SpeechResult(None, SpeechResultState.ERROR)

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
            "temperature": float(
                opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            ),
        }

        prompt = opts.get(CONF_PROMPT, DEFAULT_PROMPT)
        if prompt:
            kwargs["prompt"] = prompt

        if metadata.language:
            # OpenAI/Whisper expects ISO-639-1 (e.g. "en", not "en-US").
            kwargs["language"] = metadata.language.split("-")[0]

        result = await client.audio.transcriptions.create(**kwargs)
        return getattr(result, "text", None)


async def _collect_until_silence(
    stream: AsyncIterable[bytes],
    *,
    silence_seconds: float,
    min_speech_seconds: float,
    speech_threshold: float,
) -> bytes:
    """Read PCM frames from the stream until the user stops speaking.

    The HA pipeline streams 16 kHz mono int16 PCM. We feed the data through
    pymicro-vad in 10 ms chunks and stop once we have observed enough trailing
    silence after speech to consider the utterance complete. If the stream
    ends before that condition is met, we return whatever we have.
    """
    vad = MicroVad()
    recorded = bytearray()
    leftover = bytearray()

    speech_started = False
    speech_seconds = 0.0
    trailing_silence = 0.0
    finished = False

    async for chunk in stream:
        if not chunk:
            continue
        recorded.extend(chunk)

        if finished:
            # Drain remaining bytes from the iterator to let the producer close
            # cleanly, but do not record them or run VAD.
            continue

        leftover.extend(chunk)
        while len(leftover) >= BYTES_PER_VAD_CHUNK:
            frame = bytes(leftover[:BYTES_PER_VAD_CHUNK])
            del leftover[:BYTES_PER_VAD_CHUNK]

            prob = vad.Process10ms(frame)

            if prob >= speech_threshold:
                speech_started = True
                speech_seconds += VAD_CHUNK_SECONDS
                trailing_silence = 0.0
            elif speech_started:
                trailing_silence += VAD_CHUNK_SECONDS

            if (
                speech_started
                and speech_seconds >= min_speech_seconds
                and trailing_silence >= silence_seconds
            ):
                finished = True
                # Whisper benefits from ~200 ms of trailing silence; drop the rest.
                extra = max(0.0, trailing_silence - 0.2)
                drop = int(extra * SAMPLE_RATE) * SAMPLE_WIDTH
                if 0 < drop < len(recorded):
                    del recorded[len(recorded) - drop :]
                break

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
