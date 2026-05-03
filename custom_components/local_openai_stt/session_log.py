"""Per-session diagnostic logger for STT VAD sessions.

When enabled (via the ``debug_log`` option), each call to
:func:`open_session_logger` creates one file under
``<config>/local_openai_stt_sessions/`` named after the session's start time
in ISO 8601. Every received chunk, VAD probability, and decision event is
written there; older sessions beyond ``keep`` are trimmed when a new one
starts. When disabled, every method is a no-op and the hot path is free.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.components.stt import SpeechMetadata
    from homeassistant.core import HomeAssistant

from .const import LOG_DIR_NAME

_LOGGER = logging.getLogger(__name__)


class SessionLogger:
    """Writes one VAD session's events to a single log file.

    All public methods are safe to call when ``enabled`` is False; in that
    mode they short-circuit immediately. Timestamps in the file are seconds
    relative to ``open_session_logger`` being called.
    """

    def __init__(self, fp: io.TextIOBase | None, t0: float) -> None:
        self._fp = fp
        self._t0 = t0
        self._closed_collection = False
        self._chunk_count = 0
        self._min_prob = 1.0
        self._max_prob = 0.0

    @property
    def enabled(self) -> bool:
        """Return whether this logger is writing to a file."""
        return self._fp is not None

    def write_event(self, line: str) -> None:
        """Write a free-form event line."""
        if self._fp is None:
            return
        self._fp.write(f"t={time.monotonic() - self._t0:.3f} {line}\n")

    def log_chunk(
        self,
        *,
        index: int,
        prob: float,
        state: str,
        speech_started: bool,
        speech_seconds: float,
        trailing_silence: float,
    ) -> None:
        """Write a single VAD-chunk record."""
        if self._fp is None:
            return
        self._chunk_count += 1
        if prob < self._min_prob:
            self._min_prob = prob
        if prob > self._max_prob:
            self._max_prob = prob
        self._fp.write(
            f"t={time.monotonic() - self._t0:.3f} "
            f"CHUNK i={index} prob={prob:.3f} {state} "
            f"speech_started={speech_started} "
            f"speech_total={speech_seconds:.3f} "
            f"trailing_silence={trailing_silence:.3f}\n"
        )

    def close_collection(self, *, audio_bytes: int | None) -> None:
        """Mark the end of the audio-collection phase. Idempotent."""
        if self._fp is None or self._closed_collection:
            return
        self._closed_collection = True
        self._fp.write(
            f"t={time.monotonic() - self._t0:.3f} "
            f"COLLECTION_DONE chunks={self._chunk_count} "
            f"prob_min={self._min_prob:.3f} prob_max={self._max_prob:.3f} "
            f"audio_bytes={audio_bytes}\n"
        )

    def close(self) -> None:
        """Close the underlying file."""
        if self._fp is None:
            return
        try:
            self._fp.write(f"t={time.monotonic() - self._t0:.3f} CLOSE\n")
            self._fp.flush()
            self._fp.close()
        except OSError:
            pass
        self._fp = None


def open_session_logger(
    *,
    hass: HomeAssistant,
    enabled: bool,
    keep: int,
    metadata: SpeechMetadata,
    chunk_samples: int,
    chunk_bytes: int,
    silence_seconds: float,
    threshold: float,
    silence_prob_threshold: float,
    mic_gain: float,
) -> SessionLogger:
    """Open a per-session log file if debug logging is enabled.

    The file name is the ISO 8601 UTC timestamp of session start, e.g.
    ``2026-05-02T14:30:15.123456+00:00.log``. Returns a no-op logger if
    ``enabled`` is False or the directory cannot be created.
    """
    t0 = time.monotonic()
    if not enabled:
        return SessionLogger(fp=None, t0=t0)

    try:
        log_dir = Path(hass.config.path(LOG_DIR_NAME))
        log_dir.mkdir(parents=True, exist_ok=True)
        if keep > 0:
            existing = sorted(log_dir.glob("*.log"))
            for old in existing[: max(0, len(existing) - (keep - 1))]:
                try:
                    old.unlink()
                except OSError:
                    pass

        started = datetime.now(timezone.utc)
        path = log_dir / f"{started.isoformat()}.log"
        fp = path.open("w", encoding="utf-8", buffering=1)
    except OSError as err:
        _LOGGER.warning("Could not create session log file: %s", err)
        return SessionLogger(fp=None, t0=t0)

    fp.write(
        f"# local_openai_stt session\n"
        f"# started={started.isoformat()}\n"
        f"# language={metadata.language}\n"
        f"# sample_rate={metadata.sample_rate} channel={metadata.channel} "
        f"codec={metadata.codec} format={metadata.format}\n"
        f"# vad: chunk_samples={chunk_samples} chunk_bytes={chunk_bytes} "
        f"threshold={threshold:.3f} silence_prob={silence_prob_threshold:.3f} "
        f"silence_seconds={silence_seconds:.3f} "
        f"mic_gain={mic_gain:.2f}\n"
    )
    return SessionLogger(fp=fp, t0=t0)


__all__ = ["SessionLogger", "open_session_logger"]
