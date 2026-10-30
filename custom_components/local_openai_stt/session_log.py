"""Per-session diagnostic logger for STT VAD sessions.

When enabled (via the ``debug_log`` option), each call to
:func:`async_open_session_logger` creates one file under
``<config>/local_openai_stt_sessions/`` named after the session's start time
in ISO 8601. Every received chunk, VAD probability, and decision event is
written there; older sessions beyond ``keep`` are trimmed when a new one
starts. When disabled, every method is a no-op and the hot path is free.

No file I/O happens on the event loop. The file is created in the executor,
and each session then gets a writer thread fed by a queue, so events still
reach disk in order as they happen but the VAD loop never waits on the disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from queue import SimpleQueue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.components.stt import SpeechMetadata
    from homeassistant.core import HomeAssistant

from .const import LOG_DIR_NAME

_LOGGER = logging.getLogger(__name__)

# A log line, a WAV to save, or ``None`` to close the file and stop the writer.
_WriterItem = str | tuple[Path, bytes] | None


class SessionLogger:
    """Writes one VAD session's events to a single log file.

    All public methods are safe to call when ``enabled`` is False; in that
    mode they short-circuit immediately. Otherwise they only enqueue work for
    the session's writer thread. Timestamps in the file are seconds relative
    to ``async_open_session_logger`` being called.
    """

    def __init__(
        self,
        queue: SimpleQueue[_WriterItem] | None,
        t0: float,
        audio_path: Path | None = None,
    ) -> None:
        self._queue = queue
        self._t0 = t0
        self._audio_path = audio_path
        self._closed_collection = False
        self._chunk_count = 0
        self._min_prob = 1.0
        self._max_prob = 0.0

    @property
    def enabled(self) -> bool:
        """Return whether this logger is writing to a file."""
        return self._queue is not None

    def write_event(self, line: str) -> None:
        """Write a free-form event line."""
        if self._queue is None:
            return
        self._queue.put(f"t={time.monotonic() - self._t0:.3f} {line}\n")

    def log_chunk(
        self,
        *,
        index: int,
        prob: float,
        vad_gain: float,
        state: str,
        speech_started: bool,
        speech_seconds: float,
        trailing_silence: float,
    ) -> None:
        """Write a single VAD-chunk record."""
        if self._queue is None:
            return
        self._chunk_count += 1
        if prob < self._min_prob:
            self._min_prob = prob
        if prob > self._max_prob:
            self._max_prob = prob
        self._queue.put(
            f"t={time.monotonic() - self._t0:.3f} "
            f"CHUNK i={index} prob={prob:.3f} gain={vad_gain:.2f} {state} "
            f"speech_started={speech_started} "
            f"speech_total={speech_seconds:.3f} "
            f"trailing_silence={trailing_silence:.3f}\n"
        )

    def save_audio(self, wav_bytes: bytes, *, variant: str | None = None) -> None:
        """Write the captured WAV next to the log file for later replay.

        ``variant`` adds an infix before ``.wav`` (e.g. ``"compressed"`` →
        ``<stem>.compressed.wav``) so multiple takes from one session can be
        saved side by side.
        """
        if self._queue is None or self._audio_path is None or not wav_bytes:
            return
        if variant:
            path = self._audio_path.with_name(
                f"{self._audio_path.stem}.{variant}{self._audio_path.suffix}"
            )
        else:
            path = self._audio_path
        self._queue.put((path, wav_bytes))

    def close_collection(self, *, audio_bytes: int | None) -> None:
        """Mark the end of the audio-collection phase. Idempotent."""
        if self._queue is None or self._closed_collection:
            return
        self._closed_collection = True
        self._queue.put(
            f"t={time.monotonic() - self._t0:.3f} "
            f"COLLECTION_DONE chunks={self._chunk_count} "
            f"prob_min={self._min_prob:.3f} prob_max={self._max_prob:.3f} "
            f"audio_bytes={audio_bytes}\n"
        )

    def close(self) -> None:
        """Close the file once queued writes finish. Idempotent."""
        if self._queue is None:
            return
        self._queue.put(f"t={time.monotonic() - self._t0:.3f} CLOSE\n")
        self._queue.put(None)
        self._queue = None


def _create_log_file(path: Path, keep: int) -> io.TextIOWrapper:
    """Open ``path`` for writing, first trimming older sessions to fit ``keep``."""
    log_dir = path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    if keep > 0:
        existing = sorted(log_dir.glob("*.log"))
        for old in existing[: max(0, len(existing) - (keep - 1))]:
            try:
                old.unlink()
            except OSError:
                pass
            for wav_file in log_dir.glob(f"{old.stem}*.wav"):
                try:
                    wav_file.unlink()
                except OSError:
                    pass
    return path.open("w", encoding="utf-8", buffering=1)


def _run_writer(
    fp: io.TextIOWrapper, t0: float, queue: SimpleQueue[_WriterItem]
) -> None:
    """Drain ``queue`` into ``fp`` until the ``None`` sentinel arrives."""
    try:
        with fp:
            while (item := queue.get()) is not None:
                if isinstance(item, str):
                    fp.write(item)
                    continue
                path, wav_bytes = item
                try:
                    path.write_bytes(wav_bytes)
                except OSError as err:
                    _LOGGER.warning("Could not save session audio: %s", err)
                    continue
                fp.write(
                    f"t={time.monotonic() - t0:.3f} "
                    f"AUDIO_SAVED path={path.name} "
                    f"bytes={len(wav_bytes)}\n"
                )
    except OSError as err:
        _LOGGER.warning("Could not write session log: %s", err)


async def async_open_session_logger(
    *,
    hass: HomeAssistant,
    enabled: bool,
    keep: int,
    metadata: SpeechMetadata,
    chunk_samples: int,
    chunk_bytes: int,
    sensitivity: str,
    threshold: float,
    silence_prob_threshold: float,
    mic_gain: float,
) -> SessionLogger:
    """Open a per-session log file if debug logging is enabled.

    The file name is the ISO 8601 UTC timestamp of session start, e.g.
    ``2026-05-02T14:30:15.123456+00:00.log``. Returns a no-op logger if
    ``enabled`` is False or the file cannot be created. The caller must call
    :meth:`SessionLogger.close` to stop the writer thread.
    """
    t0 = time.monotonic()
    if not enabled:
        return SessionLogger(queue=None, t0=t0)

    started = datetime.now(timezone.utc)
    path = Path(hass.config.path(LOG_DIR_NAME)) / f"{started.isoformat()}.log"
    try:
        fp = await hass.async_add_executor_job(_create_log_file, path, keep)
    except OSError as err:
        _LOGGER.warning("Could not create session log file: %s", err)
        return SessionLogger(queue=None, t0=t0)

    queue: SimpleQueue[_WriterItem] = SimpleQueue()
    queue.put(
        f"# local_openai_stt session\n"
        f"# started={started.isoformat()}\n"
        f"# language={metadata.language}\n"
        f"# sample_rate={metadata.sample_rate} channel={metadata.channel} "
        f"codec={metadata.codec} format={metadata.format}\n"
        f"# vad: chunk_samples={chunk_samples} chunk_bytes={chunk_bytes} "
        f"threshold={threshold:.3f} silence_prob={silence_prob_threshold:.3f} "
        f"sensitivity={sensitivity} "
        f"mic_gain={mic_gain:.2f}\n"
    )
    # Daemon so a writer stuck on a slow disk can't hold up shutdown.
    threading.Thread(
        target=_run_writer,
        args=(fp, t0, queue),
        name="local_openai_stt_session_log",
        daemon=True,
    ).start()
    return SessionLogger(queue=queue, t0=t0, audio_path=path.with_suffix(".wav"))


__all__ = ["SessionLogger", "async_open_session_logger"]
