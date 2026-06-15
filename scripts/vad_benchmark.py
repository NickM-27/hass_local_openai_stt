"""Compare VAD backends on the captured sample clips.

Replays every WAV under ``scripts/samples`` through several voice-activity
detectors as if they were streaming off the Home Assistant pipeline, then runs
the *exact* end-of-speech decision state machine the integration uses
(:func:`custom_components.local_openai_stt.stt._collect_until_silence`) on each
detector's probability stream. For every clip/backend pair it reports:

* whether the 5 s no-speech timeout would fire (the failure mode we care about:
  VAD never detected that the user was talking),
* when speech latched and when end-of-speech fired (so you can see cut-off
  behaviour and trailing-silence latency),
* CPU cost: total inference time and real-time factor (compute / audio).

The decision logic is held constant across backends on purpose -- this measures
each VAD as a *drop-in replacement* feeding the existing state machine, which is
what swapping the library would actually do. Only the per-frame probabilities
(and native frame size) differ.

Backends auto-skip if their package/weights aren't installed, so you can run
with whatever subset you have. Install everything with::

    .venv/bin/pip install numpy "pysilero-vad==3.0.1" onnxruntime ten-vad \\
        silero-vad torch huggingface_hub
    # FireRedVAD weights:
    .venv/bin/python -c "from huggingface_hub import snapshot_download as d; \\
        d(repo_id='FireRedTeam/FireRedVAD', local_dir='scripts/.fireredvad_models')"

Usage::

    .venv/bin/python scripts/vad_benchmark.py                 # full table
    .venv/bin/python scripts/vad_benchmark.py --trace CLIP    # per-frame trace
    .venv/bin/python scripts/vad_benchmark.py --threshold 0.4 --sensitivity dynamic
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

# --- Decision-logic constants, mirrored from custom_components/.../stt.py ---
NO_SPEECH_TIMEOUT_SECONDS = 5.0
MIN_SPEECH_SECONDS_TO_LATCH = 0.3
MIN_POST_LATCH_SECONDS = 0.7

SAMPLES_DIR = Path(__file__).parent / "samples"
FIRERED_MODEL_DIR = Path(__file__).parent / ".fireredvad_models" / "Stream-VAD"


def trailing_silence_cutoff(sensitivity: str, speech_seconds: float) -> float:
    """Port of ``VadSensitivity.to_seconds`` from the integration's const.py."""
    if sensitivity == "dynamic":
        if speech_seconds <= 1.5:
            return 0.5
        if speech_seconds >= 4.0:
            return 1.25
        return 0.5 + (speech_seconds - 1.5) * (1.25 - 0.5) / (4.0 - 1.5)
    if sensitivity == "relaxed":
        return 1.25
    if sensitivity == "aggressive":
        return 0.25
    return 0.7  # default


# --------------------------------------------------------------------------- #
# VAD adapters: each yields one probability per native frame for an int16 clip.
# --------------------------------------------------------------------------- #
@dataclass
class FrameProbs:
    """A backend's probability stream for one clip, plus its CPU cost."""

    name: str
    frame_seconds: float
    probs: list[float]
    infer_seconds: float  # wall time spent purely inside vad inference
    audio_seconds: float

    @property
    def rtf(self) -> float:
        return self.infer_seconds / self.audio_seconds if self.audio_seconds else 0.0


# An adapter is a callable: (int16 samples) -> FrameProbs, or None if unavailable.
Adapter = Callable[[np.ndarray], FrameProbs]


def _stream(samples: np.ndarray, frame: int, step_fn) -> tuple[list[float], float]:
    """Feed non-overlapping ``frame``-sample windows to ``step_fn``; time it."""
    probs: list[float] = []
    infer = 0.0
    for j in range(0, len(samples) - frame + 1, frame):
        window = samples[j : j + frame]
        t0 = time.perf_counter()
        probs.append(step_fn(window))
        infer += time.perf_counter() - t0
    return probs, infer


def make_pysilero() -> Adapter | None:
    try:
        from pysilero_vad import SileroVoiceActivityDetector
    except ImportError:
        return None
    frame = SileroVoiceActivityDetector.chunk_samples()  # 512

    def run(samples: np.ndarray) -> FrameProbs:
        vad = SileroVoiceActivityDetector()  # fresh state per clip
        raw = samples.tobytes()
        nbytes = frame * SAMPLE_WIDTH
        # warm up (first call pays one-time setup we don't want in the timing)
        vad(raw[:nbytes]) if len(raw) >= nbytes else None
        vad = SileroVoiceActivityDetector()

        def step(window: np.ndarray) -> float:
            return vad(window.tobytes())

        probs, infer = _stream(samples, frame, step)
        return FrameProbs("pysilero-vad 3.0.1 (silero v6.2 ggml)", frame / SAMPLE_RATE,
                          probs, infer, len(samples) / SAMPLE_RATE)

    return run


def make_silero_official() -> Adapter | None:
    try:
        import torch
        from silero_vad import load_silero_vad
    except ImportError:
        return None
    try:
        model = load_silero_vad(onnx=True)
    except Exception:
        return None
    frame = 512
    import importlib.metadata as md
    try:
        ver = md.version("silero-vad")
    except md.PackageNotFoundError:
        ver = "?"

    def run(samples: np.ndarray) -> FrameProbs:
        model.reset_states()
        warm = torch.from_numpy(samples[:frame].astype("float32") / 32768.0)
        model(warm, SAMPLE_RATE)
        model.reset_states()

        def step(window: np.ndarray) -> float:
            t = torch.from_numpy(window.astype("float32") / 32768.0)
            return model(t, SAMPLE_RATE).item()

        probs, infer = _stream(samples, frame, step)
        return FrameProbs(f"silero-vad {ver} (official onnx)", frame / SAMPLE_RATE,
                          probs, infer, len(samples) / SAMPLE_RATE)

    return run


def make_ten_vad() -> Adapter | None:
    try:
        from ten_vad import TenVad
    except ImportError:
        return None
    frame = 256  # 16 ms hop

    def run(samples: np.ndarray) -> FrameProbs:
        vad = TenVad(hop_size=frame)
        vad.process(samples[:frame])  # warm up
        vad = TenVad(hop_size=frame)

        def step(window: np.ndarray) -> float:
            return float(vad.process(window)[0])

        probs, infer = _stream(samples, frame, step)
        return FrameProbs("ten-vad", frame / SAMPLE_RATE, probs, infer,
                          len(samples) / SAMPLE_RATE)

    return run


def make_fireredvad() -> Adapter | None:
    if not (FIRERED_MODEL_DIR / "model.pth.tar").exists():
        return None
    try:
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
    except ImportError:
        return None
    frame = 400  # FRAME_LENGTH_SAMPLE, 25 ms

    def run(samples: np.ndarray) -> FrameProbs:
        cfg = FireRedStreamVadConfig(use_gpu=False)
        vad = FireRedStreamVad.from_pretrained(str(FIRERED_MODEL_DIR), cfg)
        vad.reset()
        if len(samples) >= frame:
            vad.detect_frame(samples[:frame])  # warm up
        vad.reset()

        def step(window: np.ndarray) -> float:
            return float(vad.detect_frame(window).raw_prob)

        probs, infer = _stream(samples, frame, step)
        return FrameProbs("fireredvad (stream)", frame / SAMPLE_RATE, probs, infer,
                          len(samples) / SAMPLE_RATE)

    return run


ADAPTERS: dict[str, Callable[[], Adapter | None]] = {
    "pysilero": make_pysilero,
    "silero-official": make_silero_official,
    "ten-vad": make_ten_vad,
    "fireredvad": make_fireredvad,
}


# --------------------------------------------------------------------------- #
# Decision state machine (offline port of _collect_until_silence).
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    speech_start_s: float | None = None  # when SPEECH_START latched (audio time)
    end_of_speech_s: float | None = None  # when end-of-speech fired
    timed_out: bool = False  # 5 s no-speech timeout fired (the bad case)
    exhausted: bool = False  # ran out of audio before end-of-speech
    speech_seconds: float = 0.0  # accumulated in-speech time at termination
    captured_s: float = 0.0  # audio time captured before terminating

    @property
    def end_latency_s(self) -> float | None:
        """Seconds from speech start to end-of-speech (trailing-silence wait)."""
        if self.speech_start_s is None or self.end_of_speech_s is None:
            return None
        return self.end_of_speech_s - self.speech_start_s


def decide(fp: FrameProbs, *, speech_threshold: float, sensitivity: str) -> Decision:
    silence_prob_threshold = max(0.1, speech_threshold * 0.4)
    dt = fp.frame_seconds
    d = Decision()

    speech_started = False
    speech_seconds = 0.0
    trailing_silence = 0.0
    pending_speech_seconds = 0.0
    post_latch_seconds = 0.0

    for i, prob in enumerate(fp.probs):
        elapsed = (i + 1) * dt
        if prob >= speech_threshold:
            state = "speech"
        elif prob < silence_prob_threshold:
            state = "silence"
        else:
            state = "uncertain"

        if state == "speech":
            if speech_started:
                speech_seconds += dt
                trailing_silence = 0.0
            else:
                pending_speech_seconds += dt
                if pending_speech_seconds >= MIN_SPEECH_SECONDS_TO_LATCH:
                    speech_started = True
                    speech_seconds = pending_speech_seconds
                    trailing_silence = 0.0
                    d.speech_start_s = elapsed
        elif state == "silence":
            if speech_started:
                trailing_silence += dt
            else:
                pending_speech_seconds = 0.0
        # "uncertain": hold

        if speech_started:
            post_latch_seconds += dt

        if not speech_started and elapsed >= NO_SPEECH_TIMEOUT_SECONDS:
            d.timed_out = True
            d.captured_s = elapsed
            return d

        cutoff = trailing_silence_cutoff(sensitivity, speech_seconds)
        if speech_started and trailing_silence >= cutoff and post_latch_seconds >= MIN_POST_LATCH_SECONDS:
            d.end_of_speech_s = elapsed
            d.speech_seconds = speech_seconds
            d.captured_s = elapsed
            return d

    d.exhausted = True
    d.speech_seconds = speech_seconds
    d.captured_s = len(fp.probs) * dt
    return d


# --------------------------------------------------------------------------- #
def load_clip(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, SAMPLE_WIDTH):
            raise ValueError(f"{path.name}: expected 16k mono 16-bit")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16)


def fmt(v: float | None, suffix: str = "") -> str:
    return f"{v:.2f}{suffix}" if v is not None else "—"


def sparkline(probs: list[float]) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[min(8, int(p * 8.999))] for p in probs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.4, help="speech threshold (default 0.4)")
    ap.add_argument("--sensitivity", default="dynamic",
                    choices=["dynamic", "default", "relaxed", "aggressive"])
    ap.add_argument("--trace", metavar="CLIP",
                    help="print per-frame probability sparklines for clips whose "
                         "name contains CLIP, then exit")
    args = ap.parse_args(argv)

    if not SAMPLES_DIR.is_dir():
        print(f"missing samples dir: {SAMPLES_DIR}", file=sys.stderr)
        return 1
    clips = sorted(SAMPLES_DIR.rglob("*.wav"))
    if not clips:
        print(f"no WAVs under {SAMPLES_DIR}", file=sys.stderr)
        return 1

    print("loading backends...", file=sys.stderr)
    backends: dict[str, Adapter] = {}
    for key, factory in ADAPTERS.items():
        adapter = factory()
        if adapter is None:
            print(f"  skip {key} (not installed / no weights)", file=sys.stderr)
        else:
            backends[key] = adapter
    if not backends:
        print("no VAD backends available", file=sys.stderr)
        return 1

    # Run every clip through every backend once. Cache the FrameProbs so the
    # trace and table share inference results.
    results: dict[Path, dict[str, FrameProbs]] = {}
    for clip in clips:
        try:
            samples = load_clip(clip)
        except ValueError as e:
            print(f"  skip {clip.name}: {e}", file=sys.stderr)
            continue
        results[clip] = {}
        for key, adapter in backends.items():
            results[clip][key] = adapter(samples)

    if args.trace:
        needle = args.trace.lower()
        traced = [c for c in results if needle in c.name.lower()]
        if not traced:
            print(f"no clip matching {args.trace!r}", file=sys.stderr)
            return 1
        for clip in traced:
            print(f"\n=== {clip.name} ({results[clip][next(iter(backends))].audio_seconds:.2f}s) ===")
            print("each cell ≈ one native frame; █=prob 1.0, space=0.0")
            for key in backends:
                fp = results[clip][key]
                d = decide(fp, speech_threshold=args.threshold, sensitivity=args.sensitivity)
                verdict = ("TIMEOUT" if d.timed_out else
                           "exhausted" if d.exhausted else "ok")
                print(f"\n  {fp.name}  [{fp.frame_seconds*1000:.0f}ms frame] {verdict}")
                print(f"    start={fmt(d.speech_start_s,'s')} end={fmt(d.end_of_speech_s,'s')} "
                      f"latency={fmt(d.end_latency_s,'s')}")
                print(f"    {sparkline(fp.probs)}")
        return 0

    # ----- Summary table: one row per backend -----
    print(f"\nthreshold={args.threshold}  sensitivity={args.sensitivity}  clips={len(results)}\n")
    keys = list(backends)
    name_w = max(len(results[next(iter(results))][k].name) for k in keys)
    hdr = (f"{'backend':<{name_w}}  {'median_rtf':>10}  {'max_rtf':>8}  "
           f"{'timeouts':>8}  {'med_end_lat':>11}  {'exhausted':>9}")
    print(hdr)
    print("-" * len(hdr))
    per_backend: dict[str, dict] = {k: {"rtf": [], "timeouts": 0, "exhausted": 0,
                                        "lat": [], "name": ""} for k in keys}
    for clip, byk in results.items():
        for k in keys:
            fp = byk[k]
            d = decide(fp, speech_threshold=args.threshold, sensitivity=args.sensitivity)
            b = per_backend[k]
            b["name"] = fp.name
            b["rtf"].append(fp.rtf)
            if d.timed_out:
                b["timeouts"] += 1
            if d.exhausted:
                b["exhausted"] += 1
            if d.end_latency_s is not None:
                b["lat"].append(d.end_latency_s)
    for k in keys:
        b = per_backend[k]
        med_rtf = float(np.median(b["rtf"]))
        max_rtf = float(np.max(b["rtf"]))
        med_lat = float(np.median(b["lat"])) if b["lat"] else None
        print(f"{b['name']:<{name_w}}  {med_rtf:>10.4f}  {max_rtf:>8.4f}  "
              f"{b['timeouts']:>8}  {fmt(med_lat,'s'):>11}  {b['exhausted']:>9}")

    # ----- Per-clip detail: speech-start / end / timeout across backends -----
    print("\nper-clip: speech-start s / end-of-speech s  (T=timeout, X=exhausted)\n")
    clip_w = max(len(c.stem) for c in results)
    head = f"{'clip':<{clip_w}}  " + "  ".join(f"{k:>18}" for k in keys)
    print(head)
    print("-" * len(head))
    for clip in sorted(results, key=lambda c: c.stem):
        cells = []
        for k in keys:
            d = decide(results[clip][k], speech_threshold=args.threshold,
                       sensitivity=args.sensitivity)
            if d.timed_out:
                cell = "T (no speech)"
            elif d.exhausted:
                cell = f"X {fmt(d.speech_start_s)}/—"
            else:
                cell = f"{fmt(d.speech_start_s)}/{fmt(d.end_of_speech_s)}"
            cells.append(f"{cell:>18}")
        print(f"{clip.stem:<{clip_w}}  " + "  ".join(cells))

    print("\nnotes:")
    print("  rtf = vad inference time / audio duration (lower = cheaper CPU)")
    print("  timeouts = clips where VAD never latched speech -> 5s wait (the bad case)")
    print("  med_end_lat = median seconds of trailing-silence wait after speech start")
    print("  all backends share the integration's hysteresis/latch/timeout logic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
