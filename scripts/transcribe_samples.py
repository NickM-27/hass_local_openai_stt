"""Send the captured WAV variants to an OpenAI-compatible STT backend.

Reads every WAV under ``scripts/samples`` and posts each through the same
``audio.transcriptions.create`` call the integration uses. Files are grouped
by clip (sharing a base name, e.g. ``foo.wav`` and ``foo.compressed.wav``)
so the transcription of each variant can be compared side-by-side, across
every configured backend.

Add entries to ``BACKENDS`` to compare multiple endpoints/models at once.
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import NOT_GIVEN, OpenAI, OpenAIError

# Qwen3-ASR emits a trained-in prefix like ``language English<asr_text>...`` on
# every response. The reference servers (vLLM, qwen-asr-server) strip it before
# returning to the client, but llama-server's /v1/audio/transcriptions handler
# doesn't, so we strip it here.
QWEN_ASR_PREFIX = re.compile(r"^language\s+\S+\s*<asr_text>", re.IGNORECASE)

API_KEY = "not-needed"

PROMPT = (
    "You transcribe English voice input for a Home Assistant smart-home system. Utterances are usually smart-home commands (lights, switches, thermostat, locks, media, timers), weather questions, or general knowledge questions, but may be arbitrary."
)

# Backends to transcribe each clip through, as ``(base_url, model, prompt)``;
# a ``None`` prompt omits the field. Add entries here to compare several at once.
#
# LFM-ASR needs a ``Perform ASR in <language>`` system prompt. llama-server's
# transcription handler already builds that from ``language``, and appends any
# ``prompt`` as extra user text, which LFM then tends to echo back as the
# transcript. So it must be sent without a prompt.
BACKENDS: list[tuple[str, str, str | None]] = [
    ("https://chat.storagewhacker.com/v1", "Qwen3-ASR", PROMPT),
    ("https://chat.storagewhacker.com/v1", "Nemotron-Bolt", PROMPT),
    ("https://chat.storagewhacker.com/v1", "LFM-ASR", None),
]

SAMPLES_DIR = Path(__file__).parent / "samples"

# Suffixes the integration tags onto a clip's stem when it writes a processed
# variant alongside the plain pass-through. The base WAV (``<clip>.wav``) has
# no suffix and is reported as ``plain``.
VARIANT_SUFFIXES = ("compressed",)


def variant_of(path: Path) -> tuple[str, str]:
    """Return ``(clip_id, variant)`` for a sample file."""
    stem = path.name[: -len(".wav")]
    for suffix in VARIANT_SUFFIXES:
        tail = f".{suffix}"
        if stem.endswith(tail):
            return stem[: -len(tail)], suffix
    return stem, "plain"


def main() -> int:
    if not SAMPLES_DIR.is_dir():
        print(f"missing samples dir: {SAMPLES_DIR}", file=sys.stderr)
        return 1

    wavs = sorted(SAMPLES_DIR.rglob("*.wav"))
    if not wavs:
        print(f"no WAV files under {SAMPLES_DIR}", file=sys.stderr)
        return 1

    if not BACKENDS:
        print("no backends configured in BACKENDS", file=sys.stderr)
        return 1

    # One client per backend, paired with its model and prompt.
    backends = [
        (model, prompt, OpenAI(base_url=base_url, api_key=API_KEY))
        for base_url, model, prompt in BACKENDS
    ]
    for base_url, model, _ in BACKENDS:
        print(f"backend {model} @ {base_url}")
    print()

    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in wavs:
        clip_id, variant = variant_of(p)
        groups[clip_id].append((variant, p))

    m_width = max(len(model) for model, _, _ in backends)
    for clip_id in sorted(groups):
        variants = sorted(groups[clip_id], key=lambda vp: (vp[0] != "plain", vp[0]))
        print(f"clip {clip_id}")
        v_width = max(len(v) for v, _ in variants)
        for variant, p in variants:
            with p.open("rb") as fp:
                data = fp.read()
            for model, prompt, client in backends:
                start = time.monotonic()
                try:
                    result = client.audio.transcriptions.create(
                        model=model,
                        file=(p.name, data, "audio/wav"),
                        response_format="json",
                        temperature=0.0,
                        language="en",
                        prompt=prompt if prompt is not None else NOT_GIVEN,
                    )
                    text = (getattr(result, "text", "") or "").strip()
                    text = QWEN_ASR_PREFIX.sub("", text).strip()
                    elapsed = time.monotonic() - start
                    tag = "OK " if text else "EMPTY"
                    print(
                        f"  {variant:<{v_width}}  {model:<{m_width}}  {elapsed:5.2f}s  {tag}  {text!r}"
                    )
                except OpenAIError as err:
                    print(f"  {variant:<{v_width}}  {model:<{m_width}}  ERR  {err!r}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
