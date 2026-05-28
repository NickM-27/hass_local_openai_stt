"""Send the captured WAV variants to an OpenAI-compatible STT backend.

Reads every WAV under ``scripts/samples`` and posts each through the same
``audio.transcriptions.create`` call the integration uses. Files are grouped
by clip (sharing a base name, e.g. ``foo.wav`` and ``foo.compressed.wav``)
so the transcription of each variant can be compared side-by-side.

Defaults match the user's running backend. Override via env vars
``STT_BASE_URL`` / ``STT_MODEL`` / ``STT_API_KEY`` if needed.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI, OpenAIError

# Qwen3-ASR emits a trained-in prefix like ``language English<asr_text>...`` on
# every response. The reference servers (vLLM, qwen-asr-server) strip it before
# returning to the client, but llama-server's /v1/audio/transcriptions handler
# doesn't, so we strip it here.
QWEN_ASR_PREFIX = re.compile(r"^language\s+\S+\s*<asr_text>", re.IGNORECASE)

BASE_URL = "https://stt.storagewhacker.com/v1"
API_KEY = "not-needed"
MODELS = ("Gemma4-ASR", "Qwen3-ASR")

PROMPT = (
    "You transcribe English voice input for a Home Assistant smart-home "
    "system. Utterances are usually smart-home commands (lights, switches, "
    "thermostat, locks, media, timers), weather questions, or general "
    "knowledge questions, but may be arbitrary. Output only the exact words "
    "spoken, verbatim, in English. Do not translate, paraphrase, summarize, "
    "or add commentary. Do not invent words. If the audio is silent, "
    "unintelligible, or contains only background noise, output an empty string."
)

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

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    print(f"endpoint={BASE_URL} models={','.join(MODELS)}\n")

    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in wavs:
        clip_id, variant = variant_of(p)
        groups[clip_id].append((variant, p))

    for clip_id in sorted(groups):
        variants = sorted(groups[clip_id], key=lambda vp: (vp[0] != "plain", vp[0]))
        print(f"clip {clip_id}")
        v_width = max(len(v) for v, _ in variants)
        m_width = max(len(m) for m in MODELS)
        for variant, p in variants:
            with p.open("rb") as fp:
                data = fp.read()
            for model in MODELS:
                try:
                    result = client.audio.transcriptions.create(
                        model=model,
                        file=(p.name, data, "audio/wav"),
                        response_format="json",
                        temperature=0.0,
                        language="en",
                        prompt=PROMPT,
                    )
                    text = (getattr(result, "text", "") or "").strip()
                    text = QWEN_ASR_PREFIX.sub("", text).strip()
                    tag = "OK " if text else "EMPTY"
                    print(f"  {variant:<{v_width}}  {model:<{m_width}}  {tag}  {text!r}")
                except OpenAIError as err:
                    print(f"  {variant:<{v_width}}  {model:<{m_width}}  ERR  {err!r}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
