"""Send the captured WAV variants to an OpenAI-compatible STT backend.

Reads every WAV under ``scripts/samples`` (and ``scripts/samples/processed``)
and posts each through the same ``audio.transcriptions.create`` call the
integration uses, then prints a table of model output per file so we can
see which preprocessing variants survive transcription.

Defaults match the user's running backend. Override via env vars
``STT_BASE_URL`` / ``STT_MODEL`` / ``STT_API_KEY`` if needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI, OpenAIError

BASE_URL = os.environ.get("STT_BASE_URL", "https://stt.storagewhacker.com/v1")
MODEL = os.environ.get("STT_MODEL", "Gemma4-ASR")
API_KEY = os.environ.get("STT_API_KEY", "not-needed")

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


def main() -> int:
    if not SAMPLES_DIR.is_dir():
        print(f"missing samples dir: {SAMPLES_DIR}", file=sys.stderr)
        return 1

    wavs = sorted(SAMPLES_DIR.rglob("*.wav"))
    if not wavs:
        print(f"no WAV files under {SAMPLES_DIR}", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    print(f"endpoint={BASE_URL} model={MODEL}\n")

    width = max(len(str(p.relative_to(SAMPLES_DIR))) for p in wavs)
    for p in wavs:
        rel = str(p.relative_to(SAMPLES_DIR))
        try:
            with p.open("rb") as fp:
                result = client.audio.transcriptions.create(
                    model=MODEL,
                    file=(p.name, fp.read(), "audio/wav"),
                    response_format="json",
                    temperature=0.0,
                    language="en",
                    prompt=PROMPT,
                )
            text = (getattr(result, "text", "") or "").strip()
            tag = "OK " if text else "EMPTY"
            print(f"{rel:<{width}}  {tag}  {text!r}")
        except OpenAIError as err:
            print(f"{rel:<{width}}  ERR  {err!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
