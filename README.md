# Local OpenAI STT <small>_(Custom Integration for Home Assistant)_</small>

**Allows use of generic OpenAI-compatible speech-to-text services, such as (but not limited to):**

- faster-whisper-server / speaches
- whisper.cpp `server`
- LocalAI
- vLLM (Whisper)
- openedai-whisper

**Highlights:**

- Sets `requires_external_vad=False` per [home-assistant/core#167246](https://github.com/home-assistant/core/pull/167246), so the Assist pipeline does not run a second VAD on top of this integration's own.
- Internal end-of-speech detection using [Silero VAD](https://github.com/snakers4/silero-vad) via [pysilero-vad](https://github.com/rhasspy/pysilero-vad).
- Hysteresis-based segmentation so quiet mid-sentence dips do not cut the utterance off.
- Hardcoded 5-second fallback that still ships audio to Whisper if VAD never declares speech.
- Configurable end-of-speech sensitivity matching HA's `Relaxed` / `Default` / `Aggressive` levels.
- Configurable software microphone gain for setups where the satellite's audio is too quiet.
- Custom prompt and temperature per request.
- Per-session VAD diagnostic logs you can opt into when tuning.

---

## Installation

### Install via HACS (recommended)

Have [HACS](https://hacs.xyz/) installed; this will allow you to update easily.

[![Open in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=nickmowen&repository=hass_local_openai_stt&category=integration)

> [!NOTE]
> If the button above doesn't work, add `https://github.com/nickmowen/hass_local_openai_stt` as a custom repository of type Integration in HACS.

- Click install on the `Local OpenAI STT` integration.
- Restart Home Assistant.

<details><summary>Manual Install</summary>

- Copy the `local_openai_stt` folder from [latest release](https://github.com/nickmowen/hass_local_openai_stt/releases/latest) to the [`custom_components` folder](https://developers.home-assistant.io/docs/creating_integration_file_structure/#where-home-assistant-looks-for-integrations) in your config directory.
- Restart Home Assistant.

</details>

> [!NOTE]
> Requires a Home Assistant version that includes [#167246](https://github.com/home-assistant/core/pull/167246) (2026.5 or newer). The integration depends on `pysilero-vad`, which only ships wheels for manylinux x86_64/aarch64, macOS arm64, and Windows. Home Assistant OS (Alpine/musl) is not currently supported.

## Integration Configuration

After installation, configure the integration through Home Assistant's UI:

1. Go to `Settings` → `Devices & Services`.
2. Click `Add Integration`.
3. Search for `Local OpenAI STT`.
4. Enter the base URL of your OpenAI-compatible server and (optionally) an API key.
5. Pick the model, prompt, and temperature.

Once configured, the integration appears as an STT entity that you can select in any **Voice Assistant** pipeline (Settings → Voice assistants).

### Configuration Notes

- The Base URL must be the OpenAI-compatible API root, typically ending in `/v1`.
- The API key is optional; many local servers ignore it. If your server requires one, supply it here.
- The model selector is populated from `GET /v1/models`. If your server doesn't implement that endpoint, the model field falls back to a free-text input — type the model name your server expects.
- The Prompt is sent as the Whisper `prompt` field on every request. Useful for biasing toward unusual vocabulary, names, or acronyms.
- Language is derived from the active pipeline's `metadata.language` (sent to Whisper as ISO-639-1, e.g. `en` from `en-US`).

### VAD Tuning

The integration owns end-of-speech detection. Four knobs in the options flow:

- **End-of-speech sensitivity** — `Relaxed` / `Default` / `Aggressive`, matching HA's own values (1.25 s / 0.7 s / 0.25 s of trailing silence). Mirrors `homeassistant.components.assist_pipeline.vad.VadSensitivity`.
- **Speech detection threshold** — Silero probability above which a frame is treated as speech. The "silence" threshold is derived as `max(0.1, threshold * 0.4)` so probabilities between the two are treated as "uncertain" and don't cut the sentence off mid-utterance.
- **Minimum speech duration** — how much accumulated speech must be observed before end-of-speech can fire. Guards against ending the recording on a single click, breath, or one-word false start. Default 0.3 s.
- **Microphone gain** — software amplification applied to incoming audio before VAD _and_ before the Whisper request. Increase if quiet voices are missed; decrease if loud speech sounds distorted.

If VAD never confidently detects speech, a hardcoded 5-second timeout still ships the buffered audio to Whisper. Long utterances are unbounded as long as voice activity continues.

### Diagnostic Logs

Enable `Write per-session VAD logs` in the options flow to dump one log file per STT request to `<config>/local_openai_stt_sessions/<ISO-timestamp>.log`. Each file records every VAD chunk's probability, classification, and accumulated speech/silence — useful when tuning thresholds for an unusual room or microphone. Older sessions are pruned automatically once you exceed the keep count.

## Acknowledgements

- [Home Assistant](https://www.home-assistant.io/) and the assist pipeline team for [#167246](https://github.com/home-assistant/core/pull/167246), which made this kind of STT-side VAD ownership possible.
- [Silero](https://github.com/snakers4/silero-vad) for the VAD model, and [Rhasspy](https://github.com/rhasspy/pysilero-vad) for the Python packaging.
