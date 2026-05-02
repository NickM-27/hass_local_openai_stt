# Local OpenAI STT

A Home Assistant custom integration that turns any local OpenAI-compatible
speech-to-text server (faster-whisper-server, whisper.cpp `server`, LocalAI,
vLLM-whisper, openedai-whisper, etc.) into a Home Assistant voice STT
provider.

Unlike the built-in cloud OpenAI integration, this one:

- Talks to a server you run yourself, on your LAN.
- Performs **its own end-of-speech detection** using
  [Silero VAD](https://github.com/snakers4/silero-vad), so the Home Assistant
  Assist pipeline does not run a second VAD on top of yours. This is enabled by
  the `SpeechAudioProcessing.requires_external_vad` flag added in
  [home-assistant/core#167246](https://github.com/home-assistant/core/pull/167246).

## Requirements

- Home Assistant **2026.5** or newer (must include core PR #167246).
- A reachable OpenAI-compatible STT endpoint, e.g. one of:
  - [faster-whisper-server](https://github.com/fedirz/faster-whisper-server)
    / [speaches](https://github.com/speaches-ai/speaches)
  - [whisper.cpp `server`](https://github.com/ggerganov/whisper.cpp/tree/master/examples/server)
  - [LocalAI](https://localai.io/features/audio-to-text/)
  - [vLLM with Whisper](https://docs.vllm.ai/)
  - [openedai-whisper](https://github.com/matatonic/openedai-whisper)

### Platform support

The integration depends on `pysilero-vad==3.0.1`, which ships native wheels
for:

| Platform | Wheel available |
|---|---|
| Linux x86_64 (manylinux) | yes |
| Linux aarch64 (manylinux) | yes |
| macOS arm64 | yes |
| Windows x86_64 | yes |
| Linux musllinux (Home Assistant OS) | **no** |
| Linux armv7 | **no** |

Most users running Home Assistant Container or Supervised on Debian-based
hosts (including Raspberry Pi OS 64-bit) are fine. Home Assistant OS uses
Alpine/musl and currently has no wheel; on those installs the requirement
will fail to install until upstream ships a musllinux wheel.

## Installation

### HACS (recommended)

1. In HACS, go to **Integrations** -> the three-dot menu -> **Custom
   repositories**.
2. Add `https://github.com/nickmowen/hass_local_openai_stt` as an
   **Integration**.
3. Install **Local OpenAI STT** from HACS.
4. Restart Home Assistant.

### Manual

Copy `custom_components/local_openai_stt/` into your Home Assistant
`config/custom_components/` directory and restart.

## Configuration

After install, go to **Settings -> Devices & services -> Add integration ->
Local OpenAI STT**.

### Step 1: connect

| Field | Description |
|---|---|
| Base URL | Full URL of the server's OpenAI API root, e.g. `http://localhost:8000/v1` |
| API key | Optional. Many local servers ignore this; leave blank if yours does. |

The integration calls `GET /v1/models` to verify the connection and to
populate the model picker on the next step. If the server does not implement
`/v1/models`, you can still continue and type the model name as free text.

### Step 2: STT settings

| Field | Description |
|---|---|
| Model | Whisper model to transcribe with (e.g. `Systran/faster-whisper-large-v3`, `whisper-1`). Picked from the server's `/v1/models` list, or typed directly. |
| Prompt | Optional bias text passed with each request. Useful for unusual vocabulary, names, or acronyms. |
| Temperature | Sampling temperature (0 – 1). 0 is deterministic. |

After saving, the integration appears as an STT entity that you can select
in any **Voice Assistant** pipeline (Settings -> Voice assistants).

### Options (post-setup)

Open the integration and click **Configure** to retune any of the above and
to adjust the internal VAD:

| Field | Default | Description |
|---|---|---|
| End-of-speech silence | 0.8 s | Trailing silence required to end the utterance. |
| Minimum speech duration | 0.3 s | Speech detected before end-of-speech can fire. Avoids ending on a click or short noise. |
| Speech detection threshold | 0.5 | Silero VAD probability above which a frame is considered speech (0 - 1). Lower if it stops too early on quiet voices; raise if room noise keeps it open. |

## How the audio path works

```
Voice satellite (mic)
  -> Home Assistant Assist pipeline (skips its VoiceCommandSegmenter)
      -> Local OpenAI STT entity
          -> Silero VAD: stop when trailing silence > threshold
          -> POST /v1/audio/transcriptions
              -> Whisper text back into the pipeline
```

The pipeline streams 16 kHz mono int16 PCM. The integration runs Silero VAD
on 32 ms (512-sample) chunks, accumulates the audio, and once it has seen at
least `vad_min_speech_seconds` of speech followed by `vad_silence_seconds`
of silence, trims trailing silence to ~200 ms, wraps the PCM in a WAV
container, and POSTs to the server's `audio.transcriptions` endpoint with
the configured `model`, `prompt`, `temperature`, and language hint derived
from the pipeline metadata.

## Troubleshooting

### Stream never ends

Enable debug logging:

```yaml
logger:
  logs:
    custom_components.local_openai_stt: debug
```

When end-of-speech fires you will see a line like:

```
End of speech: 1.43s speech, 0.83s trailing silence, 53760 bytes
```

If you never see that line, VAD is not declaring silence. Common causes:

- Mic auto-gain is amplifying room noise above the speech threshold. Raise
  the threshold, or disable the satellite's auto-gain.
- The user spoke for less than `vad_min_speech_seconds`. Lower it, or speak
  longer.

### `RequirementsNotFound: ['pysilero-vad==3.0.1']`

You are likely on Home Assistant OS or armv7 — no wheel exists for those
platforms. See [Platform support](#platform-support).

### Models dropdown is empty

The server probably does not implement `/v1/models`. The model field falls
back to a free-text input — type the model name your server expects.

### Wrong or empty transcription

- Confirm the language hint matches what your server expects. The
  integration sends ISO-639-1 (`en`, not `en-US`) derived from
  `metadata.language`.
- Try `temperature: 0`.
- Verify the server transcribes the same WAV correctly via `curl`:

  ```sh
  curl http://localhost:8000/v1/audio/transcriptions \
    -F file=@sample.wav \
    -F model=Systran/faster-whisper-large-v3
  ```

## License

MIT.
