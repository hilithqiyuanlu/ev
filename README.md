# EV

> Your personal, always-on voice assistant that lives locally on your Mac.

EV is a local-first voice assistant that listens, understands who's talking, and responds —
all without leaving your machine. No accounts, no telemetry, no cloud: audio, models, and
history stay on your device.

## Features

- macOS menu bar app (SwiftUI)
- Real-time streaming transcription with endpoint detection
- Automatic speaker recognition — it learns your voice as you use it
- Voice onboarding: guided core-sample enrollment before automatic learning kicks in
- Wake word "小E" (ASR homophone tolerance, 嗨/hi/hai greeting handling)
- Local recording history with playback, instant filtering, and one-click correction
- Personal hotword lexicon (manual + auto-learned) injected into ASR
- Optional Qwen3-ASR (0.6B/1.7B) for mixed Chinese-English transcription, with anchor-based hotword boosting
- Far-field pickup: AGC/pre-emphasis/noise-gate preprocessing, composite FSMN+energy VAD
- Real-time speaker-turn tracking ("second-ear" mode) with dual WAV archive (processed + raw)

What's not here yet (but coming): LLM reasoning, text-to-speech, multi-speaker diarization,
environment awareness, standalone distribution.

## Quick Start

EV requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:hilithqiyuanlu/ev.git
cd ev
uv sync
uv pip install funasr torch torchaudio
```

Build and run the macOS app:

```bash
xcodebuild -project apps/macos/EV.xcodeproj -scheme EV \
  -configuration Debug -derivedDataPath /tmp/ev-derived CODE_SIGNING_ALLOWED=NO build
open /tmp/ev-derived/Build/Products/Debug/EV.app
```

Open the app, go to **Models** to download the required models on first launch,
select your microphone on **Live**, and start talking. Say "小E" followed by a command
to interact.

## Documentation

- [audio-flows.md](docs/audio-flows.md) — Audio processing pipeline

## Repository Layout

```
src/ev/       Python engine, audio pipeline, ML inference, storage
apps/macos/   Native macOS SwiftUI client
tests/        Automated tests
docs/         Design notes and research
```

## Privacy

EV never sends your audio, voiceprints, or conversation history to any server. All processing
happens locally using on-device models. The `data/` directory contains all your recordings,
database, and model files — you can delete everything at any time from the app.
