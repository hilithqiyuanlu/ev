# EV

> Your personal, always-on voice assistant that lives locally on your Mac.

EV is a local-first voice assistant that listens, understands who's talking, and responds —
all without leaving your machine. No accounts, no telemetry, no cloud: audio, models, and
history stay on your device.

## Features

- Native macOS menu bar app (SwiftUI) with a local Python engine
- Real-time streaming transcription, endpoint detection, and "小E" wake word
- Speaker recognition that learns your voice — guided onboarding, then automatic learning
- Far-field pickup: AGC/noise-gate preprocessing, composite VAD, DFSMN-ANS denoising
- Human-voice confirmation rejects non-speech noise (fan / typing / music)
- Environment awareness: YAMNet sound classification, logged separately from voice data
- Local history with playback, filtering, correction, and a personal hotword lexicon

Not here yet: LLM reasoning, text-to-speech, multi-speaker diarization, standalone distribution.

## Quick Start

EV requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:hilithqiyuanlu/ev.git
cd ev
uv sync
uv pip install funasr torch torchaudio "modelscope[framework]" speechbrain
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

- [audio-flow.md](docs/audio-flow.md) — Audio processing pipeline
- [voiceprint.md](docs/voiceprint.md) — Voiceprint flow and speaker verification
- [voice-learning.md](docs/voice-learning.md) — Voice sample tiering, cluster competition, pending confirmation

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
