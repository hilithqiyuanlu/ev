# EV

> Your personal, always-on voice assistant that lives locally on your Mac.

EV is an experimental local-first voice assistant that listens, understands who's talking,
and responds — all without leaving your machine. No cloud required, no data leaves your device.

## Vision

Voice interaction should feel natural. You should be able to talk to your computer the way
you talk to a person — without wake words that sound like code names, without thinking about
who's listening, without worrying about recordings being sent somewhere you can't see.

EV is being built around a few core ideas:

- **Local-first.** Every part of the pipeline runs on your hardware. Audio, models, and history
  stay on your machine.
- **It knows your voice.** Speaker verification happens automatically as you use it — no
  awkward enrollment flows, no repeating phrases 8 times.
- **Just works.** No complicated setup. Select your mic, start talking. It gets better as you use it.
- **Private by design.** No accounts, no telemetry, no cloud API calls. The only person who hears
  your voice is you.

## Current Status

This is an early development version. The speech input pipeline is working end-to-end today:

- macOS menu bar app with SwiftUI interface
- Real-time streaming transcription with endpoint detection
- Automatic speaker recognition — it learns your voice as you use it
- Wake word "小E" (with ASR homophone tolerance for natural speech)
- Local recording history with playback and instant filtering
- Ctrl+T global shortcut to toggle listening
- Adjustable sensitivity threshold

What's not here yet (but coming):
- LLM integration for actually answering queries
- Text-to-speech responses
- Multi-speaker diarization for distinguishing different people
- Environment awareness and contextual memory
- Standalone app distribution without a Python development environment

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

Detailed setup and troubleshooting: [Chinese User Guide](docs/user-guide.md)

## Documentation

Design notes and implementation details live in [`docs/`](docs/):

- [user-guide.md](docs/user-guide.md) — Complete operation guide in Chinese
- [phase1a-plan.md](docs/phase1a-plan.md) — Core pipeline architecture
- [phase1b-gui.md](docs/phase1b-gui.md) — Engine-client protocol
- [audio-flows.md](docs/audio-flows.md) — Audio processing pipeline
- [research.md](docs/research.md) — Research notes and references

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

## Status & Roadmap

EV is in active early development. The speech input foundation is solid, but the assistant
itself doesn't do much yet. Right now it reliably captures, transcribes, identifies speakers,
and archives — the "ears" work. Next steps are building the "brain" (LLM reasoning),
"mouth" (natural TTS responses), and context awareness.

Contributions, feedback, and testing welcome.
