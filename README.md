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

## The Promise

Imagine an assistant that doesn't beep or flash waiting for a command. It's just there —
quiet, attentive, in the background of your digital life:

- **Proactive, not reactive.** It notices patterns before you ask. If you always open your
  calendar and Spotify when you sit down Monday morning, it has them ready. If a meeting
  is running long and your next one starts in five minutes, it gently chimes in without
  needing to be summoned.
- **Knows when to stay silent.** When you're on a call, when it hears frustration or
  concentration in your voice, when you're talking to someone else in the room — it listens
  but never interrupts. It learns your rhythm.
- **Reads the room.** Not just words — tone, pause, hesitation, laughter. It can tell
  when you're stressed from the way you're speaking, or tired, or excited, and responds
  accordingly. A flat robotic answer isn't always what you need.
- **Abductive reasoning.** It connects dots you didn't explicitly state. When you say
  "I'm cold" while you're in your home office, it knows to check the thermostat — not
  pull up a dictionary definition. It remembers context across hours and days.
- **Always on your side.** Because everything is local, it never phones home, never
  sells your data, never optimizes for someone else's metrics. It works for you and
  only you.

That's the north star. Today's version is the ears — it can hear you, recognize you,
remember what was said. The brain and the rest come next.

## Current Status

This is an early development version. The speech input pipeline is working end-to-end today:

- macOS menu bar app with SwiftUI interface
- Real-time streaming transcription with endpoint detection
- Automatic speaker recognition — it learns your voice as you use it
- Wake word "小E" (with ASR homophone tolerance, and 嗨/hi/hai greeting handling)
- Local recording history with playback, instant filtering, and one-click correction
- Personal hotword lexicon — manual + auto-learned words injected into ASR
- Ctrl+T global shortcut to toggle listening
- Adjustable sensitivity threshold
- Optional Qwen3-ASR (0.6B/1.7B) for mixed Chinese-English transcription, with anchor-based hotword boosting
- Voice onboarding: guided core-sample enrollment before automatic learning kicks in
- Far-field pickup: AGC/pre-emphasis/noise-gate preprocessing, composite FSMN+energy VAD
- Real-time speaker-turn tracking ("second-ear" mode) with dual WAV archive (processed + raw)

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

## Documentation

Design notes and implementation details live in [`docs/`](docs/):

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

## Status & Roadmap

EV is in active early development. The speech input foundation is solid, but the assistant
itself doesn't do much yet. Right now it reliably captures, transcribes, identifies speakers,
and archives — the "ears" work. Next steps are building the "brain" (LLM reasoning),
"mouth" (natural TTS responses), and context awareness.

Contributions, feedback, and testing welcome.
