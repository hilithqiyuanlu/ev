# EV

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange)](#project-status)

[Quick start](#quick-start) | [Commands](#commands) | [Configuration](#configuration) | [Project status](#project-status)

EV is an experimental, local-first personal voice assistant. The current development
version combines a measurable speech input pipeline with a native macOS client. Python
owns audio capture, FunASR inference, speaker verification, model management, WAV
archives, and SQLite; SwiftUI provides the menu bar and desktop interface.

Audio and runtime data stay on the local machine. Cloud services are not required.

## Current capabilities

- Discover available audio input devices.
- Select an input device by name.
- Capture audio as asynchronous, fixed-size frames.
- Capture 16 kHz mono audio for downstream speech processing.
- Run a short microphone diagnostic with a live level meter.
- Save diagnostic recordings as WAV files for playback and inspection.
- Load default, local, and environment-based configuration.
- Archive every VAD speech segment and mark `EV + user` segments as query candidates.
- Run a native macOS SwiftUI client with menu bar controls, live transcription,
  history, voice enrollment, model management, and a manual query queue.

## Quick start

EV requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone git@github.com:hilithqiyuanlu/ev.git
cd ev
uv sync
uv run pytest
```

Install FunASR separately when the local model release is available (it is kept
out of the base lockfile so audio-only development stays lightweight):

```bash
uv pip install funasr torch torchaudio
```

The development macOS app launches the repository's `.venv/bin/python`, so run these
commands from the same checkout before opening the app.

## Commands

Show the active configuration and local data paths:

```bash
uv run python -m ev info
```

List audio input devices:

```bash
uv run python -m ev audio devices
```

Record a five-second microphone diagnostic:

```bash
uv run python -m ev audio test
```

Select a device by a case-insensitive name fragment or change the duration:

```bash
uv run python -m ev audio test --device "MacBook" --seconds 10
```

Diagnostic recordings are written to `data/audio-test/`. The entire `data/`
directory is excluded from Git.

Verify the local model release before starting the pipeline:

```bash
uv run python -m ev models verify --model-root data/models
```

Download the fixed `models-v0.1.0` release with checksum verification and atomic
installation. Existing valid models are kept and skipped:

```bash
uv run python -m ev models download --model-root data/models
```

Run enrollment and continuous transcription:

```bash
uv run python -m ev voice enroll --device "MacBook" --segments 8
uv run python -m ev transcribe --device "MacBook" --model-root data/models
```

Start the versioned JSONL engine used by the macOS app:

```bash
uv run python -m ev engine serve
```

Build the development macOS app:

```bash
xcodebuild -project apps/macos/EV.xcodeproj -scheme EV \
  -configuration Debug -derivedDataPath /tmp/ev-derived CODE_SIGNING_ALLOWED=NO build
open /tmp/ev-derived/Build/Products/Debug/EV.app
```

See [`docs/phase1b-gui.md`](docs/phase1b-gui.md) for the engine protocol and
development-client boundaries.

Models are versioned in the
[models-v0.1.0 release](https://github.com/hilithqiyuanlu/ev/releases/tag/models-v0.1.0).
The CLI and macOS client download each fixed asset to a temporary file, verify
SHA256, validate the extracted structure, and atomically install it under the
configured model root. A failed or cancelled download does not replace an
existing valid model. See
[`docs/phase1a-plan.md`](docs/phase1a-plan.md) for the required archive names and
SHA256 values.

The engine also exposes history and query operations for the macOS client. The
following commands are available through the JSONL protocol:

```text
list_segments, delete_segment, delete_all_segments
submit_manual_query, delete_query, delete_all_queries
```

## Configuration

Defaults live in [`ev.toml`](ev.toml). Create an ignored `ev.local.toml` file for
machine-specific overrides.

```toml
log_level = "INFO"

[audio]
sample_rate = 16000
channels = 1

[paths]
data_dir = "data"

[models]
root = "data/models"
vad = "ev-fsmn-vad-zh-16k"
asr_streaming = "ev-paraformer-zh-streaming-16k"
asr_final = "ev-sensevoice-small"
speaker = "ev-eres2netv2-zh-16k"
```

The following environment variables are also supported:

- `EV_DATA_DIR` changes the runtime data directory.
- `EV_MODEL_ROOT` changes the model installation directory.
- `EV_LOG_LEVEL` changes the logging level.

The macOS development client defaults to `~/Library/Application Support/EV/` for
models, archives, SQLite, and logs. CLI commands use the paths from `ev.toml` unless
an environment variable or command option overrides them.

## Repository layout

```text
src/ev/       Application package and CLI
apps/macos/   Native SwiftUI development client
tests/        Automated tests
docs/         Internal design and research notes
ev.toml       Default configuration
```

## Project status

Phase 1a and the repository-based Phase 1b client are implemented. Automated Python
and Swift tests pass, the four release models load locally, and the Debug macOS app
builds on Apple Silicon. Real microphone calibration, latency and accuracy benchmarks,
and packaging a standalone `.app` without the repository `.venv` remain in progress.
LLM, NLP, TTS, independent KWS, and multi-speaker modeling are not part of the current
runtime.

Design and implementation notes:

- [`docs/audio-flows.md`](docs/audio-flows.md)
- [`docs/phase1a-plan.md`](docs/phase1a-plan.md)
- [`docs/phase1b-gui.md`](docs/phase1b-gui.md)
- [`docs/research.md`](docs/research.md)
