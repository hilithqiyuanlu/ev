# EV

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange)](#project-status)

[Quick start](#quick-start) | [Commands](#commands) | [Configuration](#configuration) | [Project status](#project-status)

EV is an experimental, local-first personal voice assistant. Phase 1a focuses on a
reliable, measurable speech input loop: VAD, streaming ASR partials, high-quality
final transcription, user voice verification, EV wake-prefix matching, WAV archive,
and SQLite metadata.

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

## Quick start

EV requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone git@github.com:hilithqiyuanlu/mylyra.git
cd mylyra
uv sync
uv run pytest
```

Install FunASR separately when the local model release is available (it is kept
out of the base lockfile so audio-only development stays lightweight):

```bash
uv pip install funasr
```

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

Verify a manually downloaded model release before starting the pipeline:

```bash
uv run python -m ev models verify --model-root data/models
```

Run enrollment and continuous transcription:

```bash
uv run python -m ev voice enroll --device "MacBook" --segments 8
uv run python -m ev transcribe --device "MacBook" --model-root data/models
```

Models are downloaded manually from the
[models-v0.1.0 release](https://github.com/hilithqiyuanlu/mylyra/releases/tag/models-v0.1.0).
Each archive must be extracted into its matching directory under `data/models`
(the archives do not contain a top-level directory). The application never
downloads models automatically. See
[`docs/phase1a-plan.md`](docs/phase1a-plan.md) for the required archive names and
SHA256 check.

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
- `EV_LOG_LEVEL` changes the logging level.

## Repository layout

```text
src/ev/       Application package and CLI
tests/        Automated tests
docs/         Internal design and research notes
ev.toml       Default configuration
```

## Project status

EV is in early development. The audio capture foundation is available, while the
end-to-end voice assistant is not yet complete. Interfaces and data formats may
change as the project evolves.
