# EV

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange)](#project-status)

[Quick start](#quick-start) | [Commands](#commands) | [Configuration](#configuration) | [Project status](#project-status)

EV is an experimental, local-first personal voice assistant. The project currently
focuses on building a reliable audio input layer for real-time speech processing.

Audio and runtime data stay on the local machine. Cloud services are not required.

## Current capabilities

- Discover available audio input devices.
- Select an input device by name.
- Capture audio as asynchronous, fixed-size frames.
- Capture 16 kHz mono audio for downstream speech processing.
- Run a short microphone diagnostic with a live level meter.
- Save diagnostic recordings as WAV files for playback and inspection.
- Load default, local, and environment-based configuration.

## Quick start

EV requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone git@github.com:hilithqiyuanlu/mylyra.git
cd mylyra
uv sync
uv run pytest
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
