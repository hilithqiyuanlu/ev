# EV

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange)](#project-status)

[Quick start](#quick-start) | [Using the app](#using-the-app) | [Commands](#commands) | [Project status](#project-status)

EV is an experimental, local-first personal voice assistant. The current development
version combines a measurable speech input pipeline with a native macOS client. Python
owns audio capture, FunASR inference, speaker verification, model management, WAV
archives, and SQLite; SwiftUI provides the menu bar and desktop interface.

Audio and runtime data stay on the local machine. Cloud services are not required.

## Current capabilities

- Native macOS menu bar app and SwiftUI control window.
- Streaming microphone transcription with FSMN-VAD and Paraformer.
- High-quality final transcription with Paraformer Large (SeACo Paraformer v2).
- RMS loudness normalization to reduce volume/distance impact on recognition.
- ERes2NetV2 speaker embedding with K-means multi-centroid templates (1-3 centroids auto-selected).
- Binary speaker verification (user / non-user) with adjustable threshold (default 0.50).
- Automatic continuous voiceprint learning during normal usage: no explicit enrollment needed.
- Hierarchical sample library: 20 core samples (modeling) + 50 cache samples (FIFO recording).
- Local WAV archive and SQLite history for every detected speech segment.
- Wake word "小E" detection (with homophone tolerance for ASR misrecognition); only verified user
  speech becomes a voice query candidate.
- Ctrl+T global shortcut to toggle listening on/off.
- Model verification and fixed-release downloads with SHA256 and atomic install.
- Manual query queue for the future LLM integration.

## Quick start

EV requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone git@github.com:hilithqiyuanlu/mylyra.git
cd ev
uv sync
uv run pytest
```

Install the local inference runtime (kept out of the base lockfile so lightweight
development does not install PyTorch):

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

## Using the app

1. Open **Models** and verify that all four models are ready. Download them from
   the app if they are missing.
2. Allow microphone access and select the input device on **Live**.
3. Voice profile builds automatically as you use the app (starts at "未建立" with 0 core samples;
   transitions to "学习中" at 1-2 samples; "已就绪" at 3+ samples when speaker gating activates).
   You can also manually add samples from Voice Profile settings.
4. Return to **Live** and start listening (or press Ctrl+T to toggle). Partial text appears while
   speaking; the final text and speaker score appear after the endpoint.
5. Say "小E" (or "嗨小易"/"喂小艺" - common ASR variants are handled) followed by your query.
   Only your own voice triggers commands.
6. Use **History** to filter ("我"/"他人"/"全部"), play, reveal in Finder, or delete recordings
   with one click (no confirmation dialog).
7. Adjust the voiceprint threshold in **Settings** if you encounter false accepts/rejects.

Closing the main window leaves the menu bar engine running. Stop listening from
the menu bar or press Ctrl+T before privacy-sensitive situations. See the
[Chinese user guide](docs/user-guide.md) for complete operation and troubleshooting.

See [`docs/phase1b-gui.md`](docs/phase1b-gui.md) for the engine protocol and
development-client boundaries.

Models are versioned in the
[models-v0.1.0 release](https://github.com/hilithqiyuanlu/mylyra/releases/tag/models-v0.1.0).
The CLI and macOS client download each fixed asset to a temporary file, verify
SHA256, validate the extracted structure, and atomically install it under the
configured model root. A failed or cancelled download does not replace an
existing valid model. See
[`docs/phase1a-plan.md`](docs/phase1a-plan.md) for the required archive names and
SHA256 values.

The engine exposes history, query, threshold, and profile operations for the macOS client. The
following commands are available through the JSONL protocol:

```text
list_segments, delete_segment, delete_all_segments
submit_manual_query, delete_query, delete_all_queries
set_thresholds, get_profile_status, list_speaker_samples
delete_speaker_sample, promote_speaker_sample
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

[speaker]
threshold = 0.50
max_core_samples = 20
max_cache_samples = 50
max_centroids = 3
loudness_normalize = true
```

The following environment variables are also supported:

- `EV_DATA_DIR` changes the runtime data directory.
- `EV_MODEL_ROOT` changes the model installation directory.
- `EV_LOG_LEVEL` changes the logging level.

The macOS development client defaults to `~/Library/Application Support/EV/` for
models, archives, SQLite, and logs. CLI commands use the paths from `ev.toml` unless
an environment variable or command option overrides them.

## Keyboard shortcuts

- **Ctrl+T**: Toggle listening on/off (can be disabled in Settings)

## Repository layout

```text
src/ev/       Application package and CLI
apps/macos/   Native SwiftUI development client
tests/        Automated tests
docs/         Internal design and research notes
ev.toml       Default configuration
```

## Project status

Phase 1a core pipeline and Phase 1b macOS GUI client are implemented with automated tests.
Current runtime features: streaming ASR with VAD, final large-model transcription,
binary speaker gating with multi-centroid templates, automatic continuous voiceprint learning,
hierarchical sample management, wake word detection with homophone tolerance, loudness normalization,
history timeline with instant filtering, one-click deletion, and global Ctrl+T shortcut.

Remaining in progress: latency/accuracy benchmarks, standalone `.app` packaging without repository
`.venv`, and hotword customization. LLM, NLP, TTS, environmental awareness, computer vision,
biosignal integration, cognitive/behavior layers, and non-user speaker diarization are planned
for future phases.

Design and implementation notes:

- [`docs/audio-flows.md`](docs/audio-flows.md)
- [`docs/phase1a-plan.md`](docs/phase1a-plan.md)
- [`docs/phase1b-gui.md`](docs/phase1b-gui.md)
- [`docs/user-guide.md`](docs/user-guide.md)
- [`docs/research.md`](docs/research.md)
