"""EV 命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import Settings, load_settings
from .logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev", description="EV 个人语音助手(Phase 1a)")
    parser.add_argument("--version", action="version", version=f"ev {__version__}")
    parser.add_argument("--log-path", default=None, help="日志文件路径(默认 data/logs/ev.log)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="显示当前配置与路径")

    audio = sub.add_parser("audio", help="音频采集(T2')")
    audio_sub = audio.add_subparsers(dest="audio_command")
    audio_sub.add_parser("devices", help="列出输入设备")
    test = audio_sub.add_parser("test", help="采集 N 秒,显示电平并保存 wav 供回听")
    test.add_argument("--device", default=None, help="设备名字子串(默认:系统默认输入)")
    test.add_argument("--seconds", type=float, default=5.0, help="采集时长(秒)")

    models = sub.add_parser("models", help="本地模型")
    models_sub = models.add_subparsers(dest="models_command")
    verify = models_sub.add_parser("verify", help="校验本地模型 Release 解压结果")
    verify.add_argument("--model-root", default=None)
    download = models_sub.add_parser("download", help="下载并校验固定版本模型 Release")
    download.add_argument("--model-root", default=None)

    voice = sub.add_parser("voice", help="用户声纹")
    voice_sub = voice.add_subparsers(dest="voice_command")
    enroll = voice_sub.add_parser("enroll", help="录入用户声纹 profile")
    enroll.add_argument("--device", default=None)
    enroll.add_argument("--segments", type=int, default=8)
    enroll.add_argument("--model-root", default=None)

    transcribe = sub.add_parser("transcribe", help="持续 VAD/ASR/声纹/归档")
    transcribe.add_argument("--device", default=None)
    transcribe.add_argument("--model-root", default=None)

    engine = sub.add_parser("engine", help="GUI JSONL engine")
    engine_sub = engine.add_subparsers(dest="engine_command")
    engine_sub.add_parser("serve", help="通过 stdin/stdout 提供版本化 JSONL 协议")

    return parser


def _cmd_info(settings: Settings) -> int:
    settings.ensure_dirs()
    print(f"ev {__version__}")
    print(f"data_dir    : {settings.data_dir}")
    print(f"models_dir  : {settings.models_dir}")
    print(f"model_root  : {settings.models.root}")
    print(f"archive_dir : {settings.archive_dir}")
    print(f"db_path     : {settings.db_path}")
    print(f"audio       : {settings.audio.sample_rate} Hz x {settings.audio.channels} ch")
    return 0


def _cmd_audio_devices() -> int:
    from .audio.devices import list_input_devices

    devices = list_input_devices()
    if not devices:
        print("未发现输入设备")
        return 1
    for d in devices:
        mark = "*" if d.is_default else " "
        print(
            f"{mark} [{d.index}] {d.name}  "
            f"({d.max_input_channels}ch, {d.default_samplerate:.0f} Hz)"
        )
    print("(* = 系统默认输入)")
    return 0


def _cmd_audio_test(settings: Settings, device: str | None, seconds: float) -> int:
    from .audio.diag import run_capture_test

    settings.ensure_dirs()
    run_capture_test(settings, device, seconds)
    return 0


def _settings_with_model_root(settings: Settings, model_root: str | None) -> Settings:
    if not model_root:
        return settings
    return replace(settings, models=replace(settings.models, root=Path(model_root).expanduser().resolve()))


def _cmd_models_verify(settings: Settings, model_root: str | None) -> int:
    from .models import verify_models

    settings = _settings_with_model_root(settings, model_root)
    checks = verify_models(settings.models)
    for check in checks:
        if check.ok:
            print(f"OK   {check.key}: {check.path}")
        else:
            print(f"FAIL {check.key}: {check.path} ({'; '.join(check.errors)})")
    return 0 if all(check.ok for check in checks) else 1


def _cmd_models_download(settings: Settings, model_root: str | None) -> int:
    from .model_download import ModelDownloader

    settings = _settings_with_model_root(settings, model_root)

    last_percent = -1

    def report(kind: str, payload: dict) -> None:
        nonlocal last_percent
        if kind == "model_status":
            key = payload.get("key", "all")
            print(f"{key}: {payload.get('status')}")
        elif kind == "download_progress":
            size = max(int(payload.get("total_size", 1)), 1)
            current = int(payload.get("total_downloaded", 0))
            percent = int(current * 100 / size)
            if percent != last_percent:
                last_percent = percent
                print(f"\r下载进度 {percent}%", end="", flush=True)

    ModelDownloader(settings.models, report).download_all()
    print()
    return 0


def _cmd_voice_enroll(settings: Settings, device: str | None, segments: int, model_root: str | None) -> int:
    if segments < 1:
        print("--segments 必须大于 0")
        return 2
    from .audio.capture import AudioCapture
    from .audio.devices import resolve_device
    from .models import require_models
    from .speaker.verification import build_profile
    from .speaker.verification import normalize_embedding
    from .asr.adapters import SpeakerEmbeddingAdapter
    from .store.db import Store
    import numpy as np

    settings = _settings_with_model_root(settings, model_root)
    settings.ensure_dirs()
    paths = require_models(settings.models)
    adapter = SpeakerEmbeddingAdapter(str(paths["speaker"]))
    embeddings = []
    print(f"将录制 {segments} 段，每段约 4 秒；请按提示自然说话。Ctrl-C 可停止。")
    for index in range(segments):
        input(f"按回车开始第 {index + 1}/{segments} 段，然后说话 4 秒...")
        capture = AudioCapture(settings.audio, device=resolve_device(device))
        try:
            capture.start()
            async def collect() -> np.ndarray:
                chunks: list[np.ndarray] = []
                async for frame in capture.frames():
                    chunks.append(frame)
                    if sum(len(x) for x in chunks) >= settings.audio.sample_rate * 4:
                        break
                return np.concatenate(chunks)[: settings.audio.sample_rate * 4]
            audio = asyncio.run(collect())
        finally:
            capture.stop()
        embeddings.append(normalize_embedding(adapter.embed(audio, settings.audio.sample_rate)))
        print("已提取 embedding")
    profile = build_profile(embeddings)
    with Store(settings.db_path) as store:
        store.save_profile(
            "user-v1", "user", device, settings.models.speaker, profile, len(embeddings)
        )
    print(f"用户声纹已保存: {settings.db_path} (samples={len(embeddings)}, dim={profile.size})")
    return 0


def _cmd_transcribe(settings: Settings, device: str | None, model_root: str | None) -> int:
    from .audio.devices import resolve_device
    from .pipeline.runtime import transcribe_forever

    settings = _settings_with_model_root(settings, model_root)
    try:
        asyncio.run(transcribe_forever(settings, resolve_device(device)))
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    if args.command == "engine" and args.engine_command == "serve" and not args.log_path:
        log_path = None
    else:
        log_path = Path(args.log_path).expanduser().resolve() if args.log_path else settings.logs_dir / "ev.log"
    setup_logging(settings.log_level, log_path)

    if args.command == "info":
        return _cmd_info(settings)
    if args.command == "audio":
        if args.audio_command == "devices":
            return _cmd_audio_devices()
        if args.audio_command == "test":
            return _cmd_audio_test(settings, args.device, args.seconds)
        print("用法: ev audio {devices,test}")
        return 1
    if args.command == "models":
        if args.models_command == "verify":
            return _cmd_models_verify(settings, args.model_root)
        if args.models_command == "download":
            return _cmd_models_download(settings, args.model_root)
        print("用法: ev models {verify,download}")
        return 1
    if args.command == "voice":
        if args.voice_command == "enroll":
            return _cmd_voice_enroll(settings, args.device, args.segments, args.model_root)
        print("用法: ev voice enroll")
        return 1
    if args.command == "transcribe":
        return _cmd_transcribe(settings, args.device, args.model_root)
    if args.command == "engine":
        if args.engine_command == "serve":
            from .engine.service import EngineService

            return EngineService(settings).serve()
        print("用法: ev engine serve")
        return 1

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
