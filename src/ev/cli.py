"""命令行入口。子命令随任务推进增加:T2' audio,T6 enroll,T9 transcribe。"""

from __future__ import annotations

import argparse

from . import __version__
from .config import Settings, load_settings
from .logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev", description="EV 个人语音助手(Phase 1a)")
    parser.add_argument("--version", action="version", version=f"ev {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="显示当前配置与路径")

    audio = sub.add_parser("audio", help="音频采集(T2')")
    audio_sub = audio.add_subparsers(dest="audio_command")
    audio_sub.add_parser("devices", help="列出输入设备")
    test = audio_sub.add_parser("test", help="采集 N 秒,显示电平并保存 wav 供回听")
    test.add_argument("--device", default=None, help="设备名字子串(默认:系统默认输入)")
    test.add_argument("--seconds", type=float, default=5.0, help="采集时长(秒)")

    return parser


def _cmd_info(settings: Settings) -> int:
    settings.ensure_dirs()
    print(f"ev {__version__}")
    print(f"data_dir    : {settings.data_dir}")
    print(f"models_dir  : {settings.models_dir}")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_level, settings.logs_dir / "ev.log")

    if args.command == "info":
        return _cmd_info(settings)
    if args.command == "audio":
        if args.audio_command == "devices":
            return _cmd_audio_devices()
        if args.audio_command == "test":
            return _cmd_audio_test(settings, args.device, args.seconds)
        print("用法: ev audio {devices,test}")
        return 1

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
