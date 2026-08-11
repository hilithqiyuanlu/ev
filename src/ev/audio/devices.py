"""输入设备枚举与解析。内置麦与 DJI 接收器仅差一个名字子串。"""

from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    is_default: bool


def list_input_devices() -> list[InputDevice]:
    # 每次实时查询系统默认输入设备（而非缓存 sd.default），
    # 保证热插拔新默认设备后 is_default 立即正确。
    try:
        default_input = sd.query_devices(kind="input")["index"]
    except Exception:
        default_input = None
    devices: list[InputDevice] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        devices.append(
            InputDevice(
                index=i,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                default_samplerate=d["default_samplerate"],
                is_default=(i == default_input),
            )
        )
    return devices


def resolve_device(selector: str | None) -> int | None:
    """按名字子串(大小写不敏感)解析输入设备;None → 系统默认。"""
    if not selector:
        return None
    sel = selector.lower()
    for dev in list_input_devices():
        if sel in dev.name.lower():
            return dev.index
    raise ValueError(f"找不到输入设备 {selector!r},用 `ev audio devices` 查看可用设备")
