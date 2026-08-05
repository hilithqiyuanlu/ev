import asyncio

import numpy as np

from ev.audio.capture import AudioCapture
from ev.config import AudioSettings


def test_frame_chunking():
    """不同长度的输入块应被归一为定长帧。"""
    cap = AudioCapture(AudioSettings(sample_rate=16000), frame_ms=30)
    assert cap.frame_samples == 480
    cap._queue.put(np.full(480, 0.5, dtype=np.float32))
    cap._queue.put(np.full(500, 0.5, dtype=np.float32))  # 跨块拼出第二帧

    async def collect():
        out = []
        async for f in cap.frames():
            out.append(f)
            if len(out) == 2:
                break
        return out

    frames = asyncio.run(collect())
    assert all(len(f) == 480 for f in frames)
    assert np.allclose(frames[1][0], 0.5)
