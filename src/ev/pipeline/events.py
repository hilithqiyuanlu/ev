"""跨模块事件对象，供未来 GUI/LLM 订阅。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    captured_at: datetime
    sample_rate: int = 16000


@dataclass(frozen=True)
class QueryCandidate:
    segment_id: str
    query_text: str
    created_at: datetime
