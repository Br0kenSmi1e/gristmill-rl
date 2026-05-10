from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampledAction:
    decision: dict[str, Any]
    prior: float
