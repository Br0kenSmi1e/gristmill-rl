from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class ExpressionModel(Protocol):
    @property
    def batch_size(self) -> int:
        ...

    def init_params(self, rng) -> object:
        ...

    def sample_with_logp_grad(
        self,
        params,
        rng,
        row,
    ) -> tuple[object, object, object, Mapping[str, object]]:
        ...
