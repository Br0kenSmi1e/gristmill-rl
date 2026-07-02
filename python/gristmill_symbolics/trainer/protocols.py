from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class Trainer(Protocol):
    @property
    def batch_size(self) -> int:
        ...

    def init_opt_state(self, params) -> object:
        ...

    def update(
        self,
        params,
        opt_state,
        batch: Sequence[object],
        model: object,
        rng,
    ) -> tuple[object, object, Mapping[str, object]]:
        ...
