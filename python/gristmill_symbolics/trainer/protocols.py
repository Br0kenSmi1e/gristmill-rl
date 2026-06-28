from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from gristmill_symbolics.model.protocols import ExpressionModel


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
        model: ExpressionModel,
        rng,
    ) -> tuple[object, object, Mapping[str, object]]:
        ...
