from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class ExpressionModel(Protocol):
    def sample_with_logp_grad(
        self,
        params,
        rng,
        row,
        config,
    ) -> tuple[object, object, object, Mapping[str, object]]:
        ...


class Trainer(Protocol):
    def update(
        self,
        params,
        opt_state,
        batch,
        model: ExpressionModel,
        rng,
        config,
    ) -> tuple[object, object, Mapping[str, object]]:
        ...
