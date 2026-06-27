from __future__ import annotations

from gristmill_symbolics import RewriteStateRow

from .rollout import _sample_static_model_rollout
from .types import CurrentTransformerModelConfig, validate_model_config


class CurrentTransformerModel:
    def sample_with_logp_grad(
        self,
        params,
        rng,
        row: RewriteStateRow,
        config: CurrentTransformerModelConfig,
    ):
        validate_model_config(config)
        result = _sample_static_model_rollout(params, rng, row, config)
        return result.out_row, result.logp, result.grad_logp, {
            "stopped": result.stopped,
        }
