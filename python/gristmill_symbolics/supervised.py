from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from .grammar import FlatDefinitionGrammar
from .scoring import constrained_sequence_log_prob

__all__ = (
    "SupervisedTrainer",
    "accumulate_weighted_nll_grad",
    "weighted_nll",
)


def accumulate_weighted_nll_grad(
    accumulated,
    weighted_nll_sum: jax.Array,
    weight_sum: jax.Array,
    grads,
):
    if accumulated is None:
        return weighted_nll_sum, weight_sum, grads

    accumulated_nll, accumulated_weight, accumulated_grads = accumulated
    return (
        accumulated_nll + weighted_nll_sum,
        accumulated_weight + weight_sum,
        jax.tree.map(
            lambda total, current: total + current,
            accumulated_grads,
            grads,
        ),
    )


class SupervisedTrainer:
    def __init__(
        self,
        grammar: FlatDefinitionGrammar,
        *,
        deterministic: bool = False,
        mean_epsilon: float = 1e-8,
    ):
        self.grammar = grammar
        self.deterministic = deterministic
        self.mean_epsilon = mean_epsilon

        def objective(model: nnx.Module, batch: dict[str, jax.Array]):
            return weighted_nll(
                model,
                batch,
                grammar,
                deterministic=deterministic,
            )

        self._loss_and_grad = nnx.jit(
            nnx.value_and_grad(
                objective,
                argnums=nnx.DiffState(0, nnx.Param),
                has_aux=True,
            )
        )

    def update(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batches: Sequence[dict[str, jax.Array]],
    ) -> dict[str, jax.Array | int]:
        if not batches:
            raise ValueError("batches must not be empty")

        accumulated = None
        for batch in batches:
            (weighted_nll_sum, weight_sum), grads = self._loss_and_grad(model, batch)
            accumulated = accumulate_weighted_nll_grad(
                accumulated,
                weighted_nll_sum,
                weight_sum,
                grads,
            )

        total_nll, total_weight, total_grads = accumulated
        scaled_grads = jax.tree.map(
            lambda grad: grad / jnp.maximum(total_weight, self.mean_epsilon),
            total_grads,
        )
        optimizer.update(model, scaled_grads)
        return self._metrics(
            total_nll,
            total_weight,
            num_batches=len(batches),
        )

    def epoch(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        update_batches: Sequence[Sequence[dict[str, jax.Array]]],
    ) -> dict[str, jax.Array | int]:
        if not update_batches:
            raise ValueError("update_batches must not be empty")

        total_nll = None
        total_weight = None
        num_batches = 0
        for batches in update_batches:
            update_metrics = self.update(model, optimizer, batches)
            if total_nll is None:
                total_nll = update_metrics["weighted_nll_sum"]
                total_weight = update_metrics["weight_sum"]
            else:
                total_nll = total_nll + update_metrics["weighted_nll_sum"]
                total_weight = total_weight + update_metrics["weight_sum"]
            num_batches += update_metrics["num_batches"]

        metrics = self._metrics(
            total_nll,
            total_weight,
            num_batches=num_batches,
        )
        metrics["num_updates"] = len(update_batches)
        return metrics

    def _metrics(
        self,
        weighted_nll_sum: jax.Array,
        weight_sum: jax.Array,
        *,
        num_batches: int,
    ) -> dict[str, jax.Array | int]:
        mean_nll = weighted_nll_sum / jnp.maximum(weight_sum, self.mean_epsilon)
        return {
            "weighted_nll_sum": weighted_nll_sum,
            "weight_sum": weight_sum,
            "mean_nll": mean_nll,
            "num_batches": num_batches,
        }


def weighted_nll(
    model: nnx.Module,
    batch: dict[str, jax.Array],
    grammar: FlatDefinitionGrammar,
    *,
    deterministic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    logits = model(
        batch["source_ids"],
        batch["decoder_input_ids"],
        deterministic=deterministic,
    )
    sequence_logp = constrained_sequence_log_prob(
        logits,
        batch["decoder_input_ids"],
        batch["target_ids"],
        batch["target_mask"],
        grammar,
    )
    example_nll = -sequence_logp
    weighted_nll_sum = jnp.sum(batch["example_weight"] * example_nll)
    weight_sum = jnp.sum(batch["example_weight"])
    return weighted_nll_sum, weight_sum
