from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from gristmill_rl.actions import SampledAction
from gristmill_rl.features import CANDIDATE_DIM, STATE_DIM, TERM_DIM, FeatureBatch


@dataclass(frozen=True)
class ModelOutputs:
    candidate_logits: jax.Array
    left_logits: jax.Array
    right_logits: jax.Array
    value: jax.Array


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3


class PolicyValueModule(nnx.Module):
    def __init__(self, *, hidden_dim: int, rngs: nnx.Rngs):
        self.state_1 = nnx.Linear(STATE_DIM, hidden_dim, rngs=rngs)
        self.state_2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.candidate_1 = nnx.Linear(CANDIDATE_DIM, hidden_dim, rngs=rngs)
        self.candidate_2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.left_term_1 = nnx.Linear(TERM_DIM, hidden_dim, rngs=rngs)
        self.right_term_1 = nnx.Linear(TERM_DIM, hidden_dim, rngs=rngs)
        self.candidate_head = nnx.Linear(hidden_dim * 2, 1, rngs=rngs)
        self.left_head = nnx.Linear(hidden_dim * 3, 1, rngs=rngs)
        self.right_head = nnx.Linear(hidden_dim * 4, 1, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def _state_embed(self, state: jax.Array) -> jax.Array:
        x = nnx.relu(self.state_1(state))
        return nnx.relu(self.state_2(x))

    def _candidate_embed(self, candidates: jax.Array) -> jax.Array:
        x = nnx.relu(self.candidate_1(candidates))
        return nnx.relu(self.candidate_2(x))

    def _term_embed(self, terms: jax.Array, layer: nnx.Linear) -> jax.Array:
        return nnx.relu(layer(terms))

    def __call__(self, features: FeatureBatch) -> ModelOutputs:
        state = jnp.asarray(features.state)
        candidates = jnp.asarray(features.candidates)
        left_terms = jnp.asarray(features.left_terms)
        right_terms = jnp.asarray(features.right_terms)
        candidate_mask = jnp.asarray(features.candidate_mask)
        left_mask = jnp.asarray(features.left_term_mask)
        right_mask = jnp.asarray(features.right_term_mask)

        state_embed = self._state_embed(state)
        candidate_embed = self._candidate_embed(candidates)
        repeated_state = jnp.repeat(state_embed[None, :], candidates.shape[0], axis=0)

        candidate_logits = self.candidate_head(
            jnp.concatenate([repeated_state, candidate_embed], axis=-1)
        ).squeeze(-1)
        candidate_logits = jnp.where(candidate_mask, candidate_logits, -1.0e9)

        left_embed = self._term_embed(left_terms, self.left_term_1)
        left_state = jnp.repeat(repeated_state[:, None, :], left_terms.shape[1], axis=1)
        left_candidate = jnp.repeat(
            candidate_embed[:, None, :], left_terms.shape[1], axis=1
        )
        left_logits = self.left_head(
            jnp.concatenate([left_state, left_candidate, left_embed], axis=-1)
        ).squeeze(-1)
        left_logits = jnp.where(left_mask, left_logits, -1.0e9)

        right_embed = self._term_embed(right_terms, self.right_term_1)
        left_counts = jnp.maximum(left_mask.sum(axis=1, keepdims=True), 1)
        left_summary = (left_embed * left_mask[..., None]).sum(axis=1) / left_counts
        right_state = jnp.repeat(
            repeated_state[:, None, :], right_terms.shape[1], axis=1
        )
        right_candidate = jnp.repeat(
            candidate_embed[:, None, :], right_terms.shape[1], axis=1
        )
        right_left = jnp.repeat(left_summary[:, None, :], right_terms.shape[1], axis=1)
        right_logits = self.right_head(
            jnp.concatenate(
                [right_state, right_candidate, right_left, right_embed], axis=-1
            )
        ).squeeze(-1)
        right_logits = jnp.where(right_mask, right_logits, -1.0e9)

        value = self.value_head(state_embed).squeeze()
        return ModelOutputs(
            candidate_logits=candidate_logits,
            left_logits=left_logits,
            right_logits=right_logits,
            value=value,
        )


class PolicyValueModel:
    def __init__(self, *, hidden_dim: int = 32, rng_seed: int = 0):
        self.module = PolicyValueModule(hidden_dim=hidden_dim, rngs=nnx.Rngs(rng_seed))

    def __call__(self, features: FeatureBatch) -> ModelOutputs:
        return self.module(features)


def _mask_log_prob(logits: jax.Array, mask: list[bool]) -> jax.Array:
    if len(mask) != logits.shape[0]:
        raise ValueError(
            f"mask length {len(mask)} does not match represented logits length "
            f"{logits.shape[0]}"
        )
    if logits.shape[0] == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    bits = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(
        jax.nn.log_sigmoid(logits) * bits
        + jax.nn.log_sigmoid(-logits) * (1.0 - bits)
    )


def action_log_prob(
    model: PolicyValueModel | PolicyValueModule,
    features: FeatureBatch,
    action: SampledAction,
) -> jax.Array:
    module = model.module if isinstance(model, PolicyValueModel) else model
    outputs = module(features)
    decision = action.decision
    candidate_index = int(decision["candidate_index"])
    candidate_mask = features.candidate_mask
    if candidate_index < 0 or candidate_index >= candidate_mask.shape[0]:
        raise ValueError(
            f"candidate_index {candidate_index} is outside represented candidates "
            f"0..{candidate_mask.shape[0] - 1}"
        )
    if not bool(candidate_mask[candidate_index]):
        raise ValueError(f"candidate_index {candidate_index} is not represented")
    candidate_log_probs = jax.nn.log_softmax(outputs.candidate_logits)
    return (
        candidate_log_probs[candidate_index]
        + _mask_log_prob(outputs.left_logits[candidate_index], decision["left_mask"])
        + _mask_log_prob(outputs.right_logits[candidate_index], decision["right_mask"])
    )


def _loss_for_batch(
    module: PolicyValueModule, batch: list[dict[str, Any]]
) -> tuple[jax.Array, dict[str, jax.Array]]:
    policy_losses = []
    value_losses = []
    for item in batch:
        features = item["features"]
        outputs = module(features)
        action_log_probs = jnp.asarray(
            [action_log_prob(module, features, action) for action in item["actions"]]
        )
        target = jnp.asarray(item["policy_target"])
        policy_losses.append(-jnp.sum(target * action_log_probs))
        value_target = jnp.asarray(item["value_target"], dtype=jnp.float32)
        value_losses.append(jnp.square(outputs.value - value_target))
    policy_loss = jnp.mean(jnp.asarray(policy_losses))
    value_loss = jnp.mean(jnp.asarray(value_losses))
    total_loss = policy_loss + value_loss
    return total_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "total_loss": total_loss,
    }


def _flat_param_values(module: PolicyValueModule) -> list[np.ndarray]:
    state = nnx.state(module, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    values = []
    for leaf in leaves:
        value = getattr(leaf, "value", leaf)
        values.append(np.asarray(value).copy())
    return values


def train_step(
    model: PolicyValueModel,
    *,
    batch: list[dict[str, Any]],
    config: TrainConfig,
) -> dict[str, float | bool]:
    before = _flat_param_values(model.module)
    optimizer = nnx.Optimizer(
        model.module,
        optax.adam(config.learning_rate),
        wrt=nnx.Param,
    )

    grad_fn = nnx.value_and_grad(_loss_for_batch, has_aux=True)
    (loss, aux), grads = grad_fn(model.module, batch)
    optimizer.update(model.module, grads)

    after = _flat_param_values(model.module)
    params_changed = any(
        not np.array_equal(left, right) for left, right in zip(before, after, strict=True)
    )
    return {
        "policy_loss": float(aux["policy_loss"]),
        "value_loss": float(aux["value_loss"]),
        "total_loss": float(loss),
        "params_changed": params_changed,
    }
