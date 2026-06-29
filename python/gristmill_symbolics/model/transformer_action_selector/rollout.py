from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import sys
import time
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics import RewriteStateRow
from gristmill_symbolics._training import TrainingError

from .batched import (
    batched_sample_action,
    batched_sample_target,
    batched_score_action_grad,
    batched_score_target_grad,
)
from .tokenize import (
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from .tree import stack_token_trees
from .types import TokenTree, action_choice_to_python

DECISION_TARGET = 0
DECISION_ACTION = 1
_PROFILE_ROLLOUT_ENV = "GRISTMILL_PROFILE_ROLLOUT"
_PROFILE_ROLLOUT_SYNC_ENV = "GRISTMILL_PROFILE_ROLLOUT_SYNC"
_FALSE_ENV_VALUES = {
    "",
    "0",
    "false",
    "False",
    "FALSE",
    "no",
    "No",
    "NO",
    "off",
    "Off",
    "OFF",
}


@dataclass(frozen=True)
class _StaticModelRolloutResult:
    out_row: RewriteStateRow
    logp: jax.Array
    grad_logp: object
    stopped: np.ndarray


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in _FALSE_ENV_VALUES


def _profile_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array):
        if value.shape == ():
            return np.asarray(value).item()
        return f"jax.Array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, tuple | list):
        return [_profile_json_value(item) for item in value]
    return str(value)


def _block_profile_value(value):
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return value


class _RolloutProfiler:
    def __init__(self) -> None:
        self.enabled = _env_flag(_PROFILE_ROLLOUT_ENV)
        self.sync_jax = _env_flag(_PROFILE_ROLLOUT_SYNC_ENV, default=True)

    @contextmanager
    def phase(
        self, phase: str, step: int, **fields: Any
    ) -> Iterator[dict[str, Any]]:
        profile_fields = dict(fields)
        if not self.enabled:
            yield profile_fields
            return

        started_at = time.perf_counter()
        try:
            yield profile_fields
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            payload = {
                "event": "rollout_phase",
                "phase": phase,
                "step": int(step),
                "elapsed_ms": elapsed_ms,
            }
            payload.update(
                {
                    key: _profile_json_value(value)
                    for key, value in profile_fields.items()
                }
            )
            print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)

    def block_until_ready(self, value):
        if self.enabled and self.sync_jax:
            return _block_profile_value(value)
        return value


def _make_decision_rng_grid(rng, max_steps: int, batch_size: int):
    flat_keys = jax.random.split(rng, max_steps * batch_size * 2)
    return flat_keys.reshape((max_steps, batch_size, 2, *flat_keys.shape[1:]))


_DUMMY_STATE_SNAPSHOT = {
    "ranges": [],
    "tensors": [{"id": 0, "symmetry": []}],
    "definitions": [
        {
            "base": 0,
            "ext_indices": [],
            "terms": [],
        }
    ],
}

_DUMMY_TERM = {
    "coeff": {"numer": 1, "denom": 1},
    "sum_indices": [],
    "factors": [{"tensor": 0, "indices": []}],
}

_DUMMY_DEFINITION = {
    "base": 0,
    "ext_indices": [],
    "terms": [_DUMMY_TERM],
}

_DUMMY_ACTION_SPACE_SNAPSHOT = {
    "def_index": 0,
    "candidate_templates": [
        {
            "left_definition": _DUMMY_DEFINITION,
            "right_definition": _DUMMY_DEFINITION,
            "rewritten_definition": _DUMMY_DEFINITION,
        }
    ],
}


def _dummy_state_policy_item() -> tuple[TokenTree, jax.Array]:
    return tokenize_state_snapshot(_DUMMY_STATE_SNAPSHOT)


def _dummy_definition_mask() -> jax.Array:
    return jnp.zeros((1,), dtype=jnp.bool_)


def _dummy_action_policy_item() -> tuple[TokenTree, jax.Array]:
    return tokenize_action_space_snapshot(_DUMMY_ACTION_SPACE_SNAPSHOT)


def _sample_static_model_rollout(
    params,
    rng,
    row: RewriteStateRow,
    model,
) -> _StaticModelRolloutResult:
    _validate_gradient_param_dtypes(params)

    if int(row.len()) != model.batch_size:
        raise TrainingError(
            f"row batch size {row.len()} differs from batch_size {model.batch_size}"
        )

    rng_grid = _make_decision_rng_grid(
        rng,
        max_steps=model.max_steps,
        batch_size=model.batch_size,
    )
    active = [True] * model.batch_size
    stopped = [False] * model.batch_size
    exact_empty_def_masks: list[jax.Array | None] = [None] * model.batch_size
    trajectory_logp = jnp.zeros((model.batch_size,), dtype=jnp.float32)
    trajectory_grad_logp = _zero_trajectory_grad(params, model.batch_size)
    static_state_pad_to = model.state_token_pad_to
    static_definition_pad_to = model.definition_pad_to
    static_action_pad_to = model.action_token_pad_to
    profiler = _RolloutProfiler()

    for step in range(model.max_steps):
        active_indices = [sample for sample, is_active in enumerate(active) if is_active]
        if not active_indices:
            continue

        with profiler.phase(
            "row_snapshots", step, active_count=len(active_indices)
        ):
            snapshots = row.snapshots()
            definition_masks = row.definition_masks()
        state_items: list[tuple[TokenTree, jax.Array]] = []
        target_def_masks: list[jax.Array] = []
        target_keys: list[jax.Array] = []
        replay_exact_empty_samples: set[int] = set()
        target_policy_samples = list(range(model.batch_size))

        with profiler.phase(
            "tokenize_state", step, active_count=len(active_indices)
        ) as profile_fields:
            for sample in target_policy_samples:
                if not active[sample]:
                    state_items.append(_dummy_state_policy_item())
                    target_def_masks.append(_dummy_definition_mask())
                    target_keys.append(rng_grid[step, sample, DECISION_TARGET])
                    continue
                state_tokens, state_token_mask = tokenize_state_snapshot(snapshots[sample])
                row_def_mask = jnp.asarray(definition_masks[sample], dtype=jnp.bool_)
                target_def_mask = row_def_mask
                if (
                    exact_empty_def_masks[sample] is not None
                    and not _has_target_definition(row_def_mask)
                ):
                    target_def_mask = exact_empty_def_masks[sample]
                    replay_exact_empty_samples.add(sample)
                state_items.append((state_tokens, state_token_mask))
                target_def_masks.append(target_def_mask)
                target_keys.append(rng_grid[step, sample, DECISION_TARGET])
            profile_fields["state_token_len_max"] = _max_tree_length(state_items)

        with profiler.phase(
            "stack_state_tokens", step, active_count=len(active_indices)
        ) as profile_fields:
            state_tokens_batch, state_mask_batch = _stack_token_trees_for_policy(
                state_items,
                pad_to=static_state_pad_to,
                dimension="state token",
                config_field="state_token_pad_to",
            )
            target_def_mask_batch = _stack_bool_masks(
                target_def_masks,
                pad_to=static_definition_pad_to,
            )
            profile_fields["state_token_len_max"] = int(state_mask_batch.shape[1])
            profile_fields["definition_count_max"] = int(target_def_mask_batch.shape[1])
        with profiler.phase(
            "sample_target",
            step,
            active_count=len(active_indices),
            state_token_len_max=int(state_mask_batch.shape[1]),
        ):
            target_choices = batched_sample_target(
                params,
                state_tokens_batch,
                state_mask_batch,
                target_def_mask_batch,
                jnp.stack(target_keys, axis=0),
            )
            profiler.block_until_ready(target_choices)
        with profiler.phase(
            "score_target_grad",
            step,
            active_count=len(active_indices),
            state_token_len_max=int(state_mask_batch.shape[1]),
        ):
            target_logps, target_grads = batched_score_target_grad(
                params,
                state_tokens_batch,
                state_mask_batch,
                target_def_mask_batch,
                target_choices,
            )
            profiler.block_until_ready((target_logps, target_grads))
        target_active_mask = jnp.asarray(active, dtype=jnp.bool_)
        target_choices = jnp.where(target_active_mask, target_choices, -1)
        target_logps = jnp.where(target_active_mask, target_logps, 0.0)
        target_grads = _mask_tree_rows(target_grads, target_active_mask)
        target_scatter_samples = target_policy_samples
        trajectory_logp = trajectory_logp.at[jnp.asarray(target_scatter_samples)].add(
            target_logps
        )
        trajectory_grad_logp = _scatter_add_grad(
            trajectory_grad_logp, target_scatter_samples, target_grads
        )

        target_choice_list = [-1] * model.batch_size
        query_active = active.copy()
        active_position_by_sample = {
            sample: position for position, sample in enumerate(target_policy_samples)
        }

        for sample in active_indices:
            target_position = active_position_by_sample[sample]
            target_choice = int(np.asarray(target_choices[target_position]))
            target_choice_list[sample] = target_choice
            if target_choice == -1:
                active[sample] = False
                query_active[sample] = False
                stopped[sample] = True
                exact_empty_def_masks[sample] = None
                continue
            if sample in replay_exact_empty_samples:
                query_active[sample] = False

        spaces = None
        space_kinds = None
        space_snapshots = None
        if any(query_active):
            with profiler.phase(
                "row_query_action_spaces",
                step,
                active_count=len(active_indices),
                query_count=sum(query_active),
                target_choices=target_choice_list,
            ):
                spaces = row.query_action_spaces_for_row(target_choice_list, query_active)
                space_kinds = spaces.entry_kinds()
                space_snapshots = spaces.snapshots()

        non_empty_samples: list[int] = []
        selected_def_indices: list[int] = []
        action_items: list[tuple[TokenTree, jax.Array]] = []

        with profiler.phase(
            "tokenize_action_space", step, active_count=len(active_indices)
        ) as profile_fields:
            for sample in active_indices:
                if not query_active[sample]:
                    continue
                if spaces is None or space_kinds is None or space_snapshots is None:
                    raise TrainingError("active target query was not performed")
                target_choice = target_choice_list[sample]
                target_def_mask = target_def_masks[active_position_by_sample[sample]]

                if space_kinds[sample] == "exact_empty":
                    exact_empty_def_masks[sample] = _one_hot_def_mask(
                        target_choice, int(target_def_mask.shape[0])
                    )
                    continue

                if space_kinds[sample] != "non_empty":
                    raise TrainingError(
                        f"sample {sample} target {target_choice} produced "
                        f"unexpected action-space kind {space_kinds[sample]!r}"
                    )

                action_space_snapshot = space_snapshots[sample]
                if action_space_snapshot is None:
                    raise TrainingError(
                        f"sample {sample} has non_empty action-space kind but no snapshot"
                    )
                action_tokens, action_token_mask = tokenize_action_space_snapshot(
                    action_space_snapshot
                )
                non_empty_samples.append(sample)
                selected_def_indices.append(target_choice)
                action_items.append((action_tokens, action_token_mask))
                exact_empty_def_masks[sample] = None
            profile_fields["non_empty_count"] = len(non_empty_samples)
            profile_fields["action_token_len_max"] = _max_tree_length(action_items)

        non_empty_sample_set = set(non_empty_samples)
        non_empty_position_by_sample = {
            sample: position for position, sample in enumerate(non_empty_samples)
        }
        target_position_by_sample = active_position_by_sample
        action_policy_samples = list(range(model.batch_size))
        action_state_items: list[tuple[TokenTree, jax.Array]] = []
        action_policy_items: list[tuple[TokenTree, jax.Array]] = []
        selected_def_policy_indices: list[int] = []
        action_keys_for_policy: list[jax.Array] = []
        for sample in action_policy_samples:
            if sample in non_empty_sample_set:
                target_position = target_position_by_sample[sample]
                action_state_items.append(
                    (
                        _slice_tree(state_tokens_batch, target_position),
                        state_mask_batch[target_position],
                    )
                )
                action_position = non_empty_position_by_sample[sample]
                action_policy_items.append(action_items[action_position])
                selected_def_policy_indices.append(
                    selected_def_indices[action_position]
                )
            else:
                action_state_items.append(_dummy_state_policy_item())
                action_policy_items.append(_dummy_action_policy_item())
                selected_def_policy_indices.append(0)
            action_keys_for_policy.append(rng_grid[step, sample, DECISION_ACTION])

        with profiler.phase(
            "stack_action_tokens",
            step,
            non_empty_count=len(non_empty_samples),
        ) as profile_fields:
            action_state_tokens, action_state_mask = _stack_token_trees_for_policy(
                action_state_items,
                pad_to=static_state_pad_to,
                dimension="state token",
                config_field="state_token_pad_to",
            )
            action_tokens_batch, action_mask_batch = _stack_token_trees_for_policy(
                action_policy_items,
                pad_to=static_action_pad_to,
                dimension="action token",
                config_field="action_token_pad_to",
            )
            selected = jnp.asarray(selected_def_policy_indices, dtype=jnp.int32)
            stacked_action_keys = jnp.stack(action_keys_for_policy, axis=0)
            action_position_by_sample = {
                sample: sample for sample in range(model.batch_size)
            }
            profile_fields["state_token_len_max"] = int(action_state_mask.shape[1])
            profile_fields["action_token_len_max"] = int(action_mask_batch.shape[1])
        with profiler.phase(
            "sample_action",
            step,
            non_empty_count=len(non_empty_samples),
            action_token_len_max=int(action_mask_batch.shape[1]),
        ):
            action_choices = batched_sample_action(
                params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                stacked_action_keys,
            )
            profiler.block_until_ready(action_choices)
        with profiler.phase(
            "score_action_grad",
            step,
            non_empty_count=len(non_empty_samples),
            action_token_len_max=int(action_mask_batch.shape[1]),
        ):
            action_logps, action_grads = batched_score_action_grad(
                params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                action_choices,
            )
            profiler.block_until_ready((action_logps, action_grads))
        action_active_mask = jnp.asarray(
            [sample in non_empty_sample_set for sample in action_policy_samples],
            dtype=jnp.bool_,
        )
        action_logps = jnp.where(action_active_mask, action_logps, 0.0)
        action_grads = _mask_tree_rows(action_grads, action_active_mask)

        trajectory_logp = trajectory_logp.at[jnp.asarray(action_policy_samples)].add(
            action_logps
        )
        trajectory_grad_logp = _scatter_add_grad(
            trajectory_grad_logp, action_policy_samples, action_grads
        )

        if non_empty_samples:
            action_choices_for_row: list[dict[str, object] | None] = [
                None
            ] * model.batch_size
            action_score_mask = [False] * model.batch_size
            with profiler.phase(
                "action_choice_to_python",
                step,
                non_empty_count=len(non_empty_samples),
            ):
                for sample in non_empty_samples:
                    action_choices_for_row[sample] = action_choice_to_python(
                        _slice_tree(action_choices, action_position_by_sample[sample])
                    )
                    action_score_mask[sample] = True

            with profiler.phase(
                "row_validate_apply_actions",
                step,
                non_empty_count=len(non_empty_samples),
            ):
                validated = row.validate_actions_for_row(
                    spaces, action_choices_for_row, action_score_mask
                )
                applied = row.apply_validated_actions_for_row(validated)
                for sample in non_empty_samples:
                    if not bool(applied[sample]):
                        raise TrainingError(
                            f"validated action for sample {sample} was not applied"
                        )

    return _StaticModelRolloutResult(
        out_row=row,
        logp=trajectory_logp,
        grad_logp=trajectory_grad_logp,
        stopped=np.asarray(stopped, dtype=bool),
    )


def _validate_gradient_param_dtypes(params) -> None:
    for leaf in jax.tree_util.tree_leaves(params):
        try:
            dtype = jnp.asarray(leaf).dtype
        except (TypeError, ValueError) as exc:
            raise TrainingError(
                "model params must contain only floating arrays for log-prob gradients"
            ) from exc
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TrainingError(
                "model params must contain only floating arrays for log-prob gradients"
            )


def _zero_trajectory_grad(params, batch_size: int):
    def zero_leaf(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype)

    return jax.tree_util.tree_map(zero_leaf, params)


def _scatter_add_grad(accum, sample_indices: list[int], step_grad):
    indices = jnp.asarray(sample_indices, dtype=jnp.int32)
    return jax.tree_util.tree_map(
        lambda accum_leaf, grad_leaf: accum_leaf.at[indices].add(grad_leaf),
        accum,
        step_grad,
    )


def _mask_tree_rows(tree, row_mask: jax.Array):
    row_mask = jnp.asarray(row_mask, dtype=jnp.bool_)

    def mask_leaf(leaf):
        leaf = jnp.asarray(leaf)
        scale = row_mask.astype(leaf.dtype).reshape(
            (row_mask.shape[0],) + (1,) * (leaf.ndim - 1)
        )
        return leaf * scale

    return jax.tree_util.tree_map(mask_leaf, tree)


def _slice_tree(tree, index: int):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _raise_static_pad_too_small(
    *, dimension: str, config_field: str, observed: int, configured: int
) -> None:
    raise TrainingError(
        f"{dimension} length {observed} exceeds {config_field} {configured}"
    )


def _validate_static_pad_limit(
    *,
    dimension: str,
    config_field: str,
    observed: int,
    configured: int | None,
) -> None:
    if configured is not None and observed > configured:
        _raise_static_pad_too_small(
            dimension=dimension,
            config_field=config_field,
            observed=observed,
            configured=configured,
        )


def _stack_token_trees_for_policy(
    items: list[tuple[TokenTree, jax.Array]],
    *,
    pad_to: int | None,
    dimension: str,
    config_field: str,
):
    if pad_to is not None:
        for _tokens, mask in items:
            _validate_static_pad_limit(
                dimension=dimension,
                config_field=config_field,
                observed=int(mask.shape[0]),
                configured=pad_to,
            )
    return stack_token_trees(items, pad_to=pad_to)


def _stack_bool_masks(masks: list[jax.Array], pad_to: int | None = None) -> jax.Array:
    length = int(pad_to) if pad_to is not None else _max_mask_length(masks)
    if pad_to is not None:
        for mask in masks:
            _validate_static_pad_limit(
                dimension="definition mask",
                config_field="definition_pad_to",
                observed=int(mask.shape[0]),
                configured=length,
            )
    return jnp.stack([_pad_bool_mask(mask, length) for mask in masks], axis=0)


def _pad_bool_mask(mask: jax.Array, length: int) -> jax.Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if int(mask.shape[0]) > length:
        raise ValueError(
            f"cannot pad bool mask of length {mask.shape[0]} to shorter length {length}"
        )
    return jnp.pad(mask, (0, length - int(mask.shape[0])), constant_values=False)


def _has_target_definition(mask: jax.Array) -> bool:
    return bool(np.asarray(jnp.any(jnp.asarray(mask, dtype=jnp.bool_))))


def _one_hot_def_mask(target_choice: int, length: int) -> jax.Array:
    if target_choice < 0 or target_choice >= length:
        raise TrainingError(
            f"target_choice {target_choice} is outside definition mask length {length}"
        )
    return jnp.zeros((length,), dtype=jnp.bool_).at[target_choice].set(True)


def _max_tree_length(items: list[tuple[TokenTree, jax.Array]]) -> int:
    return max((int(mask.shape[0]) for _tokens, mask in items), default=0)


def _max_mask_length(masks: list[jax.Array]) -> int:
    return max(1, *(int(mask.shape[0]) for mask in masks))
