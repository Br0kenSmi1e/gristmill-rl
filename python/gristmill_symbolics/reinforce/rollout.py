from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics import RewriteStateRow
from gristmill_symbolics.policy import (
    action_choice_to_python,
    sample_action,
    sample_target,
    score_action,
    score_target,
    stack_token_trees,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from gristmill_symbolics.policy.types import ActionChoiceTree, TokenTree

from .types import (
    DECISION_ACTION,
    DECISION_TARGET,
    FinalColumnMetrics,
    PolicyState,
    RolloutConfig,
    TrainingError,
    validate_policy_state,
    validate_rollout_config,
)


@dataclass(frozen=True)
class _StreamedRolloutResult:
    final: FinalColumnMetrics
    trajectory_logp: jax.Array
    trajectory_grad_logp: object
    valid_action_count: int
    stop_count: int
    empty_action_space_count: int
    finished_count: int
    target_score_count: int
    action_score_count: int
    target_logp_sum: float
    action_logp_sum: float


def make_rng_grid(root_key, update_index: int, max_steps: int, batch_size: int):
    update_key = jax.random.fold_in(root_key, int(update_index))
    flat_keys = jax.random.split(update_key, max_steps * batch_size * 2)
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


def _collect_streamed_rollout_gradients(
    policy: PolicyState,
    initial_states,
    config: RolloutConfig,
    *,
    update_index,
    root_key,
) -> _StreamedRolloutResult:
    validate_policy_state(policy)
    validate_rollout_config(config)
    _validate_streamed_gradient_param_dtypes(policy.params)
    initial_states = list(initial_states)
    if len(initial_states) != config.batch_size:
        raise TrainingError(
            f"initial_states length {len(initial_states)} differs from "
            f"batch_size {config.batch_size}"
        )

    initial_log_flops = np.asarray(
        [state.log_total_flops() for state in initial_states], dtype=np.float64
    )
    row = RewriteStateRow.from_states(initial_states)
    rng_grid = make_rng_grid(
        root_key,
        update_index=int(update_index),
        max_steps=config.max_steps,
        batch_size=config.batch_size,
    )

    active = [True] * config.batch_size
    stopped = [False] * config.batch_size
    exact_empty_def_masks: list[jax.Array | None] = [None] * config.batch_size
    trajectory_logp = jnp.zeros((config.batch_size,), dtype=jnp.float32)
    trajectory_grad_logp = _zero_trajectory_grad(policy.params, config.batch_size)

    valid_action_count = 0
    stop_count = 0
    empty_action_space_count = 0
    finished_count = 0
    target_score_count = 0
    action_score_count = 0
    target_logp_sum = 0.0
    action_logp_sum = 0.0
    static_policy_batch = config.static_policy_batch
    static_state_pad_to = config.state_token_pad_to if static_policy_batch else None
    static_definition_pad_to = (
        config.definition_pad_to if static_policy_batch else None
    )
    static_action_pad_to = config.action_token_pad_to if static_policy_batch else None

    for step in range(config.max_steps):
        active_indices = [sample for sample, is_active in enumerate(active) if is_active]
        finished_count += config.batch_size - len(active_indices)
        if not active_indices and not static_policy_batch:
            continue

        snapshots = row.snapshots()
        definition_masks = row.definition_masks()
        state_items: list[tuple[TokenTree, jax.Array]] = []
        target_def_masks: list[jax.Array] = []
        target_keys: list[jax.Array] = []
        replay_exact_empty_samples: set[int] = set()
        target_policy_samples = (
            list(range(config.batch_size)) if static_policy_batch else active_indices
        )

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
        target_choices = jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
            policy.params,
            state_tokens_batch,
            state_mask_batch,
            target_def_mask_batch,
            jnp.stack(target_keys, axis=0),
        )
        target_logps, target_grads = jax.vmap(
            jax.value_and_grad(score_target, argnums=0),
            in_axes=(None, 0, 0, 0, 0),
        )(
            policy.params,
            state_tokens_batch,
            state_mask_batch,
            target_def_mask_batch,
            target_choices,
        )
        if static_policy_batch:
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
        target_score_count += len(active_indices)
        target_logp_sum += float(np.asarray(jnp.sum(target_logps)))

        target_choice_list = [-1] * config.batch_size
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
                stop_count += 1
                continue
            if sample in replay_exact_empty_samples:
                query_active[sample] = False
                empty_action_space_count += 1

        spaces = None
        space_kinds = None
        space_snapshots = None
        if any(query_active):
            spaces = row.query_action_spaces_for_row(target_choice_list, query_active)
            space_kinds = spaces.entry_kinds()
            space_snapshots = spaces.snapshots()

        non_empty_samples: list[int] = []
        non_empty_active_positions: list[int] = []
        selected_def_indices: list[int] = []
        action_items: list[tuple[TokenTree, jax.Array]] = []
        action_keys: list[jax.Array] = []

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
                empty_action_space_count += 1
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
            non_empty_active_positions.append(active_position_by_sample[sample])
            selected_def_indices.append(target_choice)
            action_items.append((action_tokens, action_token_mask))
            action_keys.append(rng_grid[step, sample, DECISION_ACTION])
            exact_empty_def_masks[sample] = None

        if non_empty_samples or static_policy_batch:
            if static_policy_batch:
                non_empty_sample_set = set(non_empty_samples)
                non_empty_position_by_sample = {
                    sample: position
                    for position, sample in enumerate(non_empty_samples)
                }
                target_position_by_sample = active_position_by_sample
                action_policy_samples = list(range(config.batch_size))
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
                    action_keys_for_policy.append(
                        rng_grid[step, sample, DECISION_ACTION]
                    )

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
                    sample: sample for sample in range(config.batch_size)
                }
            else:
                action_policy_samples = non_empty_samples
                action_state_tokens = _take_tree_rows(
                    state_tokens_batch, non_empty_active_positions
                )
                action_state_mask = state_mask_batch[
                    jnp.asarray(non_empty_active_positions)
                ]
                action_tokens_batch, action_mask_batch = stack_token_trees(action_items)
                selected = jnp.asarray(selected_def_indices, dtype=jnp.int32)
                stacked_action_keys = jnp.stack(action_keys, axis=0)
                action_position_by_sample = {
                    sample: position
                    for position, sample in enumerate(non_empty_samples)
                }
            action_choices = jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
                policy.params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                stacked_action_keys,
            )
            action_logps, action_grads = jax.vmap(
                jax.value_and_grad(score_action, argnums=0),
                in_axes=(None, 0, 0, 0, 0, 0, 0),
            )(
                policy.params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                action_choices,
            )
            if static_policy_batch:
                action_active_mask = jnp.asarray(
                    [
                        sample in set(non_empty_samples)
                        for sample in action_policy_samples
                    ],
                    dtype=jnp.bool_,
                )
                action_logps = jnp.where(action_active_mask, action_logps, 0.0)
                action_grads = _mask_tree_rows(action_grads, action_active_mask)

            trajectory_logp = trajectory_logp.at[
                jnp.asarray(action_policy_samples)
            ].add(action_logps)
            trajectory_grad_logp = _scatter_add_grad(
                trajectory_grad_logp, action_policy_samples, action_grads
            )
            action_score_count += len(non_empty_samples)
            action_logp_sum += float(np.asarray(jnp.sum(action_logps)))

            action_choices_for_row: list[dict[str, object] | None] = [
                None
            ] * config.batch_size
            action_score_mask = [False] * config.batch_size
            for sample in non_empty_samples:
                action_choices_for_row[sample] = action_choice_to_python(
                    _slice_tree(action_choices, action_position_by_sample[sample])
                )
                action_score_mask[sample] = True

            validated = row.validate_actions_for_row(
                spaces, action_choices_for_row, action_score_mask
            )
            applied = row.apply_validated_actions_for_row(validated)
            for sample in non_empty_samples:
                if not bool(applied[sample]):
                    raise TrainingError(
                        f"validated action for sample {sample} was not applied"
                    )
            valid_action_count += len(non_empty_samples)

    final = FinalColumnMetrics(
        initial_log_flops=initial_log_flops,
        final_log_flops=np.asarray(row.log_total_flops(), dtype=np.float64),
        stopped=np.asarray(stopped, dtype=bool),
        max_steps=np.asarray(active, dtype=bool),
    )
    return _StreamedRolloutResult(
        final=final,
        trajectory_logp=trajectory_logp,
        trajectory_grad_logp=trajectory_grad_logp,
        valid_action_count=valid_action_count,
        stop_count=stop_count,
        empty_action_space_count=empty_action_space_count,
        finished_count=finished_count,
        target_score_count=target_score_count,
        action_score_count=action_score_count,
        target_logp_sum=target_logp_sum,
        action_logp_sum=action_logp_sum,
    )


def _validate_streamed_gradient_param_dtypes(params) -> None:
    for leaf in jax.tree_util.tree_leaves(params):
        try:
            dtype = jnp.asarray(leaf).dtype
        except (TypeError, ValueError) as exc:
            raise TrainingError(
                "policy params must contain only floating arrays for streamed gradients"
            ) from exc
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TrainingError(
                "policy params must contain only floating arrays for streamed gradients"
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


def _take_tree_rows(tree, indices: list[int]):
    index_array = jnp.asarray(indices, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda value: value[index_array], tree)


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


def _max_mask_length(masks: list[jax.Array]) -> int:
    return max(1, *(int(mask.shape[0]) for mask in masks))
