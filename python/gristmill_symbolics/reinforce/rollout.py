from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics import RewriteStateRow
from gristmill_symbolics.policy import (
    action_choice_to_python,
    pad_token_tree,
    sample_action,
    sample_target,
    stack_token_trees,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from gristmill_symbolics.policy.constants import (
    ACTION_TOKEN_FIELDS,
    SENTINEL,
    STATE_TOKEN_FIELDS,
    TOKEN_KIND,
)
from gristmill_symbolics.policy.types import ActionChoiceTree, TokenTree

from .types import (
    CASE_ALREADY_FINISHED,
    CASE_EMPTY_ACTION_SPACE,
    CASE_STOP,
    CASE_VALID_ACTION,
    DECISION_ACTION,
    DECISION_TARGET,
    FinalColumnMetrics,
    PolicyState,
    RolloutConfig,
    RolloutTable,
    TrainingError,
    validate_policy_state,
    validate_rollout_config,
)


@dataclass(frozen=True)
class _SampleRecord:
    state_tokens: TokenTree
    state_token_mask: jax.Array
    target_def_mask: jax.Array
    target_choice: jax.Array
    target_score_mask: jax.Array
    selected_def_index: jax.Array
    action_space_tokens: TokenTree
    action_space_token_mask: jax.Array
    action_choice: ActionChoiceTree
    action_score_mask: jax.Array
    step_case: jax.Array
    sampled_target_logp: jax.Array
    sampled_action_logp: jax.Array


def make_rng_grid(root_key, update_index: int, max_steps: int, batch_size: int):
    update_key = jax.random.fold_in(root_key, int(update_index))
    flat_keys = jax.random.split(update_key, max_steps * batch_size * 2)
    return flat_keys.reshape((max_steps, batch_size, 2, *flat_keys.shape[1:]))


def collect_rollout_batch(
    policy: PolicyState,
    initial_states,
    config: RolloutConfig,
    *,
    update_index,
    root_key,
) -> tuple[RolloutTable, FinalColumnMetrics]:
    validate_policy_state(policy)
    validate_rollout_config(config)
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
    records_by_step: list[list[_SampleRecord]] = []
    action_width = int(policy.config.max_side_terms)

    for step in range(config.max_steps):
        step_records = [
            _already_finished_record(action_width) for _ in range(config.batch_size)
        ]
        active_indices = [sample for sample, is_active in enumerate(active) if is_active]

        if active_indices:
            snapshots = row.snapshots()
            definition_masks = row.definition_masks()
            state_items: list[tuple[TokenTree, jax.Array]] = []
            target_def_masks: list[jax.Array] = []
            target_keys: list[jax.Array] = []
            state_by_sample: dict[int, tuple[TokenTree, jax.Array]] = {}
            def_mask_by_sample: dict[int, jax.Array] = {}

            for sample in active_indices:
                state_tokens, state_token_mask = tokenize_state_snapshot(snapshots[sample])
                target_def_mask = jnp.asarray(
                    definition_masks[sample], dtype=jnp.bool_
                )
                state_by_sample[sample] = (state_tokens, state_token_mask)
                def_mask_by_sample[sample] = target_def_mask
                state_items.append((state_tokens, state_token_mask))
                target_def_masks.append(target_def_mask)
                target_keys.append(rng_grid[step, sample, DECISION_TARGET])

            target_choices, target_logps = _sample_targets_for_active(
                policy,
                state_items,
                target_def_masks,
                target_keys,
            )
            target_choice_list = [-1] * config.batch_size
            target_logp_by_sample: dict[int, jax.Array] = {}

            for position, sample in enumerate(active_indices):
                target_choice = int(np.asarray(target_choices[position]))
                target_choice_list[sample] = target_choice
                target_logp_by_sample[sample] = target_logps[position]

            spaces = row.query_action_spaces_for_row(target_choice_list, active)
            space_kinds = spaces.entry_kinds()
            space_snapshots = spaces.snapshots()

            non_empty_samples: list[int] = []
            non_empty_state_items: list[tuple[TokenTree, jax.Array]] = []
            selected_def_indices: list[int] = []
            action_items: list[tuple[TokenTree, jax.Array]] = []
            action_keys: list[jax.Array] = []

            for sample in active_indices:
                target_choice = target_choice_list[sample]
                target_logp = target_logp_by_sample[sample]
                state_tokens, state_token_mask = state_by_sample[sample]
                target_def_mask = def_mask_by_sample[sample]

                if target_choice == -1:
                    step_records[sample] = _record_without_action(
                        state_tokens=state_tokens,
                        state_token_mask=state_token_mask,
                        target_def_mask=target_def_mask,
                        target_choice=target_choice,
                        target_score_mask=True,
                        selected_def_index=0,
                        step_case=CASE_STOP,
                        sampled_target_logp=target_logp,
                        action_width=action_width,
                    )
                    active[sample] = False
                    stopped[sample] = True
                    continue

                if space_kinds[sample] == "exact_empty":
                    step_records[sample] = _record_without_action(
                        state_tokens=state_tokens,
                        state_token_mask=state_token_mask,
                        target_def_mask=target_def_mask,
                        target_choice=target_choice,
                        target_score_mask=True,
                        selected_def_index=target_choice,
                        step_case=CASE_EMPTY_ACTION_SPACE,
                        sampled_target_logp=target_logp,
                        action_width=action_width,
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
                        f"sample {sample} has non_empty action-space kind "
                        "but no snapshot"
                    )
                action_tokens, action_token_mask = tokenize_action_space_snapshot(
                    action_space_snapshot
                )
                non_empty_samples.append(sample)
                non_empty_state_items.append((state_tokens, state_token_mask))
                selected_def_indices.append(target_choice)
                action_items.append((action_tokens, action_token_mask))
                action_keys.append(rng_grid[step, sample, DECISION_ACTION])

            action_choices_for_row: list[dict[str, object] | None] = [
                None
            ] * config.batch_size
            action_score_mask = [False] * config.batch_size

            if non_empty_samples:
                sampled_action_choices, sampled_action_logps = (
                    _sample_actions_for_non_empty(
                        policy,
                        non_empty_state_items,
                        selected_def_indices,
                        action_items,
                        action_keys,
                    )
                )

                for position, sample in enumerate(non_empty_samples):
                    action_choice = _ensure_width(
                        _slice_action_choice(sampled_action_choices, position),
                        action_width,
                    )
                    action_logp = sampled_action_logps[position]
                    state_tokens, state_token_mask = state_by_sample[sample]
                    action_tokens, action_token_mask = action_items[position]
                    target_choice = selected_def_indices[position]
                    target_def_mask = def_mask_by_sample[sample]
                    target_logp = target_logp_by_sample[sample]

                    step_records[sample] = _SampleRecord(
                        state_tokens=state_tokens,
                        state_token_mask=state_token_mask,
                        target_def_mask=target_def_mask,
                        target_choice=jnp.asarray(target_choice, dtype=jnp.int32),
                        target_score_mask=jnp.asarray(True, dtype=jnp.bool_),
                        selected_def_index=jnp.asarray(target_choice, dtype=jnp.int32),
                        action_space_tokens=action_tokens,
                        action_space_token_mask=action_token_mask,
                        action_choice=action_choice,
                        action_score_mask=jnp.asarray(True, dtype=jnp.bool_),
                        step_case=jnp.asarray(CASE_VALID_ACTION, dtype=jnp.int32),
                        sampled_target_logp=target_logp,
                        sampled_action_logp=action_logp,
                    )
                    action_choices_for_row[sample] = action_choice_to_python(action_choice)
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

        records_by_step.append(step_records)

    table = _assemble_rollout(records_by_step, action_width)
    final = FinalColumnMetrics(
        initial_log_flops=initial_log_flops,
        final_log_flops=np.asarray(row.log_total_flops(), dtype=np.float64),
        stopped=np.asarray(stopped, dtype=bool),
        max_steps=np.asarray(active, dtype=bool),
    )
    return table, final


def _sample_targets_for_active(
    policy: PolicyState,
    state_items: list[tuple[TokenTree, jax.Array]],
    target_def_masks: list[jax.Array],
    keys: list[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    if not state_items:
        return (
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.float32),
        )

    state_tokens, state_token_mask = stack_token_trees(state_items)
    def_length = _max_mask_length(target_def_masks)
    target_def_mask = jnp.stack(
        [_pad_bool_mask(mask, def_length) for mask in target_def_masks], axis=0
    )
    return jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
        policy.params,
        state_tokens,
        state_token_mask,
        target_def_mask,
        jnp.stack(keys, axis=0),
    )


def _sample_actions_for_non_empty(
    policy: PolicyState,
    state_items: list[tuple[TokenTree, jax.Array]],
    selected_def_indices: list[int],
    action_items: list[tuple[TokenTree, jax.Array]],
    keys: list[jax.Array],
) -> tuple[ActionChoiceTree, jax.Array]:
    if not action_items:
        return _empty_batched_action_choice(0, int(policy.config.max_side_terms)), jnp.zeros(
            (0,), dtype=jnp.float32
        )

    state_tokens, state_token_mask = stack_token_trees(state_items)
    action_tokens, action_token_mask = stack_token_trees(action_items)
    selected = jnp.asarray(selected_def_indices, dtype=jnp.int32)
    return jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        policy.params,
        state_tokens,
        state_token_mask,
        selected,
        action_tokens,
        action_token_mask,
        jnp.stack(keys, axis=0),
    )


def _assemble_rollout(
    records_by_step: list[list[_SampleRecord]],
    width: int,
) -> RolloutTable:
    flat_records = [record for step_records in records_by_step for record in step_records]
    state_length = max(int(record.state_token_mask.shape[0]) for record in flat_records)
    action_length = max(
        int(record.action_space_token_mask.shape[0]) for record in flat_records
    )
    def_length = max(int(record.target_def_mask.shape[0]) for record in flat_records)

    state_tokens_by_step: list[TokenTree] = []
    state_mask_by_step: list[jax.Array] = []
    action_tokens_by_step: list[TokenTree] = []
    action_mask_by_step: list[jax.Array] = []
    target_def_mask_by_step: list[jax.Array] = []
    action_choice_by_step: list[ActionChoiceTree] = []

    for step_records in records_by_step:
        state_tokens, state_mask = stack_token_trees(
            [
                _pad_state_tree(record.state_tokens, record.state_token_mask, state_length)
                for record in step_records
            ],
            pad_to=state_length,
        )
        action_tokens, action_mask = stack_token_trees(
            [
                _pad_action_tree(
                    record.action_space_tokens,
                    record.action_space_token_mask,
                    action_length,
                )
                for record in step_records
            ],
            pad_to=action_length,
        )
        state_tokens_by_step.append(state_tokens)
        state_mask_by_step.append(state_mask)
        action_tokens_by_step.append(action_tokens)
        action_mask_by_step.append(action_mask)
        target_def_mask_by_step.append(
            jnp.stack(
                [
                    _pad_bool_mask(record.target_def_mask, def_length)
                    for record in step_records
                ],
                axis=0,
            )
        )
        action_choice_by_step.append(_stack_action_choices(step_records, width))

    return RolloutTable(
        state_tokens=_stack_trees(state_tokens_by_step),
        state_token_mask=jnp.stack(state_mask_by_step, axis=0),
        target_def_mask=jnp.stack(target_def_mask_by_step, axis=0),
        target_choice=_stack_record_scalar(records_by_step, "target_choice", jnp.int32),
        target_score_mask=_stack_record_scalar(
            records_by_step, "target_score_mask", jnp.bool_
        ),
        selected_def_index=_stack_record_scalar(
            records_by_step, "selected_def_index", jnp.int32
        ),
        action_space_tokens=_stack_trees(action_tokens_by_step),
        action_space_token_mask=jnp.stack(action_mask_by_step, axis=0),
        action_choice=_stack_trees(action_choice_by_step),
        action_score_mask=_stack_record_scalar(
            records_by_step, "action_score_mask", jnp.bool_
        ),
        step_case=_stack_record_scalar(records_by_step, "step_case", jnp.int32),
        sampled_target_logp=_stack_record_scalar(
            records_by_step, "sampled_target_logp", jnp.float32
        ),
        sampled_action_logp=_stack_record_scalar(
            records_by_step, "sampled_action_logp", jnp.float32
        ),
    )


def _record_without_action(
    *,
    state_tokens: TokenTree,
    state_token_mask: jax.Array,
    target_def_mask: jax.Array,
    target_choice: int,
    target_score_mask: bool,
    selected_def_index: int,
    step_case: int,
    sampled_target_logp: jax.Array,
    action_width: int,
) -> _SampleRecord:
    action_tokens, action_token_mask = _dummy_action_tree()
    return _SampleRecord(
        state_tokens=state_tokens,
        state_token_mask=state_token_mask,
        target_def_mask=target_def_mask,
        target_choice=jnp.asarray(target_choice, dtype=jnp.int32),
        target_score_mask=jnp.asarray(target_score_mask, dtype=jnp.bool_),
        selected_def_index=jnp.asarray(selected_def_index, dtype=jnp.int32),
        action_space_tokens=action_tokens,
        action_space_token_mask=action_token_mask,
        action_choice=_empty_action_choice(action_width),
        action_score_mask=jnp.asarray(False, dtype=jnp.bool_),
        step_case=jnp.asarray(step_case, dtype=jnp.int32),
        sampled_target_logp=sampled_target_logp,
        sampled_action_logp=jnp.asarray(0.0, dtype=jnp.float32),
    )


def _already_finished_record(action_width: int) -> _SampleRecord:
    state_tokens, state_token_mask = _dummy_state_tree()
    action_tokens, action_token_mask = _dummy_action_tree()
    return _SampleRecord(
        state_tokens=state_tokens,
        state_token_mask=state_token_mask,
        target_def_mask=jnp.zeros((1,), dtype=jnp.bool_),
        target_choice=jnp.asarray(-1, dtype=jnp.int32),
        target_score_mask=jnp.asarray(False, dtype=jnp.bool_),
        selected_def_index=jnp.asarray(0, dtype=jnp.int32),
        action_space_tokens=action_tokens,
        action_space_token_mask=action_token_mask,
        action_choice=_empty_action_choice(action_width),
        action_score_mask=jnp.asarray(False, dtype=jnp.bool_),
        step_case=jnp.asarray(CASE_ALREADY_FINISHED, dtype=jnp.int32),
        sampled_target_logp=jnp.asarray(0.0, dtype=jnp.float32),
        sampled_action_logp=jnp.asarray(0.0, dtype=jnp.float32),
    )


def _dummy_state_tree() -> tuple[TokenTree, jax.Array]:
    return _dummy_token_tree(STATE_TOKEN_FIELDS)


def _dummy_action_tree() -> tuple[TokenTree, jax.Array]:
    return _dummy_token_tree(ACTION_TOKEN_FIELDS)


def _dummy_token_tree(fields: tuple[str, ...]) -> tuple[TokenTree, jax.Array]:
    tokens = {
        field: jnp.asarray(
            [int(TOKEN_KIND.PAD) if field == "token_kind" else SENTINEL],
            dtype=jnp.int32,
        )
        for field in fields
    }
    return tokens, jnp.zeros((1,), dtype=jnp.bool_)


def _pad_state_tree(
    tokens: TokenTree, mask: jax.Array, length: int
) -> tuple[TokenTree, jax.Array]:
    return pad_token_tree(tokens, mask, length)


def _pad_action_tree(
    tokens: TokenTree, mask: jax.Array, length: int
) -> tuple[TokenTree, jax.Array]:
    return pad_token_tree(tokens, mask, length)


def _empty_action_choice(width: int) -> ActionChoiceTree:
    return {
        "candidate_index": jnp.asarray(0, dtype=jnp.int32),
        "left_mask": jnp.zeros((width,), dtype=jnp.bool_),
        "left_valid_mask": jnp.zeros((width,), dtype=jnp.bool_),
        "right_mask": jnp.zeros((width,), dtype=jnp.bool_),
        "right_valid_mask": jnp.zeros((width,), dtype=jnp.bool_),
    }


def _empty_batched_action_choice(batch_size: int, width: int) -> ActionChoiceTree:
    return {
        "candidate_index": jnp.zeros((batch_size,), dtype=jnp.int32),
        "left_mask": jnp.zeros((batch_size, width), dtype=jnp.bool_),
        "left_valid_mask": jnp.zeros((batch_size, width), dtype=jnp.bool_),
        "right_mask": jnp.zeros((batch_size, width), dtype=jnp.bool_),
        "right_valid_mask": jnp.zeros((batch_size, width), dtype=jnp.bool_),
    }


def _slice_tree(tree, index: int):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _slice_action_choice(choice: ActionChoiceTree, index: int) -> ActionChoiceTree:
    return _slice_tree(choice, index)


def _ensure_width(choice: ActionChoiceTree, width: int) -> ActionChoiceTree:
    for key in ("left_mask", "left_valid_mask", "right_mask", "right_valid_mask"):
        if int(choice[key].shape[0]) != width:
            raise TrainingError(
                f"{key} width {choice[key].shape[0]} differs from policy width {width}"
            )
    return choice


def _pad_bool_mask(mask: jax.Array, length: int) -> jax.Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if int(mask.shape[0]) > length:
        raise ValueError(
            f"cannot pad bool mask of length {mask.shape[0]} to shorter length {length}"
        )
    return jnp.pad(mask, (0, length - int(mask.shape[0])), constant_values=False)


def _max_mask_length(masks: list[jax.Array]) -> int:
    return max(1, *(int(mask.shape[0]) for mask in masks))


def _stack_action_choices(records: list[_SampleRecord], width: int) -> ActionChoiceTree:
    choices = [_ensure_width(record.action_choice, width) for record in records]
    return {
        key: jnp.stack([choice[key] for choice in choices], axis=0)
        for key in choices[0]
    }


def _stack_trees(trees: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    return {
        key: jnp.stack([tree[key] for tree in trees], axis=0)
        for key in trees[0]
    }


def _stack_record_scalar(
    records_by_step: list[list[_SampleRecord]], field: str, dtype
) -> jax.Array:
    return jnp.asarray(
        [
            [getattr(record, field) for record in step_records]
            for step_records in records_by_step
        ],
        dtype=dtype,
    )
