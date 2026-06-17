# Streamed REINFORCE Policy Refactor Design

## Goal

Replace the current rollout-table-centered REINFORCE stack with a breaking, clean API built around:

- policy sampling functions that only sample choices;
- policy scoring functions that are the sole log-probability authority;
- action-space batching as `vector of scalar inputs -> local padding -> vmap(scalar policy function)`;
- streamed REINFORCE gradients accumulated during rollout instead of materializing `RolloutTable`.

This is a breaking refactor. The implementation must remove historical compatibility paths rather than preserve old aliases, wrappers, or hidden table-based training routes.

## Non-Goals

- Do not preserve the old `(choice, logp)` return contract for `sample_target` or `sample_action`.
- Do not keep `RolloutTable` as a public or internal training dependency.
- Do not add a public `PolicyGradientBatch` class.
- Do not add gradient microbatching in v1.
- Do not design checkpoint migration for old parameter pytrees. Checkpoint tests should be updated to the new parameter shape; old checkpoints may fail naturally.
- Do not address the separate full-attention action-space memory issue here.

## Current Problems

`PolicyConfig.max_candidates` and `PolicyConfig.max_side_terms` are global model caps, not real policy semantics. The tokenizer serializes all candidates and all side terms, but the action head only scores candidate indices `0..max_candidates-1` and side term positions `0..max_side_terms-1`. This can ignore valid candidates and make large candidates unrepresentable.

These caps exist because the current implementation uses fixed-shape action heads and padded rollout tables. They should be removed. Candidate and side-term capacities must come from the current action-space inputs, with padding only in the local batching adapter.

The current trainer also materializes a padded `RolloutTable`, then replays `score_rollout` under `jax.value_and_grad`. This makes memory scale with the worst padded rollout shape and performs a second full scoring pass.

## Milestone 1: Policy Model Cleanup

### Public Policy API

Sampling functions return choices only:

```python
def sample_target(
    params,
    state_tokens,
    state_token_mask,
    def_mask,
    rng,
) -> jax.Array:
    ...

def sample_action(
    params,
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
    rng,
) -> ActionChoiceTree:
    ...
```

Scoring functions remain the only log-probability APIs:

```python
def score_target(params, state_tokens, state_token_mask, def_mask, target_choice):
    ...

def score_action(
    params,
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
    action_choice,
):
    ...
```

Tests that need sampled log-probabilities should replay explicitly:

```python
choice = sample_action(...)
logp = score_action(..., choice)
```

### Policy Config And Params

Remove these fields from `PolicyConfig`:

```python
max_candidates
max_side_terms
```

Remove these cap-shaped action parameters:

```python
candidate_slot_bias
left_position_bias
right_position_bias
```

The replacement action heads must be shape-independent. Candidate and term identities are already represented through token fields, embeddings, and pooled encoded representations. If scalar biases are useful, they must be shared parameters such as `candidate_bias`, `left_bias`, and `right_bias`, not per-global-slot vectors.

### Action-Space Batching Contract

The policy stack should separate scalar policy semantics from batching:

```text
list of scalar action-space inputs
    -> local padding/stacking adapter
    -> vmap(scalar sample_action or score_action)
```

The scalar policy functions operate on one already-padded row of input arrays. They do not own global padding and do not read model-config caps.

The batching adapter chooses local shapes for the current active group. For example:

```text
state_tokens:        [active, local_state_token_len]
action_tokens:       [active, local_action_token_len]
candidate slots:     local to current action-space batch
left term slots:     local to current action-space batch
right term slots:    local to current action-space batch
```

The tokenizer continues to preserve all candidates and all side terms. Missing local batch entries are masked false. Padding is ephemeral for the current `vmap` call; it is not persisted as a rollout table and is not encoded in `PolicyConfig`.

### Action Choice Shape

`ActionChoiceTree` keeps the existing semantic fields:

```python
{
    "candidate_index": int32[],
    "left_mask": bool[local_left_capacity],
    "left_valid_mask": bool[local_left_capacity],
    "right_mask": bool[local_right_capacity],
    "right_valid_mask": bool[local_right_capacity],
}
```

The side mask widths are local batch capacities, not global model caps. `candidate_index` remains the real candidate index expected by Rust validation. Rust row validation already trims padded masks through the valid masks, so this remains compatible with the row environment.

### Milestone 1 Acceptance Criteria

- `sample_target` and `sample_action` return choices only.
- `score_target` and `score_action` replay sampled choices and produce finite log-probabilities for valid fixtures.
- `PolicyConfig` no longer has `max_candidates` or `max_side_terms`.
- Policy params no longer include `candidate_slot_bias`, `left_position_bias`, or `right_position_bias`.
- Policy action tests cover action spaces with candidate and side-term counts exceeding the old caps.
- Policy batching tests use local padding followed by `vmap`.
- Policy tests pass after update.
- Reinforce/training tests may be temporarily broken after this milestone because the trainer still depends on old policy and table assumptions.

## Milestone 2: Streamed REINFORCE Trainer

### Public Reinforce API

`train_update` becomes:

```python
def train_update(...) -> tuple[TrainState, UpdateMetrics]:
    ...
```

The public reinforce surface keeps trainer and configuration concepts, but removes table-objective APIs:

Remove from public exports:

```text
RolloutTable
ScoreOutputs
LossDiagnostics
collect_rollout_batch
score_rollout
reinforce_loss
```

No deprecation wrappers or legacy aliases should remain.

### Streamed Gradient Collection

For one optimizer update, freeze params for the whole rollout batch. The rollout loop keeps the current row-environment flow and RNG grid, but scores each sampled decision immediately.

For target decisions:

```python
target_choice = sample_target(...)
target_logp, target_grad_logp = batched_target_logp_and_grad(
    params,
    state_tokens,
    state_token_mask,
    target_def_mask,
    target_choice,
)
```

For valid action decisions:

```python
action_choice = sample_action(...)
action_logp, action_grad_logp = batched_action_logp_and_grad(
    params,
    state_tokens,
    state_token_mask,
    selected_def_index,
    action_space_tokens,
    action_space_token_mask,
    action_choice,
)
```

The batched gradient helpers are conceptually:

```python
jax.jit(jax.vmap(jax.value_and_grad(score_target, argnums=0), ...))
jax.jit(jax.vmap(jax.value_and_grad(score_action, argnums=0), ...))
```

Only `params` are differentiated. Choices, rewards, baselines, advantages, token arrays, row transitions, and selected definition indices are treated as stopped values.

### Per-Sample Accumulation

For each sample `i`, accumulate through the rollout:

```text
S_i = sum_t log pi_theta(decision_it | state_it)
G_i = sum_t grad log pi_theta(decision_it | state_it)
R_i = initial_log_flops_i - final_log_flops_i
```

Implementation-wise this means:

```python
trajectory_logp[i] += step_logp[i]
trajectory_grad_logp[i] += step_grad_logp[i]
```

where `trajectory_grad_logp` is a params-shaped pytree with leading sample axis on floating leaves:

```text
leaf shape: [batch_size, *param_leaf_shape]
```

Each batched `value_and_grad` call returns a temporary gradient pytree with leading active axis:

```text
leaf shape: [active_count, *param_leaf_shape]
```

The collector scatters/adds those active rows into the per-sample trajectory accumulator and then drops the temporary. It must not store per-step token tables, choices, or gradient pytrees.

### Gradient And Metrics

Advantages use the existing batch-baseline behavior:

```text
A_i = R_i - mean_j R_j
```

If `BaselineConfig.standardize` is enabled, standardize the stopped advantages exactly as the current `compute_advantages` does.

The gradient passed to Optax is:

```text
grad_loss = -mean_i stop(A_i) * G_i
```

The REINFORCE surrogate is a diagnostic:

```text
surrogate_loss = -mean_i stop(A_i) * S_i
```

Reported objective metrics must be reward-based, not surrogate-based:

```text
reward_mean = mean_i R_i
reward_std = std_i R_i
reward_stderr = reward_std / sqrt(batch_size)
objective_loss_mean = -reward_mean
objective_loss_stderr = reward_stderr
```

`UpdateMetrics` must make the distinction explicit. If a field named `loss` remains for CLI compatibility, it must mean the true objective loss estimate `objective_loss_mean`, not the surrogate.

### Memory Behavior

Removed:

- persistent padded rollout tables across all steps and samples;
- replay scoring pass over the whole rollout table.

Kept:

- current-step active token batches;
- temporary current-decision batched gradient pytrees;
- per-sample trajectory-gradient accumulators.

V1 does not include microbatching or sufficient-stat-only gradient accumulation. If temporary `[active, params]` gradients or `[batch, params]` trajectory accumulators become the next bottleneck, that should be a follow-up issue with profiling evidence.

### Milestone 2 Acceptance Criteria

- `train_update` returns `(TrainState, UpdateMetrics)`.
- Training does not materialize or depend on `RolloutTable`.
- Public reinforce exports do not include old table-objective APIs.
- Sampled choices are treated as stopped values.
- Params are fixed throughout one rollout batch and Optax is applied once after gradient accumulation.
- One-step and multi-step tests verify streamed `sum_t grad_logp_t` behavior.
- Advantage-weighting and batch-baseline tests verify `grad_loss = -mean_i A_i * G_i`.
- Metrics tests verify reward-based objective reporting and surrogate-loss diagnostics.
- Full test suite passes after the streamed trainer milestone.

## Testing Strategy

Policy tests should be updated first. They should assert literal sampling/scoring separation, dynamic action capacities, and local batch padding before `vmap`.

Training tests should then verify streamed-gradient semantics. Direct score-based scalar fixtures are preferred over preserving the table implementation as an oracle. If a temporary implementation-local oracle is used during development, it must be removed before the streamed trainer milestone is considered complete. The final code and public API must not retain `RolloutTable`.

Checkpoint tests should be updated to the new parameter shape. No old-checkpoint migration test is required.

## Sequencing

1. Implement Milestone 1 and restore policy tests.
2. Implement Milestone 2 and restore reinforce/training tests.
3. Remove stale tests and exports tied to the old public API.
4. Confirm no compatibility wrappers, old aliases, or hidden table-training routes remain.
