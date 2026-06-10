# REINFORCE Training Implementation Design

Status: planned
Phase: 3 of 3
Feeds implementation plan: yes

## Summary

This spec defines the third implementation phase for the REINFORCE system: a
runnable on-policy training loop over the symbolic rewrite kernel.

The trainer integrates the row environment and policy model. For each optimizer
update, it collects fresh rollout rows, stores immutable model inputs and sampled
choices, computes terminal rewards and per-column advantages, recomputes
differentiable target/action log probabilities from stored data, assembles a
column-normalized REINFORCE loss, applies one optimizer update, records metrics,
and optionally writes checkpoints.

## Goals

- Build row rollout orchestration over the row environment and policy model.
- Store rollout data as a rectangular table indexed by row and sample position.
- Preserve sample position as the reward and advantage column identity.
- Recompute differentiable logp from immutable stored arrays and choices.
- Compute rewards and advantages in `float64` through baseline calculation.
- Use column-normalized REINFORCE loss by default.
- Report metrics that expose STOP collapse, exact-empty probes, reward variance,
  score counts, and parameter updates.
- Save and load checkpoints with schema validation.

## Non-Goals

- Implementing the row environment or policy model.
- Adding supervised warm start, value heads, critics, replay, MCTS, GAE, or
  off-policy correction.
- Differentiating through rewrite application, action-space generation, reward,
  baseline, or advantage computation.
- Matching Gristmill's optimizer exactly in the first runnable version.
- Solving distributed or production-scale training.
- Preserving deprecated `gristmill_rl` training APIs.

## Inputs From Earlier Phases

The trainer depends on Phase 1 row environment APIs:

```text
PyRewriteStateRow.definition_masks()
PyRewriteStateRow.snapshots()
PyRewriteStateRow.query_action_spaces_for_row(target_choices, active_mask)
PyActionSpaceRow.entry_kinds()
PyActionSpaceRow.snapshots()
PyRewriteStateRow.validate_actions_for_row(action_space_row, action_choices, action_score_mask)
PyRewriteStateRow.apply_validated_actions_for_row(validated_action_row)
```

The trainer depends on Phase 2 policy APIs:

```text
tokenize_state_snapshot(snapshot)
tokenize_action_space_snapshot(snapshot)
sample_target(params, state_tokens, state_token_mask, def_mask, rng)
score_target(params, state_tokens, state_token_mask, def_mask, target_choice)
sample_action(params, state_tokens, state_token_mask, selected_def_index, action_space_tokens, action_space_token_mask, rng)
score_action(params, state_tokens, state_token_mask, selected_def_index, action_space_tokens, action_space_token_mask, action_choice)
```

## Public Trainer Contracts

The trainer should be organized around one update:

```text
collect_rollout_batch(policy, initial_states, rollout_config, rng)
  -> RolloutTable
  -> FinalColumnMetrics

compute_rewards(final_column_metrics, reward_config)
  -> reward[sample]

compute_advantages(reward, baseline_config)
  -> advantage[sample]

score_rollout(policy, rollout_table)
  -> target_logp[row, sample]
  -> action_logp[row, sample]

reinforce_loss(rollout_table, score_outputs, advantage, loss_config)
  -> loss
  -> diagnostics

train_update(policy, optimizer, batch, configs, rng)
  -> updated_policy
  -> updated_optimizer
  -> metrics
```

A CLI may wrap these operations, but the library contracts should be testable
without invoking the CLI.

## Rollout Configuration

Rollout config includes:

```text
batch_size
max_steps
seed
```

`batch_size` is the logical number of sample columns per optimizer update.

## Rollout Table

The rollout table is rectangular over row and sample axes. A sample position is
the column identity for reward assignment.

Target fields:

```text
target_state_tokens.<leaf>[row, sample, token, ...]
target_state_token_mask[row, sample, token]
target_def_mask[row, sample, def]
target_choice[row, sample]
target_score_mask[row, sample]
```

Action fields:

```text
action_state_tokens.<leaf>[row, sample, token, ...]
action_state_token_mask[row, sample, token]
selected_def_index[row, sample]
action_space_tokens.<leaf>[row, sample, token, ...]
action_space_token_mask[row, sample, token]
candidate_index[row, sample]
left_mask[row, sample, bit]
left_valid_mask[row, sample, bit]
right_mask[row, sample, bit]
right_valid_mask[row, sample, bit]
action_score_mask[row, sample]
```

Shared metadata:

```text
step_case[row, sample]
diagnostics[row, sample]
```

Score-mask mapping:

```text
case                  target_score_mask    action_score_mask
already finished      false                false
STOP                  true                 false
empty action space    true                 false
valid action          true                 true
```

Masked entries must contain safe padded values so scoring can run over
rectangular arrays. Masked entries contribute no logp, loss, or metric totals.

## Row Rollout Algorithm

For one row:

```text
1. Identify active and finished sample positions.
2. Snapshot active row states and tokenize target arrays.
3. Sample target choices with per-sample RNG streams.
4. Store target arrays, target choices, and target score masks.
5. Query row action spaces for selected non-STOP targets.
6. Mark STOP and exact-empty cases in the stored row.
7. Tokenize non-empty action-space snapshots into action arrays.
8. Sample action choices with per-sample RNG streams.
9. Store action arrays, action choices, and action score masks.
10. Validate sampled actions for the row.
11. Apply validated rewrites through the row environment.
12. Scatter all results back to stable sample positions.
```

There is no inner retry loop for exact-empty action spaces. An exact-empty target
choice is a scored target probe with no action score. The sample remains active
unless STOP, max steps, or another stopping rule ends it.

Target arrays must be captured before exact action-space generation mutates a
lazy definition mask. Action arrays must be captured before rewrite application.

## Randomness

Each sample position must receive an independent and reproducible random stream.

The mapping from sample position to RNG stream must not depend on private active
sample compaction. STOP, exact-empty, or already-finished samples must not change
later samples' random choices.

## Reward

Rewards are computed per sample column after rollout finishes.

The default reward is symbolic cost improvement:

```text
reward[s] = initial_log_flops[s] - final_log_flops[s]
```

This gives positive reward for cost reduction and is less sensitive to absolute
problem scale than `-final_log_flops`.

Reward calculation must use `float64` at least through baseline and advantage
calculation.

## Baseline And Advantage

The default baseline is the logical batch mean reward:

```text
baseline = mean_s reward[s]
advantage[s] = reward[s] - baseline
```

Optional advantage standardization may be supported:

```text
advantage = (advantage - mean) / (std + epsilon)
```

Baseline and advantage values are stop-gradient constants. The trainer must not
differentiate through reward, baseline, or advantage computation.

## Logp Recompute

Training recomputes:

```text
target_logp[row, sample] =
  log p(target_choice[row, sample] | target arrays[row, sample])

action_logp[row, sample] =
  log p(action_choice[row, sample] | action arrays[row, sample])
```

Only recomputed logp terms are differentiable. Sampled rollout logp may be
stored for diagnostics, but it must not be the gradient source.

## Loss

The default loss normalizes by sample columns, not by scored decisions:

```text
column_logp_sum[s] =
  sum over rows r (
    target_score_mask[r, s] * target_logp[r, s]
  + action_score_mask[r, s] * action_logp[r, s]
  )

loss =
  -mean over samples s (
    stop_gradient(advantage[s]) * column_logp_sum[s]
  )
```

Each rollout column receives one unit of batch weight regardless of how many
target/action terms it emitted. Metrics may report per-decision averages, but
the optimizer objective uses the column-normalized default.

## Optimizer Update Unit

One optimizer update uses one logical rollout batch:

```text
freeze current policy parameters
collect batch_size sample columns
compute rewards and advantages over the full logical batch
score stored rows
apply one optimizer update
discard rollout rows unless debugging/checkpoint config retains them
```

## Metrics

Each update should report:

```text
update_index
batch_size
max_steps
initial_log_flops_mean
final_log_flops_mean
final_log_flops_best
reward_mean
reward_std
advantage_mean
advantage_std
valid_action_count
stop_count
empty_action_space_count
finished_count
max_steps_count
target_score_count
action_score_count
loss
target_logp_mean
action_logp_mean
params_changed
```

Metrics should make stalled training visible, especially immediate STOP
collapse, zero reward variance, exact-empty target probes, zero score counts,
non-finite loss, and unchanged parameters.

## Checkpoints

Checkpoints should contain:

- policy model config;
- policy parameters;
- optimizer config and state;
- tokenizer/model schema version;
- rollout/training config;
- update index;
- RNG state or seed sequence;
- recent metrics.

Loading a checkpoint must validate schema version and fail clearly for unknown or
incompatible versions.

## Error Handling

Training should fail clearly when:

- a rollout row changes width;
- stored row data has a true score mask but missing inputs or choices;
- `action_score_mask=true` without `target_score_mask=true`;
- reward or advantage length differs from rollout table width;
- scoring returns non-finite logp for a real score term;
- loss is non-finite;
- reward computation returns non-finite values;
- no scored policy terms exist in a library training batch;
- checkpoint schema is unknown or incompatible.

For "no scored policy terms", a CLI may skip the optimizer update with an
explicit metric. Library code must make the behavior explicit.

## Testing Requirements

Rollout table tests:

- width-1 table stores the masks, choices, and immutable arrays for STOP,
  exact-empty, valid-action, and already-finished cases;
- multi-sample rows preserve sample positions across multiple steps;
- finished samples remain aligned in later rows;
- masked padded values do not affect row loss or metrics;
- stored arrays can be scored after row states advance.

Rollout orchestration tests:

- width-1 row rollout exercises all scalar cases through the row environment;
- multi-sample row rollout preserves scalar-equivalent behavior for each sample;
- target selection does not construct unselected action spaces;
- exact-empty action spaces produce target score only and no rewrite;
- valid actions are validated before row rewrite application;
- RNG assignment is stable when preceding samples finish.

Objective tests:

- deterministic fake logp and advantage arrays produce the expected loss;
- masked entries do not affect loss;
- column-normalized loss differs from decision-normalized loss on a crafted
  example;
- baseline and advantage are treated as stop-gradient constants.

Reward tests:

- improvement reward uses `initial_log_flops - final_log_flops`;
- rewards and advantages preserve small `float64` differences;
- batch-mean baseline produces zero-mean advantages.

Integration tests:

- width-1 row rollout with injected advantage changes policy parameters;
- tiny multi-sample row rollout computes finite loss;
- negative STOP bias makes immediate STOP rare in a deterministic fixture;
- checkpoint save/load round trips model and optimizer state.

CLI smoke tests:

- a tiny training run completes one update;
- metrics include reward variance, score counts, STOP counts, exact-empty counts,
  and `params_changed`;
- checkpoint output can be loaded for a second update.

## Exit Criteria

Phase 3 is complete when:

- the trainer can collect a width-1 rollout table over the real symbolic kernel;
- the trainer can collect a multi-sample rollout table while preserving sample
  alignment;
- rewards and advantages use `float64` through baseline calculation;
- training recomputes target/action logp from stored immutable arrays;
- loss normalization is column-based and covered by tests;
- one width-1 update changes policy parameters for nonzero advantage;
- one multi-sample update produces finite loss and metrics;
- metrics expose STOP count, exact-empty count, score counts, reward variance,
  and parameter-change status;
- checkpoint save/load supports continued training;
- tests do not import deprecated `gristmill_rl` training APIs.
