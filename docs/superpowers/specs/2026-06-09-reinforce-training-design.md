# REINFORCE Training Design

Status: planned
Supersedes: `2026-06-03-naive-reinforce-training-design.md` for the runnable
post-refactor training path
Depends on: `2026-06-09-reinforce-policy-model-design.md`,
`2026-06-09-reinforce-row-table-overview-design.md`,
`2026-06-09-reinforce-parallel-row-wrapper-design.md`
Feeds implementation plan: yes

## Summary

This spec defines the first runnable on-policy REINFORCE training loop over the
attention policy and symbolic rewrite kernel.

The trainer collects fresh rollout rows under a frozen policy, computes a reward
and advantage per sample column, recomputes differentiable target/action logp
from stored row data, assembles a normalized REINFORCE loss, applies one
optimizer update, reports diagnostics, and optionally writes checkpoints.

## Goals

- Define the on-policy training update unit.
- Use recomputed logp from immutable stored row data as the gradient source.
- Compute rewards and advantages with enough precision for symbolic cost
  differences.
- Make immediate STOP unlikely at initialization without adding trainer-side STOP
  mask modes.
- Pin down the default REINFORCE loss normalization.
- Support bounded scoring chunks independent from logical batch size.
- Define required metrics and checkpoint contents.

## Non-Goals

- Implementing supervised warm start in this first runnable REINFORCE loop.
- Adding a value head, critic, GAE, replay, MCTS, or off-policy correction.
- Differentiating through Rust rewrite application, action-space generation, cost,
  sampled choices, baseline, or advantage computation.
- Solving production-scale distributed training.
- Preserving deprecated `gristmill_rl` training APIs.

## Public Contracts

The trainer API should be organized around one update:

```text
collect_rollout_batch(policy, initial_states, rollout_config, rng)
  -> RolloutTable
  -> FinalColumnMetrics

compute_rewards(final_column_metrics, reward_config)
  -> reward[sample]

compute_advantages(reward, baseline_config)
  -> advantage[sample]

score_rollout(policy, rollout_table, score_config)
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

The trainer may expose a CLI around these operations, but the internal contracts
should be testable without invoking the CLI.

## Rollout Configuration

Rollout config includes:

```text
batch_size
max_steps
rollout_microbatch_size
seed
```

`batch_size` is the logical number of sample columns per optimizer update.

`rollout_microbatch_size` controls how many samples are stepped or collected
before returning progress. It must not change the logical baseline or optimizer
update unit.

## STOP Initialization

STOP is always part of the target distribution. The trainer does not provide a
separate STOP mask mode.

To reduce immediate STOP during early training, the policy model should
initialize the STOP head bias to a negative value, as defined in the policy model
spec. This keeps the rollout interface simple while making STOP effectively rare
when legal definition choices are available.

When no definition is allowed by the current `RewriteState.definition_mask()`,
STOP is the only unmasked target choice and the sample can terminate.

## Reward

Rewards are computed per sample column after rollout finishes.

The default reward should be improvement in symbolic cost:

```text
reward[s] = initial_log_flops[s] - final_log_flops[s]
```

This gives positive reward for cost reduction and is less sensitive to absolute
problem scale than `-final_log_flops`.

Reward calculation must use `float64` at least through baseline and advantage
calculation. The previous prototype observed `float32` collapse for small
symbolic cost differences; the runnable implementation should avoid that failure
mode.

Reward config may later add shaping terms, but v1 should keep the default
terminal reward simple.

## Baseline And Advantage

The default baseline is the logical batch mean reward:

```text
baseline = mean_s reward[s]
advantage[s] = reward[s] - baseline
```

The implementation may support optional advantage standardization:

```text
advantage = (advantage - mean) / (std + epsilon)
```

Baseline and advantage values are stop-gradient constants. The trainer must not
differentiate through reward, baseline, or advantage computation.

## Logp Recompute

Training recomputes:

```text
target_logp[t, s] =
  log p(target_choice[t, s] | target arrays[t, s])

action_logp[t, s] =
  log p(action_choice[t, s] | action arrays[t, s])
```

Only recomputed target/action logp terms are differentiable. Sampled rollout logp
may be stored for diagnostics, but it must not be used as the gradient source.

Scoring may run in chunks:

```text
target_score_chunk_size
action_score_chunk_size
```

Chunking must not change logp values.

## Loss Normalization

The default loss normalizes by sample columns, not by number of scored decisions:

```text
column_logp_sum[s] =
  sum over rows t (
    target_score_mask[t, s] * target_logp[t, s]
  + action_score_mask[t, s] * action_logp[t, s]
  )

loss =
  -mean over samples s (
    stop_gradient(advantage[s]) * column_logp_sum[s]
  )
```

This gives each rollout column one unit of batch weight regardless of how many
target/action terms it emitted. Metrics may additionally report per-decision
averages, but the optimizer objective should use the column-normalized default
unless a later design changes it.

Optional entropy or step penalties may be added later, but they are not required
for the first runnable implementation.

## Optimizer Update Unit

One optimizer update uses one logical rollout batch:

```text
freeze current policy parameters
collect batch_size sample columns
compute rewards and advantages over the full logical batch
score stored rows in chunks
accumulate gradients across chunks if needed
apply one optimizer update
discard rollout rows unless debugging/checkpoint config retains them
```

If gradient accumulation is needed, it must preserve the same objective as
scoring the full logical batch at once.

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
score_chunk_count
params_changed
```

Metrics should make stalled training visible, especially immediate STOP collapse,
zero reward variance, exact-empty target probes, and unchanged parameters.

## Checkpoints

Checkpoints should contain:

- policy model config;
- policy parameters;
- optimizer config and state;
- tokenizer/model schema version;
- rollout/training config;
- update index and RNG state or seed sequence;
- recent metrics.

Loading a checkpoint should validate schema version and fail clearly for unknown
or incompatible versions.

## Invariants

- Rollout rows are collected under the policy parameters being trained for that
  update.
- Rewards and advantages are per sample column.
- Advantage length equals row width.
- Score masks determine exactly which logp terms enter `column_logp_sum`.
- Masked target/action entries do not affect loss or metrics.
- Baseline and advantage are not differentiable.
- One logical batch produces one optimizer update.
- Scoring chunks are equivalent to full-batch scoring.

## Error Handling

Training should fail clearly when:

- rollout table width differs from reward or advantage length;
- scoring returns non-finite logp for a real score term;
- loss is non-finite;
- reward computation returns non-finite values;
- checkpoint schema is unknown;
- no scored policy terms exist in a training batch.

For "no scored policy terms", the CLI may skip the optimizer update with an
explicit metric, but library code should make the behavior explicit.

## Testing Requirements

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

- width-1 rollout with injected advantage changes policy parameters;
- a tiny multi-sample row rollout computes finite loss;
- score chunking matches unchunked scoring;
- negative STOP bias makes immediate STOP rare in a deterministic model fixture;
- checkpoint save/load round trips model and optimizer state.

CLI smoke tests:

- a tiny training run completes one update;
- metrics include reward variance, score counts, STOP counts, and
  `params_changed`;
- checkpoint output can be loaded for a second update.

## Acceptance Criteria

- The trainer can run one width-1 REINFORCE update over the real symbolic kernel.
- The trainer can run one multi-sample row update with recomputed target/action
  logp.
- Loss normalization is column-based and covered by tests.
- Rewards and advantages use `float64` through baseline calculation.
- STOP initialization is configured and STOP counts are visible in metrics.
- A checkpoint can be written and loaded for continued training.
