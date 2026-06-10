# REINFORCE Scalar Step Design

Status: planned
Supersedes: earlier scalar portions of `2026-06-03-naive-reinforce-training-design.md`
Depends on: `2026-06-09-reinforce-policy-model-design.md`
Feeds implementation plan: yes

## Summary

This spec defines the authoritative behavior for one sample step in the
REINFORCE rollout. The parallel row wrapper must preserve these semantics for
every sample position.

The scalar step is:

```text
step_sample(sample_t) -> sample_t_plus_1, stored sample data for row t
```

The policy model owns how target/action choices are sampled and scored. This spec
owns when those policy calls happen, when Rust action spaces are generated, how
STOP and empty action spaces behave, and what data is stored for later training.

## Goals

- Define the exact one-sample rollout order.
- Keep target selection and action selection separate.
- Ensure target selection never constructs unselected action spaces.
- Define the four scalar cases: already finished, STOP, empty action space, valid
  action.
- Store immutable width-1 row data for recomputed logp.
- Define scalar score-mask semantics used by row and training specs.

## Non-Goals

- Defining the attention model architecture.
- Defining row-level parallel execution.
- Choosing padding values for row batches.
- Defining reward, advantage, optimizer, or checkpoint behavior.
- Differentiating through rewrite application, action-space generation, rewards,
  or sampled choices.
- Preserving the previous transformer/reinforce prototype API.

## Dependencies

The scalar step depends on these contracts:

- target/action array and choice shapes from the policy model spec.
- `RewriteState.definition_mask()`, `action_space_for_def`, and
  `step_with_space` from the rewrite-state API.

## Public Contract

The scalar public contract is:

```text
step_sample(sample_t, policy, rng, rollout_config)
  -> sample_t_plus_1
  -> StoredSampleStep
```

`StoredSampleStep` is the width-1 form of the row table arrays:

```text
StoredSampleStep {
  target_state_tokens
  target_state_token_mask
  target_def_mask
  target_choice
  target_score_mask

  action_state_tokens
  action_state_token_mask
  selected_def_index
  action_space_tokens
  action_space_token_mask
  action_choice
  action_score_mask

  step_case
  diagnostics
}
```

Fields named `_tokens` are token pytrees in the policy-model sense: rectangular
JAX-compatible leaves plus explicit padding masks, not necessarily raw integer
token-id vectors.

If a score mask is false, the corresponding arrays and choice are ignored by loss
and metrics. They must still contain safe padded values if the implementation
stores rectangular arrays eagerly.

## Sample State

A scalar sample contains:

```text
Sample {
  rewrite_state
  finished
  terminal_reason
  step_index
  rollout_metadata
}
```

`finished` means the sample emits no new policy decisions. `terminal_reason`
records why a sample became finished, such as STOP, max steps, no target
available, or environment error. The concrete representation can evolve, but the
step cases below are public behavior.

## Algorithm

For an already-finished sample:

```text
1. Return the same finished sample.
2. Store target_score_mask=false and action_score_mask=false.
```

For an active sample:

```text
1. Build immutable target arrays from the current sample state.
2. Sample target_choice from the target distribution.
   target_choice = -1 means STOP; otherwise selected_def_index = target_choice.
3. Store target arrays, target choice, and target_score_mask=true.
4. If target_choice == -1, mark the sample terminal and end the step.
5. Call action_space_for_def(selected_def_index) for the selected definition only.
6. If the action space is empty, keep Rust's refined definition mask and end the step.
7. Build immutable action arrays from the current state, selected_def_index, and action space.
8. Sample candidate_index, left_mask, and right_mask from the action distribution.
9. Store action arrays, action choice, and action_score_mask=true.
10. Apply the action through step_with_space to produce the next sample state.
```

There is no inner retry loop. If a selected definition has an empty action space,
the next target selection happens in the next row.

## Step Cases

```text
case                  target_score_mask    action_score_mask
already finished      false                false
STOP                  true                 false
empty action space    true                 false
valid action          true                 true
```

### Already Finished

If the input sample is already finished:

- no target choice is sampled;
- no action choice is sampled;
- no target or action logp contributes to training;
- the sample remains finished.

### STOP

If target selection chooses STOP:

- the target choice is stored and scored;
- no action space is generated;
- no action choice exists;
- the sample becomes terminal.

STOP is always part of the target distribution. It is not gated by rollout config,
the definition mask, or a separate trainer-side legality mask. The model makes
immediate STOP rare through its configured negative STOP bias.

### Empty Action Space

If target selection chooses a definition whose exact action space is empty:

- the target choice is stored and scored;
- no action input or action choice exists;
- `action_space_for_def` refines the selected definition's mask to false when
  the kernel reports exact empty;
- the sample continues unless another stopping rule applies.

This case is a scored target probe with no environment rewrite. The training spec
may add a step penalty or other reward shaping, but the scalar mask behavior is
fixed here.

### Valid Action

If target selection chooses a definition with a non-empty action space:

- the target choice is stored and scored;
- the action choice is stored and scored;
- `step_with_space` applies the rewrite;
- the sample remains active unless max steps or another stopping rule finishes it
  after the step.

## Immutable Storage Requirements

Target arrays must be captured before exact action-space generation for the
selected definition mutates the lazy definition mask.

Action arrays must be captured before applying the rewrite. They must contain
plain immutable data sufficient to score the selected action later. They must not
depend on the live `ActionSpace` handle remaining valid.

Rollout may store sampled logp for diagnostics, but training must recompute
differentiable target/action logp from stored arrays and choices.

## Scalar REINFORCE Loss Shape

For one sample column, scoring recomputes:

```text
target_logp[t] =
  log p(stored target_choice[t] | stored target arrays[t])

action_logp[t] =
  log p(stored action_choice[t] | stored action arrays[t])
```

The scalar column logp sum is:

```text
column_logp_sum =
  sum over rows t (
    target_score_mask[t] * target_logp[t]
  + action_score_mask[t] * action_logp[t]
  )
```

The training spec owns advantage weighting and normalization.

## Invariants

- Active scalar target selection builds no unselected action spaces.
- `target_score_mask=true` exactly when a target choice was sampled.
- `action_score_mask=true` exactly when an action distribution existed and an
  action choice was sampled.
- Empty action space does not apply a rewrite.
- STOP does not generate an action space.
- Stored arrays are immutable snapshots.
- Already-finished samples produce no score terms.

## Error Handling

The scalar step should fail clearly when:

- target sampling returns an illegal choice;
- `def_index` is out of range;
- action sampling returns a candidate or bit mask illegal for the generated
  action space;
- `step_with_space` rejects the sampled action;
- required stored data is missing for a true score mask.

## Testing Requirements

- Already-finished input emits no target/action scores.
- STOP emits target score data, emits no action score data, and marks the sample
  terminal.
- Empty action space emits target score data, emits no action score data, refines
  the target mask, and leaves the sample active.
- Valid action emits target and action score data and applies one rewrite.
- Target input construction and sampling do not call action-space generation.
- Action input construction happens only for the selected non-empty action space.
- Stored target/action data can be rescored after the live `RewriteState` has
  moved on.

## Acceptance Criteria

- Width-1 rollout can execute all four step cases.
- Scoring recomputes target/action logp from stored immutable data.
- Scalar mask semantics match the row-table mask mapping.
- Scalar behavior is complete enough for width-1 REINFORCE tests with injected
  advantages.
