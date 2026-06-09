# REINFORCE Scalar Step Design

## Summary

This spec defines the scalar behavior for one sample and one rollout step. It is
the semantic source of truth for the row-level wrapper.

The scalar shape is:

```text
one sample at step t -> same sample at step t + 1
```

During that update, the scalar step also produces the per-sample data that will
be placed into the current rollout row for later scoring.

## Goals

- Define the exact target/action order for one sample step.
- Define the first-version policy model architecture behind target/action
  sampling and scoring.
- Define the scalar REINFORCE objective used to train that model.
- Ensure target selection never constructs or inspects action spaces.
- Define scoring meaning for valid action, STOP, empty action space, and
  already-finished samples.
- Store enough per-sample row data to recompute target/action logp later.
- Use target/action score masks so scalar training is the width-1 version of
  row training.
- Keep scalar behavior independent of any row-level execution mechanics.

## Non-Goals

- Defining row-level parallel execution.
- Defining row-level padding or batching mechanics beyond the width-1 scalar
  case.
- Defining the final model architecture.
- Running CCSD-scale batches.
- Solving warm start.
- Keeping the previous transformer/reinforce prototype API.

## Scalar API Shape

The scalar step has this conceptual shape:

```text
step_sample(sample_t) -> sample_t_plus_1, data for this sample in row t
```

The returned row data means the target/action inputs, choices, status, and
sample-column identity needed to fill this sample's position in the current
rollout row. It is not a training gradient source.

For scalar training, this is the width-1 version of the row-table format:

```text
target_input[t, sample 0]
target_choice[t, sample 0]
target_score_mask[t, sample 0]

action_input[t, sample 0]
action_choice[t, sample 0]
action_score_mask[t, sample 0]
```

If a score mask is false, the corresponding input and choice are ignored by
loss and metrics. Masked values still need to be safe padded values.

Policy scoring is separate:

```text
score_target(stored target input, stored target choice) -> target logp
score_action(stored action input, stored action choice) -> action logp
```

Training uses these scoring functions to recompute logp from stored row data.

## Target Stage

For an active sample, the first policy stage is target selection:

```text
TargetInput -> STOP or def_index
```

`TargetInput` is derived from the current sample state. It may include state
tokens, definition metadata, target masks, and STOP legality.

It must not include action-space data. The scalar target stage must not call
action-space generation.

The first-version target model is:

```text
state tokens + definition metadata + target legality mask
  -> attention context
  -> logits over STOP plus definition indices
```

The target logits are masked by target legality before sampling and before
scoring. STOP is represented as one target output position, not as an action.

Target scoring recomputes:

```text
target_logp = log_softmax(masked_target_logits)[stored target choice]
```

The model must not create logits by first expanding action spaces for every
definition. That would violate the target/action split.

## Action-Space Stage

If target selection chooses a definition, the environment builds an action
space only for that selected definition:

```text
current sample state + selected def_index -> action space or empty
```

This is environment work, not policy scoring.

## Action Stage

If the selected definition has a non-empty action space, the second policy stage
is action selection:

```text
ActionInput -> candidate_index, left_mask, right_mask
```

`ActionInput` is derived from:

- current sample state;
- selected `def_index`;
- selected definition's action space.

The resulting action choice is then applied by the environment to produce the
next sample state.

The first-version action model is:

```text
state tokens + selected def_index + selected action-space tokens
  -> attention context
  -> candidate logits
  -> left-mask logits
  -> right-mask logits
```

The action distribution is autoregressive:

```text
candidate_index
left_mask conditioned on candidate_index
right_mask conditioned on candidate_index and left_mask
```

`candidate_index` is sampled from logits over the selected definition's action
space. `left_mask` and `right_mask` are sampled from logits over the selected
candidate's left and right term positions. Invalid padded positions are masked
out. The sampled action must be representable as:

```text
candidate_index, left_mask, right_mask
```

Action scoring recomputes the same conditional distribution:

```text
candidate_logp = log p(stored candidate_index | ActionInput)
left_mask_logp = log p(stored left_mask | ActionInput, stored candidate_index)
right_mask_logp =
  log p(stored right_mask | ActionInput, stored candidate_index, stored left_mask)

action_logp = candidate_logp + left_mask_logp + right_mask_logp
```

This architecture uses attention over the state and selected action-space
tokens, then directly decodes logits over the selected action space. It does not
use the old token-decoder grammar where choices live as detached extra tokens.

## Step Cases

### Already Finished

If the input sample is already finished:

- no target choice is made;
- no action choice is made;
- no target or action score is produced;
- the sample remains finished in the next step.

Its row entry has `target_score_mask = false` and
`action_score_mask = false`.

### STOP

If target selection chooses STOP:

- store the target input and STOP choice for scoring;
- store no action input or action choice;
- mark the sample terminal for the next step.

STOP is scored by the target scorer.

Its row entry has `target_score_mask = true` and
`action_score_mask = false`.

### Empty Action Space

If target selection chooses a definition and the selected definition has an
empty action space:

- store the target input and selected definition for scoring;
- store no action input or action choice;
- update the sample so that this definition is unavailable in the next target
  selection from this state;
- continue the sample unless another stopping rule applies.

The target choice is scored. No action score exists.

There is no retry target selection inside the same scalar step.

Its row entry has `target_score_mask = true` and
`action_score_mask = false`.

### Valid Action

If target selection chooses a definition with a non-empty action space:

- store the target input and selected definition for scoring;
- build the action input for the selected definition and action space;
- sample the action choice;
- store the action input and action choice for scoring;
- apply the selected action to produce the next sample state.

Both the target choice and action choice are scored.

Its row entry has `target_score_mask = true` and
`action_score_mask = true`.

## Scalar Step Order

The active-sample order is:

```text
1. Build TargetInput from the current sample state.
2. Sample STOP or def_index.
3. Store target scoring data.
4. If STOP, mark terminal and end the step.
5. Build action space for the selected def_index only.
6. If action space is empty, refine target legality and end the step.
7. Build ActionInput from current state, def_index, and action space.
8. Sample action choice.
9. Store action scoring data.
10. Apply the action to produce the next sample state.
```

This order is the scalar contract. The later row-level wrapper may batch or
filter work internally, but it must preserve these semantics for each sample.

## Recompute-Based Scoring

Rollout may compute logp for metrics, but rollout logp is not used as the
gradient source.

Training recomputes:

```text
target_logp = score_target(stored target input, stored target choice)
action_logp = score_action(stored action input, stored action choice)
```

The REINFORCE loss includes only real scores:

- active STOP sample: target score only;
- active empty-action sample: target score only;
- active valid-action sample: target score and action score;
- already-finished sample: no score.

Reward or advantage is assigned by sample column.

For a scalar width-1 rollout:

```text
loss =
  -advantage[sample 0] *
  sum over rows (
    target_score_mask[t, 0] * target_logp[t, 0]
  + action_score_mask[t, 0] * action_logp[t, 0]
  )
```

For example:

```text
row 0: valid action        target mask true   action mask true
row 1: empty action space  target mask true   action mask false
row 2: valid action        target mask true   action mask true
row 3: STOP                target mask true   action mask false
```

No later finished row is required in the scalar width-1 rollout unless a test
explicitly wants to exercise masked finished entries.

## Scalar REINFORCE Algorithm

The scalar trainer uses the width-1 rollout table. It repeatedly applies the
scalar step to one sample column until the sample becomes terminal or a rollout
limit is reached:

```text
row 0 -> row 1 -> row 2 -> ... -> terminal row
```

After rollout, the trainer computes a scalar reward for the sample column from
the configured reward evaluator. The reward and advantage are not
differentiable. The first scalar implementation may use a baseline value of
zero, so advantage equals reward, or an externally supplied advantage in
deterministic tests.

For one sample column, the REINFORCE loss is:

```text
loss =
  -advantage *
  sum over rows t (
    target_score_mask[t, 0] * target_logp[t, 0]
  + action_score_mask[t, 0] * action_logp[t, 0]
  )
```

The differentiable quantities are the recomputed target and action logp terms.
The trainer must not backpropagate through:

- sampled choices;
- rewrite application;
- action-space generation;
- reward or advantage computation.

For a scalar implementation, one optimizer update may use one or more completed
sample columns. If more than one column is used, this is still just multiple
width-1 scalar trajectories accumulated before the update; row-level parallel
execution is not required.

## Relationship To Rows

The scalar step produces one sample's contribution to a row.

The row-level wrapper will update:

```text
row t -> row t + 1
```

by applying scalar semantics to each sample position. The wrapper may use
private mechanics for efficiency, but higher-level code should still see whole
rows with stable sample positions.

## Acceptance Criteria

- Target input construction and target sampling do not call action-space
  generation.
- Already-finished samples emit no target or action score.
- STOP emits target scoring data, emits no action scoring data, and makes the
  sample terminal.
- Empty action space emits target scoring data, emits no action scoring data,
  and makes the selected definition unavailable for the next target selection.
- Valid action emits target and action scoring data and applies one rewrite.
- Scalar training uses target/action score masks to include exactly the real
  target and action logp terms.
- Training recomputes target/action logp from stored row data and sampled
  choices.
- Target scoring uses masked logits over STOP plus definition indices.
- Action scoring uses the autoregressive candidate, left-mask, right-mask
  distribution over the selected action space.
- The scalar REINFORCE loss uses recomputed logp terms weighted by the sample
  column advantage.
- The scalar design uses the overview vocabulary: sample, row, and column.
