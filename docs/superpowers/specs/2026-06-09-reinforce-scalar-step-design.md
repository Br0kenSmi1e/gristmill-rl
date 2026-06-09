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
- Ensure target selection never constructs or inspects action spaces.
- Define scoring meaning for valid action, STOP, empty action space, and
  already-finished samples.
- Store enough per-sample row data to recompute target/action logp later.
- Keep scalar behavior independent of any row-level execution mechanics.

## Non-Goals

- Defining row-level parallel execution.
- Choosing the concrete padded-row representation for STOP, empty action space,
  or finished samples.
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

## Step Cases

### Already Finished

If the input sample is already finished:

- no target choice is made;
- no action choice is made;
- no target or action score is produced;
- the sample remains finished in the next step.

The concrete way this appears inside a row is deferred.

### STOP

If target selection chooses STOP:

- store the target input and STOP choice for scoring;
- store no action input or action choice;
- mark the sample terminal for the next step.

STOP is scored by the target scorer.

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

### Valid Action

If target selection chooses a definition with a non-empty action space:

- store the target input and selected definition for scoring;
- build the action input for the selected definition and action space;
- sample the action choice;
- store the action input and action choice for scoring;
- apply the selected action to produce the next sample state.

Both the target choice and action choice are scored.

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
- Training recomputes target/action logp from stored row data and sampled
  choices.
- The scalar design uses the overview vocabulary: sample, row, and column.
