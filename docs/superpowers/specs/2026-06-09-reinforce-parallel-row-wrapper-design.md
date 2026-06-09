# REINFORCE Parallel Row-Wrapper Design

## Summary

This spec defines the parallel wrapper around the scalar REINFORCE sample step.
It does not introduce new public rollout terminology. The public vocabulary
remains:

- A sample is one rollout instance.
- A row is all samples at the same step.
- A column is one sample from the first step to the last step.

The scalar spec defines:

```text
step_sample(sample_t) -> sample_t_plus_1, data for this sample in row t
```

The parallel wrapper lifts that to:

```text
step_row(row_t) -> row_t_plus_1, stored_row_t
```

The row wrapper is reachable because `stored_row_t` is already rectangular:

```text
target_input[t, sample]
target_choice[t, sample]
target_score_mask[t, sample]

action_input[t, sample]
action_choice[t, sample]
action_score_mask[t, sample]
```

The wrapper may use private mechanics to avoid wasted work, but callers only see
whole rows with stable sample positions.

## Goals

- Define the public contract for updating one row by one step.
- Preserve scalar sample-step semantics for every sample position.
- Keep row width and sample-column alignment stable.
- Store target/action tables and score masks for each row.
- Define row scoring and loss assembly from stored row data.
- Keep any row execution mechanics private.

## Non-Goals

- Choosing the concrete internal scheduling strategy for row execution.
- Choosing low-level padding values for masked entries.
- Defining model architecture.
- Solving warm start.
- Differentiating through rewrite application or action-space generation.
- Changing scalar behavior.

## Row Step Contract

The public row update shape is:

```text
step_row(row_t) -> row_t_plus_1, stored_row_t
```

`row_t` contains all sample positions at step `t`.

`row_t_plus_1` contains the same sample positions at step `t + 1`.

`stored_row_t` contains the target/action inputs, choices, and masks needed to
score the policy choices made while moving from `row_t` to `row_t_plus_1`.

The row wrapper must preserve these invariants:

- the row width is unchanged;
- sample position `s` in the input row corresponds to sample position `s` in the
  output row;
- each sample position behaves according to the scalar sample-step spec;
- masked target/action entries do not contribute to loss, metrics, or score
  totals;
- masked target/action data uses safe padded values.

## Scalar Equivalence

The row wrapper is semantically equivalent to applying the scalar step to each
sample position:

```text
for each sample position s:
  step_row(row_t)[s] == step_sample(row_t[s])
```

The equality is semantic, not necessarily byte-for-byte for masked padded data.
For masked entries, only the masks and ignored-score behavior matter.

The policy parameters are shared across all samples. Each sample position has
its own policy choices for the current step. One sample's STOP, empty action
space, or finished status must not change another sample's scalar behavior.

## Stored Row Format

For every row `t`, the wrapper stores:

```text
target_input[t, sample]
target_choice[t, sample]
target_score_mask[t, sample]

action_input[t, sample]
action_choice[t, sample]
action_score_mask[t, sample]
```

The mask mapping is:

```text
case                  target_score_mask    action_score_mask
already finished      false                false
STOP                  true                 false
empty action space    true                 false
valid action          true                 true
```

This is the same mapping as the scalar width-1 case. The parallel wrapper only
adds more sample columns.

## Row Scoring

The scorer consumes one stored row:

```text
score_row(stored_row_t)
  -> target_logp[t, sample]
  -> action_logp[t, sample]
```

The row loss uses score masks and sample-column advantages:

```text
row_loss[t] =
  -sum over samples s (
    advantage[s] *
    (
      target_score_mask[t, s] * target_logp[t, s]
    + action_score_mask[t, s] * action_logp[t, s]
    )
  )
```

Across the rollout table, training sums or averages the real masked score terms
according to the trainer's chosen normalization rule. The normalization rule
must not give weight to masked target/action entries.

## STOP, Empty, And Finished In A Row

Different samples in the same row may be in different cases.

Example:

```text
            sample 0          sample 1          sample 2          sample 3
row t       valid action      STOP              empty action      finished

target mask true              true              true              false
action mask true              false             false             false
```

The next row keeps the same sample positions:

```text
            sample 0          sample 1          sample 2          sample 3
row t + 1   updated sample    terminal sample   updated sample    finished
```

The concrete representation of terminal or finished sample state is not part of
this public contract. The scoring behavior is fully determined by the masks.

## Relationship To Policy And Trainer

The trainer calls row-level rollout and row-level scoring:

```text
step_row(row_t) -> row_t_plus_1, stored_row_t
score_row(stored_row_t) -> target/action logp per sample
```

The policy still exposes target and action behavior:

```text
target: TargetInput -> STOP or def_index
action: ActionInput -> candidate_index, left_mask, right_mask
```

The row wrapper does not merge target and action into one policy decision. It
preserves the scalar target/action order for each sample.

## Private Row Mechanics

The public API must not expose extra rollout concepts beyond sample, row, and
column.

The row wrapper may privately avoid work for samples that cannot emit a score,
and may privately organize target and action work in any way that preserves the
row contract. These choices are not visible to the trainer or policy API.

## Acceptance Criteria

- Width-1 row stepping matches the scalar sample step.
- Multi-sample row stepping matches independent scalar sample steps for each
  sample position.
- Row width and sample positions remain stable across steps.
- A row containing valid action, STOP, empty action space, and already-finished
  samples produces the expected target/action score masks.
- Row scoring includes only masked-in target/action logp terms.
- Masked padded values do not affect loss or metrics.
- Target selection still does not construct or inspect action spaces before a
  target definition is chosen.
- The parallel wrapper spec uses only the public vocabulary: sample, row, and
  column.

