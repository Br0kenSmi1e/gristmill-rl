# REINFORCE Row-Table Overview Design

## Summary

This spec defines the public mental model for the next REINFORCE refactor. The
goal is to keep the policy/trainer boundary simple while leaving room for
efficient row-level execution later.

The public rollout shape is a table:

```text
              sample 0    sample 1    sample 2    ...    sample N
row 0         ...         ...         ...
row 1         ...         ...         ...
row 2         ...         ...         ...
...
```

The vocabulary is:

- A sample is one rollout instance.
- A row is all samples at the same step.
- A column is one sample from the first step to the last step.

The public step shapes are:

```text
sample step:
  one sample at step t -> same sample at step t + 1

row step:
  row t -> row t + 1
```

The scalar sample step defines the semantics. The row step applies those
semantics to all samples in a row. Any internal filtering, compaction, padding,
or scheduling used by a row step is private and must not change the public row
shape or sample alignment.

## Goals

- Use only `sample`, `row`, and `column` for the public rollout vocabulary.
- Make target selection and action selection separate policy stages.
- Make target selection depend only on the current sample state and target
  legality.
- Build an action space only after a target definition has been selected.
- Store rollout rows so training can recompute logp later.
- Keep `STOP`, empty action space, and finished samples conceptually clear.
- Represent scored decisions with target/action tables and score masks.
- Keep row-level execution details in the parallel row-wrapper spec.

## Non-Goals

- Specifying how row-level execution filters or compacts active samples.
- Choosing the low-level padding values for masked target/action entries.
- Solving warm start or supervised pretraining.
- Differentiating through Rust rewrite application or action-space generation.
- Preserving compatibility with the previous transformer/reinforce prototype.

## Public Vocabulary

### Sample

A sample is one rollout instance. It has a current rewrite state, policy-relevant
status, and any state needed to continue the rollout.

A sample may be active or finished:

- An active sample may emit a target choice and possibly an action choice.
- A finished sample does not emit new policy choices.

The exact in-memory representation is not part of this overview.

### Row

A row is all samples at one synchronized rollout step.

The row width is stable across the rollout. If a sample is finished, the row
still contains that sample position so the column remains aligned.

A row contains enough stored data to score the policy choices made at that step.
Within a row, model-facing inputs may already be padded so the scorer can
operate on the row directly.

### Column

A column is one sample through time.

Training uses the column to associate policy choices with the reward or
advantage for that sample. Scoring may be row-parallel, but credit assignment is
still per sample column.

## Policy Boundary

The policy has two stages.

### Target Selection

Target selection decides where to act:

```text
TargetInput -> STOP or def_index
```

Target selection may use:

- current state tokens or token features;
- definition positions or equivalent target metadata;
- target legality mask;
- STOP legality.

Target selection must not use:

- action-space generation;
- action-space tokens;
- candidate information;
- left/right mask information.

This separation exists to prevent the model from materializing every
definition's action space before choosing a target.

### Action Selection

Action selection decides how to act after one target definition has been
selected:

```text
ActionInput -> candidate_index, left_mask, right_mask
```

Action selection exists only when the selected target definition has a non-empty
action space.

## Rollout Table Storage

When a row is updated, the rollout stores the data needed for later scoring.
The stored table should be rectangular at the public level:

```text
for each row t:
  target_input[t, sample]
  target_choice[t, sample]
  target_score_mask[t, sample]

  action_input[t, sample]
  action_choice[t, sample]
  action_score_mask[t, sample]
```

The score masks define which entries are real policy decisions. If a score mask
is false, the corresponding input and choice are ignored by loss and metrics.
Masked entries still need safe padded values so the scorer can run over the
rectangular row without out-of-range indexing or shape errors.

The sample position is the column identity used for reward assignment.

Training recomputes logp from stored row data:

```text
stored row -> target scorer -> target logp per scored sample
stored row -> action scorer -> action logp per scored sample
```

The trainer should not backpropagate through sampled logp values saved during
rollout. Rollout stores inputs and choices; training recomputes differentiable
logp.

## Trainer Boundary

The trainer owns:

- rollout control at the row level;
- reward and advantage computation per sample column;
- scoring stored rows by recomputation;
- combining target and action logp terms into a REINFORCE loss;
- optimizer updates and checkpoints.

The policy owns:

- constructing target/action model inputs;
- sampling target/action choices;
- scoring stored target/action choices.

The rewrite environment owns:

- action-space generation for one selected definition;
- rewrite application;
- cost or reward evaluation.

## STOP, Empty, And Finished Semantics

The public row representation uses target/action score masks:

```text
case                  target_score_mask    action_score_mask
already finished      false                false
STOP                  true                 false
empty action space    true                 false
valid action          true                 true
```

The masked target/action data may contain any safe padded values. It must not
contribute to loss, metrics, or score totals.

### Finished Sample

A finished sample remains aligned in later rows but emits no new target or
action score.

### STOP

STOP is a target choice.

When STOP is selected:

- the target choice is scored;
- no action choice is scored;
- the sample becomes terminal.

### Empty Action Space

An empty action space happens after target selection chooses a definition whose
action space is empty.

When this happens:

- the target choice is scored;
- no action choice is scored;
- the selected definition becomes unavailable for the affected sample state;
- the sample continues unless another stopping rule applies.

There is no inner retry loop that chooses another target inside the same sample
step. The next target selection happens in the next row.

## Row-Level Privacy

The public abstraction is:

```text
row t -> row t + 1
```

A row-level implementation may internally skip finished samples, filter active
samples, compact valid action inputs, split target/action work, or use other
memory-saving mechanics. Those mechanics must not be visible to the trainer or
policy API. The output remains a whole row with stable sample positions.

## Spec Split

The design is split into three documents:

- This overview spec defines the public vocabulary and module boundary.
- The scalar spec defines the authoritative behavior for one sample step.
- The parallel row-wrapper spec defines how to update and score rows without
  changing the public abstraction.
