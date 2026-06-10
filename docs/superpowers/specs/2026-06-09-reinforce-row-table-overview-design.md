# REINFORCE Row Table Design

Status: planned
Supersedes: the public table portions of earlier REINFORCE prototype specs
Depends on: `2026-06-09-reinforce-scalar-step-design.md`
Feeds implementation plan: yes

## Summary

This spec defines the public rollout storage model for REINFORCE training. The
storage model is a rectangular table indexed by row and sample position:

```text
              sample 0    sample 1    sample 2    ...    sample N
row 0         ...         ...         ...
row 1         ...         ...         ...
row 2         ...         ...         ...
...
```

Rows store immutable target/action inputs, sampled choices, and score masks so
training can recompute differentiable logp later.

The scalar step defines per-sample semantics. The row table defines how those
semantics are stored across time and samples.

## Goals

- Use only `sample`, `row`, and `column` as public rollout vocabulary.
- Keep row width stable across rollout.
- Preserve sample position as the column identity for reward assignment.
- Store target/action inputs, choices, and score masks in rectangular form.
- Define padding and mask responsibilities without choosing low-level sentinel
  values.
- Keep row storage independent from private parallel scheduling mechanics.

## Non-Goals

- Defining model architecture or policy logits.
- Defining scalar environment behavior beyond referencing the scalar spec.
- Defining private active-sample compaction or scheduling.
- Choosing exact padding values for every tensor field.
- Defining reward, advantage, optimizer, or checkpoint behavior.

## Public Vocabulary

### Sample

A sample is one rollout instance. It has a current rewrite state, policy-relevant
status, and any metadata needed to continue rollout.

A sample may be active or finished:

- an active sample may emit a target choice and possibly an action choice;
- a finished sample emits no new policy choices.

### Row

A row is all samples at one synchronized rollout step.

The row width is stable. Finished samples still occupy their sample positions so
columns remain aligned.

### Column

A column is one sample through time.

Training assigns rewards and advantages per column. Scoring may run row-parallel,
but credit assignment uses sample position.

## Stored Row Contract

For every rollout row `t`, the table stores:

```text
target_record[t, sample]
target_choice[t, sample]
target_score_mask[t, sample]

action_record[t, sample]
action_choice[t, sample]
action_score_mask[t, sample]

step_case[t, sample]
diagnostics[t, sample]
```

The model spec defines the structure of `target_record`, `action_record`,
`target_choice`, and `action_choice`.

The scalar spec defines which masks are true for each step case.

## Mask Mapping

```text
case                  target_score_mask    action_score_mask
already finished      false                false
STOP                  true                 false
empty action space    true                 false
valid action          true                 true
```

If a score mask is false, the corresponding input and choice are ignored by loss,
metrics, and score totals.

Masked entries still need safe padded values so model scorers can run over
rectangular batches without out-of-range indexing or shape errors.

## Immutable Input Requirement

Stored row inputs are immutable snapshots:

- `target_record` must not reference a mutable `RewriteState`;
- `action_record` must not require a live `ActionSpace` handle;
- stored choices must be plain data;
- row data must remain scorable after rollout has advanced or finished.

This requirement is part of the table contract because row scoring happens after
environment mutation.

## Padding Requirements

The row table may contain ragged symbolic structures internally, but any
model-facing row batch must provide safe padding for:

- token sequences;
- definition positions;
- target choices;
- action-space candidate positions;
- left/right side term positions;
- left/right bit sequences;
- legality masks.

Padding must obey these rules:

- masked score entries contribute no logp, loss, or metrics;
- padding indices never point outside padded arrays;
- illegal padded logits are masked before sampling or scoring;
- sample positions remain stable after padding.

The concrete sentinel values are implementation details and belong in the
implementation plan.

## Row-To-Column Assignment

The sample index in a row is the column identity:

```text
stored_row[t].target_record[s] belongs to sample column s
```

Rewards and advantages are computed per sample column and then broadcast to that
sample's scored target/action terms.

The table must not reorder columns after rollout begins. Private row mechanics
may compact active samples internally, but stored rows are scattered back to the
original sample positions.

## Row Scoring Interface

Training recomputes logp from stored rows:

```text
score_target_row(stored_row_t)
  -> target_logp[t, sample]

score_action_row(stored_row_t)
  -> action_logp[t, sample]
```

The returned arrays have row width. Values for masked entries may be arbitrary
finite padding values because masks exclude them from loss and metrics.

## Invariants

- Every row has the same width.
- Every sample column has at most one target choice per row.
- Every sample column has at most one action choice per row.
- `action_score_mask=true` implies `target_score_mask=true`.
- `action_score_mask=true` implies the corresponding action input and choice are
  valid for scoring.
- Masked entries are safe to batch but invisible to objective terms.
- Row storage does not expose private active-sample scheduling.

## Error Handling

The row table builder should fail clearly when:

- a row has a different width from previous rows;
- a true score mask lacks input or choice data;
- an action score is present without a target score;
- a stored choice is out of range for its stored input;
- reward or advantage assignment uses a different width than the row table.

## Testing Requirements

- A width-1 table stores the same data as scalar step output.
- A multi-sample row containing valid action, STOP, empty action space, and
  already-finished samples produces the expected masks.
- Finished samples remain aligned in later rows.
- Masked padded values do not affect row loss or metrics.
- Stored inputs can be scored after all samples finish.
- Reward/advantage arrays align by sample position.

## Acceptance Criteria

- The row table can represent scalar and multi-sample rollout without changing
  public vocabulary.
- Stored rows contain all data needed for recomputed target/action logp.
- The table contract is independent from private row execution mechanics.
- Training can assemble per-column REINFORCE terms from row masks and logp arrays.
