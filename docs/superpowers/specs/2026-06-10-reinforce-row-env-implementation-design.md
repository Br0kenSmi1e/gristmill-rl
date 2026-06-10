# REINFORCE Rust/PyO3 Row Environment Implementation Design

Status: planned
Phase: 1 of 3
Feeds implementation plan: yes

## Summary

This spec defines the first implementation phase for the REINFORCE system: a
Rust-owned row rewrite environment exposed through PyO3.

The row environment batches the rewrite-kernel operations needed during rollout:
selected action-space generation, action-space snapshot export, sampled action
validation, and rewrite application. It preserves scalar `RewriteState` behavior
for every sample position while crossing the Python/Rust boundary once per row
operation.

This phase intentionally uses injected target and action choices in tests. It
does not require the policy model, rollout table, reward logic, optimizer, or
training loop.

## Goals

- Add row-owned Rust structures for batched rewrite-state operations.
- Preserve scalar rewrite behavior for every sample position.
- Query exact action spaces only for selected, non-STOP definitions.
- Preserve sample order and row width across all row operations.
- Keep exact-empty definition-mask refinement inside the owning scalar state.
- Export deterministic action-space snapshots as plain host data for tokenizers.
- Validate all sampled row actions before mutating any state.
- Apply validated rewrites through Rust-side row operations.
- Release the Python GIL around Rust row work that can run independently per
  sample.
- Provide tests that prove row operations match scalar calls.

## Non-Goals

- Implementing the policy model or any JAX code.
- Implementing the rollout table, rewards, loss, optimizer, metrics, or
  checkpointing.
- Moving tokenization into Rust.
- Defining model padding sentinels or policy logits.
- Replacing scalar `RewriteState` and `ActionSpace` APIs.
- Guaranteeing transactional recovery after an internal rewrite-application bug.

## Public Vocabulary

### Sample Position

A sample position is the row index for one scalar `RewriteState`. It is stable
from input to output for every row operation.

### STOP Target

The target choice `-1` means STOP. The row environment skips action-space
generation for STOP entries.

### Inactive Entry

An inactive entry is a sample position the caller does not want to step in the
current row, such as an already-finished sample. The row environment skips
action-space generation and rewrite application for inactive entries.

### Exact-Empty Entry

An exact-empty entry is a selected definition whose exact action space is empty.
The owning scalar `RewriteState` keeps the refined definition mask. No action is
validated or applied for that sample position.

## Rust Structures

The implementation target is:

```text
RewriteStateRow {
  states: Vec<RewriteState>
}

ActionSpaceRow {
  entries: Vec<ActionSpaceEntry>
}

ActionSpaceEntry =
  skipped
  exact_empty
  non_empty(ActionSpace)

ValidatedActionRow {
  entries: Vec<ValidatedActionEntry>
}
```

`ActionSpaceRow` and `ValidatedActionRow` are live runtime handles. They must
not be stored in rollout data or exposed to JAX.

The scalar `RewriteState` and `ActionSpace` APIs remain available for scalar
tests, width-1 equivalence checks, and debugging.

## PyO3 Surface

The Python bindings should expose thin row-shaped classes:

```text
PyRewriteStateRow
PyActionSpaceRow
PyValidatedActionRow
```

The required methods are:

```text
PyRewriteStateRow.from_states(states)
  -> PyRewriteStateRow

PyRewriteStateRow.len()
  -> int

PyRewriteStateRow.definition_masks()
  -> mask[sample, def]

PyRewriteStateRow.snapshots()
  -> state_snapshot[sample]

PyRewriteStateRow.query_action_spaces_for_row(target_choices, active_mask)
  -> PyActionSpaceRow

PyActionSpaceRow.len()
  -> int

PyActionSpaceRow.entry_kinds()
  -> kind[sample]

PyActionSpaceRow.snapshots()
  -> action_space_snapshot_or_none[sample]

PyRewriteStateRow.validate_actions_for_row(
  action_space_row,
  action_choices,
  action_score_mask,
)
  -> PyValidatedActionRow

PyRewriteStateRow.apply_validated_actions_for_row(validated_action_row)
  -> step_result[sample]
```

The concrete Python object shapes may follow existing binding conventions, but
the semantic inputs and outputs above are required.

## Target Choice Contract

`query_action_spaces_for_row` receives row-aligned inputs:

```text
target_choices[sample]: int32
active_mask[sample]: bool
```

For each sample position:

- if `active_mask` is false, produce `skipped`;
- if `target_choice == -1`, produce `skipped`;
- if `target_choice` is a legal definition with an empty exact action space,
  produce `exact_empty`;
- if `target_choice` is a legal definition with candidates, produce
  `non_empty(ActionSpace)`;
- if `target_choice` is out of range or masked by the scalar state, fail clearly.

The method must call scalar action-space generation only for selected,
non-STOP, active entries. It must never query unselected definitions.

## Exact-Empty Semantics

When scalar action-space generation discovers that a selected definition is
exact-empty, the owning `RewriteState` may refine its definition mask. That
refinement is preserved in the row state for later target selection.

An exact-empty entry does not apply a rewrite. It remains active from the row
environment's perspective; the rollout orchestrator decides whether other
stopping rules end the sample.

## Snapshot Contract

`ActionSpaceRow.snapshots()` returns deterministic host-side data:

```text
action_space_snapshot_or_none[sample]
```

For `non_empty` entries, the snapshot contains enough plain data for the policy
tokenizer to serialize:

- candidate order;
- left side term order per candidate;
- right side term order per candidate;
- rewritten definition data per candidate;
- any graph or incidence metadata exposed by the scalar action-space snapshot.

For `skipped` and `exact_empty` entries, the snapshot is absent or explicitly
marked as not scoreable.

Snapshot data must not hold live Rust handles. It must remain valid after the
owning row state advances.

## Action Choice Contract

The row validator consumes row-aligned action choices:

```text
ActionChoice {
  candidate_index: int32
  left_mask: bool[bit]
  left_valid_mask: bool[bit]
  right_mask: bool[bit]
  right_valid_mask: bool[bit]
}

action_score_mask[sample]: bool
```

For each sample position:

- if `action_score_mask` is false, skip validation and produce a skipped
  validated entry;
- otherwise the corresponding action-space entry must be `non_empty`;
- `candidate_index` must select an existing candidate;
- left and right bit sequences must match the selected candidate side lengths;
- left and right selected masks must each be non-empty;
- padded bits outside the valid masks are ignored.

Invalid choices indicate a row or policy contract bug. Validation must fail
clearly with sample position information.

## Validation Before Mutation

`validate_actions_for_row` must not mutate any `RewriteState`.

If any scored action choice is invalid, validation fails and no rewrite is
applied for the row. This all-or-nothing validation boundary protects rollout
from partially applying a row after one bad sampled action.

`apply_validated_actions_for_row` may mutate states only after validation has
succeeded for the row. Application mutates each owning scalar `RewriteState` for
valid-action entries and skips all other entries.

If application fails after validation because of an internal state inconsistency,
the error must include the sample position when available. The caller must treat
the partially advanced row as invalid and must not use it for training.

## Parallelism And Boundary Rules

Row action-space generation and row rewrite application should parallelize inside
Rust over independent sample positions, for example with Rayon. The PyO3 binding
should release the Python GIL around these Rust sections.

The binding should not use Python-level threads over individual scalar calls as
the primary parallelism mechanism.

The first implementation may be serial internally if needed for correctness
bring-up, but the public API and tests must be compatible with Rust-side
parallel execution.

## Error Handling

The row environment should fail clearly when:

- row-aligned input lengths differ from `RewriteStateRow.len()`;
- a target choice is invalid for its scalar state;
- action-space row length differs from the rewrite-state row length;
- an action choice is scored for a skipped or exact-empty entry;
- an action choice selects an invalid candidate;
- left or right bit lengths do not match the selected candidate;
- a scored left or right mask is empty;
- validation is asked to use an action-space row from a different state row;
- rewrite application is asked to use a validated-action row from a different
  state row.

Errors should include sample position and operation name when available.

## Testing Requirements

Contract tests:

- constructing a row from scalar states preserves row length and sample order;
- row state snapshots match scalar state snapshots for each position;
- definition masks match scalar `RewriteState.definition_mask()` for each
  position;
- row-aligned input length mismatches fail clearly.

Action-space query tests:

- width-1 row query matches scalar `action_space_for_def`;
- multi-sample row query preserves sample order;
- STOP and inactive entries are skipped;
- unselected definitions are not queried;
- exact-empty selected definitions refine only the owning scalar state mask;
- non-empty action-space snapshots match scalar `ActionSpace.snapshot()`.

Validation tests:

- valid injected choices produce a `ValidatedActionRow`;
- invalid candidate indices fail before any state mutation;
- wrong-length masks fail before any state mutation;
- empty left or right masks fail before any state mutation;
- `action_score_mask=false` skips validation for padded or non-action entries.

Application tests:

- width-1 validated application matches scalar `step_with_space`;
- multi-sample validated application mutates only valid-action positions;
- skipped, STOP, inactive, and exact-empty positions do not apply rewrites;
- validation failure leaves all row states unchanged;
- row application reports sample position on failure when possible.

Binding tests:

- PyO3 methods expose deterministic Python data without live action-space handles
  in snapshots;
- row Rust work releases the GIL where the binding can do so safely;
- scalar APIs remain available after adding row APIs.

## Exit Criteria

Phase 1 is complete when:

- `RewriteStateRow`, `ActionSpaceRow`, and `ValidatedActionRow` exist in Rust.
- PyO3 exposes the row methods needed by later phases.
- Row action-space queries skip STOP and inactive samples, preserve sample order,
  and query only selected definitions.
- Exact-empty selected definitions refine only the owning scalar state.
- Action-space snapshots are deterministic plain host data.
- Row action validation is all-or-nothing and does not mutate state.
- Validated row rewrite application matches scalar behavior for width-1 and
  mixed multi-sample rows.
- Tests cover STOP, inactive, exact-empty, invalid action, and valid rewrite
  cases without importing the policy model or trainer.
