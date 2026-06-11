# REINFORCE Rust/PyO3 Row Environment Implementation Design

Status: planned
Phase: 1 of 3
Feeds implementation plan: yes

## Summary

This spec defines the first implementation phase for the REINFORCE system: thin
Rust/PyO3 row wrappers over the existing scalar rewrite API.

The row environment is not a new rewrite subsystem. Phase 1 first refactors
`src/rewrite.rs` to expose the existing scalar workflow at cleaner boundaries,
without changing rewrite semantics, then batches those scalar boundaries with
Rayon:

```text
RewriteState::action_space_for_def
validate_decision for one Decision and one ActionSpace without mutation
apply one already-validated Decision through one RewriteState
ActionSpace snapshot conversion used by PyO3
RewriteState snapshot/cost helpers used by PyO3
```

It preserves scalar `RewriteState` behavior for every sample position while
crossing the Python/Rust boundary once per row operation.

This phase intentionally uses injected target and action choices in tests. It
does not require the policy model, rollout table, reward logic, optimizer, or
training loop.

## Goals

- Add row-owned Rust structures as thin wrappers around `Vec<RewriteState>`,
  row-aligned `ActionSpace` entries, and row-aligned validated
  `ActionSpace`/`Decision` pairs.
- Refactor `src/rewrite.rs` to expose scalar generate, decision validation, and
  apply boundaries without semantic changes.
- Preserve scalar rewrite behavior for every sample position.
- Query exact action spaces only for selected, non-STOP definitions.
- Preserve sample order and row width across all row operations.
- Keep exact-empty definition-mask refinement inside the owning scalar state.
- Export deterministic action-space snapshots as plain host data for tokenizers.
- Validate all sampled row actions before mutating any state.
- Apply validated rewrites by calling the refactored scalar apply boundary on
  each owning state.
- Use Rayon inside Rust for row action-space generation, row action validation,
  and row rewrite application parallelism.
- Release the Python GIL around Rust row work that can run independently per
  sample.
- Provide tests that prove row operations match scalar calls.

## Non-Goals

- Implementing the policy model or any JAX code.
- Implementing the rollout table, rewards, loss, optimizer, metrics, or
  checkpointing.
- Moving tokenization into Rust.
- Defining model padding sentinels or policy logits.
- Replacing or removing scalar `RewriteState` and `ActionSpace` APIs.
- Adding a second rewrite planner, new action-space semantics, or a separate row
  rewrite algorithm.
- Moving JAX padding concerns into core `src/rewrite.rs` scalar types.
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

ValidatedActionEntry =
  skipped
  valid { space: ActionSpace, decision: Decision }
```

`ActionSpaceRow` and `ValidatedActionRow` are live runtime handles. They must
not be stored in rollout data or exposed to JAX.

These row structures should live in or near `src/rewrite.rs` so they can reuse
the existing scalar types and refactored scalar helpers directly. If the row code
lives outside `rewrite.rs`, expose only the smallest required scalar boundary
rather than duplicating validation or apply logic.

The refactored scalar `RewriteState`, `ActionSpace`, and `Decision` boundaries
are the authoritative behavior for row tests, width-1 equivalence checks, and
debugging.

## File Layout

Phase 1 should split rewrite code enough to make the scalar and row boundaries
clear without redesigning the whole rewrite module:

```text
src/rewrite.rs
src/rewrite/scalar.rs
src/rewrite/row.rs

python/src/lib.rs
python/src/rewrite_bindings.rs
```

`src/rewrite.rs` should become the public facade for the rewrite module:

```text
mod scalar;
mod row;

pub use scalar::{
  ActionSpace,
  Decision,
  Factorization,
  RewriteError,
  RewriteState,
  validate_decision,
};

pub use row::{
  ActionSpaceEntry,
  ActionSpaceRow,
  RewriteStateRow,
  ValidatedActionEntry,
  ValidatedActionRow,
};
```

`src/rewrite/scalar.rs` should contain today's scalar rewrite implementation,
plus only the scalar boundary refactor described below.

`src/rewrite/row.rs` should stay small. It owns row-aligned entry enums and Rayon
wrappers over scalar generation, validation, and apply boundaries. It must not
duplicate scalar rewrite planning logic.

`python/src/rewrite_bindings.rs` should own rewrite-related PyO3 classes and
conversion:

- `PyRewriteState`;
- `PyActionSpace`;
- `PyRewriteStateRow`;
- `PyActionSpaceRow`;
- `PyValidatedActionRow`;
- Python action input parsing;
- padded row action mask trimming into exact Rust `Decision` values.

`python/src/lib.rs` should keep shared module setup and class registration. It
should not keep growing as the home for rewrite binding implementation details.

## Scalar Rewrite Boundary Refactor

Before adding row wrappers, refactor the current scalar workflow in
`src/rewrite.rs` into explicit boundaries:

```text
generate:
  RewriteState::action_space_for_def(def_index)
    -> Result<Option<ActionSpace>, RewriteError>

validate:
  validate_decision(space, decision)
    -> Result<(), RewriteError>

apply:
  apply one already-validated Decision with one ActionSpace to one RewriteState
    -> Result<(), RewriteError>
```

The refactor is a reordering and boundary extraction of the current workflow. The
validation currently inside `build_rewrite` and `step_with_space` should become
an explicit public scalar validation boundary. The scalar apply boundary should
assume the caller already validated the decision and should not call
`validate_decision` internally.

The refactor must not change:

- action-space generation results;
- exact-empty definition-mask refinement;
- decision validation errors;
- current computation checks in rewrite building/application, such as
  definition-index bounds;
- rewrite application output;
- definition-mask refresh after rewrite;
- current caller-owned provenance behavior: callers are responsible for applying
  an `ActionSpace` to the intended compatible `RewriteState`, and the rewrite
  module does not add an identity/provenance check;
- scalar PyO3 behavior except for the intentional API rename/split needed to
  expose the new boundaries.

This phase should replace the old scalar Rust/PyO3 apply functions after tests
and callers are updated to the new boundary. In particular,
`RewriteState::step_with_space(&ActionSpace, &Decision)` and the scalar PyO3
`RewriteState.step_with_space(...)` method should be removed or renamed instead
of kept as compatibility wrappers.

The apply boundary may continue using private helpers such as `build_rewrite` and
`apply_rewrite`. Those helpers should preserve their current computation checks,
but should not repeat `validate_decision` after the caller has already validated.

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
the semantic inputs and outputs above are required. PyO3 code should parse Python
target/action choices into the existing Rust scalar types, call the Rust row
wrapper methods, and convert snapshots with the same conversion helpers used by
the scalar bindings.

Padded policy outputs are a PyO3 conversion concern. Core `src/rewrite.rs`
should accept exact scalar `Decision` values only. PyO3 should trim padded
left/right masks with `left_valid_mask` and `right_valid_mask`, check row input
shape consistency, and then construct exact `Decision` values before calling
Rust row validation.

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

The method must be a Rayon iterator over sample positions that calls
`RewriteState::action_space_for_def` only for selected, non-STOP, active entries.
It must never query unselected definitions.

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

`validate_actions_for_row` must be a Rayon iterator over row action entries that
uses scalar `validate_decision` and does not mutate any `RewriteState`.

If any scored action choice is invalid, validation fails and no rewrite is
applied for the row. This all-or-nothing validation boundary protects rollout
from partially applying a row after one bad sampled action.

`ValidatedActionRow` should store exact scalar `Decision` values and the matching
`ActionSpace` values needed to apply them, not a new row-specific rewrite plan.
`apply_validated_actions_for_row` may mutate states only after validation has
succeeded for the row. Application mutates each owning scalar `RewriteState` by
calling the refactored scalar apply boundary; it skips all other entries.

If application fails after validation because of an internal state inconsistency,
the error must include the sample position when available. The caller must treat
the partially advanced row as invalid and must not use it for training.

## Parallelism And Boundary Rules

Row action-space generation, action validation, and rewrite application must use
Rayon inside Rust to parallelize over independent sample positions. The PyO3
binding should release the Python GIL around these Rust sections.

Rayon work must preserve row semantics:

- output order stays aligned by sample position;
- each worker reads or mutates only the owning sample state and matching
  row-aligned action-space entry;
- validation remains non-mutating even when it runs in parallel;
- validation errors include sample position and prevent later rewrite
  application;
- rewrite application runs only after full-row validation succeeds;
- row code delegates legality and mutation semantics to existing scalar
  boundaries instead of reimplementing them.

The binding should not use Python-level threads over individual scalar calls as
the primary parallelism mechanism.

A serial implementation may exist only as a narrow test/debug fallback. The
implementation path used by the PyO3 row methods must exercise Rayon for row
action-space generation, validation, and application.

## Error Handling

The row environment should fail clearly when:

- row-aligned input lengths differ from `RewriteStateRow.len()`;
- a target choice is invalid for its scalar state;
- action-space row length differs from the rewrite-state row length;
- an action choice is scored for a skipped or exact-empty entry;
- padded PyO3 action mask and valid-mask lengths differ;
- padded PyO3 valid masks select no usable side bits for a scored action;
- an action choice selects an invalid candidate;
- left or right bit lengths do not match the selected candidate;
- a scored left or right mask is empty;
- scalar apply reports a current-computation error such as an out-of-range
  definition index.

Errors should include sample position and operation name when available.

## Testing Requirements

Contract tests:

- constructing a row from scalar states preserves row length and sample order;
- row state snapshots match scalar state snapshots for each position;
- definition masks match scalar `RewriteState.definition_mask()` for each
  position;
- row-aligned input length mismatches fail clearly.

Scalar refactor tests:

- public `validate_decision` accepts and rejects the same decisions as the
  previous private scalar validation behavior;
- `build_rewrite` and the scalar apply boundary do not call `validate_decision`
  internally after the refactor;
- the new scalar validate/apply sequence produces the same rewrite result
  as the previous `RewriteState::step_with_space` workflow;
- scalar apply refreshes definition masks exactly as
  the previous scalar apply workflow did;
- current caller-owned provenance behavior is preserved: tests cover that an
  `ActionSpace` generated from one equivalent state can be applied to another
  equivalent state, and no new identity/provenance check rejects it;
- scalar PyO3 tests cover the replacement boundary instead of requiring the old
  `RewriteState.step_with_space` method to remain.

Action-space query tests:

- width-1 row query matches scalar `action_space_for_def`;
- multi-sample row query preserves sample order;
- STOP and inactive entries are skipped;
- unselected definitions are not queried;
- exact-empty selected definitions refine only the owning scalar state mask;
- non-empty action-space snapshots match scalar `ActionSpace.snapshot()`;
- the PyO3 row query path uses the Rayon-backed Rust implementation.

Validation tests:

- valid injected choices produce a `ValidatedActionRow`;
- validation accepts and rejects the same choices as scalar `validate_decision`;
- invalid candidate indices fail before any state mutation;
- wrong-length masks fail before any state mutation;
- empty left or right masks fail before any state mutation;
- `action_score_mask=false` skips validation for padded or non-action entries;
- the PyO3 row validation path uses the Rayon-backed Rust implementation.

Application tests:

- width-1 validated application matches the new scalar validate/apply sequence;
- multi-sample validated application mutates only valid-action positions;
- row application delegates each valid rewrite to the scalar apply boundary;
- skipped, STOP, inactive, and exact-empty positions do not apply rewrites;
- validation failure leaves all row states unchanged;
- row application reports sample position on failure when possible;
- the PyO3 row application path uses the Rayon-backed Rust implementation.

Binding tests:

- PyO3 methods expose deterministic Python data without live action-space handles
  in snapshots;
- PyO3 trims padded left/right masks with valid masks before constructing exact
  Rust `Decision` values;
- PyO3 rejects inconsistent padded action input shapes before calling Rust row
  validation;
- row Rust work releases the GIL where the binding can do so safely;
- scalar PyO3 bindings expose the new scalar boundary and do not keep old
  functions solely for backward compatibility.

## Exit Criteria

Phase 1 is complete when:

- rewrite code is split into a scalar implementation file, a small row wrapper
  file, and a facade `src/rewrite.rs`;
- rewrite PyO3 code is moved into a rewrite-specific binding module instead of
  continuing to grow inside `python/src/lib.rs`;
- `src/rewrite.rs` exposes scalar generate, public `validate_decision`, and apply
  boundaries without changing scalar rewrite semantics.
- Old scalar Rust/PyO3 apply functions are removed or renamed when replaced by
  the new validate/apply boundary; compatibility wrappers are not kept.
- `RewriteStateRow`, `ActionSpaceRow`, and `ValidatedActionRow` exist in Rust as
  thin wrappers around existing scalar rewrite types.
- PyO3 exposes the row methods needed by later phases.
- Row action-space queries use Rayon over existing `action_space_for_def` calls,
  skip STOP and inactive samples, preserve sample order, and query only selected
  definitions.
- Exact-empty selected definitions refine only the owning scalar state.
- Action-space snapshots are deterministic plain host data.
- Row action validation uses Rayon over scalar `validate_decision`, is
  all-or-nothing, and does not mutate state.
- Validated row rewrite application uses Rayon over the scalar apply boundary
  and matches scalar behavior for width-1 and mixed multi-sample rows.
- The PyO3 row query, validation, and application paths use Rayon-backed Rust
  parallelism.
- PyO3 owns padded action-input conversion and core `src/rewrite.rs` only sees
  exact scalar `Decision` values.
- Tests cover STOP, inactive, exact-empty, invalid action, and valid rewrite
  cases without importing the policy model or trainer.
