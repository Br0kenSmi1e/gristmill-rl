# REINFORCE Parallel Row-Wrapper Design

Status: planned
Supersedes: row-parallel portions of earlier REINFORCE prototype specs
Depends on: `2026-06-09-reinforce-scalar-step-design.md`,
`2026-06-09-reinforce-row-table-overview-design.md`
Feeds implementation plan: yes

## Summary

This spec defines how to lift the scalar sample step to a whole row while
preserving scalar behavior for every sample position.

The public contract is:

```text
step_row(row_t) -> row_t_plus_1, stored_row_t
score_row(stored_row_t) -> target/action logp arrays
```

The wrapper may privately filter, compact, batch, and scatter active samples, but
callers only see whole rows with stable sample positions.

The implementation target is a Rust-owned row environment exposed through a
row-shaped PyO3 binding:

```text
RewriteStateRow {
  states: Vec<RewriteState>
}
```

The scalar `RewriteState` and `ActionSpace` APIs remain the reference behavior
for width-1 tests and debugging. Production row rollout should use the row API
so action-space generation and rewrite application cross the Python/Rust
boundary once per row.

## Goals

- Preserve scalar semantics for every sample position.
- Keep row width and sample-column alignment stable.
- Store one rectangular row after each row step.
- Allow private batching of target sampling, action-space generation, action
  sampling, and scoring.
- Support bounded score chunks so logical row width is independent from physical
  model batch size.
- Keep row execution details out of policy and trainer public APIs.

## Non-Goals

- Defining model architecture or probability semantics.
- Changing scalar STOP, empty action space, or valid-action behavior.
- Choosing exact low-level padding sentinel values.
- Defining reward, advantage, optimizer, or checkpoint behavior.
- Requiring full environment rollout to be vectorized in JAX.
- Requiring the first implementation to export Rust-built token arrays directly.
  Python snapshot tokenization is acceptable for the first runnable path.

## Row Step Contract

`row_t` contains all sample positions at rollout step `t`.

`row_t_plus_1` contains the same sample positions at step `t + 1`.

`stored_row_t` contains the row-table data needed to score policy choices made
while moving from `row_t` to `row_t_plus_1`.

The wrapper must preserve:

- unchanged row width;
- stable sample positions;
- scalar-equivalent behavior for each sample;
- safe masked padded data;
- no score contribution from masked entries.

## Scalar Equivalence

The row wrapper is semantically equivalent to independent scalar stepping:

```text
for each sample position s:
  step_row(row_t)[s] == step_sample(row_t[s])
```

The equality is semantic, not necessarily byte-for-byte for padded masked data or
diagnostic ordering.

One sample's STOP, exact-empty target, valid rewrite, or already-finished status
must not change another sample's scalar behavior.

## Private Row Mechanics

The row wrapper may implement the step in phases:

```text
1. Identify active and finished sample positions.
2. Build target inputs for active samples.
3. Batch target sampling where possible.
4. Scatter STOP samples to stored row and next row.
5. Query selected action spaces through `RewriteStateRow`.
6. Scatter exact-empty samples to stored row and next row.
7. Build action inputs by tokenizing non-empty `ActionSpaceRow` snapshots.
8. Batch action sampling where possible.
9. Validate sampled actions through `RewriteStateRow` without mutating state.
10. Apply validated rewrites through `RewriteStateRow`.
11. Scatter all results into row_t_plus_1 and stored_row_t.
```

This phase structure is an implementation suggestion, not a public API. Any
mechanic is acceptable if it preserves scalar equivalence and row-table storage.

Action arrays must be captured before rewrite application mutates the owning
sample state.

## Randomness

Each sample position must receive an independent and reproducible random stream.

The row wrapper may split a row-level RNG into per-sample RNGs. The mapping from
sample position to RNG stream must not depend on private active-sample
compaction, otherwise STOP or finished samples could change later samples'
choices.

Tests should be able to run width-1 and multi-sample row stepping with fixed RNGs
and compare semantic outputs.

## Rust Row Environment

The row wrapper should build on two Rust-side row objects:

```text
RewriteStateRow {
  states: Vec<RewriteState>
}

ActionSpaceRow {
  entries: Vec<ActionSpaceEntry>
}

ActionSpaceEntry =
  skipped          # STOP, finished, or inactive sample
  exact_empty      # selected definition has no exact action space
  non_empty(ActionSpace)
```

`ActionSpaceRow` is aligned by sample position. It is a live runtime handle used
only while executing one row step. It must not be stored in the rollout table and
training must not depend on it.

The intended Rust/PyO3 surface is:

```text
RewriteStateRow.query_action_spaces_for_row(target_choices)
  -> ActionSpaceRow

ActionSpaceRow.snapshots()
  -> host-side Python data for tokenization/debugging

RewriteStateRow.validate_actions_for_row(action_space_row, action_choices, action_score_mask)
  -> ValidatedActionRow

RewriteStateRow.apply_validated_actions_for_row(validated_action_row)
  -> step_result[sample]
```

`ValidatedActionRow` may contain Rust-only rewrite plans. It is also a live
runtime handle and must not be stored in the rollout table.

PyO3 should expose row-shaped classes such as `PyRewriteStateRow` and
`PyActionSpaceRow`. The binding should release the Python GIL around Rust row
work that can run independently per sample. The binding itself should remain a
thin bridge; parallelism belongs in Rust.

The scalar `RewriteState.action_space_for_def` and `RewriteState.step_with_space`
APIs remain available for scalar tests, width-1 equivalence checks, and
debugging.

## Action-Space Generation

Target selection must not generate action spaces for unselected definitions.

After target sampling, the wrapper queries exact action spaces only for selected
non-STOP targets. The intended implementation is a Rust-side row batch query
over sample positions:

```text
RewriteStateRow.query_action_spaces_for_row(target_choices)
  -> ActionSpaceRow
```

The batch query should:

- skip `target_choice == -1` for STOP samples;
- call `action_space_for_def(target_choice)` only on the owning sample state;
- parallelize over sample positions inside Rust, for example with Rayon over
  mutable per-sample states;
- release the Python GIL around the Rust batch query;
- preserve output order by sample position;
- keep exact-empty definition-mask refinement inside the corresponding sample
  state.

The row wrapper should not use Python-level threads over individual
`action_space_for_def` calls as the primary parallelism mechanism. A serial Rust
fallback may exist for debugging, but the implementation target is Rust-side
row-parallel generation.

An exact-empty result follows scalar semantics:

- target choice is scored;
- no action choice is scored;
- Rust's refined definition mask is kept in that sample state;
- the sample remains active unless another stopping rule applies.

## Action Input Tokenization And Boundary Movement

For the first runnable implementation, Python may tokenize action spaces from
`ActionSpaceRow.snapshots()`:

```text
ActionSpaceRow.snapshots()
  -> snapshot[sample]
tokenize_action_space_row(snapshot)
  -> action_space_tokens.<leaf>[sample, token, ...]
  -> action_space_token_mask[sample, token]
```

This tokenization is host-side Python code that produces NumPy/JAX-compatible
rectangular arrays. JAX must not trace through PyO3 `ActionSpace` or
`ActionSpaceRow` objects; JAX only consumes the resulting token arrays and masks.

The live `ActionSpaceRow` handle must remain available until validation and
rewrite application finish:

```text
action_space_row = query selected action spaces
action arrays = tokenize action_space_row snapshots
action choices = jax.vmap(sample_action)(action arrays)
validated = validate action choices against action_space_row
apply validated rewrites
```

The Python/Rust boundary should move only coarse row data:

- Rust to Python/JAX: state snapshots, action-space snapshots, and eventually
  numeric token/mask arrays;
- JAX to Rust: compact sampled choices such as `candidate_index`, `left_mask`,
  and `right_mask` arrays.

The first implementation may use nested Python snapshot data for tokenization,
matching the current scalar `ActionSpace.snapshot()` style. A later performance
optimization may move tokenization into Rust and export numeric buffers directly,
without changing the policy API or row-table storage contract.

## Parallel Rewrite Application

Rewrite application should also happen through the Rust row environment. The row
wrapper should validate sampled actions for all valid-action samples before any
sample state is mutated:

```text
validate_actions_for_row(action_space_row, action_choices, action_score_mask)
  -> ValidatedActionRow

apply_validated_actions_for_row(validated_action_row)
  -> step_result[sample]
```

Validation must:

- skip samples whose `action_score_mask` is false;
- reject candidate indices outside the corresponding non-empty action space;
- reject left or right bit sequences with the wrong length;
- reject empty left or right masks;
- preserve sample order in errors and results;
- avoid mutating any `RewriteState`.

Only after validation succeeds should the wrapper apply rewrites. Application
may parallelize over sample positions inside Rust because each worker mutates
only its owning `RewriteState`.

If validation fails, no rewrite should be applied for that row. If application
fails after validation because of an internal state inconsistency, the error
should include sample position and row index when available; the partially
advanced row state should be treated as invalid and not reused for training.

## Stored Row Format

The wrapper stores exactly the row-table fields. Fields named `_tokens` are token
pytrees whose leaves share the shown `[t, sample, token, ...]` leading axes:

```text
target_state_tokens.<leaf>[t, sample, token, ...]
target_state_token_mask[t, sample, token]
target_def_mask[t, sample, def]
target_choice[t, sample]
target_score_mask[t, sample]

action_state_tokens.<leaf>[t, sample, token, ...]
action_state_token_mask[t, sample, token]
selected_def_index[t, sample]
action_space_tokens.<leaf>[t, sample, token, ...]
action_space_token_mask[t, sample, token]
action_choice[t, sample]
action_score_mask[t, sample]

step_case[t, sample]
diagnostics[t, sample]
```

Private compacted batches must be scattered back to these sample positions before
returning.

## Row Scoring

The row scorer consumes one stored row and returns row-width arrays:

```text
score_row(stored_row_t)
  -> target_logp[t, sample]
  -> action_logp[t, sample]
```

Scoring may internally split target and action entries into chunks:

```text
target_score_chunk_size
action_score_chunk_size
```

Chunking must not change results. The same stored arrays and choice should produce
the same logp whether scored alone, in a width-1 row, in a full row, or in a
chunk.

## Row Loss Inputs

The row wrapper does not own reward or advantage calculation, but it provides the
arrays needed by the training spec:

```text
target_logp[t, s]
action_logp[t, s]
target_score_mask[t, s]
action_score_mask[t, s]
```

The training spec combines these with per-column advantages.

## Invariants

- Row width is unchanged by `step_row`.
- Stored row sample positions match input row sample positions.
- Finished samples emit no score terms.
- STOP emits target score only.
- Empty action space emits target score only.
- Valid action emits target and action scores.
- Valid action rewrite application happens only after row validation succeeds.
- Private active-sample compaction is invisible outside the wrapper.
- Live `ActionSpaceRow` and `ValidatedActionRow` handles are runtime-only and are
  never stored in rollout tables.
- RNG assignment is stable by sample position.
- Score chunks are result-equivalent to scalar scoring.

## Error Handling

The wrapper should fail clearly when:

- scalar stepping one sample fails;
- a compacted result cannot be scattered to its original sample position;
- vectorized policy output length differs from active input length;
- row scoring returns arrays with wrong width;
- a score chunk contains an invalid stored choice.
- row action validation rejects a sampled choice.
- row rewrite application fails after validation.

Errors should include sample position and row index when available.

## Testing Requirements

- Width-1 row stepping matches scalar sample stepping.
- Multi-sample row stepping matches independent scalar stepping for each sample.
- Row width and sample positions remain stable across multiple steps.
- A mixed row with valid action, STOP, empty action space, and already-finished
  samples produces the expected masks.
- RNG assignment is stable when preceding samples finish.
- Target selection still does not construct unselected action spaces.
- Rust-side row action-space batch querying skips STOP samples, preserves sample
  order, and mutates exact-empty definition masks only on the owning sample
  state.
- `ActionSpaceRow.snapshots()` can be tokenized into rectangular action arrays
  without exposing live handles to JAX.
- Row action validation rejects bad candidate or bit-mask choices before any
  rewrite is applied.
- Row rewrite application mutates only valid-action sample states and preserves
  scalar semantics.
- Row scoring with chunks matches row scoring without chunks.
- Masked padded values do not affect loss inputs or metrics.

## Acceptance Criteria

- The row wrapper can step a row of samples over the real `RewriteState`.
- Selected action spaces are generated through Rust-side row-parallel batch
  querying while preserving scalar semantics.
- Valid sampled actions are validated first and then applied through Rust-side
  row rewrite application.
- The stored row matches the row-table contract.
- The wrapper preserves scalar semantics while allowing private batching.
- Row scoring returns target/action logp arrays usable by the training spec.
