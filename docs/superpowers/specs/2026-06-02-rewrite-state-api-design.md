# RewriteState API Refactor Design

## Summary

Replace the cursor-based rewrite action API with a Rust-owned `RewriteState`
boundary for reinforcement-learning use.

The new API lets callers choose which `TensorDef` index to query through a
per-state definition mask. Candidate generation remains lazy and exact. The
state transition API is mutation-first so linear callers avoid extra cloning,
while branching callers such as search algorithms can clone explicitly.

This slice updates Rust rewrite APIs, PyO3 bindings, the random rewrite CLI, and
their tests. The existing Python RL package is allowed to break because a later
RL refactor will rebuild it on top of this boundary.

## Goals

- Replace the public `start_from` cursor workflow with explicit definition
  choice.
- Add a Rust-owned `RewriteState` that contains a `TensorComputation` and a
  per-definition cheap/lazily-refined mask.
- Keep exact candidate generation in Rust.
- Expose the new state/action boundary through PyO3 as a faithful wrapper.
- Make the random rewrite CLI choose definitions randomly from the current mask.
- Update Rust and PyO3 tests to match the new boundary.

## Non-Goals

- Repairing or redesigning `python/gristmill_rl`.
- Adding REINFORCE, MCTS, `STOP`, replay, or model changes.
- Adding stale-action-space tracking.
- Adding Python-side actionability or mask logic.
- Replacing the current `ActionSpace`, `Decision`, `build_rewrite`, or
  `apply_rewrite` internals beyond what the new state boundary needs.
- Implementing a richer cheap actionability predicate than term count.

## Existing Problem

The current rewrite API exposes:

```rust
next_action_space(comp, start_from) -> Option<ActionSpace>
apply_decision_with_space(comp, space, decision)
```

This makes the action space a left-to-right cursor sweep. It prevents RL callers
from treating the definition index as a policy choice. The Python RL prototype
then has to model only the selected candidate and term masks for the first
actionable definition found by the cursor.

The new boundary should make the full conceptual macro-action explicit:

```text
(def_index, candidate_index, left_mask, right_mask)
```

Operationally, transitions still use `ActionSpace + Decision` so the exact Rust
candidate graphs and bicliques are reused.

## Rust API

Add `RewriteState` in `src/rewrite.rs`:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewriteState {
    comp: TensorComputation,
    def_mask: Vec<bool>,
}

impl RewriteState {
    pub fn new(comp: TensorComputation) -> Self;
    pub fn computation(&self) -> &TensorComputation;
    pub fn into_computation(self) -> TensorComputation;
    pub fn definition_mask(&self) -> &[bool];

    pub fn action_space_for_def(
        &mut self,
        def_index: usize,
    ) -> Result<Option<ActionSpace>, RewriteError>;

    pub fn step_with_space(
        &mut self,
        space: &ActionSpace,
        decision: &Decision,
    ) -> Result<(), RewriteError>;
}
```

`computation()` provides a read-only view for scoring, snapshots, and
serialization. `into_computation()` consumes the state and returns the owned
computation for final output writing or callers that need to leave the rewrite
state abstraction.

`step_with_space` mutates the state in place. Linear callers can step without
cloning. Branching callers clone explicitly before stepping:

```rust
let mut child = parent.clone();
child.step_with_space(&space, &decision)?;
```

## Mask Semantics

The v1 cheap predicate is:

```rust
cheap_possible(def) = def.terms.len() >= 2
```

`RewriteState::new(comp)` initializes:

```rust
def_mask[i] = cheap_possible(comp.definitions()[i])
```

`definition_mask()` returns the current cheap/lazily-refined mask.

`action_space_for_def(def_index)` is the exact lazy query:

- if `def_index` is out of range, return
  `RewriteError::DefinitionIndexOutOfRange`
- if `def_mask[def_index]` is false, return `Ok(None)`
- if the mask is true, enumerate exact candidates for only that definition
- if exact candidate enumeration returns no candidates, set
  `def_mask[def_index] = false` and return `Ok(None)`
- if candidates exist, return `Ok(Some(ActionSpace { def_index, ... }))`

`RewriteState` does not cache `ActionSpace` values. Callers decide whether to
query again or retain a returned `ActionSpace`.

## Step Semantics

`step_with_space(space, decision)` applies the provided action space and decision
to the state's current computation. It uses the existing lower-level
`build_rewrite` and `apply_rewrite` machinery.

The method does not check that `space` was produced by this exact state, and it
does not require `def_mask[space.def_index]` to be true. The API assumes the
caller passes a corresponding action space. Invalid or stale inputs are guarded
only by the existing structural validation that rewrite construction already
performs.

After a successful step, the mask is updated cheaply:

- preserve existing mask entries for definitions not structurally affected
- recompute the cheap mask for the rewritten `space.def_index`
- append cheap-mask entries for the two new helper definitions created by the
  rewrite
- do not run exact candidate enumeration during mask update

## Removed Cursor API

The cursor API is removed from public rewrite and PyO3 use:

```rust
next_action_space(comp, start_from)
```

Callers that previously scanned for the next actionable definition should now
use `RewriteState::definition_mask()` and `action_space_for_def(i)`.

## PyO3 API

Expose a new `RewriteState` Python class:

```python
from gristmill_symbolics import TensorComputation, RewriteState, ActionSpace

comp = TensorComputation.load_json(path)
state = RewriteState.from_computation(comp)

mask = state.definition_mask()
space = state.action_space_for_def(def_index)
state.step_with_space(space, decision)

state.snapshot()
state.log_total_flops()
state.to_json_string()
state.write_json(path)
```

PyO3 remains a faithful converter:

- `RewriteState.from_computation(comp)` calls Rust `RewriteState::new` with a
  cloned `TensorComputation`
- `definition_mask()` converts Rust `&[bool]` to a fresh Python `list[bool]`
- `action_space_for_def(i)` directly calls Rust
  `state.action_space_for_def(i)`
- `step_with_space(space, decision)` parses the Python decision dict into a
  Rust `Decision`, then calls Rust `state.step_with_space(...)`
- `snapshot`, `log_total_flops`, `to_json_string`, and `write_json` delegate to
  the state computation through existing Rust helpers

PyO3 must not recompute masks, infer actionability, detect stale action spaces,
or implement transition logic in Python-facing glue code.

Remove these Python methods:

```python
TensorComputation.next_action_space(start_from)
TensorComputation.apply_decision_with_space(space, decision)
```

`ActionSpace` stays mostly unchanged:

```python
space.def_index
space.candidate_count
space.snapshot()
```

## Random Rewrite CLI

`src/bin/random-rewrite.rs` should use `RewriteState`.

Instead of scanning with `start_from`, each step chooses a definition uniformly
from the current true mask entries:

```rust
let mut state = RewriteState::new(comp);

for step in 0..steps {
    let Some(space) = random_action_space_from_mask(&mut state, &mut rng)? else {
        stop_reason = StopReason::NoActionSpace;
        break;
    };

    let decision = random_decision(&space, random_subsets, &mut rng);
    state.step_with_space(&space, &decision)?;
}

let comp = state.into_computation();
```

`random_action_space_from_mask` should:

- read `state.definition_mask()`
- collect indices where the mask is true
- choose one uniformly at random
- call `state.action_space_for_def(index)`
- return the space if present
- retry if the lazy exact query returned `None` and refined the mask
- return `None` only when no true mask entries remain

This changes CLI behavior intentionally: definition choice becomes random, not
left-to-right cursor based.

## Testing

Rust rewrite tests should cover:

- `RewriteState::new` initializes masks from term count
- `definition_mask()` reflects lazy false updates
- `action_space_for_def` checks only the requested definition
- querying a false mask entry returns `None`
- querying an exact-empty true entry sets that entry false and returns `None`
- querying an actionable entry returns an `ActionSpace` with the requested
  `def_index`
- `step_with_space` mutates the computation
- `step_with_space` updates the mask length for helper definitions
- the old cursor tests are removed or rewritten around `RewriteState`

PyO3 tests should cover:

- module exports `RewriteState`
- `RewriteState.from_computation` creates a state without Python-side logic
- `definition_mask()` returns a copy
- `action_space_for_def` returns `None` and mutates the mask for inactive or
  exact-empty definitions
- `step_with_space` mutates the state and returns `None`
- `snapshot`, `log_total_flops`, `to_json_string`, and `write_json` work through
  `RewriteState`
- old `TensorComputation.next_action_space` and
  `TensorComputation.apply_decision_with_space` expectations are removed

CLI tests should cover:

- `random-rewrite` terminates when the mask is exhausted
- `random-rewrite` writes a valid final JSON computation
- seeded runs remain deterministic under the new random-definition policy

Python RL tests are expected to fail or be rewritten later.

## Acceptance Criteria

The foundation refactor is complete when:

- Rust exposes `RewriteState` with mutation-first stepping
- public cursor-based action-space lookup is removed from Rust/PyO3 usage
- PyO3 exposes only faithful wrappers around Rust `RewriteState` behavior
- `random-rewrite` chooses definition indices randomly from the current mask
- Rust, PyO3, and CLI tests match the new state/action API
- existing Python RL breakage is acknowledged as follow-up scope, not repaired
  in this slice
