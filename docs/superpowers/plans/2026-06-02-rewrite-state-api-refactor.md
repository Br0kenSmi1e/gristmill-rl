# RewriteState API Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cursor-based rewrite action selection with a mutation-first Rust `RewriteState` API, faithful PyO3 bindings, and random definition choice in the CLI.

**Architecture:** `src/rewrite.rs` owns all state, mask, candidate-generation, and transition behavior. PyO3 only converts Python objects to Rust calls and Rust results to Python values. `random-rewrite` chooses definition indices from the current state mask and delegates exact checks to `RewriteState`.

**Tech Stack:** Rust 2024, existing rewrite/cost/io modules, Clap, Rand, PyO3 0.28, pythonize 0.28, maturin, uv, pytest.

---

## File Structure

- Modify `src/rewrite.rs`: add `RewriteState`, cheap mask helpers, exact `action_space_for_def`, mutation-first `step_with_space`, and remove the public cursor lookup after consumers migrate.
- Modify `tests/rewrite.rs`: replace cursor-oriented integration tests with `RewriteState` tests for mask initialization, lazy refinement, exact action spaces, errors, and stepping.
- Modify `src/bin/random-rewrite.rs`: replace `start_from` scanning with `random_action_space_from_mask(&mut RewriteState, rng)`.
- Modify `tests/random_rewrite_cli.rs`: keep existing output/snapshot coverage and add seeded determinism for the new random-definition policy.
- Modify `python/src/lib.rs`: expose `RewriteState` as a faithful PyO3 wrapper and remove old `TensorComputation` cursor/step methods.
- Modify `python/tests/test_bindings.py`: update binding tests to the new `RewriteState` surface.

`python/gristmill_rl` and its tests are not part of this plan. They will still reference the old API until the RL package is refactored.

---

### Task 1: Add Rust `RewriteState` Core

**Files:**
- Modify: `tests/rewrite.rs`
- Modify: `src/rewrite.rs`

- [ ] **Step 1: Write failing `RewriteState` tests**

In `tests/rewrite.rs`, change the rewrite import to include `RewriteState` and keep the old cursor import temporarily so the file still compiles while the new API is added:

```rust
use gristmill_symbolics::rewrite::{
    Decision, Factorization, FactorizationRewrite, RewriteError, RewriteState, apply_rewrite,
    build_rewrite, next_action_space,
};
```

Add this helper after `comp_with_shared_left_candidate()`:

```rust
fn comp_with_two_unsplittable_terms() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);

    comp.add_definition(
        out,
        vec![idx(0)],
        vec![
            term(vec![], vec![factor(a, &[0])]),
            term(vec![], vec![factor(a, &[0])]),
        ],
    );

    comp
}

fn comp_with_unsplittable_then_actionable_definition() -> TensorComputation {
    let mut comp = comp_with_shared_left_candidate();
    let extra_base = comp.add_tensor(vec![]);
    comp.definitions_mut().insert(
        0,
        gristmill_symbolics::repr::TensorDef {
            base: extra_base,
            ext_indices: vec![idx(0)],
            terms: vec![
                term(vec![], vec![factor(TensorId(0), &[0])]),
                term(vec![], vec![factor(TensorId(0), &[0])]),
            ],
        },
    );
    comp
}
```

Add these tests after `first_full_decision()`:

```rust
#[test]
fn rewrite_state_initializes_definition_mask_from_term_count() {
    let basic = {
        let mut comp = TensorComputation::new();
        comp.add_range(8);
        let a = comp.add_tensor(vec![]);
        let out = comp.add_tensor(vec![]);
        comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);
        comp
    };
    let exact_empty = comp_with_two_unsplittable_terms();
    let actionable = comp_with_shared_left_candidate();

    assert_eq!(RewriteState::new(basic).definition_mask(), &[false]);
    assert_eq!(RewriteState::new(exact_empty).definition_mask(), &[true]);
    assert_eq!(RewriteState::new(actionable).definition_mask(), &[true]);
}

#[test]
fn action_space_for_def_returns_none_for_false_mask_entry() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);
    let mut state = RewriteState::new(comp);

    assert_eq!(state.definition_mask(), &[false]);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    assert_eq!(state.definition_mask(), &[false]);
}

#[test]
fn action_space_for_def_refines_exact_empty_definition_to_false() {
    let mut state = RewriteState::new(comp_with_two_unsplittable_terms());

    assert_eq!(state.definition_mask(), &[true]);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    assert_eq!(state.definition_mask(), &[false]);
}

#[test]
fn action_space_for_def_returns_requested_definition_without_scanning() {
    let mut state = RewriteState::new(comp_with_unsplittable_then_actionable_definition());

    assert_eq!(state.definition_mask(), &[true, true]);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    assert_eq!(state.definition_mask(), &[false, true]);

    let space = state.action_space_for_def(1).unwrap().unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}

#[test]
fn action_space_for_def_rejects_out_of_range_definition_index() {
    let mut state = RewriteState::new(comp_with_shared_left_candidate());

    assert_eq!(
        state.action_space_for_def(7),
        Err(RewriteError::DefinitionIndexOutOfRange { index: 7, len: 1 })
    );
}

#[test]
fn rewrite_state_step_with_space_mutates_computation_and_updates_mask() {
    let original = comp_with_unsplittable_then_actionable_definition();
    let original_tensors = original.tensors().len();
    let original_definitions = original.definitions().len();
    let mut state = RewriteState::new(original);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    let space = state.action_space_for_def(1).unwrap().unwrap();
    let decision = first_full_decision(&space);

    state.step_with_space(&space, &decision).unwrap();

    assert_eq!(state.computation().tensors().len(), original_tensors + 2);
    assert_eq!(state.computation().definitions().len(), original_definitions + 2);
    assert_eq!(state.computation().validate(), Ok(()));
    assert_eq!(state.definition_mask().len(), state.computation().definitions().len());
    assert!(!state.definition_mask()[0]);
}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
cargo test --test rewrite -- --nocapture
```

Expected: FAIL to compile because `RewriteState` is not exported by `gristmill_symbolics::rewrite`.

- [ ] **Step 3: Implement `RewriteState` in `src/rewrite.rs`**

Add `RewriteState` after `Decision`:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewriteState {
    comp: TensorComputation,
    def_mask: Vec<bool>,
}

impl RewriteState {
    pub fn new(comp: TensorComputation) -> Self {
        let def_mask = comp.definitions().iter().map(cheap_possible).collect();
        Self { comp, def_mask }
    }

    pub fn computation(&self) -> &TensorComputation {
        &self.comp
    }

    pub fn into_computation(self) -> TensorComputation {
        self.comp
    }

    pub fn definition_mask(&self) -> &[bool] {
        &self.def_mask
    }

    pub fn action_space_for_def(
        &mut self,
        def_index: usize,
    ) -> Result<Option<ActionSpace>, RewriteError> {
        if def_index >= self.def_mask.len() {
            return Err(RewriteError::DefinitionIndexOutOfRange {
                index: def_index,
                len: self.def_mask.len(),
            });
        }
        if !self.def_mask[def_index] {
            return Ok(None);
        }

        let Some(space) = action_space_for_definition(&self.comp, def_index)? else {
            self.def_mask[def_index] = false;
            return Ok(None);
        };
        Ok(Some(space))
    }

    pub fn step_with_space(
        &mut self,
        space: &ActionSpace,
        decision: &Decision,
    ) -> Result<(), RewriteError> {
        let rewrite = build_rewrite(&self.comp, space, decision)?;
        let def_index = rewrite.def_index;
        apply_rewrite(&mut self.comp, rewrite)?;
        self.refresh_mask_after_rewrite(def_index);
        Ok(())
    }

    fn refresh_mask_after_rewrite(&mut self, def_index: usize) {
        let replacement_mask: Vec<bool> = self.comp.definitions()[def_index..def_index + 3]
            .iter()
            .map(cheap_possible)
            .collect();
        self.def_mask.remove(def_index);
        for (offset, mask_value) in replacement_mask.into_iter().enumerate() {
            self.def_mask.insert(def_index + offset, mask_value);
        }
    }
}

fn cheap_possible(def: &TensorDef) -> bool {
    def.terms.len() >= 2
}
```

Replace the body of `next_action_space` with a temporary shim that delegates to a new exact helper:

```rust
pub fn next_action_space(
    comp: &TensorComputation,
    start_from: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    for def_index in start_from..comp.definitions().len() {
        if let Some(space) = action_space_for_definition(comp, def_index)? {
            return Ok(Some(space));
        }
    }
    Ok(None)
}

fn action_space_for_definition(
    comp: &TensorComputation,
    def_index: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    let Some(def) = comp.definitions().get(def_index) else {
        return Err(RewriteError::DefinitionIndexOutOfRange {
            index: def_index,
            len: comp.definitions().len(),
        });
    };
    if !cheap_possible(def) {
        return Ok(None);
    }

    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
    let (candidate_graphs, candidate_bicliques) = enumerate_candidates(comp, def)?;
    if candidate_bicliques.is_empty() {
        return Ok(None);
    }

    let candidate_templates = candidate_graphs
        .iter()
        .zip(&candidate_bicliques)
        .map(|(graph, biclique)| build_factorization(def, graph, biclique, left_tid, right_tid))
        .collect();

    Ok(Some(ActionSpace {
        def_index,
        candidate_templates,
        candidate_graphs,
        candidate_bicliques,
    }))
}
```

- [ ] **Step 4: Run the targeted Rust tests**

Run:

```bash
cargo test --test rewrite -- --nocapture
```

Expected: PASS for the new `RewriteState` tests.

- [ ] **Step 5: Run all rewrite integration tests**

Run:

```bash
cargo test --test rewrite -- --nocapture
```

Expected: PASS. The old cursor tests still pass through the temporary shim.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rewrite.rs tests/rewrite.rs
git commit -m "feat: add rewrite state boundary"
```

---

### Task 2: Migrate `random-rewrite` To Definition-Mask Selection

**Files:**
- Modify: `src/bin/random-rewrite.rs`
- Modify: `tests/random_rewrite_cli.rs`

- [ ] **Step 1: Add CLI unit tests for mask-driven action-space choice**

In `src/bin/random-rewrite.rs`, update the imports:

```rust
use gristmill_symbolics::rewrite::{ActionSpace, Decision, Factorization, RewriteError, RewriteState};
```

Inside the `#[cfg(test)] mod tests` block, add these imports:

```rust
use gristmill_symbolics::repr::{Factor, Index, IndexId, RangeId};
```

Add these helpers inside the test module:

```rust
fn idx(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(0),
    }
}

fn factor(tensor: TensorId, indices: &[u32]) -> Factor {
    Factor {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn integration_term(sum_indices: Vec<Index>, factors: Vec<Factor>) -> Term {
    Term {
        coeff: Rational::new(1, 1),
        sum_indices,
        factors,
    }
}

fn comp_without_exact_action_space() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    comp.add_definition(
        out,
        vec![idx(0)],
        vec![
            integration_term(vec![], vec![factor(a, &[0])]),
            integration_term(vec![], vec![factor(a, &[0])]),
        ],
    );
    comp
}

fn comp_with_empty_and_actionable_spaces() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let b = comp.add_tensor(vec![]);
    let c = comp.add_tensor(vec![]);
    let empty_out = comp.add_tensor(vec![]);
    let actionable_out = comp.add_tensor(vec![]);

    comp.add_definition(
        empty_out,
        vec![idx(0)],
        vec![
            integration_term(vec![], vec![factor(a, &[0])]),
            integration_term(vec![], vec![factor(a, &[0])]),
        ],
    );
    comp.add_definition(
        actionable_out,
        vec![idx(0), idx(1)],
        vec![
            integration_term(vec![idx(2)], vec![factor(a, &[0, 2]), factor(b, &[2, 1])]),
            integration_term(vec![idx(3)], vec![factor(a, &[0, 3]), factor(c, &[3, 1])]),
        ],
    );

    comp
}
```

Add these tests inside the test module:

```rust
#[test]
fn random_action_space_from_mask_refines_empty_mask_entries() {
    let mut state = RewriteState::new(comp_without_exact_action_space());
    let mut rng = StdRng::seed_from_u64(0);

    let space = random_action_space_from_mask(&mut state, &mut rng).unwrap();

    assert!(space.is_none());
    assert_eq!(state.definition_mask(), &[false]);
}

#[test]
fn random_action_space_from_mask_returns_available_space() {
    let mut state = RewriteState::new(comp_with_empty_and_actionable_spaces());
    let mut rng = StdRng::seed_from_u64(4);

    let space = random_action_space_from_mask(&mut state, &mut rng)
        .unwrap()
        .unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}
```

- [ ] **Step 2: Run CLI unit tests to verify they fail**

Run:

```bash
cargo test --bin random-rewrite random_action_space_from_mask -- --nocapture
```

Expected: FAIL to compile because `random_action_space_from_mask` does not exist.

- [ ] **Step 3: Implement random definition selection**

Add this helper above `random_decision`:

```rust
fn random_action_space_from_mask(
    state: &mut RewriteState,
    rng: &mut impl Rng,
) -> Result<Option<ActionSpace>, RewriteError> {
    loop {
        let available: Vec<usize> = state
            .definition_mask()
            .iter()
            .enumerate()
            .filter_map(|(index, possible)| possible.then_some(index))
            .collect();
        if available.is_empty() {
            return Ok(None);
        }

        let def_index = available[rng.gen_range(0..available.len())];
        if let Some(space) = state.action_space_for_def(def_index)? {
            return Ok(Some(space));
        }
    }
}
```

Replace `run` with this state-based version:

```rust
fn run(args: Args) -> Result<RunSummary, CliError> {
    let comp = io::read_json(&args.input).map_err(|source| CliError::ReadInput {
        path: args.input.clone(),
        source,
    })?;
    let mut state = RewriteState::new(comp);

    if let Some(snapshot_dir) = &args.snapshot_dir {
        fs::create_dir_all(snapshot_dir).map_err(|source| CliError::CreateSnapshotDir {
            path: snapshot_dir.clone(),
            source,
        })?;
        write_snapshot(snapshot_dir, 0, state.computation())?;
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut applied_rewrites = 0;
    let mut stop_reason = StopReason::ReachedStepLimit;

    for step in 0..args.steps {
        let Some(space) = random_action_space_from_mask(&mut state, &mut rng)
            .map_err(|source| CliError::Rewrite { step, source })?
        else {
            stop_reason = StopReason::NoActionSpace;
            break;
        };

        let decision = random_decision(&space, args.random_subsets, &mut rng);
        state
            .step_with_space(&space, &decision)
            .map_err(|source| CliError::Rewrite { step, source })?;

        applied_rewrites += 1;

        if let Some(snapshot_dir) = &args.snapshot_dir {
            write_snapshot(snapshot_dir, applied_rewrites, state.computation())?;
        }
    }

    let comp = state.into_computation();
    io::write_json(&args.output, &comp).map_err(|source| CliError::WriteOutput {
        path: args.output.clone(),
        source,
    })?;

    Ok(RunSummary {
        seed: args.seed,
        requested_steps: args.steps,
        applied_rewrites,
        stop_reason,
    })
}
```

- [ ] **Step 4: Run CLI unit tests**

Run:

```bash
cargo test --bin random-rewrite -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Add seeded determinism integration test**

Append this test to `tests/random_rewrite_cli.rs`:

```rust
#[test]
fn seeded_runs_are_deterministic_under_random_definition_policy() {
    let case = TempCase::new("seeded-determinism");
    let input = case.path("input.json");
    let left_output = case.path("left-output.json");
    let right_output = case.path("right-output.json");
    let comp = comp_with_shared_left_candidate();
    write_json(&input, &comp).unwrap();

    run_random_rewrite_with_options(
        &["--random-subsets", "--seed", "17", "--steps", "2"],
        &[&input, &left_output],
    );
    run_random_rewrite_with_options(
        &["--random-subsets", "--seed", "17", "--steps", "2"],
        &[&input, &right_output],
    );

    assert_eq!(read_json(&left_output).unwrap(), read_json(&right_output).unwrap());
}
```

- [ ] **Step 6: Run CLI integration tests**

Run:

```bash
cargo test --test random_rewrite_cli -- --nocapture
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/bin/random-rewrite.rs tests/random_rewrite_cli.rs
git commit -m "feat: choose rewrite definitions from state mask"
```

---

### Task 3: Expose `RewriteState` Through PyO3

**Files:**
- Modify: `python/tests/test_bindings.py`
- Modify: `python/src/lib.rs`

- [ ] **Step 1: Rewrite binding tests around `RewriteState`**

In `python/tests/test_bindings.py`, update the import:

```python
from gristmill_symbolics import (
    ActionSpace,
    GristmillSymbolicsError,
    RewriteState,
    TensorComputation,
)
```

Update `test_module_exports_core_types`:

```python
def test_module_exports_core_types():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "RewriteState")
    assert hasattr(gristmill_symbolics, "ActionSpace")
    assert hasattr(gristmill_symbolics, "GristmillSymbolicsError")
```

Add this fixture helper after `actionable_json()`:

```python
def exact_empty_json() -> str:
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 8}],
            "tensors": [
                {"id": 0, "symmetry": []},
                {"id": 1, "symmetry": []},
            ],
            "definitions": [
                {
                    "base": 1,
                    "ext_indices": [{"id": 0, "range": 0}],
                    "terms": [
                        {
                            "coeff": [1, 1],
                            "sum_indices": [],
                            "factors": [{"tensor": 0, "indices": [0]}],
                        },
                        {
                            "coeff": [1, 1],
                            "sum_indices": [],
                            "factors": [{"tensor": 0, "indices": [0]}],
                        },
                    ],
                }
            ],
        }
    )
```

Replace the old action-space and apply tests with these `RewriteState` tests:

```python
def test_rewrite_state_from_computation_clones_input_computation():
    comp = TensorComputation.from_json_string(actionable_json())
    before = comp.snapshot()
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)

    state.step_with_space(space, first_full_decision(space))

    assert comp.snapshot() == before
    assert state.snapshot() != before


def test_rewrite_state_returns_none_for_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)
    state = RewriteState.from_computation(comp)

    assert state.definition_mask() == [False]
    assert state.action_space_for_def(0) is None


def test_rewrite_state_definition_mask_returns_copy():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    mask = state.definition_mask()

    mask[0] = False

    assert state.definition_mask() == [True]


def test_rewrite_state_refines_exact_empty_mask_to_false():
    comp = TensorComputation.from_json_string(exact_empty_json())
    state = RewriteState.from_computation(comp)

    assert state.definition_mask() == [True]
    assert state.action_space_for_def(0) is None
    assert state.definition_mask() == [False]


def test_rewrite_state_action_space_handle_and_public_snapshot():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)

    space = state.action_space_for_def(0)
    snapshot = space.snapshot()

    assert isinstance(space, ActionSpace)
    assert space.def_index == 0
    assert space.candidate_count == len(snapshot["candidate_templates"])
    assert space.candidate_count > 0
    assert set(snapshot) == {"def_index", "candidate_templates"}
    assert snapshot["def_index"] == 0
    first = snapshot["candidate_templates"][0]
    assert set(first) == {
        "left_definition",
        "right_definition",
        "rewritten_definition",
    }
    assert first["left_definition"]["terms"]
    assert first["right_definition"]["terms"]
    assert first["rewritten_definition"]["terms"]


def test_rewrite_state_step_with_space_mutates_state_and_returns_none():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)
    before = state.snapshot()
    decision = first_full_decision(space)

    result = state.step_with_space(space, decision)
    after = state.snapshot()

    assert result is None
    assert len(after["tensors"]) == len(before["tensors"]) + 2
    assert len(after["definitions"]) == len(before["definitions"]) + 2
    assert after != before
    assert len(state.definition_mask()) == len(after["definitions"])


def test_invalid_decision_raises_and_does_not_mutate():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)
    before = state.snapshot()
    bad_decision = {
        "candidate_index": 0,
        "left_mask": [],
        "right_mask": [True],
    }

    with pytest.raises(GristmillSymbolicsError):
        state.step_with_space(space, bad_decision)

    assert state.snapshot() == before


def test_malformed_decision_shape_raises_type_or_value_error():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)

    with pytest.raises(TypeError):
        state.step_with_space(space, "not a dict")

    with pytest.raises(ValueError):
        state.step_with_space(
            space,
            {"candidate_index": 0, "left_mask": [True]},
        )

    with pytest.raises(TypeError):
        state.step_with_space(
            space,
            {"candidate_index": True, "left_mask": [True], "right_mask": [True]},
        )

    with pytest.raises(ValueError):
        state.step_with_space(
            space,
            {"candidate_index": -1, "left_mask": [True], "right_mask": [True]},
        )

    with pytest.raises(ValueError):
        state.step_with_space(
            space,
            {"candidate_index": 2**128, "left_mask": [True], "right_mask": [True]},
        )

    with pytest.raises(TypeError):
        state.step_with_space(
            space,
            {"candidate_index": 0, "left_mask": True, "right_mask": [True]},
        )

    with pytest.raises(TypeError):
        state.step_with_space(
            space,
            {"candidate_index": 0, "left_mask": [1], "right_mask": [True]},
        )


def test_action_space_handle_is_reusable_on_multiple_states():
    comp = TensorComputation.from_json_string(actionable_json())
    source_state = RewriteState.from_computation(comp)
    space = source_state.action_space_for_def(0)
    decision = first_full_decision(space)
    left = RewriteState.from_computation(comp)
    right = RewriteState.from_computation(comp)

    left.step_with_space(space, decision)
    right.step_with_space(space, decision)

    assert left.snapshot() == right.snapshot()
```

Update `test_write_json_round_trips_rewritten_computation`:

```python
def test_write_json_round_trips_rewritten_computation(tmp_path):
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)
    assert space is not None
    template = space.snapshot()["candidate_templates"][0]
    state.step_with_space(
        space,
        {
            "candidate_index": 0,
            "left_mask": [True] * len(template["left_definition"]["terms"]),
            "right_mask": [True] * len(template["right_definition"]["terms"]),
        },
    )
    output = tmp_path / "rewritten.json"

    state.write_json(output)
    loaded = TensorComputation.load_json(output)

    assert loaded.snapshot() == state.snapshot()
```

- [ ] **Step 2: Run binding tests to verify they fail**

Run from `python/`:

```bash
uv run pytest tests/test_bindings.py -q
```

Expected: FAIL because `RewriteState` is not exported by the extension.

- [ ] **Step 3: Implement PyO3 `RewriteState` wrapper**

In `python/src/lib.rs`, add `RewriteState as RustRewriteState` to the rewrite import:

```rust
use ::gristmill_symbolics::rewrite::{
    ActionSpace as RustActionSpace, Decision, Factorization, RewriteState as RustRewriteState,
};
```

Remove `next_action_space` and `apply_decision_with_space` from `impl PyTensorComputation`.

Add this class after `impl PyTensorComputation`:

```rust
#[pyclass(name = "RewriteState")]
struct PyRewriteState {
    inner: RustRewriteState,
}

#[pymethods]
impl PyRewriteState {
    #[staticmethod]
    fn from_computation(comp: &PyTensorComputation) -> Self {
        Self {
            inner: RustRewriteState::new(comp.inner.clone()),
        }
    }

    fn definition_mask(&self) -> Vec<bool> {
        self.inner.definition_mask().to_vec()
    }

    fn action_space_for_def(&mut self, def_index: usize) -> PyResult<Option<PyActionSpace>> {
        self.inner
            .action_space_for_def(def_index)
            .map(|space| space.map(|inner| PyActionSpace { inner }))
            .map_err(py_gristmill_error)
    }

    fn step_with_space(
        &mut self,
        space: &PyActionSpace,
        decision: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let decision = parse_decision(decision)?;
        self.inner
            .step_with_space(&space.inner, &decision)
            .map_err(py_gristmill_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &computation_value(self.inner.computation()))
            .map_err(py_gristmill_display_error)
    }

    fn log_total_flops(&self) -> PyResult<f64> {
        cost::log_total_flops(self.inner.computation()).map_err(py_gristmill_error)
    }

    fn to_json_string(&self) -> PyResult<String> {
        io::to_json(self.inner.computation()).map_err(py_gristmill_display_error)
    }

    fn write_json(&self, path: PathBuf) -> PyResult<()> {
        io::write_json(path, self.inner.computation()).map_err(py_gristmill_display_error)
    }
}
```

Update the module registration:

```rust
module.add_class::<PyTensorComputation>()?;
module.add_class::<PyRewriteState>()?;
module.add_class::<PyActionSpace>()?;
```

- [ ] **Step 4: Build the extension and run binding tests**

Run from `python/`:

```bash
uv run maturin develop
uv run pytest tests/test_bindings.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add python/src/lib.rs python/tests/test_bindings.py
git commit -m "feat: expose rewrite state bindings"
```

---

### Task 4: Remove Cursor API And Finish Migration

**Files:**
- Modify: `src/rewrite.rs`
- Modify: `tests/rewrite.rs`

- [ ] **Step 1: Rewrite remaining Rust integration tests away from `next_action_space`**

In `tests/rewrite.rs`, replace the import with:

```rust
use gristmill_symbolics::rewrite::{
    Decision, Factorization, FactorizationRewrite, RewriteError, RewriteState, apply_rewrite,
    build_rewrite,
};
```

Replace `next_action_space_returns_none_when_no_definition_is_actionable` with:

```rust
#[test]
fn rewrite_state_returns_none_when_no_definition_is_actionable() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);
    let mut state = RewriteState::new(comp);

    assert_eq!(state.action_space_for_def(0), Ok(None));
}
```

Replace `next_action_space_returns_first_actionable_definition` with:

```rust
#[test]
fn rewrite_state_returns_action_space_for_selected_definition() {
    let mut state = RewriteState::new(comp_with_unsplittable_then_actionable_definition());

    assert_eq!(state.action_space_for_def(0), Ok(None));
    let space = state.action_space_for_def(1).unwrap().unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}
```

Replace the three error propagation tests with:

```rust
#[test]
fn action_space_for_def_propagates_split_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    let many_factors = (0..65).map(|_| factor(a, &[])).collect();

    comp.add_definition(
        out,
        vec![],
        vec![
            term(vec![], many_factors),
            term(vec![], vec![factor(a, &[])]),
        ],
    );
    let mut state = RewriteState::new(comp);

    assert_eq!(
        state.action_space_for_def(0),
        Err(RewriteError::Split(SplitError::TooManyFactors {
            len: 65,
            max: 64,
        }))
    );
}

#[test]
fn action_space_for_def_propagates_canon_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let out = comp.add_tensor(vec![]);
    let missing = TensorId(99);

    comp.add_definition(
        out,
        vec![],
        vec![
            term(vec![], vec![factor(missing, &[]), factor(missing, &[])]),
            term(vec![], vec![factor(missing, &[]), factor(missing, &[])]),
        ],
    );
    let mut state = RewriteState::new(comp);

    assert_eq!(
        state.action_space_for_def(0),
        Err(RewriteError::Canon(CanonError::MissingTensorSymmetry {
            tensor: missing,
        }))
    );
}

#[test]
fn action_space_for_def_propagates_graph_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(128);
    let a = comp.add_tensor(vec![]);
    let b = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    let terms: Vec<_> = (0..65)
        .map(|offset| {
            let sum_id = 2 + offset;
            term(
                vec![idx(sum_id)],
                vec![factor(a, &[0, sum_id]), factor(b, &[sum_id, 1])],
            )
        })
        .collect();

    comp.add_definition(out, vec![idx(0), idx(1)], terms);
    let mut state = RewriteState::new(comp);

    assert_eq!(
        state.action_space_for_def(0),
        Err(RewriteError::Graph(GraphError::TooManyTerms {
            len: 65,
            max: 64,
        }))
    );
}
```

In `apply_rewrite_registers_tensors_inserts_definitions_and_validates` and `apply_rewrite_only_checks_definition_index_after_rewrite_construction`, replace:

```rust
let space = next_action_space(&comp, 0).unwrap().unwrap();
```

with:

```rust
let mut state = RewriteState::new(comp.clone());
let space = state.action_space_for_def(0).unwrap().unwrap();
```

- [ ] **Step 2: Run tests to verify migration still passes before deletion**

Run:

```bash
cargo test --test rewrite -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Delete the cursor API from `src/rewrite.rs`**

Remove this function from `src/rewrite.rs`:

```rust
pub fn next_action_space(
    comp: &TensorComputation,
    start_from: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    for def_index in start_from..comp.definitions().len() {
        if let Some(space) = action_space_for_definition(comp, def_index)? {
            return Ok(Some(space));
        }
    }
    Ok(None)
}
```

Keep `action_space_for_definition` as a private helper used by `RewriteState`.

- [ ] **Step 4: Confirm no Rust/PyO3/CLI/test cursor usage remains**

Run:

```bash
rg -n "next_action_space|apply_decision_with_space" src tests python/src python/tests/test_bindings.py
```

Expected: no matches. Matches under `python/gristmill_rl` or other Python RL tests are not part of this task.

- [ ] **Step 5: Run migrated test surfaces**

Run:

```bash
cargo test --test rewrite -- --nocapture
cargo test --test random_rewrite_cli -- --nocapture
```

Expected: PASS.

Run from `python/`:

```bash
uv run maturin develop
uv run pytest tests/test_bindings.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/rewrite.rs tests/rewrite.rs python/src/lib.rs python/tests/test_bindings.py
git commit -m "refactor: remove cursor rewrite api"
```

---

### Task 5: Final Verification

**Files:**
- No planned code changes.

- [ ] **Step 1: Format Rust code**

Run:

```bash
cargo fmt
```

Expected: command exits 0 and formats touched Rust files.

- [ ] **Step 2: Run full Rust tests**

Run:

```bash
cargo test
```

Expected: PASS.

- [ ] **Step 3: Rebuild PyO3 extension and run binding tests**

Run from `python/`:

```bash
uv run maturin develop
uv run pytest tests/test_bindings.py -q
```

Expected: PASS. Do not run the full Python test suite for this refactor because `python/gristmill_rl` intentionally still references the removed cursor API.

- [ ] **Step 4: Check removed public cursor surface**

Run:

```bash
rg -n "next_action_space|apply_decision_with_space" src tests python/src python/tests/test_bindings.py
```

Expected: no matches.

- [ ] **Step 5: Commit formatting-only changes if any exist**

Run:

```bash
git status --short
```

If only formatting changes from `cargo fmt` remain, commit them:

```bash
git add src/rewrite.rs src/bin/random-rewrite.rs tests/rewrite.rs tests/random_rewrite_cli.rs python/src/lib.rs python/tests/test_bindings.py
git commit -m "style: format rewrite state refactor"
```

If `git status --short` is clean or only shows unrelated pre-existing files such as `.superpowers/`, do not create a commit.

---

## Self-Review

- Spec coverage: Task 1 implements Rust `RewriteState`, lazy exact definition queries, mask refinement, mutation-first stepping, and cheap mask updates. Task 2 implements random definition choice in the CLI. Task 3 implements faithful PyO3 conversion. Task 4 removes the old public cursor API from Rust/PyO3 usage. Task 5 verifies the accepted surfaces while leaving Python RL repair out of scope.
- Scope check: the plan is one cohesive foundation refactor. It does not implement REINFORCE, MCTS changes, `STOP`, replay, model changes, or `TensorDef` embedding.
- Type consistency: Rust uses `RewriteState`, `ActionSpace`, `Decision`, `RewriteError`, `action_space_for_def`, `step_with_space`, `definition_mask`, `computation`, and `into_computation` consistently. Python uses `RewriteState.from_computation`, `definition_mask`, `action_space_for_def`, `step_with_space`, `snapshot`, `log_total_flops`, `to_json_string`, and `write_json` consistently.
