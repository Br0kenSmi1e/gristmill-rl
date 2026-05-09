# repr Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `repr` module as the JSON-compatible symbolic data model for the rewrite kernel, including builders, symmetry helpers, and structural validation.

**Architecture:** Turn the package into a library crate that exposes a single `repr` module through `src/lib.rs`. Keep the schema, helper methods, and validation in `src/repr.rs`, and keep compatibility checks in integration tests so they exercise the public API exactly as callers will use it.

**Tech Stack:** Rust 2024, `serde`, `serde_json`, `num` with serde support, standard library collections.

---

### Task 1: Scaffold the library and the core data model

**Files:**
- Modify: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/repr.rs`
- Create: `tests/repr_schema.rs`

- [ ] **Step 1: Write the failing test**

```rust
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, SymAction, SymGenerator, TensorComputation,
    TensorId, Term,
};

#[test]
fn builders_assign_ids_and_accessors_work() {
    let mut comp = TensorComputation::new();
    assert_eq!(TensorComputation::default(), TensorComputation::new());
    assert_eq!(comp.ranges(), &[]);
    assert_eq!(comp.tensors(), &[]);
    assert_eq!(comp.definitions(), &[]);
    assert_eq!(comp.next_tensor_id(), TensorId(0));

    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![]);
    comp.add_definition(tensor_id, vec![], vec![]);

    assert_eq!(range_id, RangeId(0));
    assert_eq!(tensor_id, TensorId(0));
    assert_eq!(comp.ranges()[0].size, 3);
    assert_eq!(comp.tensors()[0].id, tensor_id);
    assert_eq!(comp.definitions()[0].base, tensor_id);
    assert_eq!(comp.next_tensor_id(), TensorId(1));
}

#[test]
fn serde_keeps_compatibility_fields() {
    let mut comp = TensorComputation::new();
    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![SymGenerator {
        perm: vec![0],
        action: SymAction::Identity,
    }]);

    comp.add_definition(
        tensor_id,
        vec![Index {
            id: IndexId(0),
            range: range_id,
        }],
        vec![Term {
            coeff: num::rational::Ratio::new(1, 1),
            sum_indices: vec![],
            factors: vec![Factor {
                tensor: tensor_id,
                indices: vec![IndexId(0)],
            }],
        }],
    );

    let json = serde_json::to_string(&comp).unwrap();
    for field in [
        "ranges",
        "tensors",
        "definitions",
        "id",
        "size",
        "symmetry",
        "perm",
        "action",
        "range",
        "base",
        "ext_indices",
        "terms",
        "coeff",
        "sum_indices",
        "factors",
        "tensor",
        "indices",
    ] {
        assert!(json.contains(&format!("\"{field}\"")));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test repr_schema -- --nocapture`

Expected: compile failure because the `repr` module and library API do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```toml
# Cargo.toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
num = { version = "0.4", features = ["serde"] }
```

```rust
// src/lib.rs
pub mod repr;
```

```rust
// src/repr.rs
use num::rational::Ratio;
use serde::{Deserialize, Serialize};

pub type Rational = Ratio<i64>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RangeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymAction {
    Identity,
    Negate,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymGenerator {
    pub perm: Vec<usize>,
    pub action: SymAction,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Range {
    pub id: RangeId,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Index {
    pub id: IndexId,
    pub range: RangeId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub id: TensorId,
    pub symmetry: Vec<SymGenerator>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Factor {
    pub tensor: TensorId,
    pub indices: Vec<IndexId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Term {
    pub coeff: Rational,
    pub sum_indices: Vec<Index>,
    pub factors: Vec<Factor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDef {
    pub base: TensorId,
    pub ext_indices: Vec<Index>,
    pub terms: Vec<Term>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorComputation {
    ranges: Vec<Range>,
    tensors: Vec<TensorInfo>,
    definitions: Vec<TensorDef>,
}

impl TensorComputation {
    pub fn new() -> Self {
        Self {
            ranges: vec![],
            tensors: vec![],
            definitions: vec![],
        }
    }

    pub fn add_range(&mut self, size: u64) -> RangeId {
        let id = RangeId(self.ranges.len() as u32);
        self.ranges.push(Range { id, size });
        id
    }

    pub fn add_tensor(&mut self, symmetry: Vec<SymGenerator>) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorInfo { id, symmetry });
        id
    }

    pub fn add_definition(
        &mut self,
        base: TensorId,
        ext_indices: Vec<Index>,
        terms: Vec<Term>,
    ) {
        self.definitions.push(TensorDef {
            base,
            ext_indices,
            terms,
        });
    }

    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn definitions(&self) -> &[TensorDef] {
        &self.definitions
    }

    pub fn definitions_mut(&mut self) -> &mut Vec<TensorDef> {
        &mut self.definitions
    }

    pub fn next_tensor_id(&self) -> TensorId {
        TensorId(self.tensors.len() as u32)
    }
}

impl Default for TensorComputation {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test repr_schema -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/lib.rs src/repr.rs tests/repr_schema.rs
git commit -m "feat: add repr schema scaffold"
```

### Task 2: Add symmetry helpers and helper-failure errors

**Files:**
- Modify: `src/repr.rs`
- Create: `tests/repr_symmetry.rs`

- [ ] **Step 1: Write the failing test**

```rust
use gristmill_symbolics::repr::{ReprError, SymAction, SymGenerator};

#[test]
fn sym_action_combines_signs() {
    assert_eq!(SymAction::Identity.combine(SymAction::Identity), SymAction::Identity);
    assert_eq!(SymAction::Identity.combine(SymAction::Negate), SymAction::Negate);
    assert_eq!(SymAction::Negate.combine(SymAction::Identity), SymAction::Negate);
    assert_eq!(SymAction::Negate.combine(SymAction::Negate), SymAction::Identity);
}

#[test]
fn sym_generator_applies_permutation_and_action() {
    let generator = SymGenerator {
        perm: vec![1, 0],
        action: SymAction::Negate,
    };

    let (indices, action) = generator.apply(&[10, 20]).unwrap();
    assert_eq!(indices, vec![20, 10]);
    assert_eq!(action, SymAction::Negate);
}

#[test]
fn sym_generator_apply_rejects_arity_mismatch() {
    let generator = SymGenerator {
        perm: vec![0, 1],
        action: SymAction::Identity,
    };

    assert_eq!(
        generator.apply(&[7]),
        Err(ReprError::SymmetryArityMismatch {
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn sym_generator_apply_rejects_invalid_permutation() {
    let generator = SymGenerator {
        perm: vec![0, 0],
        action: SymAction::Identity,
    };

    assert_eq!(
        generator.apply(&[7, 8]),
        Err(ReprError::InvalidPermutation { perm: vec![0, 0] })
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test repr_symmetry -- --nocapture`

Expected: compile failure because `combine`, `apply`, and `ReprError` are not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReprError {
    RangeIdMismatch { position: usize, found: RangeId },
    TensorIdMismatch { position: usize, found: TensorId },
    UnknownRange { range: RangeId },
    UnknownTensor { tensor: TensorId },
    UnknownIndex {
        def_index: usize,
        term_index: usize,
        index: IndexId,
    },
    InconsistentIndexRange {
        def_index: usize,
        index: IndexId,
        first: RangeId,
        second: RangeId,
    },
    DuplicateExternalIndex {
        def_index: usize,
        index: IndexId,
    },
    ExternalAndSumIndexOverlap {
        def_index: usize,
        index: IndexId,
    },
    DuplicateSumIndex {
        def_index: usize,
        term_index: usize,
        index: IndexId,
    },
    InvalidPermutation {
        perm: Vec<usize>,
    },
    SymmetryArityMismatch {
        expected: usize,
        got: usize,
    },
}

impl SymAction {
    pub fn combine(self, other: SymAction) -> SymAction {
        match (self, other) {
            (SymAction::Identity, rhs) => rhs,
            (SymAction::Negate, SymAction::Identity) => SymAction::Negate,
            (SymAction::Negate, SymAction::Negate) => SymAction::Identity,
        }
    }
}

impl SymGenerator {
    pub fn apply<T: Copy>(&self, indices: &[T]) -> Result<(Vec<T>, SymAction), ReprError> {
        if self.perm.len() != indices.len() {
            return Err(ReprError::SymmetryArityMismatch {
                expected: self.perm.len(),
                got: indices.len(),
            });
        }

        let mut seen = vec![false; self.perm.len()];
        for &position in &self.perm {
            if position >= self.perm.len() || seen[position] {
                return Err(ReprError::InvalidPermutation {
                    perm: self.perm.clone(),
                });
            }
            seen[position] = true;
        }

        Ok((
            self.perm.iter().map(|&position| indices[position]).collect(),
            self.action,
        ))
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test repr_symmetry -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/repr.rs tests/repr_symmetry.rs
git commit -m "feat: add repr symmetry helpers"
```

### Task 3: Implement structural validation and the overlap rule

**Files:**
- Modify: `src/repr.rs`
- Create: `tests/repr_validation.rs`

- [ ] **Step 1: Write the failing test**

```rust
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, ReprError, SymAction, SymGenerator,
    TensorComputation, TensorId, Term,
};
use serde_json::json;

fn one() -> num::rational::Ratio<i64> {
    num::rational::Ratio::new(1, 1)
}

fn well_formed_computation() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![SymGenerator {
        perm: vec![0],
        action: SymAction::Identity,
    }]);

    comp.add_definition(
        tensor_id,
        vec![Index {
            id: IndexId(0),
            range: range_id,
        }],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![Factor {
                tensor: tensor_id,
                indices: vec![IndexId(0)],
            }],
        }],
    );

    comp
}

#[test]
fn validate_accepts_a_well_formed_computation() {
    well_formed_computation().validate().unwrap();
}

#[test]
fn validate_rejects_id_position_mismatches() {
    let range_bad: TensorComputation = serde_json::from_value(json!({
        "ranges": [{ "id": 7, "size": 3 }],
        "tensors": [],
        "definitions": []
    }))
    .unwrap();
    assert_eq!(
        range_bad.validate(),
        Err(ReprError::RangeIdMismatch {
            position: 0,
            found: RangeId(7),
        })
    );

    let tensor_bad: TensorComputation = serde_json::from_value(json!({
        "ranges": [],
        "tensors": [{ "id": 2, "symmetry": [] }],
        "definitions": []
    }))
    .unwrap();
    assert_eq!(
        tensor_bad.validate(),
        Err(ReprError::TensorIdMismatch {
            position: 0,
            found: TensorId(2),
        })
    );
}

#[test]
fn validate_rejects_unknown_references() {
    let mut unknown_range = well_formed_computation();
    unknown_range.definitions_mut()[0].ext_indices[0].range = RangeId(99);
    assert_eq!(
        unknown_range.validate(),
        Err(ReprError::UnknownRange { range: RangeId(99) })
    );

    let mut unknown_base = well_formed_computation();
    unknown_base.definitions_mut()[0].base = TensorId(99);
    assert_eq!(
        unknown_base.validate(),
        Err(ReprError::UnknownTensor {
            tensor: TensorId(99),
        })
    );

    let mut unknown_factor_tensor = well_formed_computation();
    unknown_factor_tensor.definitions_mut()[0].terms[0].factors[0].tensor = TensorId(99);
    assert_eq!(
        unknown_factor_tensor.validate(),
        Err(ReprError::UnknownTensor {
            tensor: TensorId(99),
        })
    );

    let mut unknown_index = well_formed_computation();
    unknown_index.definitions_mut()[0].terms[0].factors[0].indices = vec![IndexId(99)];
    assert_eq!(
        unknown_index.validate(),
        Err(ReprError::UnknownIndex {
            def_index: 0,
            term_index: 0,
            index: IndexId(99),
        })
    );
}

#[test]
fn validate_rejects_factor_index_declared_only_in_another_term() {
    let mut comp = TensorComputation::new();
    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![]);
    comp.add_definition(
        tensor_id,
        vec![],
        vec![
            Term {
                coeff: one(),
                sum_indices: vec![Index {
                    id: IndexId(1),
                    range: range_id,
                }],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![Factor {
                    tensor: tensor_id,
                    indices: vec![IndexId(1)],
                }],
            },
        ],
    );

    assert_eq!(
        comp.validate(),
        Err(ReprError::UnknownIndex {
            def_index: 0,
            term_index: 1,
            index: IndexId(1),
        })
    );
}

#[test]
fn validate_rejects_inconsistent_index_ranges() {
    let mut comp = TensorComputation::new();
    let range_0 = comp.add_range(3);
    let range_1 = comp.add_range(5);
    let tensor_id = comp.add_tensor(vec![]);
    comp.add_definition(
        tensor_id,
        vec![Index {
            id: IndexId(0),
            range: range_0,
        }],
        vec![Term {
            coeff: one(),
            sum_indices: vec![Index {
                id: IndexId(0),
                range: range_1,
            }],
            factors: vec![],
        }],
    );

    assert_eq!(
        comp.validate(),
        Err(ReprError::InconsistentIndexRange {
            def_index: 0,
            index: IndexId(0),
            first: range_0,
            second: range_1,
        })
    );
}

#[test]
fn validate_rejects_duplicate_index_declarations() {
    let mut duplicate_external = well_formed_computation();
    duplicate_external.definitions_mut()[0].ext_indices.push(Index {
        id: IndexId(0),
        range: RangeId(0),
    });
    assert_eq!(
        duplicate_external.validate(),
        Err(ReprError::DuplicateExternalIndex {
            def_index: 0,
            index: IndexId(0),
        })
    );

    let mut duplicate_sum = well_formed_computation();
    duplicate_sum.definitions_mut()[0].terms[0].sum_indices = vec![
        Index {
            id: IndexId(1),
            range: RangeId(0),
        },
        Index {
            id: IndexId(1),
            range: RangeId(0),
        },
    ];
    assert_eq!(
        duplicate_sum.validate(),
        Err(ReprError::DuplicateSumIndex {
            def_index: 0,
            term_index: 0,
            index: IndexId(1),
        })
    );
}

#[test]
fn validate_rejects_external_and_sum_overlap() {
    let mut comp = well_formed_computation();
    let def = &mut comp.definitions_mut()[0];
    def.terms[0].sum_indices = vec![Index {
        id: IndexId(0),
        range: RangeId(0),
    }];

    assert_eq!(
        comp.validate(),
        Err(ReprError::ExternalAndSumIndexOverlap {
            def_index: 0,
            index: IndexId(0),
        })
    );
}

#[test]
fn validate_rejects_invalid_symmetry_permutation() {
    let mut comp = TensorComputation::new();
    comp.add_tensor(vec![SymGenerator {
        perm: vec![0, 0],
        action: SymAction::Identity,
    }]);

    assert_eq!(
        comp.validate(),
        Err(ReprError::InvalidPermutation { perm: vec![0, 0] })
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test repr_validation -- --nocapture`

Expected: compile failure because `validate` is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```rust
impl TensorComputation {
    pub fn validate(&self) -> Result<(), ReprError> {
        use std::collections::{HashMap, HashSet};

        for (position, range) in self.ranges.iter().enumerate() {
            if range.id != RangeId(position as u32) {
                return Err(ReprError::RangeIdMismatch {
                    position,
                    found: range.id,
                });
            }
        }

        for (position, tensor) in self.tensors.iter().enumerate() {
            if tensor.id != TensorId(position as u32) {
                return Err(ReprError::TensorIdMismatch {
                    position,
                    found: tensor.id,
                });
            }

            for generator in &tensor.symmetry {
                let mut seen = vec![false; generator.perm.len()];
                for &index in &generator.perm {
                    if index >= generator.perm.len() || seen[index] {
                        return Err(ReprError::InvalidPermutation {
                            perm: generator.perm.clone(),
                        });
                    }
                    seen[index] = true;
                }
            }
        }

        let range_ids: HashSet<_> = self.ranges.iter().map(|range| range.id).collect();
        let tensor_ids: HashSet<_> = self.tensors.iter().map(|tensor| tensor.id).collect();

        for (def_index, def) in self.definitions.iter().enumerate() {
            if !tensor_ids.contains(&def.base) {
                return Err(ReprError::UnknownTensor { tensor: def.base });
            }

            let mut external_ids: HashMap<IndexId, RangeId> = HashMap::new();
            let mut sum_ids: HashMap<IndexId, RangeId> = HashMap::new();

            for index in &def.ext_indices {
                if !range_ids.contains(&index.range) {
                    return Err(ReprError::UnknownRange { range: index.range });
                }
                if external_ids.insert(index.id, index.range).is_some() {
                    return Err(ReprError::DuplicateExternalIndex {
                        def_index,
                        index: index.id,
                    });
                }
            }

            for (term_index, term) in def.terms.iter().enumerate() {
                let mut seen_sum = HashSet::new();
                let mut term_sum_ids = HashMap::new();
                for index in &term.sum_indices {
                    if !range_ids.contains(&index.range) {
                        return Err(ReprError::UnknownRange { range: index.range });
                    }
                    if !seen_sum.insert(index.id) {
                        return Err(ReprError::DuplicateSumIndex {
                            def_index,
                            term_index,
                            index: index.id,
                        });
                    }

                    term_sum_ids.insert(index.id, index.range);

                    if let Some(first) = sum_ids.insert(index.id, index.range) {
                        if first != index.range {
                            return Err(ReprError::InconsistentIndexRange {
                                def_index,
                                index: index.id,
                                first,
                                second: index.range,
                            });
                        }
                    }

                    if let Some(ext_range) = external_ids.get(&index.id) {
                        if *ext_range != index.range {
                            return Err(ReprError::InconsistentIndexRange {
                                def_index,
                                index: index.id,
                                first: *ext_range,
                                second: index.range,
                            });
                        }
                        return Err(ReprError::ExternalAndSumIndexOverlap {
                            def_index,
                            index: index.id,
                        });
                    }
                }

                for factor in &term.factors {
                    if !tensor_ids.contains(&factor.tensor) {
                        return Err(ReprError::UnknownTensor {
                            tensor: factor.tensor,
                        });
                    }

                    for index in &factor.indices {
                        if !external_ids.contains_key(index) && !term_sum_ids.contains_key(index) {
                            return Err(ReprError::UnknownIndex {
                                def_index,
                                term_index,
                                index: *index,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test repr_validation -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/repr.rs tests/repr_validation.rs
git commit -m "feat: add repr validation"
```

### Task 4: Add fixture-based compatibility coverage

**Files:**
- Create: `tests/fixtures/repr/basic.json`
- Create: `tests/fixtures/repr/legacy_conjugate.json`
- Create: `tests/repr_fixtures.rs`

- [ ] **Step 1: Write the failing test**

```rust
use gristmill_symbolics::repr::TensorComputation;

#[test]
fn compatible_fixture_round_trips() {
    let json = include_str!("fixtures/repr/basic.json");
    let comp: TensorComputation = serde_json::from_str(json).unwrap();
    let round_trip = serde_json::to_string(&comp).unwrap();
    let reparsed: TensorComputation = serde_json::from_str(&round_trip).unwrap();
    assert_eq!(comp, reparsed);
}

#[test]
fn legacy_conjugate_actions_are_rejected() {
    let json = include_str!("fixtures/repr/legacy_conjugate.json");
    assert!(serde_json::from_str::<TensorComputation>(json).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test repr_fixtures -- --nocapture`

Expected: compile failure because the `include_str!` fixture paths do not exist yet.

- [ ] **Step 3: Write minimal implementation**

No new runtime code is needed if Tasks 1-3 are complete. Add the fixture files and keep the test as pure public-API coverage.

`tests/fixtures/repr/basic.json`

```json
{
  "ranges": [
    { "id": 0, "size": 3 }
  ],
  "tensors": [
    {
      "id": 0,
      "symmetry": [
        { "perm": [0], "action": "Identity" }
      ]
    }
  ],
  "definitions": [
    {
      "base": 0,
      "ext_indices": [
        { "id": 0, "range": 0 }
      ],
      "terms": [
        {
          "coeff": [1, 1],
          "sum_indices": [],
          "factors": [
            { "tensor": 0, "indices": [0] }
          ]
        }
      ]
    }
  ]
}
```

`tests/fixtures/repr/legacy_conjugate.json`

```json
{
  "ranges": [
    { "id": 0, "size": 3 }
  ],
  "tensors": [
    {
      "id": 0,
      "symmetry": [
        { "perm": [0], "action": "Conjugate" }
      ]
    }
  ],
  "definitions": []
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test repr_fixtures -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/repr/basic.json tests/fixtures/repr/legacy_conjugate.json tests/repr_fixtures.rs
git commit -m "test: add repr compatibility fixtures"
```
