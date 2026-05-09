# split Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `split` module that enumerates structural factor bipartitions for one `Term` and emits explicit `SplitInterface` records.

**Architecture:** Add a focused `src/split.rs` module and expose it from `src/lib.rs`. The module keeps all bitmask and helper machinery private, reports fixed-width mask limits through `SplitError`, and returns public `Split` records whose interface vectors are explicit sorted `Index` values.

**Tech Stack:** Rust 2024, existing `repr` module, standard library collections, `num::rational::Ratio` through `repr::Rational`.

---

## File Structure

- Create `src/split.rs`: public `SplitInterface`, `Split`, `SplitError`, `enumerate_splits`, and private helper pipeline.
- Modify `src/lib.rs`: expose `pub mod split;`.
- Create `tests/split.rs`: integration tests for public split behavior only.

---

### Task 1: Add the public split API and empty-term behavior

**Files:**
- Modify: `src/lib.rs`
- Create: `src/split.rs`
- Create: `tests/split.rs`

- [ ] **Step 1: Write the failing test**

```rust
use gristmill_symbolics::repr::{IndexId, Rational, TensorDef, TensorId, Term};
use gristmill_symbolics::split::enumerate_splits;

fn one() -> Rational {
    Rational::new(1, 1)
}

#[test]
fn terms_with_fewer_than_two_factors_produce_no_splits() {
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![],
    };

    let zero_factor = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![],
    };
    assert_eq!(enumerate_splits(&zero_factor, &def).unwrap(), vec![]);

    let one_factor = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![gristmill_symbolics::repr::Factor {
            tensor: TensorId(0),
            indices: vec![IndexId(0)],
        }],
    };
    assert_eq!(enumerate_splits(&one_factor, &def).unwrap(), vec![]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test split terms_with_fewer_than_two_factors_produce_no_splits -- --nocapture`

Expected: compile failure because `gristmill_symbolics::split` does not exist.

- [ ] **Step 3: Add the public API and minimal implementation**

```rust
// src/lib.rs
pub mod repr;
pub mod split;
```

```rust
// src/split.rs
use crate::repr::{Index, TensorDef, Term};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SplitInterface {
    pub left_external: Vec<Index>,
    pub right_external: Vec<Index>,
    pub contracted: Vec<Index>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub left: Term,
    pub right: Term,
    pub interface: SplitInterface,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SplitError {
    TooManyFactors { len: usize, max: usize },
    TooManySumIndices { len: usize, max: usize },
    TooManyExternalIndices { len: usize, max: usize },
}

pub fn enumerate_splits(term: &Term, _def: &TensorDef) -> Result<Vec<Split>, SplitError> {
    if term.factors.len() < 2 {
        return Ok(vec![]);
    }

    Ok(vec![])
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test split terms_with_fewer_than_two_factors_produce_no_splits -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/split.rs tests/split.rs
git commit -m "feat: add split module API"
```

---

### Task 2: Enumerate one two-factor split with unit subterms

**Files:**
- Modify: `src/split.rs`
- Modify: `tests/split.rs`

- [ ] **Step 1: Add the failing test**

Append to `tests/split.rs`:

```rust
use gristmill_symbolics::repr::{Factor, Index, RangeId};
use gristmill_symbolics::split::{Split, SplitInterface};

#[test]
fn two_factor_term_produces_one_unit_coefficient_split() {
    let range = RangeId(0);
    let a = Index {
        id: IndexId(0),
        range,
    };
    let b = Index {
        id: IndexId(1),
        range,
    };
    let c = Index {
        id: IndexId(2),
        range,
    };
    let x = TensorId(0);
    let y = TensorId(1);

    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![a, b],
        terms: vec![],
    };
    let term = Term {
        coeff: Rational::new(7, 3),
        sum_indices: vec![c],
        factors: vec![
            Factor {
                tensor: x,
                indices: vec![a.id, c.id],
            },
            Factor {
                tensor: y,
                indices: vec![c.id, b.id],
            },
        ],
    };

    assert_eq!(
        enumerate_splits(&term, &def).unwrap(),
        vec![Split {
            left: Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![Factor {
                    tensor: x,
                    indices: vec![a.id, c.id],
                }],
            },
            right: Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![Factor {
                    tensor: y,
                    indices: vec![c.id, b.id],
                }],
            },
            interface: SplitInterface {
                left_external: vec![a],
                right_external: vec![b],
                contracted: vec![c],
            },
        }]
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test split two_factor_term_produces_one_unit_coefficient_split -- --nocapture`

Expected: FAIL because `enumerate_splits` still returns an empty vector for two-factor terms.

- [ ] **Step 3: Implement internal masks and two-factor-capable split construction**

Replace `src/split.rs` with:

```rust
use crate::repr::{Index, IndexId, Rational, TensorDef, Term};
use std::collections::{HashMap, HashSet};

const MAX_MASK_BITS: usize = 64;

type FactorSubset = u64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SplitInterface {
    pub left_external: Vec<Index>,
    pub right_external: Vec<Index>,
    pub contracted: Vec<Index>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub left: Term,
    pub right: Term,
    pub interface: SplitInterface,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SplitError {
    TooManyFactors { len: usize, max: usize },
    TooManySumIndices { len: usize, max: usize },
    TooManyExternalIndices { len: usize, max: usize },
}

struct TermIndexInfo {
    factor_sum_bits: Vec<u64>,
    factor_external_bits: Vec<u64>,
}

pub fn enumerate_splits(term: &Term, def: &TensorDef) -> Result<Vec<Split>, SplitError> {
    if term.factors.len() < 2 {
        return Ok(vec![]);
    }

    validate_mask_len(term.factors.len(), SplitLimit::Factors)?;
    let info = build_term_index_info(term, def)?;
    let full = full_factor_mask(term.factors.len());
    let mut out = Vec::new();

    for mut left in 1..full {
        let mut right = full ^ left;
        if left >= right {
            continue;
        }

        if subset_external_bits(&info, left) > subset_external_bits(&info, right) {
            std::mem::swap(&mut left, &mut right);
        }

        out.push(make_split(term, def, &info, left, right));
    }

    Ok(out)
}

enum SplitLimit {
    Factors,
    SumIndices,
    ExternalIndices,
}

fn validate_mask_len(len: usize, limit: SplitLimit) -> Result<(), SplitError> {
    if len <= MAX_MASK_BITS {
        return Ok(());
    }

    match limit {
        SplitLimit::Factors => Err(SplitError::TooManyFactors {
            len,
            max: MAX_MASK_BITS,
        }),
        SplitLimit::SumIndices => Err(SplitError::TooManySumIndices {
            len,
            max: MAX_MASK_BITS,
        }),
        SplitLimit::ExternalIndices => Err(SplitError::TooManyExternalIndices {
            len,
            max: MAX_MASK_BITS,
        }),
    }
}

fn build_term_index_info(term: &Term, def: &TensorDef) -> Result<TermIndexInfo, SplitError> {
    validate_mask_len(term.sum_indices.len(), SplitLimit::SumIndices)?;
    validate_mask_len(def.ext_indices.len(), SplitLimit::ExternalIndices)?;

    let sum_positions = index_positions(&term.sum_indices);
    let external_positions = index_positions(&def.ext_indices);
    let mut factor_sum_bits = Vec::with_capacity(term.factors.len());
    let mut factor_external_bits = Vec::with_capacity(term.factors.len());

    for factor in &term.factors {
        let mut sum_bits = 0;
        let mut external_bits = 0;

        for index in &factor.indices {
            if let Some(position) = sum_positions.get(index) {
                sum_bits |= bit(*position);
            }
            if let Some(position) = external_positions.get(index) {
                external_bits |= bit(*position);
            }
        }

        factor_sum_bits.push(sum_bits);
        factor_external_bits.push(external_bits);
    }

    Ok(TermIndexInfo {
        factor_sum_bits,
        factor_external_bits,
    })
}

fn index_positions(indices: &[Index]) -> HashMap<IndexId, usize> {
    indices
        .iter()
        .enumerate()
        .map(|(position, index)| (index.id, position))
        .collect()
}

fn bit(position: usize) -> u64 {
    1_u64 << position
}

fn full_factor_mask(len: usize) -> FactorSubset {
    if len == MAX_MASK_BITS {
        u64::MAX
    } else {
        (1_u64 << len) - 1
    }
}

fn subset_sum_bits(info: &TermIndexInfo, subset: FactorSubset) -> u64 {
    let mut out = 0;
    for (position, bits) in info.factor_sum_bits.iter().enumerate() {
        if subset & bit(position) != 0 {
            out |= bits;
        }
    }
    out
}

fn subset_external_bits(info: &TermIndexInfo, subset: FactorSubset) -> u64 {
    let mut out = 0;
    for (position, bits) in info.factor_external_bits.iter().enumerate() {
        if subset & bit(position) != 0 {
            out |= bits;
        }
    }
    out
}

fn contracted_sum_bits(info: &TermIndexInfo, left: FactorSubset, right: FactorSubset) -> u64 {
    subset_sum_bits(info, left) & subset_sum_bits(info, right)
}

fn indices_from_mask(source: &[Index], mask: u64) -> Vec<Index> {
    let mut out: Vec<_> = source
        .iter()
        .enumerate()
        .filter_map(|(position, index)| {
            if mask & bit(position) == 0 {
                None
            } else {
                Some(*index)
            }
        })
        .collect();
    out.sort_by_key(|index| index.id);
    out
}

fn make_subterm(term: &Term, subset: FactorSubset, contracted_sum_bits: u64) -> Term {
    let factors: Vec<_> = term
        .factors
        .iter()
        .enumerate()
        .filter_map(|(position, factor)| {
            if subset & bit(position) == 0 {
                None
            } else {
                Some(factor.clone())
            }
        })
        .collect();

    let selected_indices: HashSet<_> = factors
        .iter()
        .flat_map(|factor| factor.indices.iter().copied())
        .collect();
    let sum_indices = term
        .sum_indices
        .iter()
        .enumerate()
        .filter_map(|(position, index)| {
            if selected_indices.contains(&index.id) && contracted_sum_bits & bit(position) == 0 {
                Some(*index)
            } else {
                None
            }
        })
        .collect();

    Term {
        coeff: Rational::new(1, 1),
        sum_indices,
        factors,
    }
}

fn make_interface(
    term: &Term,
    def: &TensorDef,
    info: &TermIndexInfo,
    left: FactorSubset,
    right: FactorSubset,
) -> SplitInterface {
    SplitInterface {
        left_external: indices_from_mask(&def.ext_indices, subset_external_bits(info, left)),
        right_external: indices_from_mask(&def.ext_indices, subset_external_bits(info, right)),
        contracted: indices_from_mask(&term.sum_indices, contracted_sum_bits(info, left, right)),
    }
}

fn make_split(
    term: &Term,
    def: &TensorDef,
    info: &TermIndexInfo,
    left: FactorSubset,
    right: FactorSubset,
) -> Split {
    let contracted = contracted_sum_bits(info, left, right);

    Split {
        left: make_subterm(term, left, contracted),
        right: make_subterm(term, right, contracted),
        interface: make_interface(term, def, info, left, right),
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test split two_factor_term_produces_one_unit_coefficient_split -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Run the existing split test file**

Run: `cargo test --test split -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/split.rs tests/split.rs
git commit -m "feat: enumerate basic term splits"
```

---

### Task 3: Cover three-factor chains, private sum indices, and sorted interfaces

**Files:**
- Modify: `tests/split.rs`
- Modify: `src/split.rs` only if this test exposes an implementation defect.

- [ ] **Step 1: Add the failing test**

Append to `tests/split.rs`:

```rust
#[test]
fn three_factor_chain_emits_public_unordered_bipartitions() {
    let range = RangeId(0);
    let a = Index {
        id: IndexId(10),
        range,
    };
    let b = Index {
        id: IndexId(0),
        range,
    };
    let c = Index {
        id: IndexId(30),
        range,
    };
    let d = Index {
        id: IndexId(20),
        range,
    };
    let x = TensorId(0);
    let y = TensorId(1);
    let z = TensorId(2);

    let def = TensorDef {
        base: TensorId(3),
        ext_indices: vec![a, b],
        terms: vec![],
    };
    let term = Term {
        coeff: Rational::new(-5, 2),
        sum_indices: vec![d, c],
        factors: vec![
            Factor {
                tensor: x,
                indices: vec![a.id, c.id],
            },
            Factor {
                tensor: y,
                indices: vec![c.id, d.id],
            },
            Factor {
                tensor: z,
                indices: vec![d.id, b.id],
            },
        ],
    };

    let splits = enumerate_splits(&term, &def).unwrap();
    assert_eq!(splits.len(), 3);

    assert_eq!(splits[0].interface.left_external, vec![a]);
    assert_eq!(splits[0].interface.right_external, vec![b]);
    assert_eq!(splits[0].interface.contracted, vec![c]);
    assert_eq!(splits[0].left.factors, vec![term.factors[0].clone()]);
    assert_eq!(
        splits[0].right.factors,
        vec![term.factors[1].clone(), term.factors[2].clone()]
    );
    assert_eq!(splits[0].left.sum_indices, vec![]);
    assert_eq!(splits[0].right.sum_indices, vec![d]);

    assert_eq!(splits[1].interface.left_external, vec![]);
    assert_eq!(splits[1].interface.right_external, vec![b, a]);
    assert_eq!(splits[1].interface.contracted, vec![d, c]);
    assert_eq!(splits[1].left.factors, vec![term.factors[1].clone()]);
    assert_eq!(
        splits[1].right.factors,
        vec![term.factors[0].clone(), term.factors[2].clone()]
    );
    assert_eq!(splits[1].left.sum_indices, vec![]);
    assert_eq!(splits[1].right.sum_indices, vec![]);

    assert_eq!(splits[2].interface.left_external, vec![a]);
    assert_eq!(splits[2].interface.right_external, vec![b]);
    assert_eq!(splits[2].interface.contracted, vec![d]);
    assert_eq!(
        splits[2].left.factors,
        vec![term.factors[0].clone(), term.factors[1].clone()]
    );
    assert_eq!(splits[2].right.factors, vec![term.factors[2].clone()]);
    assert_eq!(splits[2].left.sum_indices, vec![c]);
    assert_eq!(splits[2].right.sum_indices, vec![]);

    for split in splits {
        assert_eq!(split.left.coeff, one());
        assert_eq!(split.right.coeff, one());
    }
}
```

- [ ] **Step 2: Run test to verify it passes or exposes a concrete defect**

Run: `cargo test --test split three_factor_chain_emits_public_unordered_bipartitions -- --nocapture`

Expected: PASS if Task 2 implementation is correct. If it fails, the failure should identify one of these specific defects: wrong side normalization, unsorted interface vectors, contracted indices retained in side `sum_indices`, private sum index dropped from the wrong side, or source factor order not preserved.

- [ ] **Step 3: Fix only the exposed defect if needed**

Use the existing helper responsibilities in `src/split.rs`:

```rust
fn indices_from_mask(source: &[Index], mask: u64) -> Vec<Index> {
    let mut out: Vec<_> = source
        .iter()
        .enumerate()
        .filter_map(|(position, index)| {
            if mask & bit(position) == 0 {
                None
            } else {
                Some(*index)
            }
        })
        .collect();
    out.sort_by_key(|index| index.id);
    out
}
```

```rust
fn make_subterm(term: &Term, subset: FactorSubset, contracted_sum_bits: u64) -> Term {
    let factors: Vec<_> = term
        .factors
        .iter()
        .enumerate()
        .filter_map(|(position, factor)| {
            if subset & bit(position) == 0 {
                None
            } else {
                Some(factor.clone())
            }
        })
        .collect();

    let selected_indices: HashSet<_> = factors
        .iter()
        .flat_map(|factor| factor.indices.iter().copied())
        .collect();
    let sum_indices = term
        .sum_indices
        .iter()
        .enumerate()
        .filter_map(|(position, index)| {
            if selected_indices.contains(&index.id) && contracted_sum_bits & bit(position) == 0 {
                Some(*index)
            } else {
                None
            }
        })
        .collect();

    Term {
        coeff: Rational::new(1, 1),
        sum_indices,
        factors,
    }
}
```

- [ ] **Step 4: Run all split tests**

Run: `cargo test --test split -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/split.rs tests/split.rs
git commit -m "test: cover split interfaces and side subterms"
```

---

### Task 4: Report fixed-width mask limits through `SplitError`

**Files:**
- Modify: `tests/split.rs`
- Modify: `src/split.rs` only if this test exposes an implementation defect.

- [ ] **Step 1: Add the failing tests**

Append to `tests/split.rs`:

```rust
use gristmill_symbolics::split::SplitError;

fn factor_with_index(index: IndexId) -> Factor {
    Factor {
        tensor: TensorId(0),
        indices: vec![index],
    }
}

#[test]
fn too_many_factors_returns_split_error() {
    let range = RangeId(0);
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: vec![Index {
            id: IndexId(0),
            range,
        }],
        terms: vec![],
    };
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: (0..65)
            .map(|_| factor_with_index(IndexId(0)))
            .collect(),
    };

    assert_eq!(
        enumerate_splits(&term, &def),
        Err(SplitError::TooManyFactors { len: 65, max: 64 })
    );
}

#[test]
fn too_many_sum_indices_returns_split_error() {
    let range = RangeId(0);
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: vec![],
        terms: vec![],
    };
    let term = Term {
        coeff: one(),
        sum_indices: (0..65)
            .map(|id| Index {
                id: IndexId(id),
                range,
            })
            .collect(),
        factors: vec![
            factor_with_index(IndexId(0)),
            factor_with_index(IndexId(1)),
        ],
    };

    assert_eq!(
        enumerate_splits(&term, &def),
        Err(SplitError::TooManySumIndices { len: 65, max: 64 })
    );
}

#[test]
fn too_many_external_indices_returns_split_error() {
    let range = RangeId(0);
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: (0..65)
            .map(|id| Index {
                id: IndexId(id),
                range,
            })
            .collect(),
        terms: vec![],
    };
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![
            factor_with_index(IndexId(0)),
            factor_with_index(IndexId(1)),
        ],
    };

    assert_eq!(
        enumerate_splits(&term, &def),
        Err(SplitError::TooManyExternalIndices { len: 65, max: 64 })
    );
}
```

- [ ] **Step 2: Run tests to verify the behavior**

Run: `cargo test --test split too_many -- --nocapture`

Expected: PASS if Task 2 implementation already reports all three limits. If it fails, update `validate_mask_len` and `build_term_index_info` to match the exact `SplitError` values.

- [ ] **Step 3: Ensure the implementation contains these checks**

```rust
fn build_term_index_info(term: &Term, def: &TensorDef) -> Result<TermIndexInfo, SplitError> {
    validate_mask_len(term.sum_indices.len(), SplitLimit::SumIndices)?;
    validate_mask_len(def.ext_indices.len(), SplitLimit::ExternalIndices)?;

    let sum_positions = index_positions(&term.sum_indices);
    let external_positions = index_positions(&def.ext_indices);
    let mut factor_sum_bits = Vec::with_capacity(term.factors.len());
    let mut factor_external_bits = Vec::with_capacity(term.factors.len());

    for factor in &term.factors {
        let mut sum_bits = 0;
        let mut external_bits = 0;

        for index in &factor.indices {
            if let Some(position) = sum_positions.get(index) {
                sum_bits |= bit(*position);
            }
            if let Some(position) = external_positions.get(index) {
                external_bits |= bit(*position);
            }
        }

        factor_sum_bits.push(sum_bits);
        factor_external_bits.push(external_bits);
    }

    Ok(TermIndexInfo {
        factor_sum_bits,
        factor_external_bits,
    })
}
```

- [ ] **Step 4: Run all split tests**

Run: `cargo test --test split -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/split.rs tests/split.rs
git commit -m "test: cover split mask limits"
```

---

### Task 5: Final verification

**Files:**
- Verify: `src/split.rs`
- Verify: `src/lib.rs`
- Verify: `tests/split.rs`

- [ ] **Step 1: Run the full test suite**

Run: `cargo test`

Expected: PASS for all existing `repr` tests and the new `split` tests.

- [ ] **Step 2: Run formatting check**

Run: `cargo fmt --check`

Expected: no formatting diff.

- [ ] **Step 3: Review the public API surface**

Run: `rg "pub " src/split.rs src/lib.rs`

Expected output includes only these public split items:

```text
src/lib.rs:pub mod repr;
src/lib.rs:pub mod split;
src/split.rs:pub struct SplitInterface {
src/split.rs:    pub left_external: Vec<Index>,
src/split.rs:    pub right_external: Vec<Index>,
src/split.rs:    pub contracted: Vec<Index>,
src/split.rs:pub struct Split {
src/split.rs:    pub left: Term,
src/split.rs:    pub right: Term,
src/split.rs:    pub interface: SplitInterface,
src/split.rs:pub enum SplitError {
src/split.rs:pub fn enumerate_splits(term: &Term, def: &TensorDef) -> Result<Vec<Split>, SplitError> {
```

- [ ] **Step 4: Commit final polish if formatting changed**

If `cargo fmt` changed files:

```bash
git add src/lib.rs src/split.rs tests/split.rs
git commit -m "chore: format split module"
```

If `cargo fmt --check` was already clean, no commit is needed.

---

## Self-Review Notes

- Spec coverage: public API, unordered bipartition enumeration, unit side coefficients, source factor order, explicit sorted interface vectors, contracted-index removal, private sum preservation, deterministic side normalization, and fixed-width limit errors are covered.
- Out of scope: canonical owner orientations, tensor symmetries, graph grouping, biclique enumeration, rewrite construction, and profitability filtering.
- Type consistency: all tests use existing `repr` public types and the new `split` public API exactly as specified.
