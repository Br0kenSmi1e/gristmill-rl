# canon Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `canon` module that deterministically canonicalizes standalone terms and split owner orientations under tensor symmetries, factor ordering, and dummy-index renaming.

**Architecture:** Add a focused `src/canon.rs` module exposed from `src/lib.rs`. Keep public API small, with explicit helper maps and fallible `canon_term` / `canon_split`; keep symmetry closure, structural ordering, allocation, and split owner/follower machinery private. Tests exercise only public behavior, while implementation follows the private helper pipeline from the canon design spec.

**Tech Stack:** Rust 2024, existing `repr` and `split` modules, `num::rational::Ratio` through `repr::Rational`, standard library `HashMap` / `HashSet` / `VecDeque` collections.

---

## File Structure

- Create `src/canon.rs`: public `IndexPool`, `TensorSymmetryMap`, `CanonError`, `build_index_pool`, `build_tensor_symmetry_map`, `canon_term`, `canon_split`, and private helper pipeline.
- Modify `src/lib.rs`: expose `pub mod canon;`.
- Create `tests/canon.rs`: integration tests for map builders, term canonicalization, symmetry errors, structural selection, and split canonicalization.

---

### Task 1: Add Public API And Map Builders

**Files:**
- Modify: `src/lib.rs`
- Create: `src/canon.rs`
- Create: `tests/canon.rs`

- [ ] **Step 1: Write the failing tests**

Create `tests/canon.rs`:

```rust
use gristmill_symbolics::canon::{
    build_index_pool, build_tensor_symmetry_map, canon_term, CanonError,
};
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, SymAction, SymGenerator, TensorDef, TensorId,
    TensorInfo, Term,
};

fn one() -> Rational {
    Rational::new(1, 1)
}

fn idx(id: u32, range: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(range),
    }
}

fn factor(tensor: u32, indices: &[u32]) -> Factor {
    Factor {
        tensor: TensorId(tensor),
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

#[test]
fn build_index_pool_groups_sorts_and_deduplicates_sum_indices() {
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![idx(0, 0)],
        terms: vec![
            Term {
                coeff: one(),
                sum_indices: vec![idx(5, 1), idx(2, 0)],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![idx(2, 0), idx(4, 1), idx(3, 0)],
                factors: vec![],
            },
        ],
    };

    let pool = build_index_pool(&def);

    assert_eq!(pool.get(&RangeId(0)).unwrap(), &vec![IndexId(2), IndexId(3)]);
    assert_eq!(pool.get(&RangeId(1)).unwrap(), &vec![IndexId(4), IndexId(5)]);
    assert!(!pool.values().any(|ids| ids.contains(&IndexId(0))));
}

#[test]
fn build_tensor_symmetry_map_indexes_by_tensor_id_and_preserves_order() {
    let first = SymGenerator {
        perm: vec![1, 0],
        action: SymAction::Negate,
    };
    let second = SymGenerator {
        perm: vec![0, 1],
        action: SymAction::Identity,
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(7),
            symmetry: vec![first.clone(), second.clone()],
        },
        TensorInfo {
            id: TensorId(3),
            symmetry: vec![],
        },
    ];

    let symmetry = build_tensor_symmetry_map(&tensors);

    assert_eq!(symmetry.get(&TensorId(7)).unwrap(), &vec![first, second]);
    assert_eq!(symmetry.get(&TensorId(3)).unwrap(), &Vec::<SymGenerator>::new());
}

#[test]
fn canon_term_reports_missing_tensor_symmetry() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(9, &[0])],
    };

    assert_eq!(
        canon_term(&term, &build_tensor_symmetry_map(&[]), &build_index_pool(&TensorDef {
            base: TensorId(0),
            ext_indices: vec![],
            terms: vec![term.clone()],
        })),
        Err(CanonError::MissingTensorSymmetry {
            tensor: TensorId(9),
        })
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test canon -- --nocapture`

Expected: compile failure because `gristmill_symbolics::canon` does not exist.

- [ ] **Step 3: Add minimal public API and map builders**

Modify `src/lib.rs`:

```rust
pub mod canon;
pub mod repr;
pub mod split;
```

Create `src/canon.rs`:

```rust
use crate::repr::{
    Factor, Index, IndexId, RangeId, Rational, SymAction, SymGenerator, TensorDef, TensorId,
    TensorInfo, Term,
};
use crate::split::{Split, SplitInterface};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

pub type IndexPool = HashMap<RangeId, Vec<IndexId>>;
pub type TensorSymmetryMap = HashMap<TensorId, Vec<SymGenerator>>;

type DummyRange = HashMap<IndexId, RangeId>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonError {
    MissingTensorSymmetry { tensor: TensorId },
    SymmetryArityMismatch {
        tensor: TensorId,
        expected: usize,
        got: usize,
    },
    InvalidSymmetryPermutation {
        tensor: TensorId,
        perm: Vec<usize>,
    },
    MissingIndexPool { range: RangeId },
    ExhaustedIndexPool { range: RangeId },
    EmptyCanonicalCandidates,
    InconsistentSymmetryCoefficient,
}

pub fn build_index_pool(def: &TensorDef) -> IndexPool {
    let mut pool: IndexPool = HashMap::new();

    for term in &def.terms {
        for index in &term.sum_indices {
            pool.entry(index.range).or_default().push(index.id);
        }
    }

    for ids in pool.values_mut() {
        ids.sort();
        ids.dedup();
    }

    pool
}

pub fn build_tensor_symmetry_map(tensors: &[TensorInfo]) -> TensorSymmetryMap {
    tensors
        .iter()
        .map(|tensor| (tensor.id, tensor.symmetry.clone()))
        .collect()
}

pub fn canon_term(
    term: &Term,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<Term, CanonError> {
    let dummy_range = build_term_dummy_range(term);
    let mut candidates = Vec::new();

    for sym_term in enumerate_symmetry_terms(term, symmetry)? {
        for ordered in enumerate_ordered_terms(&sym_term, &dummy_range) {
            candidates.push(rename_standalone_term(&ordered, &dummy_range, pool)?);
        }
    }

    let index = choose_min_term_index(&candidates)?;
    Ok(candidates[index].clone())
}

pub fn canon_split(
    split: &Split,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<(Split, Split), CanonError> {
    let _ = (split, symmetry, pool);
    Err(CanonError::EmptyCanonicalCandidates)
}
```

Append temporary private stubs to `src/canon.rs`; later tasks replace them with real behavior:

```rust
fn build_term_dummy_range(term: &Term) -> DummyRange {
    term.sum_indices
        .iter()
        .map(|index| (index.id, index.range))
        .collect()
}

fn enumerate_symmetry_terms(
    term: &Term,
    symmetry: &TensorSymmetryMap,
) -> Result<Vec<Term>, CanonError> {
    for factor in &term.factors {
        if !symmetry.contains_key(&factor.tensor) {
            return Err(CanonError::MissingTensorSymmetry {
                tensor: factor.tensor,
            });
        }
    }

    Ok(vec![term.clone()])
}

fn enumerate_ordered_terms(term: &Term, _dummy_range: &DummyRange) -> Vec<Term> {
    vec![term.clone()]
}

fn rename_standalone_term(
    term: &Term,
    _dummy_range: &DummyRange,
    _pool: &IndexPool,
) -> Result<Term, CanonError> {
    Ok(term.clone())
}

fn choose_min_term_index(candidates: &[Term]) -> Result<usize, CanonError> {
    if candidates.is_empty() {
        Err(CanonError::EmptyCanonicalCandidates)
    } else {
        Ok(0)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test canon build_index_pool build_tensor_symmetry_map canon_term_reports_missing_tensor_symmetry -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/canon.rs tests/canon.rs
git commit -m "feat: add canon module API"
```

---

### Task 2: Implement Tensor Symmetry Closure And Coefficient Actions

**Files:**
- Modify: `src/canon.rs`
- Modify: `tests/canon.rs`

- [ ] **Step 1: Add failing tests**

Append to `tests/canon.rs`:

```rust
#[test]
fn canon_term_applies_factor_symmetry_and_negates_coefficient() {
    let term = Term {
        coeff: Rational::new(3, 1),
        sum_indices: vec![],
        factors: vec![factor(0, &[2, 1])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Negate,
        }],
    }];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&TensorDef {
            base: TensorId(0),
            ext_indices: vec![idx(1, 0), idx(2, 0)],
            terms: vec![term.clone()],
        }),
    )
    .unwrap();

    assert_eq!(
        canonical,
        Term {
            coeff: Rational::new(-3, 1),
            sum_indices: vec![],
            factors: vec![factor(0, &[1, 2])],
        }
    );
}

#[test]
fn canon_term_reports_symmetry_arity_mismatch() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(0, &[0])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Identity,
        }],
    }];

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&tensors),
            &build_index_pool(&TensorDef {
                base: TensorId(0),
                ext_indices: vec![idx(0, 0)],
                terms: vec![term.clone()],
            }),
        ),
        Err(CanonError::SymmetryArityMismatch {
            tensor: TensorId(0),
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn canon_term_reports_invalid_symmetry_permutation() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(0, &[0, 1])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![0, 0],
            action: SymAction::Identity,
        }],
    }];

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&tensors),
            &build_index_pool(&TensorDef {
                base: TensorId(0),
                ext_indices: vec![idx(0, 0), idx(1, 0)],
                terms: vec![term.clone()],
            }),
        ),
        Err(CanonError::InvalidSymmetryPermutation {
            tensor: TensorId(0),
            perm: vec![0, 0],
        })
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test canon canon_term_applies_factor_symmetry_and_negates_coefficient canon_term_reports_symmetry_arity_mismatch canon_term_reports_invalid_symmetry_permutation -- --nocapture`

Expected: FAIL because symmetry closure is not implemented.

- [ ] **Step 3: Implement symmetry helpers**

In `src/canon.rs`, replace the temporary `enumerate_symmetry_terms` stub and add these helpers:

```rust
fn enumerate_sym_group(
    tensor: TensorId,
    generators: &[SymGenerator],
    arity: usize,
) -> Result<Vec<(Vec<usize>, SymAction)>, CanonError> {
    for generator in generators {
        validate_generator(tensor, generator, arity)?;
    }

    let identity_perm: Vec<_> = (0..arity).collect();
    let mut group = vec![(identity_perm.clone(), SymAction::Identity)];
    let mut queue = VecDeque::from([(identity_perm, SymAction::Identity)]);

    while let Some((perm, action)) = queue.pop_front() {
        for generator in generators {
            let next_perm = compose_perm(&generator.perm, &perm);
            let next_action = action.combine(generator.action);

            if !group
                .iter()
                .any(|(seen_perm, seen_action)| seen_perm == &next_perm && seen_action == &next_action)
            {
                group.push((next_perm.clone(), next_action));
                queue.push_back((next_perm, next_action));
            }
        }
    }

    Ok(group)
}

fn validate_generator(
    tensor: TensorId,
    generator: &SymGenerator,
    arity: usize,
) -> Result<(), CanonError> {
    if generator.perm.len() != arity {
        return Err(CanonError::SymmetryArityMismatch {
            tensor,
            expected: generator.perm.len(),
            got: arity,
        });
    }

    let mut seen = vec![false; generator.perm.len()];
    for &position in &generator.perm {
        if position >= generator.perm.len() || seen[position] {
            return Err(CanonError::InvalidSymmetryPermutation {
                tensor,
                perm: generator.perm.clone(),
            });
        }
        seen[position] = true;
    }

    Ok(())
}

fn compose_perm(left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter().map(|&position| right[position]).collect()
}

fn enumerate_factor_variants(
    tensor: TensorId,
    indices: &[IndexId],
    generators: &[SymGenerator],
) -> Result<Vec<(Vec<IndexId>, SymAction)>, CanonError> {
    enumerate_sym_group(tensor, generators, indices.len()).map(|group| {
        group
            .into_iter()
            .map(|(perm, action)| {
                (
                    perm.into_iter().map(|position| indices[position]).collect(),
                    action,
                )
            })
            .collect()
    })
}

fn enumerate_symmetry_terms(
    term: &Term,
    symmetry: &TensorSymmetryMap,
) -> Result<Vec<Term>, CanonError> {
    let mut factor_variants = Vec::with_capacity(term.factors.len());

    for factor in &term.factors {
        let generators =
            symmetry
                .get(&factor.tensor)
                .ok_or(CanonError::MissingTensorSymmetry {
                    tensor: factor.tensor,
                })?;
        factor_variants.push(enumerate_factor_variants(
            factor.tensor,
            &factor.indices,
            generators,
        )?);
    }

    let mut out = Vec::new();
    enumerate_symmetry_product(
        term,
        &factor_variants,
        0,
        Vec::with_capacity(term.factors.len()),
        SymAction::Identity,
        &mut out,
    );
    Ok(out)
}

fn enumerate_symmetry_product(
    term: &Term,
    factor_variants: &[Vec<(Vec<IndexId>, SymAction)>],
    position: usize,
    mut factors: Vec<Factor>,
    action: SymAction,
    out: &mut Vec<Term>,
) {
    if position == term.factors.len() {
        out.push(Term {
            coeff: apply_action_to_coeff(term.coeff.clone(), action),
            sum_indices: term.sum_indices.clone(),
            factors,
        });
        return;
    }

    for (indices, factor_action) in &factor_variants[position] {
        let mut next_factors = factors.clone();
        next_factors.push(Factor {
            tensor: term.factors[position].tensor,
            indices: indices.clone(),
        });
        enumerate_symmetry_product(
            term,
            factor_variants,
            position + 1,
            next_factors,
            action.combine(*factor_action),
            out,
        );
    }
}

fn apply_action_to_coeff(coeff: Rational, action: SymAction) -> Rational {
    match action {
        SymAction::Identity => coeff,
        SymAction::Negate => -coeff,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test canon canon_term_applies_factor_symmetry_and_negates_coefficient canon_term_reports_symmetry_arity_mismatch canon_term_reports_invalid_symmetry_permutation -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/canon.rs tests/canon.rs
git commit -m "feat: enumerate tensor symmetry candidates"
```

---

### Task 3: Implement Structural Factor Ordering And Standalone Dummy Renaming

**Files:**
- Modify: `src/canon.rs`
- Modify: `tests/canon.rs`

- [ ] **Step 1: Add failing tests**

Append to `tests/canon.rs`:

```rust
#[test]
fn canon_term_normalizes_dummy_names_and_sum_index_order() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![idx(8, 0), idx(4, 0)],
        factors: vec![factor(0, &[8, 4]), factor(1, &[4, 8])],
    };
    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![],
        terms: vec![term.clone()],
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(0),
            symmetry: vec![],
        },
        TensorInfo {
            id: TensorId(1),
            symmetry: vec![],
        },
    ];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&def),
    )
    .unwrap();

    assert_eq!(
        canonical,
        Term {
            coeff: one(),
            sum_indices: vec![idx(4, 0), idx(8, 0)],
            factors: vec![factor(0, &[4, 8]), factor(1, &[8, 4])],
        }
    );
}

#[test]
fn canon_term_orders_factors_but_preserves_external_id_distinctions() {
    let a = idx(1, 0);
    let b = idx(2, 0);
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(1, &[b.id.0]), factor(0, &[a.id.0])],
    };
    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![a, b],
        terms: vec![term.clone()],
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(0),
            symmetry: vec![],
        },
        TensorInfo {
            id: TensorId(1),
            symmetry: vec![],
        },
    ];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&def),
    )
    .unwrap();

    assert_eq!(
        canonical.factors,
        vec![factor(0, &[a.id.0]), factor(1, &[b.id.0])]
    );
}

#[test]
fn canon_term_is_deterministic_for_tied_factor_groups() {
    let term_a = Term {
        coeff: one(),
        sum_indices: vec![idx(10, 0), idx(11, 0), idx(12, 0)],
        factors: vec![
            factor(0, &[11, 10]),
            factor(0, &[12, 11]),
            factor(0, &[10, 12]),
        ],
    };
    let term_b = Term {
        coeff: one(),
        sum_indices: vec![idx(12, 0), idx(10, 0), idx(11, 0)],
        factors: vec![
            factor(0, &[10, 12]),
            factor(0, &[11, 10]),
            factor(0, &[12, 11]),
        ],
    };
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: vec![],
        terms: vec![term_a.clone(), term_b.clone()],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![],
    }];
    let symmetry = build_tensor_symmetry_map(&tensors);
    let pool = build_index_pool(&def);

    assert_eq!(
        canon_term(&term_a, &symmetry, &pool).unwrap(),
        canon_term(&term_b, &symmetry, &pool).unwrap()
    );
}

#[test]
fn canon_term_reports_missing_index_pool() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![idx(10, 4)],
        factors: vec![factor(0, &[10])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![],
    }];

    assert_eq!(
        canon_term(&term, &build_tensor_symmetry_map(&tensors), &Default::default()),
        Err(CanonError::MissingIndexPool { range: RangeId(4) })
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test canon canon_term_normalizes_dummy_names_and_sum_index_order canon_term_orders_factors_but_preserves_external_id_distinctions canon_term_is_deterministic_for_tied_factor_groups canon_term_reports_missing_index_pool -- --nocapture`

Expected: FAIL because ordering, tied-group permutation, and dummy allocation are still stubs.

- [ ] **Step 3: Implement ordering, allocator, and standalone rename helpers**

In `src/canon.rs`, replace the temporary `enumerate_ordered_terms` and `rename_standalone_term` stubs, and add these helpers:

```rust
#[derive(Clone, Copy)]
enum IndexSlot {
    Dummy(RangeId),
    External(IndexId),
}

struct PoolAllocator<'a> {
    pool: &'a IndexPool,
    used: HashMap<RangeId, HashSet<usize>>,
}

impl<'a> PoolAllocator<'a> {
    fn new(pool: &'a IndexPool) -> Self {
        Self {
            pool,
            used: HashMap::new(),
        }
    }

    fn from_base_map_for_ids(
        pool: &'a IndexPool,
        base_map: &HashMap<IndexId, IndexId>,
        original_ids: &HashSet<IndexId>,
    ) -> Result<Self, CanonError> {
        let mut allocator = Self::new(pool);
        for original_id in original_ids {
            if let Some(&new_id) = base_map.get(original_id) {
                let range = range_for_new_id(pool, new_id)
                    .ok_or(CanonError::ExhaustedIndexPool { range: RangeId(u32::MAX) })?;
                let position = pool[&range]
                    .iter()
                    .position(|&candidate| candidate == new_id)
                    .unwrap();
                allocator.used.entry(range).or_default().insert(position);
            }
        }
        Ok(allocator)
    }

    fn alloc_low(&mut self, range: RangeId) -> Result<IndexId, CanonError> {
        let ids = self
            .pool
            .get(&range)
            .ok_or(CanonError::MissingIndexPool { range })?;
        let used = self.used.entry(range).or_default();
        for (position, &id) in ids.iter().enumerate() {
            if used.insert(position) {
                return Ok(id);
            }
        }
        Err(CanonError::ExhaustedIndexPool { range })
    }

    fn alloc_high(&mut self, range: RangeId) -> Result<IndexId, CanonError> {
        let ids = self
            .pool
            .get(&range)
            .ok_or(CanonError::MissingIndexPool { range })?;
        let used = self.used.entry(range).or_default();
        for (position, &id) in ids.iter().enumerate().rev() {
            if used.insert(position) {
                return Ok(id);
            }
        }
        Err(CanonError::ExhaustedIndexPool { range })
    }
}

fn range_for_new_id(pool: &IndexPool, id: IndexId) -> Option<RangeId> {
    pool.iter()
        .find_map(|(&range, ids)| ids.contains(&id).then_some(range))
}

fn compare_factors_by_structure(
    left: &Factor,
    right: &Factor,
    dummy_range: &DummyRange,
) -> Ordering {
    left.tensor
        .cmp(&right.tensor)
        .then_with(|| compare_index_slots(&left.indices, &right.indices, dummy_range))
}

fn compare_index_slots(
    left: &[IndexId],
    right: &[IndexId],
    dummy_range: &DummyRange,
) -> Ordering {
    for (left_id, right_id) in left.iter().zip(right) {
        let ordering = index_slot(*left_id, dummy_range).cmp(&index_slot(*right_id, dummy_range));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn index_slot(index: IndexId, dummy_range: &DummyRange) -> IndexSlot {
    if let Some(&range) = dummy_range.get(&index) {
        IndexSlot::Dummy(range)
    } else {
        IndexSlot::External(index)
    }
}

impl PartialEq for IndexSlot {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for IndexSlot {}

impl PartialOrd for IndexSlot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexSlot {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (IndexSlot::Dummy(left), IndexSlot::Dummy(right)) => left.cmp(right),
            (IndexSlot::Dummy(_), IndexSlot::External(_)) => Ordering::Less,
            (IndexSlot::External(_), IndexSlot::Dummy(_)) => Ordering::Greater,
            (IndexSlot::External(left), IndexSlot::External(right)) => left.cmp(right),
        }
    }
}

fn enumerate_ordered_terms(term: &Term, dummy_range: &DummyRange) -> Vec<Term> {
    let mut factors = term.factors.clone();
    factors.sort_by(|left, right| compare_factors_by_structure(left, right, dummy_range));
    let groups = tied_groups(&factors, dummy_range);
    let mut factor_orders = Vec::new();
    enumerate_group_permutations(&factors, &groups, 0, &mut factor_orders);

    factor_orders
        .into_iter()
        .map(|factors| Term {
            coeff: term.coeff.clone(),
            sum_indices: term.sum_indices.clone(),
            factors,
        })
        .collect()
}

fn tied_groups(factors: &[Factor], dummy_range: &DummyRange) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut start = 0;

    while start < factors.len() {
        let mut end = start + 1;
        while end < factors.len()
            && compare_factors_by_structure(&factors[start], &factors[end], dummy_range)
                == Ordering::Equal
        {
            end += 1;
        }
        groups.push((start, end));
        start = end;
    }

    groups
}

fn enumerate_group_permutations(
    factors: &[Factor],
    groups: &[(usize, usize)],
    group_index: usize,
    out: &mut Vec<Vec<Factor>>,
) {
    if group_index == groups.len() {
        out.push(factors.to_vec());
        return;
    }

    let (start, end) = groups[group_index];
    for permutation in permutations(&factors[start..end]) {
        let mut next = factors.to_vec();
        next.splice(start..end, permutation);
        enumerate_group_permutations(&next, groups, group_index + 1, out);
    }
}

fn permutations(items: &[Factor]) -> Vec<Vec<Factor>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }

    let mut out = Vec::new();
    for index in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(index);
        for mut tail in permutations(&rest) {
            let mut permutation = vec![head.clone()];
            permutation.append(&mut tail);
            out.push(permutation);
        }
    }
    out
}

fn build_rename_map<F>(
    term: &Term,
    rename_ids: &HashSet<IndexId>,
    dummy_range: &DummyRange,
    base_map: &HashMap<IndexId, IndexId>,
    allocator: &mut PoolAllocator<'_>,
    mut allocate: F,
) -> Result<HashMap<IndexId, IndexId>, CanonError>
where
    F: FnMut(&mut PoolAllocator<'_>, RangeId) -> Result<IndexId, CanonError>,
{
    let mut remap = base_map.clone();

    for factor in &term.factors {
        for &index_id in &factor.indices {
            if !rename_ids.contains(&index_id) || remap.contains_key(&index_id) {
                continue;
            }
            let range = *dummy_range
                .get(&index_id)
                .expect("rename_ids must be present in dummy_range");
            remap.insert(index_id, allocate(allocator, range)?);
        }
    }

    Ok(remap)
}

fn apply_rename_map(term: &Term, remap: &HashMap<IndexId, IndexId>) -> Term {
    let mut sum_indices: Vec<_> = term
        .sum_indices
        .iter()
        .map(|index| Index {
            id: remap.get(&index.id).copied().unwrap_or(index.id),
            range: index.range,
        })
        .collect();
    sum_indices.sort_by_key(|index| index.id);

    Term {
        coeff: term.coeff.clone(),
        sum_indices,
        factors: term
            .factors
            .iter()
            .map(|factor| Factor {
                tensor: factor.tensor,
                indices: factor
                    .indices
                    .iter()
                    .map(|index| remap.get(index).copied().unwrap_or(*index))
                    .collect(),
            })
            .collect(),
    }
}

fn rename_standalone_term(
    term: &Term,
    dummy_range: &DummyRange,
    pool: &IndexPool,
) -> Result<Term, CanonError> {
    let rename_ids: HashSet<_> = term.sum_indices.iter().map(|index| index.id).collect();
    let mut allocator = PoolAllocator::new(pool);
    let remap = build_rename_map(
        term,
        &rename_ids,
        dummy_range,
        &HashMap::new(),
        &mut allocator,
        PoolAllocator::alloc_low,
    )?;
    Ok(apply_rename_map(term, &remap))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test canon canon_term_normalizes_dummy_names_and_sum_index_order canon_term_orders_factors_but_preserves_external_id_distinctions canon_term_is_deterministic_for_tied_factor_groups canon_term_reports_missing_index_pool -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/canon.rs tests/canon.rs
git commit -m "feat: canonicalize factor order and dummy names"
```

---

### Task 4: Implement Structural Selection And Inconsistent Coefficient Detection

**Files:**
- Modify: `src/canon.rs`
- Modify: `tests/canon.rs`

- [ ] **Step 1: Add failing tests**

Append to `tests/canon.rs`:

```rust
#[test]
fn canon_term_selects_representative_by_structure_not_coefficient() {
    let term = Term {
        coeff: Rational::new(5, 1),
        sum_indices: vec![],
        factors: vec![factor(0, &[2, 1])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Identity,
        }],
    }];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&TensorDef {
            base: TensorId(0),
            ext_indices: vec![idx(1, 0), idx(2, 0)],
            terms: vec![term.clone()],
        }),
    )
    .unwrap();

    assert_eq!(canonical.coeff, Rational::new(5, 1));
    assert_eq!(canonical.factors, vec![factor(0, &[1, 2])]);
}

#[test]
fn canon_term_reports_inconsistent_symmetry_coefficient() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(0, &[1, 1])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Negate,
        }],
    }];

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&tensors),
            &build_index_pool(&TensorDef {
                base: TensorId(0),
                ext_indices: vec![idx(1, 0)],
                terms: vec![term.clone()],
            }),
        ),
        Err(CanonError::InconsistentSymmetryCoefficient)
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test canon canon_term_selects_representative_by_structure_not_coefficient canon_term_reports_inconsistent_symmetry_coefficient -- --nocapture`

Expected: FAIL because `choose_min_term_index` still returns the first candidate and does not check equal structures.

- [ ] **Step 3: Implement structural comparators and selection**

In `src/canon.rs`, replace `choose_min_term_index` and add comparators:

```rust
fn choose_min_term_index(candidates: &[Term]) -> Result<usize, CanonError> {
    if candidates.is_empty() {
        return Err(CanonError::EmptyCanonicalCandidates);
    }

    for left in 0..candidates.len() {
        for right in (left + 1)..candidates.len() {
            if compare_term_structure(&candidates[left], &candidates[right]) == Ordering::Equal
                && candidates[left].coeff != candidates[right].coeff
            {
                return Err(CanonError::InconsistentSymmetryCoefficient);
            }
        }
    }

    let mut best = 0;
    for index in 1..candidates.len() {
        if compare_term_structure(&candidates[index], &candidates[best]) == Ordering::Less {
            best = index;
        }
    }
    Ok(best)
}

fn compare_terms(left: &Term, right: &Term) -> Ordering {
    compare_term_structure(left, right).then_with(|| left.coeff.cmp(&right.coeff))
}

fn compare_term_structure(left: &Term, right: &Term) -> Ordering {
    compare_indices(&left.sum_indices, &right.sum_indices)
        .then_with(|| compare_factors(&left.factors, &right.factors))
}

fn compare_indices(left: &[Index], right: &[Index]) -> Ordering {
    for (left_index, right_index) in left.iter().zip(right) {
        let ordering = left_index
            .range
            .cmp(&right_index.range)
            .then_with(|| left_index.id.cmp(&right_index.id));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn compare_factors(left: &[Factor], right: &[Factor]) -> Ordering {
    for (left_factor, right_factor) in left.iter().zip(right) {
        let ordering = left_factor
            .tensor
            .cmp(&right_factor.tensor)
            .then_with(|| left_factor.indices.cmp(&right_factor.indices));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test canon canon_term_selects_representative_by_structure_not_coefficient canon_term_reports_inconsistent_symmetry_coefficient -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/canon.rs tests/canon.rs
git commit -m "feat: select canonical structures deterministically"
```

---

### Task 5: Implement Split Canonicalization With Owner/Follower Renaming

**Files:**
- Modify: `src/canon.rs`
- Modify: `tests/canon.rs`

- [ ] **Step 1: Add failing tests**

Append to `tests/canon.rs`:

```rust
use gristmill_symbolics::canon::canon_split;
use gristmill_symbolics::split::{Split, SplitInterface};

#[test]
fn canon_split_returns_owner_orientations_without_swapping_sides() {
    let a = idx(0, 0);
    let b = idx(1, 0);
    let k = idx(5, 0);
    let l = idx(6, 0);
    let p = idx(7, 0);
    let split = Split {
        left: Term {
            coeff: one(),
            sum_indices: vec![l],
            factors: vec![factor(0, &[a.id.0, k.id.0, l.id.0])],
        },
        right: Term {
            coeff: one(),
            sum_indices: vec![p],
            factors: vec![factor(1, &[k.id.0, p.id.0, b.id.0])],
        },
        interface: SplitInterface {
            left_external: vec![a],
            right_external: vec![b],
            contracted: vec![k],
        },
    };
    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![a, b],
        terms: vec![Term {
            coeff: one(),
            sum_indices: vec![k, l, p],
            factors: vec![],
        }],
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(0),
            symmetry: vec![],
        },
        TensorInfo {
            id: TensorId(1),
            symmetry: vec![],
        },
    ];

    let (left_owner, right_owner) = canon_split(
        &split,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&def),
    )
    .unwrap();

    assert_eq!(left_owner.interface.left_external, vec![a]);
    assert_eq!(left_owner.interface.right_external, vec![b]);
    assert_eq!(right_owner.interface.left_external, vec![a]);
    assert_eq!(right_owner.interface.right_external, vec![b]);

    assert_eq!(left_owner.left.factors[0].tensor, TensorId(0));
    assert_eq!(left_owner.right.factors[0].tensor, TensorId(1));
    assert_eq!(right_owner.left.factors[0].tensor, TensorId(0));
    assert_eq!(right_owner.right.factors[0].tensor, TensorId(1));
}

#[test]
fn canon_split_remaps_contracted_id_consistently_across_sides() {
    let a = idx(0, 0);
    let b = idx(1, 0);
    let k = idx(8, 0);
    let l = idx(4, 0);
    let p = idx(6, 0);
    let split = Split {
        left: Term {
            coeff: one(),
            sum_indices: vec![l],
            factors: vec![factor(0, &[a.id.0, k.id.0, l.id.0])],
        },
        right: Term {
            coeff: one(),
            sum_indices: vec![p],
            factors: vec![factor(1, &[k.id.0, p.id.0, b.id.0])],
        },
        interface: SplitInterface {
            left_external: vec![a],
            right_external: vec![b],
            contracted: vec![k],
        },
    };
    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![a, b],
        terms: vec![Term {
            coeff: one(),
            sum_indices: vec![k, l, p],
            factors: vec![],
        }],
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(0),
            symmetry: vec![],
        },
        TensorInfo {
            id: TensorId(1),
            symmetry: vec![],
        },
    ];

    let (left_owner, right_owner) = canon_split(
        &split,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&def),
    )
    .unwrap();

    let left_owner_contracted = left_owner.interface.contracted[0].id;
    assert!(left_owner.left.factors[0].indices.contains(&left_owner_contracted));
    assert!(left_owner.right.factors[0].indices.contains(&left_owner_contracted));
    assert_eq!(left_owner.interface.contracted[0].range, RangeId(0));

    let right_owner_contracted = right_owner.interface.contracted[0].id;
    assert!(right_owner.left.factors[0].indices.contains(&right_owner_contracted));
    assert!(right_owner.right.factors[0].indices.contains(&right_owner_contracted));
    assert_eq!(right_owner.interface.contracted[0].range, RangeId(0));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test canon canon_split_returns_owner_orientations_without_swapping_sides canon_split_remaps_contracted_id_consistently_across_sides -- --nocapture`

Expected: FAIL because `canon_split` still returns `EmptyCanonicalCandidates`.

- [ ] **Step 3: Implement split dummy range, owner/follower helpers, and `canon_split`**

In `src/canon.rs`, add split helpers:

```rust
#[derive(Clone, Copy)]
enum SplitSide {
    Left,
    Right,
}

fn build_split_dummy_range(split: &Split) -> DummyRange {
    split
        .interface
        .contracted
        .iter()
        .chain(split.left.sum_indices.iter())
        .chain(split.right.sum_indices.iter())
        .map(|index| (index.id, index.range))
        .collect()
}

fn index_id_set(indices: &[Index]) -> HashSet<IndexId> {
    indices.iter().map(|index| index.id).collect()
}

fn rename_owner_term(
    term: &Term,
    side: SplitSide,
    contracted_ids: &HashSet<IndexId>,
    dummy_range: &DummyRange,
    pool: &IndexPool,
) -> Result<(Term, HashMap<IndexId, IndexId>), CanonError> {
    let private_ids: HashSet<_> = term.sum_indices.iter().map(|index| index.id).collect();
    let mut rename_ids = contracted_ids.clone();
    rename_ids.extend(private_ids.iter().copied());

    let mut allocator = PoolAllocator::new(pool);
    let contracted_map = build_rename_map(
        term,
        contracted_ids,
        dummy_range,
        &HashMap::new(),
        &mut allocator,
        PoolAllocator::alloc_low,
    )?;

    let mut remap = contracted_map.clone();
    let private_map = build_rename_map(
        term,
        &private_ids,
        dummy_range,
        &remap,
        &mut allocator,
        match side {
            SplitSide::Left => PoolAllocator::alloc_low,
            SplitSide::Right => PoolAllocator::alloc_high,
        },
    )?;
    remap.extend(private_map);

    Ok((apply_rename_map(term, &remap), contracted_map))
}

fn rename_follower_term(
    term: &Term,
    side: SplitSide,
    contracted_ids: &HashSet<IndexId>,
    contracted_map: &HashMap<IndexId, IndexId>,
    dummy_range: &DummyRange,
    pool: &IndexPool,
) -> Result<Term, CanonError> {
    let private_ids: HashSet<_> = term.sum_indices.iter().map(|index| index.id).collect();
    let mut remap = contracted_map.clone();
    let mut allocator = PoolAllocator::from_base_map_for_ids(pool, contracted_map, contracted_ids)?;
    let private_map = build_rename_map(
        term,
        &private_ids,
        dummy_range,
        &remap,
        &mut allocator,
        match side {
            SplitSide::Left => PoolAllocator::alloc_low,
            SplitSide::Right => PoolAllocator::alloc_high,
        },
    )?;
    remap.extend(private_map);
    Ok(apply_rename_map(term, &remap))
}

fn remap_interface(interface: &SplitInterface, remap: &HashMap<IndexId, IndexId>) -> SplitInterface {
    let mut contracted: Vec<_> = interface
        .contracted
        .iter()
        .map(|index| Index {
            id: remap.get(&index.id).copied().unwrap_or(index.id),
            range: index.range,
        })
        .collect();
    contracted.sort_by_key(|index| index.id);

    SplitInterface {
        left_external: interface.left_external.clone(),
        right_external: interface.right_external.clone(),
        contracted,
    }
}
```

Replace `canon_split` with:

```rust
pub fn canon_split(
    split: &Split,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<(Split, Split), CanonError> {
    let dummy_range = build_split_dummy_range(split);
    let contracted_ids = index_id_set(&split.interface.contracted);

    let left_owner = canon_split_orientation(
        split,
        SplitSide::Left,
        &contracted_ids,
        &dummy_range,
        symmetry,
        pool,
    )?;
    let right_owner = canon_split_orientation(
        split,
        SplitSide::Right,
        &contracted_ids,
        &dummy_range,
        symmetry,
        pool,
    )?;

    Ok((left_owner, right_owner))
}

fn canon_split_orientation(
    split: &Split,
    owner_side: SplitSide,
    contracted_ids: &HashSet<IndexId>,
    dummy_range: &DummyRange,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<Split, CanonError> {
    let (owner_raw, follower_raw) = match owner_side {
        SplitSide::Left => (&split.left, &split.right),
        SplitSide::Right => (&split.right, &split.left),
    };

    let mut owner_terms = Vec::new();
    let mut owner_maps = Vec::new();
    for sym_term in enumerate_symmetry_terms(owner_raw, symmetry)? {
        for ordered in enumerate_ordered_terms(&sym_term, dummy_range) {
            let (renamed, contracted_map) =
                rename_owner_term(&ordered, owner_side, contracted_ids, dummy_range, pool)?;
            owner_terms.push(renamed);
            owner_maps.push(contracted_map);
        }
    }

    let owner_index = choose_min_term_index(&owner_terms)?;
    let owner_term = owner_terms[owner_index].clone();
    let contracted_map = owner_maps[owner_index].clone();

    let follower_side = match owner_side {
        SplitSide::Left => SplitSide::Right,
        SplitSide::Right => SplitSide::Left,
    };
    let mut follower_terms = Vec::new();
    for sym_term in enumerate_symmetry_terms(follower_raw, symmetry)? {
        for ordered in enumerate_ordered_terms(&sym_term, dummy_range) {
            follower_terms.push(rename_follower_term(
                &ordered,
                follower_side,
                contracted_ids,
                &contracted_map,
                dummy_range,
                pool,
            )?);
        }
    }
    let follower_index = choose_min_term_index(&follower_terms)?;
    let follower_term = follower_terms[follower_index].clone();

    let interface = remap_interface(&split.interface, &contracted_map);
    Ok(match owner_side {
        SplitSide::Left => Split {
            left: owner_term,
            right: follower_term,
            interface,
        },
        SplitSide::Right => Split {
            left: follower_term,
            right: owner_term,
            interface,
        },
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test canon canon_split_returns_owner_orientations_without_swapping_sides canon_split_remaps_contracted_id_consistently_across_sides -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/canon.rs tests/canon.rs
git commit -m "feat: canonicalize split owner orientations"
```

---

### Task 6: Add Exhaustion Coverage And Full Canon Test Run

**Files:**
- Modify: `tests/canon.rs`

- [ ] **Step 1: Add failing exhaustion test**

Append to `tests/canon.rs`:

```rust
#[test]
fn canon_term_reports_exhausted_index_pool() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![idx(1, 0), idx(2, 0)],
        factors: vec![factor(0, &[1, 2])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![],
    }];
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![Term {
            coeff: one(),
            sum_indices: vec![idx(1, 0)],
            factors: vec![],
        }],
    };

    assert_eq!(
        canon_term(&term, &build_tensor_symmetry_map(&tensors), &build_index_pool(&def)),
        Err(CanonError::ExhaustedIndexPool { range: RangeId(0) })
    );
}
```

- [ ] **Step 2: Run test to verify it fails if exhaustion is not handled**

Run: `cargo test --test canon canon_term_reports_exhausted_index_pool -- --nocapture`

Expected: PASS if Task 3 allocator already returns `ExhaustedIndexPool`; otherwise FAIL and implement the missing allocator error branch exactly as specified in Task 3.

- [ ] **Step 3: Run all canon tests**

Run: `cargo test --test canon -- --nocapture`

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run: `cargo test`

Expected: PASS for all existing `repr`, `split`, and new `canon` tests.

- [ ] **Step 5: Commit**

```bash
git add tests/canon.rs src/canon.rs
git commit -m "test: cover canon pool exhaustion"
```

---

## Self-Review Checklist

- [ ] Spec coverage: `canon` public API, map builders, symmetry closure, deterministic factor ordering, standalone dummy rename, structure-only selection, inconsistent coefficient error, split dummy range, split owner/follower rename, contracted interface remap, and error policy are covered by tasks above.
- [ ] Placeholder scan: this plan contains no placeholder markers, no deferred edge cases, and no references to functions without a defining task.
- [ ] Type consistency: all public names match `docs/superpowers/specs/2026-05-08-canon-module-design.md`; the coefficient inconsistency error uses the accepted current name.
- [ ] Boundary check: no graph, biclique, rewrite, JSON I/O, or algebraic zero-term simplification is added to `canon`.
