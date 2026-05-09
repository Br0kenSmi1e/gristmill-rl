# `canon` Module Design

## Goal

Define deterministic canonicalization for terms and splits.

`canon` consumes structural values from `repr` and `split`, applies tensor
symmetries, orders factors, renames dummy indices, and returns canonical
representatives for downstream graph construction.

It does not enumerate raw factor bipartitions, build construction graphs,
enumerate bicliques, rank candidates, or apply rewrites.

## Scope

This module includes:

- definition-level helper maps
- standalone term canonicalization
- split canonicalization for both owner orientations
- tensor symmetry closure and application
- deterministic factor ordering
- deterministic dummy-index renaming

This module excludes:

- a `CanonContext` wrapper
- a `CanonicalSplit` or `CanonSplitPair` wrapper
- graph grouping
- biclique search
- rewrite/action construction
- algebraic term merging across a full definition

## Public API

```rust
pub type IndexPool = HashMap<RangeId, Vec<IndexId>>;
pub type TensorSymmetryMap = HashMap<TensorId, Vec<SymGenerator>>;

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

pub fn build_index_pool(def: &TensorDef) -> IndexPool;

pub fn build_tensor_symmetry_map(
    tensors: &[TensorInfo],
) -> TensorSymmetryMap;

pub fn canon_term(
    term: &Term,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<Term, CanonError>;

pub fn canon_split(
    split: &Split,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<(Split, Split), CanonError>;
```

Canonicalization is fallible because missing symmetry metadata, invalid
generator arity, and exhausted dummy pools should not panic.

## Map Builders

### `build_index_pool`

```rust
pub fn build_index_pool(def: &TensorDef) -> IndexPool;
```

Build a definition-level pool of reusable dummy index ids.

Behavior:

- scan every `term.sum_indices` in `def.terms`
- group `IndexId`s by `Index.range`
- sort each range's ids by `IndexId`
- deduplicate each range's ids
- do not include external indices from `def.ext_indices`

The pool is used only for dummy-index renaming.

### `build_tensor_symmetry_map`

```rust
pub fn build_tensor_symmetry_map(
    tensors: &[TensorInfo],
) -> TensorSymmetryMap;
```

Build a tensor-id lookup table for factor symmetry generators.

Behavior:

- map each `TensorInfo.id` to its `symmetry` vector
- preserve generator order
- do not validate arity here because tensor arity is factor-use specific

## Standalone Term Canonicalization

```rust
pub fn canon_term(
    term: &Term,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<Term, CanonError>;
```

Pipeline:

1. build `DummyRange` from `term.sum_indices`
2. enumerate all symmetry-realized candidate terms
3. enumerate deterministic factor orderings for each candidate
4. rename all dummy indices using low allocation from `pool`
5. select the minimum canonical term

Expected behavior:

- equivalent terms with different dummy names canonicalize to the same `Term`
- equivalent terms with different sum-index order canonicalize to the same
  `Term`
- tensor symmetry is applied before factor ordering
- `SymAction::Negate` toggles the sign of the term coefficient
- dummy ids are renamed by first occurrence in factor scan order
- standalone dummy allocation always uses the low end of the pool for each
  range
- factor order is deterministic after symmetry and tied-group handling

Symmetry coefficient consistency:

- if multiple canonical candidates have the same structure but different
  coefficients, return `CanonError::InconsistentSymmetryCoefficient`
- otherwise choose the structurally smallest candidate by the private term
  comparator
- coefficient values do not affect which structural representative is selected
- `canon` does not simplify or drop terms implied to be zero by inconsistent
  symmetry coefficients; callers decide how to handle this condition

## Split Canonicalization

```rust
pub fn canon_split(
    split: &Split,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<(Split, Split), CanonError>;
```

`split` emits one unordered structural split. `canon_split` creates the two
canonical owner orientations needed by graph construction:

- first return value: left side is owner
- second return value: right side is owner

No wrapper type is introduced.

Owner orientation is only the policy for choosing the contracted dummy-id rename
map. It does not change the split side roles. Both returned `Split` values keep
the original `left` and `right` side positions: `Split.left` corresponds to
`interface.left_external`, and `Split.right` corresponds to
`interface.right_external`.

### Contracted And Private Dummy IDs

For a split:

- contracted ids come from `split.interface.contracted`
- left-private dummy ids come from `split.left.sum_indices`
- right-private dummy ids come from `split.right.sum_indices`

External ids are never renamed.

`canon` must not reconstruct the split interface from factor overlap.
`SplitInterface` is the source of truth.

The owner orientation needs a contracted-id rename map:

```rust
let contracted_ids: HashSet<IndexId> =
    split.interface.contracted.iter().map(|index| index.id).collect();
```

### Left-Owner Orientation

The left-owner orientation is built as:

1. enumerate symmetry and ordering candidates for the left term
2. for each left candidate:
   - assign contracted dummy ids using low allocation
   - assign left-private dummy ids using low allocation
   - produce an owner term and the contracted rename map
3. choose the minimum owner term index
4. fetch the owner term and contracted rename map from aligned vectors
5. enumerate symmetry and ordering candidates for the right term
6. rename right contracted dummy ids through the chosen contracted map
7. assign right-private dummy ids using high allocation
8. choose the minimum right follower term index
9. remap `split.interface.contracted` through the owner contracted map
10. return a `Split` with the canonical left owner, canonical right follower,
   and remapped interface

### Right-Owner Orientation

The right-owner orientation is symmetric:

1. canonicalize the right term as owner
2. assign contracted dummy ids using low allocation
3. assign right-private dummy ids using high allocation
4. canonicalize the left term as follower using the right owner's contracted map
5. assign left-private dummy ids using low allocation
6. remap `split.interface.contracted` through the right owner's contracted map

The allocation policy intentionally makes the two owner orientations distinct
when needed:

- left-owner private ids use low allocation on the left and high allocation on
  the right
- right-owner private ids use low allocation on the left and high allocation on
  the right after the right owner fixes the contracted map

The important invariant is that both returned splits use a consistent name for
each contracted dummy id across their left and right terms.

### Interface Handling

`SplitInterface` stays explicit.

Rules:

- `left_external` passes through unchanged
- `right_external` passes through unchanged
- `contracted` is remapped through the owner contracted rename map
- remapped `contracted` values preserve `Index.range`
- remapped `contracted` is sorted by `IndexId`

No downstream module should reconstruct the interface from factor overlap.

## Internal Function Pipeline

The implementation should be organized as explicit private helpers. These names
are not public API, but they describe the intended implementation shape.

### Symmetry Helpers

```rust
fn enumerate_sym_group(
    tensor: TensorId,
    generators: &[SymGenerator],
    arity: usize,
) -> Result<Vec<(Vec<usize>, SymAction)>, CanonError>;

fn enumerate_factor_variants(
    tensor: TensorId,
    indices: &[IndexId],
    generators: &[SymGenerator],
) -> Result<Vec<(Vec<IndexId>, SymAction)>, CanonError>;

fn enumerate_symmetry_terms(
    term: &Term,
    symmetry: &TensorSymmetryMap,
) -> Result<Vec<Term>, CanonError>;

fn apply_action_to_coeff(coeff: Rational, action: SymAction) -> Rational;
```

Responsibilities:

- `enumerate_sym_group` computes the finite closure of the supplied generators
  for a concrete factor arity
- invalid generator permutations return `InvalidSymmetryPermutation`
- generator arity mismatches return `SymmetryArityMismatch`
- `enumerate_factor_variants` applies each group element to one factor's
  indices
- `enumerate_symmetry_terms` takes the Cartesian product across all factor
  variants and combines coefficient actions
- `apply_action_to_coeff` negates the coefficient only for `SymAction::Negate`

### Factor Ordering Helpers

```rust
type DummyRange = HashMap<IndexId, RangeId>;

fn build_term_dummy_range(term: &Term) -> DummyRange;

fn build_split_dummy_range(split: &Split) -> DummyRange;

fn compare_factors_by_structure(
    left: &Factor,
    right: &Factor,
    dummy_range: &DummyRange,
) -> Ordering;

fn tied_groups(
    factors: &[Factor],
    dummy_range: &DummyRange,
) -> Vec<Vec<usize>>;

fn enumerate_ordered_terms(
    term: &Term,
    dummy_range: &DummyRange,
) -> Vec<Term>;
```

Responsibilities:

- `build_term_dummy_range` maps each summed dummy id in a standalone term to
  its range
- `build_split_dummy_range` maps every renameable split dummy id to its range:
  contracted ids from `split.interface.contracted`, left-private ids from
  `split.left.sum_indices`, and right-private ids from
  `split.right.sum_indices`
- `compare_factors_by_structure` compares factor structure before dummy
  renaming
- dummy slots compare as dummy positions by `RangeId`
- external slots compare as external positions by `IndexId`
- factors are first sorted by the structural comparator
- only factors tied by structural equality are permuted against each other

This preserves external-id distinctions while treating dummy ids by range until
renaming chooses concrete ids.

For split canonicalization, contracted ids are not present in either side term's
`sum_indices`; by the `split` contract they live only in
`SplitInterface::contracted`. Therefore split factor ordering must use
`build_split_dummy_range(split)`, not a side-local term dummy range. Contracted
and private id sets used by rename helpers are derived directly from
`SplitInterface::contracted` and the corresponding side term's `sum_indices`.

### Allocation And Rename Helpers

```rust
struct PoolAllocator<'a> {
    pool: &'a IndexPool,
    low: HashMap<RangeId, usize>,
    high: HashMap<RangeId, usize>,
    used: HashMap<RangeId, HashSet<usize>>,
}

impl<'a> PoolAllocator<'a> {
    fn new(pool: &'a IndexPool) -> Self;

    fn from_base_map_for_ids(
        pool: &'a IndexPool,
        base_map: &HashMap<IndexId, IndexId>,
        original_ids: &HashSet<IndexId>,
    ) -> Result<Self, CanonError>;

    fn alloc_low(&mut self, range: RangeId) -> Result<IndexId, CanonError>;

    fn alloc_high(&mut self, range: RangeId) -> Result<IndexId, CanonError>;
}

fn build_rename_map<F>(
    term: &Term,
    rename_ids: &HashSet<IndexId>,
    dummy_range: &DummyRange,
    base_map: &HashMap<IndexId, IndexId>,
    allocator: &mut PoolAllocator<'_>,
    allocate: F,
) -> Result<HashMap<IndexId, IndexId>, CanonError>
where
    F: FnMut(&mut PoolAllocator<'_>, RangeId) -> Result<IndexId, CanonError>;

fn apply_rename_map(
    term: &Term,
    remap: &HashMap<IndexId, IndexId>,
) -> Term;
```

Responsibilities:

- `PoolAllocator` allocates from immutable sorted pool vectors
- `alloc_low` chooses the first unused id in a range
- `alloc_high` chooses the last unused id in a range
- missing range pools return `MissingIndexPool`
- no available ids return `ExhaustedIndexPool`
- `build_rename_map` scans factors left to right and slots left to right
- `apply_rename_map` renames all factor and sum indices

No split-specific rename application helper is needed. By the `split` module
contract, side-term `sum_indices` already contain only private dummy ids;
contracted ids live in `SplitInterface::contracted`. A normal
`apply_rename_map` preserves that separation.

### Canonical Selection Helpers

```rust
fn choose_min_term_index(candidates: &[Term]) -> Result<usize, CanonError>;

fn compare_terms(left: &Term, right: &Term) -> Ordering;

fn compare_term_structure(left: &Term, right: &Term) -> Ordering;
```

Responsibilities:

- return `EmptyCanonicalCandidates` if there are no candidates
- compare candidates by `compare_term_structure`
- return the index of the structurally minimum candidate
- use `compare_term_structure` to detect equal canonical structures with
  inconsistent coefficients
- detect inconsistent coefficients for equal canonical structure
- coefficient values are payload/sign transport and are not structural
  tie-breakers
- callers that need sidecar data, such as contracted rename maps, should keep
  that data in vectors aligned with the candidate term vector and fetch it by
  the returned index

### Split Rename Helpers

```rust
enum SplitSide {
    Left,
    Right,
}

fn rename_standalone_term(
    term: &Term,
    dummy_range: &DummyRange,
    pool: &IndexPool,
) -> Result<Term, CanonError>;

fn rename_owner_term(
    term: &Term,
    side: SplitSide,
    contracted_ids: &HashSet<IndexId>,
    dummy_range: &DummyRange,
    pool: &IndexPool,
) -> Result<(Term, HashMap<IndexId, IndexId>), CanonError>;

fn rename_follower_term(
    term: &Term,
    side: SplitSide,
    contracted_ids: &HashSet<IndexId>,
    contracted_map: &HashMap<IndexId, IndexId>,
    dummy_range: &DummyRange,
    pool: &IndexPool,
) -> Result<Term, CanonError>;

fn remap_interface(
    interface: &SplitInterface,
    remap: &HashMap<IndexId, IndexId>,
) -> SplitInterface;
```

Responsibilities:

- `rename_standalone_term` renames every dummy id using low allocation
- `rename_owner_term` creates the owner term and contracted-id rename map
- `rename_follower_term` uses the owner contracted map, then allocates private ids
- `remap_interface` preserves external vectors and remaps only `contracted`

## Error Policy

`canon` should not panic for malformed canonicalization inputs.

Return errors for:

- missing tensor symmetry entries
- invalid generator permutations
- generator arity mismatches for a concrete factor
- missing index pool entries
- exhausted index pools
- empty canonical candidate iterators
- equal canonical structures with inconsistent coefficients

`repr::validate` catches many structural problems earlier, but `canon` should
still defend its own assumptions where it has enough context to return a clear
error.

## Testing

Initial tests should cover:

- `build_index_pool` groups dummy ids by range, sorts, and deduplicates
- `build_tensor_symmetry_map` indexes by tensor id
- `canon_term` normalizes dummy names
- `canon_term` normalizes sum-index order
- `canon_term` applies factor symmetry before ordering
- `canon_term` applies `Negate` to coefficients
- `canon_term` distinguishes external ids with the same range
- `canon_term` is deterministic for tied factors
- `canon_term` selects representatives by structure, not coefficient value
- `canon_term` returns `InconsistentSymmetryCoefficient` instead of panicking
- `canon_split` returns two owner orientations
- `canon_split` owner orientation does not swap split left/right side roles
- split owner/follower terms use consistent shared names
- `canon_split` remaps `interface.contracted`
- `canon_split` preserves `left_external` and `right_external`
- missing tensor symmetry returns `MissingTensorSymmetry`
- invalid symmetry arity returns `SymmetryArityMismatch`
- missing or exhausted pools return pool errors

## Acceptance Criteria

The `canon` module is complete when:

- public API exposes explicit maps, `canon_term`, `canon_split`, and
  fallible canonicalization errors
- no `CanonContext` exists
- no canonical split wrapper exists
- `canon_term` and `canon_split` are fallible
- standalone terms canonicalize deterministically under symmetry, ordering, and
  renaming
- `canon_split` returns `(left_owner, right_owner)`
- `SplitInterface` remains explicit, with only `contracted` remapped
- errors are returned instead of panics for the malformed cases listed above
