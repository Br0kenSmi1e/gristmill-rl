# `repr` Module Design

## Goal

Define the minimal symbolic data model for the new rewrite kernel.

`repr` is the compatibility boundary with existing `rustymill` JSON fixtures and
the shared vocabulary used by later modules. It should stay deliberately small:
data structures, serde support, builder/accessor helpers, and optional
validation.

It should not contain formatting, canonicalization, split enumeration, graph
logic, biclique logic, rewrite logic, cost logic, or JSON file I/O.

## Scope

This module includes:

- ID newtypes
- tensor symmetry data
- ranges, indices, tensors, factors, terms, and definitions
- the `TensorComputation` container
- builders and read/write accessors for `TensorComputation`
- structural validation helpers

This module excludes:

- `Display` implementations
- algebraic simplification
- term sorting or normalization
- duplicate-term merging
- action generation
- serde adapters for changed field names

## JSON Compatibility

The new `repr` should preserve the serialized shape of `rustymill::repr` for the
fields that remain in scope:

- `ranges`
- `tensors`
- `definitions`
- `id`
- `size`
- `symmetry`
- `perm`
- `action`
- `range`
- `base`
- `ext_indices`
- `terms`
- `coeff`
- `sum_indices`
- `factors`
- `tensor`
- `indices`

The first implementation should keep Rust field names aligned with these JSON
field names instead of adding serde rename adapters.

One intentional schema simplification is symmetry action support. The new
kernel only supports sign symmetry:

```rust
pub enum SymAction {
    Identity,
    Negate,
}
```

Fixtures containing `Conjugate` or `NegateConjugate` are out of scope for this
kernel and should fail to deserialize unless a later design adds compatibility
handling.

## Public Types

ID newtypes:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RangeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorId(pub u32);
```

Symmetry:

```rust
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
```

Tensor data:

```rust
pub type Rational = Ratio<i64>;

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
```

Container:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorComputation {
    ranges: Vec<Range>,
    tensors: Vec<TensorInfo>,
    definitions: Vec<TensorDef>,
}
```

The container fields stay private so callers use the builder/accessor API.
Serde still sees the same field names.

## Symmetry Helpers

`SymAction` should support composition:

```rust
impl SymAction {
    pub fn combine(self, other: SymAction) -> SymAction;
}
```

Behavior:

- `Identity.combine(x) == x`
- `Negate.combine(Identity) == Negate`
- `Negate.combine(Negate) == Identity`

`SymGenerator` should support applying a permutation to an index-like slice:

```rust
impl SymGenerator {
    pub fn apply<T: Copy>(&self, indices: &[T]) -> Result<(Vec<T>, SymAction), ReprError>;
}
```

Unlike the old reference code, this helper should not panic on arity mismatch.
It should return a validation-style error.

## Builder And Accessor API

```rust
impl TensorComputation {
    pub fn new() -> Self;

    pub fn add_range(&mut self, size: u64) -> RangeId;

    pub fn add_tensor(&mut self, symmetry: Vec<SymGenerator>) -> TensorId;

    pub fn add_definition(
        &mut self,
        base: TensorId,
        ext_indices: Vec<Index>,
        terms: Vec<Term>,
    );

    pub fn ranges(&self) -> &[Range];
    pub fn tensors(&self) -> &[TensorInfo];
    pub fn definitions(&self) -> &[TensorDef];
    pub fn definitions_mut(&mut self) -> &mut Vec<TensorDef>;

    pub fn next_tensor_id(&self) -> TensorId;
}
```

Builder behavior:

- `add_range` assigns `RangeId(ranges.len())`
- `add_tensor` assigns `TensorId(tensors.len())`
- `add_definition` appends a `TensorDef` without validating it
- `next_tensor_id` returns `TensorId(tensors.len())`

The builders are convenience helpers, not invariant-enforcing constructors.
This keeps serde-loaded and hand-built computations on the same footing.

`Default` should delegate to `TensorComputation::new()`.

## Validation

`repr` should expose optional validation:

```rust
impl TensorComputation {
    pub fn validate(&self) -> Result<(), ReprError>;
}
```

Validation should inspect structure and references without mutating,
normalizing, sorting, or simplifying.

Checks:

- each `Range.id` equals its position in `ranges`
- each `TensorInfo.id` equals its position in `tensors`
- every `TensorDef.base` references an existing tensor
- every `Factor.tensor` references an existing tensor
- every `Index.range` references an existing range
- within each definition, every factor index ID is declared by either
  `ext_indices` or the term's `sum_indices`
- within each definition, `ext_indices` and the union of all `sum_indices` are
  disjoint
- within each definition, the same `IndexId` is not assigned two different
  ranges
- within each definition, duplicate external index IDs are rejected
- within each term, duplicate sum index IDs are rejected
- every symmetry generator permutation is a valid permutation of
  `0..perm.len()`

Validation should not check tensor arity globally. A tensor can appear with
different arities in malformed data, but the representation does not store a
single tensor arity. Canonicalization should check generator arity when applying
symmetry to a concrete factor.

Validation should not reject:

- empty computations
- empty definitions
- zero terms
- zero coefficients
- terms with no factors
- definitions with repeated algebraically equivalent terms

These cases may be uninteresting to later pipeline stages, but they are
representable.

## Error Type

Use a concrete error enum for `repr` validation and symmetry helper failures:

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
```

The exact display wording can be deferred. The first implementation only needs
structured errors for tests and callers.

## Derives And Trait Policy

Required derives:

- serde `Serialize` and `Deserialize` for all public data records
- `Clone`, `Debug`, `PartialEq`, and `Eq` for all public data records
- `Hash`, `PartialOrd`, and `Ord` for ID newtypes
- `Hash` for records used as downstream map keys: `Range`, `Index`,
  `SymGenerator`, `Factor`, and `Term`

Do not add `Display` implementations in the first version. Pretty-printing is
not needed for the kernel boundary and can be designed later as a separate
diagnostic layer.

## Testing

Initial tests should cover:

- empty computation construction
- ID assignment by builders
- accessor behavior
- serde round trip for a hand-built computation
- old-compatible JSON fields are present
- `SymAction::combine`
- `SymGenerator::apply` success and arity mismatch
- `validate` accepts a well-formed computation
- `validate` catches ID-position mismatches
- `validate` catches unknown range, tensor, and index references
- `validate` catches inconsistent index ranges
- `validate` catches duplicate external and sum index IDs
- `validate` catches external/sum index overlap
- `validate` catches invalid symmetry permutations

## Acceptance Criteria

The `repr` module is complete when:

- all approved public types exist with serde-compatible field names
- `SymAction` only supports `Identity` and `Negate`
- `TensorComputation` builders and accessors match this spec
- no `Display` implementations are added
- `TensorComputation::validate` performs the structural checks listed above
- serde round-trip tests pass for representative compatible fixtures
