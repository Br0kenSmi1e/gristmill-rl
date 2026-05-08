# `split` Module Design

## Goal

Define the structural split stage for the rewrite kernel.

`split` takes one `Term` and its parent `TensorDef`, enumerates nontrivial
factor bipartitions, and emits explicit split records for later
canonicalization.

It does not perform canonicalization, symmetry reasoning, graph grouping,
biclique enumeration, cost evaluation, action validation, or pruning.

## Scope

This module includes:

- the split record type
- the explicit split-interface type
- local factor bipartition enumeration for one term
- construction of left and right structural subterms
- construction of explicit interface index vectors

This module excludes:

- public factor-subset bitmask types
- final sorting of the returned split vector
- canonical owner-orientation logic
- tensor symmetry handling
- profitability filtering
- rewrite construction

## Public API

```rust
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

pub fn enumerate_splits(term: &Term, def: &TensorDef) -> Vec<Split>;
```

`SplitInterface` is the authoritative downstream record for the final
contraction between the two sides:

- `left_external`: parent-definition external indices used by the left side
- `right_external`: parent-definition external indices used by the right side
- `contracted`: summed indices that appear on both sides of the split

Downstream modules should trust this record. They should not reconstruct the
interface from factor overlap.

## Factor Bipartitions

For a term with `n` factors, `enumerate_splits` considers nonempty factor
bipartitions:

- left side is nonempty
- right side is nonempty
- both sides are disjoint
- both sides together cover all factors

For `n < 2`, the function returns an empty vector.

Each unordered bipartition should be emitted once. `split` should not emit both
`(left, right)` and `(right, left)` as separate records. Later
`canon::canon_split` is responsible for producing the two canonical owner
orientations.

## Internal Subset Representation

The implementation may use bitsets internally to represent factor subsets and
index membership, but no bitset type is public API.

If an internal `FactorSubset` type is useful, it should stay private to the
module.

Any practical limit from a bitset implementation is an implementation detail,
not a semantic part of the public contract. If the implementation chooses a
fixed-width bitset, it should either assert/fail clearly at that boundary or be
changed later without affecting callers.

## Side Normalization

`split` should choose one deterministic side ordering for each unordered
bipartition before building public `Vec<Index>` interface values.

Recommended normalization:

1. represent each factor subset as an internal `FactorSubset` bitset
2. for a candidate `left`, compute `right = full ^ left`
3. emit the split only when `left < right`
4. build public subterms and interface vectors from those factor subsets

This directly uses the factor-subset representation to avoid emitting both
`(left, right)` and `(right, left)`. It also avoids computing interface data
before deciding whether a candidate is the representative.

The comparison is purely a deterministic representative choice. Later stages
should not assign semantic meaning to split output position.

## Subterm Construction

For each normalized side:

- copy selected factors in their original order from `term.factors`
- collect the `IndexId`s that appear in those selected factors
- keep only the `term.sum_indices` whose IDs appear in the selected factors and
  are not contracted across the split
- set the coefficient to `1`

The original term coefficient is not stored in either subterm. It belongs to
the graph edge/candidate generation layer, where source-term provenance is
known.

Contracted sum indices are removed from both subterms and recorded only in
`SplitInterface::contracted`.

## Interface Construction

The public interface vectors should be explicit `Index` values, not masks.

Construction rules:

- `left_external` contains entries from `def.ext_indices` whose `IndexId`
  appears in any normalized left-side factor
- `right_external` contains entries from `def.ext_indices` whose `IndexId`
  appears in any normalized right-side factor
- `contracted` contains entries from `term.sum_indices` whose `IndexId` appears
  in both normalized sides
- all three vectors preserve the original `Index.range`
- all three vectors are sorted by `Index.id`

Sorting by `Index.id` is sufficient. `repr::validate` is responsible for
ensuring that a given `IndexId` has one consistent `RangeId` within a
definition.

## Output Ordering

`enumerate_splits` does not need to sort its returned `Vec<Split>`.

The output order should be deterministic as a consequence of deterministic
subset enumeration plus side normalization. Tests should assert exact ordering
only when that ordering follows directly from the chosen subset enumeration.

Later stages should not rely on split position as a semantic property.

## Example

For:

```text
term = X[a,c] * Y[c,d] * Z[d,b]
def.ext_indices = [a, b]
term.sum_indices = [c, d]
```

The three unordered bipartitions are:

```text
X | YZ
Y | XZ
Z | XY
```

After side normalization, expected split interfaces are:

```text
X | YZ:
  left_external  = [a]
  right_external = [b]
  contracted     = [c]

Y | XZ:
  left_external  = []
  right_external = [a, b]
  contracted     = [c, d]

XY | Z:
  left_external  = [a]
  right_external = [b]
  contracted     = [d]
```

For `X | YZ`, the subterms are:

```text
left:
  coeff       = 1
  sum_indices = []
  factors     = [X[a,c]]

right:
  coeff       = 1
  sum_indices = [d]
  factors     = [Y[c,d], Z[d,b]]
```

The contracted index `c` is not present in either subterm's `sum_indices`; it
is recorded in `interface.contracted`.

## Relationship To Adjacent Modules

`repr` provides the input data model and optional validation. `split` assumes
its inputs are structurally meaningful enough to inspect, but it does not call
`validate` itself.

`canon` consumes each raw `Split` and produces two canonical owner orientations:

```rust
pub fn canon_split(...) -> (Split, Split);
```

This means `split` should emit only one representative of each unordered
bipartition.

`graph` consumes already-canonicalized split streams. It should use
`SplitInterface` directly rather than deriving interface information from
subterm overlap.

## Testing

Initial tests should cover:

- terms with zero or one factor produce no splits
- a two-factor term produces one split
- a three-factor chain produces three unordered splits
- subterm coefficients are reset to `1`
- selected factors preserve original order
- contracted sum indices are removed from side subterms
- private sum indices remain on the side that uses them
- `left_external`, `right_external`, and `contracted` preserve `Index.range`
- interface vectors are sorted by `IndexId`
- side normalization emits only the `left < right` representative under
  internal `FactorSubset` ordering

## Acceptance Criteria

The `split` module is complete when:

- public API exposes only `SplitInterface`, `Split`, and `enumerate_splits`
- no public factor-subset bitmask type exists
- each unordered factor bipartition is emitted once
- subterms are structural unit-coefficient terms
- contracted indices are represented only in `SplitInterface::contracted`
- interface vectors are explicit `Index` values sorted by `IndexId`
- the returned split vector is deterministic but not explicitly sorted
