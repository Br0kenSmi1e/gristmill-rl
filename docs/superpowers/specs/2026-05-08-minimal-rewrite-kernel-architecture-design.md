# Minimal Rewrite Kernel Architecture Design

## Goal

Build a clean Rust library in this repository for the symbolic rewrite pipeline
currently represented in `~/rcode/rustymill` by:

```text
repr -> rl_parenth -> rl_canon -> biclique -> biclique_action
```

`rustymill` is a reference implementation and fixture source, not a 1:1 port
target. The new library should keep the useful pipeline shape while removing
historical naming, coupling, and transient transport types that do not carry
clear invariants.

The first usable surface is a library with JSON I/O:

- load and write `TensorComputation` JSON compatible with `rustymill::repr`
- generate rewrite candidates for a tensor computation
- validate one selected candidate decision
- build and apply the resulting rewrite

No CLI, optimizer loop, RL code, Python binding, cost model, or search policy is
included in this architecture slice.

## Design Principles

- Keep the serialized representation compatible with existing `rustymill`
  fixtures.
- Keep each pipeline stage responsible for one algebraic concept.
- Use explicit stage contracts instead of shared hidden context objects.
- Make orientation logic belong to canonicalization, not graph building.
- Keep graph and biclique code independent of `TensorComputation` mutation.
- Prefer small public APIs and plain data over broad framework abstractions.

## Crate Shape

The crate should be a Rust library. The initial module decomposition is:

```text
src/
  lib.rs
  repr.rs
  split.rs
  canon.rs
  graph.rs
  biclique.rs
  rewrite.rs
  io.rs
```

Dependencies should be minimal and match the representation needs:

- `serde`
- `serde_json`
- `num` with serde support for rational coefficients

## Module Responsibilities

### `repr`

`repr` owns the symbolic data model and JSON-compatible schema.

It should preserve the serialized shape of `rustymill::repr`, including:

- `TensorComputation`
- `TensorDef`
- `Term`
- `Factor`
- `Range`, `Index`, and ID newtypes
- `TensorInfo`
- tensor symmetry generators and actions
- rational coefficients

The Rust type and field names may stay close to `rustymill` to avoid serde
adapter complexity. Helper methods and documentation may be improved, but the
wire format should remain compatible with existing JSON fixtures.

This module should not know about split enumeration, canonicalization, graph
construction, biclique enumeration, or rewrites.

### `split`

`split` replaces the role of `rl_parenth` with clearer names.

It enumerates nontrivial bipartitions of a single term's factors and emits
explicit split records:

```rust
pub struct SplitInterface {
    pub left_external: Vec<Index>,
    pub right_external: Vec<Index>,
    pub contracted: Vec<Index>,
}

pub struct Split {
    pub left: Term,
    pub right: Term,
    pub interface: SplitInterface,
}

pub fn enumerate_splits(term: &Term, def: &TensorDef) -> Vec<Split>;
```

Expected behavior:

- terms with fewer than two factors produce no splits
- each split contains unit-coefficient left and right subterms
- factors keep their source order inside each subterm
- `left_external` and `right_external` come from `def.ext_indices`
- `contracted` comes from sum indices crossing the split
- interface vectors preserve `Index.range` and are sorted deterministically
- bitmasks may be used internally, but no bitmask should escape the module

An internal `FactorSubset` bitmask type may be useful while enumerating factor
bipartitions, but it is not part of the public stage contract.

`SplitInterface` is the source of truth for downstream stages. Later modules
must not reconstruct external or contracted indices from factor overlap.

### `canon`

`canon` replaces the role of `rl_canon`.

It canonicalizes standalone terms and split orientations using tensor
symmetries, factor ordering, and deterministic dummy-index renaming.

Public helpers remain explicit instead of wrapped in a context object:

```rust
pub type IndexPool = HashMap<RangeId, Vec<IndexId>>;
pub type TensorSymmetryMap = HashMap<TensorId, Vec<SymGenerator>>;

pub fn build_index_pool(def: &TensorDef) -> IndexPool;
pub fn build_tensor_symmetry_map(tensors: &[TensorInfo]) -> TensorSymmetryMap;

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

`CanonError` is the canon-stage error boundary. Its concrete variants are
specified in the detailed `canon` module design.

Expected behavior:

- `canon_term` returns a deterministic representative for equivalent terms
- tensor symmetry actions are applied to factor indices and coefficients
- tied factors are ordered deterministically
- dummy sum indices are renamed through the supplied index pool
- canonicalization should be deterministic across input term order
- malformed symmetry metadata or exhausted dummy-index pools return
  `CanonError` rather than panicking

`canon_split` owns all split-orientation logic. It returns:

- the canonicalized split with the left side as owner
- the canonicalized split with the right side as owner

No `CanonicalSplit` wrapper is introduced. The tuple is intentionally unpacked
by the caller into two canonical split streams before graph construction.

The split interface handling contract is:

- side external indices are carried through from the raw split
- contracted indices are remapped when dummy index names are canonicalized
- `Index.range` is preserved
- returned interface vectors remain deterministic

### `graph`

`graph` builds construction graphs from already-canonicalized split streams.

It does not enumerate splits and does not perform canonicalization or
left-owner/right-owner reasoning.

Input API:

```rust
pub struct GraphEdge {
    pub left_id: usize,
    pub right_id: usize,
    pub coeff: Rational,
    pub terms_used: u64,
}

pub struct ConstrGraph {
    pub interface: SplitInterface,
    pub left_nodes: Vec<Term>,
    pub right_nodes: Vec<Term>,
    pub edges: Vec<GraphEdge>,
}

pub fn build_graphs_from_splits(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<Vec<ConstrGraph>, GraphError>;
```

Expected behavior:

- `splits_by_term` is aligned with `def.terms`
- alignment mismatches and the `u64` provenance bitset term limit return
  `GraphError`
- each inner split list has already been canonicalized by `canon`
- graphs are bucketed by the explicit `SplitInterface`
- left and right terms are interned separately
- an edge records one canonical left/right pair in one interface bucket
- edge coefficients aggregate contributing source-term coefficients
- `terms_used` records source-term provenance with a bitset
- duplicate derivations from the same source term do not double-count
- zero-coefficient edges are omitted
- graphs with fewer than two useful edges are omitted
- output graph order is unspecified

The caller is responsible for invoking this function once for the left-owner
stream and once for the right-owner stream, then chaining the resulting graph
lists for candidate generation.

### `biclique`

`biclique` enumerates legal bicliques inside one construction graph.

It should not depend on `TensorComputation`, JSON, action masks, or rewrite
application.

API:

```rust
pub struct Biclique {
    pub left_node_ids: Vec<usize>,
    pub right_node_ids: Vec<usize>,
    pub left_coeffs: Vec<Rational>,
    pub right_coeffs: Vec<Rational>,
    pub terms_used: u64,
}

pub fn enumerate_bicliques(graph: &ConstrGraph) -> Vec<Biclique>;
```

Expected behavior:

- graphs with fewer than two edges produce no bicliques
- generated bicliques should represent valid factorizable sharing
- one side may have a single node if the other side has sharing
- node IDs and coefficient vectors stay aligned
- output order is deterministic
- bicliques are graph-local records and are not themselves public rewrite
  actions

The exact maximality and coefficient-consistency rules should be specified in
the detailed `biclique` module design.

### `rewrite`

`rewrite` is the action-facing layer above `split`, `canon`, `graph`, and
`biclique`.

It generates visible candidate templates, validates decisions, builds rewrites,
and applies rewrites to `TensorComputation`.

Public API:

```rust
pub struct Factorization {
    pub left_definition: TensorDef,
    pub right_definition: TensorDef,
    pub rewritten_definition: TensorDef,
}

pub struct ActionSpace {
    pub def_index: usize,
    pub candidate_templates: Vec<Factorization>,
    // private candidate records keep graph + biclique sidecars
}

pub struct Decision {
    pub candidate_index: usize,
    pub left_mask: Vec<bool>,
    pub right_mask: Vec<bool>,
}

pub struct FactorizationRewrite {
    pub def_index: usize,
    pub factorization: Factorization,
}

pub fn next_action_space(
    comp: &TensorComputation,
    start_from: usize,
) -> Option<ActionSpace>;

pub fn validate_decision(
    space: &ActionSpace,
    decision: &Decision,
) -> Result<(), RewriteError>;

pub fn build_rewrite(
    comp: &TensorComputation,
    space: &ActionSpace,
    decision: &Decision,
) -> Result<FactorizationRewrite, RewriteError>;

pub fn apply_rewrite(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) -> Result<(), RewriteError>;
```

`RewriteError` is intentionally left as a named boundary type in this
architecture spec. Its concrete variants should be designed when the rewrite
module receives its detailed design pass.

Expected behavior:

- `next_action_space` scans definitions from `start_from`
- definitions with no legal candidates are skipped
- the first actionable definition returns an `ActionSpace`
- candidate templates expose faithful factorization payloads
- hidden candidate records preserve the originating graph and biclique
- `Decision` masks select kept side terms from a template
- `true` means keep the corresponding side term
- both masks must have the expected length
- both sides must keep at least one term
- strict subsets of maximal bicliques are allowed
- `build_rewrite` uses hidden records to rebuild the selected sub-biclique
- `apply_rewrite` registers two fresh intermediate tensors and replaces the
  target definition with left, right, and rewritten definitions

The initial stale-decision policy is intentionally narrow. `apply_rewrite`
should verify:

- `def_index` exists in the current computation
- intermediate tensor IDs match the current fresh IDs

It does not prove that the target definition is unchanged since action-space
generation. Stronger stale-state protection can be designed later if needed.

### `io`

`io` owns JSON convenience functions for compatible fixtures.

API:

```rust
pub enum IoJsonError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

pub fn read_json(path: impl AsRef<Path>) -> Result<TensorComputation, IoJsonError>;
pub fn write_json(path: impl AsRef<Path>, comp: &TensorComputation) -> Result<(), IoJsonError>;
pub fn from_json(input: &str) -> Result<TensorComputation, serde_json::Error>;
pub fn to_json(comp: &TensorComputation) -> Result<String, serde_json::Error>;
```

JSON errors are separate from `RewriteError`; malformed files are not rewrite
validation failures.

## End-To-End Pipeline

`rewrite::next_action_space` orchestrates the pipeline:

```text
for each candidate TensorDef from start_from:
  symmetry = build_tensor_symmetry_map(comp.tensors())
  pool = build_index_pool(def)

  left_owner_splits_by_term = Vec<Vec<Split>>
  right_owner_splits_by_term = Vec<Vec<Split>>

  for each term in def.terms:
    for raw_split in split::enumerate_splits(term, def):
      (left_owner, right_owner) = canon::canon_split(raw_split, symmetry, pool)?
      push left_owner into the left-owner stream for this term
      push right_owner into the right-owner stream for this term

  graphs =
    graph::build_graphs_from_splits(def, left_owner_splits_by_term)?
    + graph::build_graphs_from_splits(def, right_owner_splits_by_term)?

  for each graph:
    for each biclique in biclique::enumerate_bicliques(graph):
      export a Factorization candidate template
      keep private graph + biclique metadata

  if any candidates were found:
    return ActionSpace

return None
```

This is the central module boundary decision:

- `canon_split` returns a fallible tuple of owner orientations
- `rewrite` unpacks that tuple into two `Vec<Vec<Split>>` streams
- `graph` consumes one canonicalized stream at a time and returns
  `Result<Vec<ConstrGraph>, GraphError>`
- `graph` never knows why a stream exists

## Testing Strategy

The first implementation should use focused tests by stage:

- `repr`: deserialize and reserialize selected `rustymill` JSON fixtures
- `split`: explicit interfaces for two-factor and three-factor terms
- `canon`: deterministic term and split canonicalization, including symmetry
- `graph`: grouping, edge merging, coefficient cancellation, provenance bits
- `biclique`: valid biclique enumeration on small hand-built graphs
- `rewrite`: action-space generation, decision validation, subset masks, apply
- integration: run the full pipeline on one or two `rustymill` fixtures

The tests should treat `rustymill` as a reference for selected behavior, not a
complete parity oracle. When the new kernel intentionally differs, the test
name or fixture comment should state the new expected behavior.

## Open Detailed-Design Items

This spec fixes the top-level decomposition and module APIs. It intentionally
does not fully specify:

- exact split ordering
- full canonicalization tie-break rules
- biclique maximality algorithm details
- candidate template ordering
- candidate profitability or ranking
- stronger stale-state checks

Those should be handled in detailed module designs or the implementation plan,
with this architecture spec as the boundary contract.

## Acceptance Criteria

This architecture is accepted when:

- the new crate is decomposed into the seven modules listed above
- `repr` can read and write the old JSON schema
- each non-`rewrite` stage exposes a narrow, independently testable API
- `canon_split` returns `Result<(Split, Split), CanonError>`
- `rewrite` unpacks canonical split tuples into two graph-building streams
- `graph` consumes `&[Vec<Split>]`, returns
  `Result<Vec<ConstrGraph>, GraphError>`, and contains no owner-orientation
  logic
- the public library API supports JSON-backed candidate generation, decision
  validation, rewrite construction, and rewrite application
