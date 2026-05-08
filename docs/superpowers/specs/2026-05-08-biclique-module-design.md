# `biclique` Module Design

## Goal

Define biclique enumeration for one construction graph.

`biclique` consumes a `ConstrGraph` from the `graph` module and enumerates
legal inclusion-maximal bicliques. The algorithm should closely follow the
carefully designed `rustymill` implementation, with only small cleanup for the
new module boundaries.

It does not build graphs, canonicalize terms, inspect `TensorComputation`,
export rewrite templates, rank candidates, or apply actions.

## Scope

This module includes:

- `Biclique`
- graph-local biclique enumeration
- coefficient-factorization checks
- source-term provenance disjointness checks
- inclusion-maximal search

This module excludes:

- graph building
- candidate template export
- action masks
- cost/profitability filtering
- output ranking
- output sorting

## Public API

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Biclique {
    pub left_node_ids: Vec<usize>,
    pub right_node_ids: Vec<usize>,
    pub left_coeffs: Vec<Rational>,
    pub right_coeffs: Vec<Rational>,
    pub terms_used: u64,
}

pub fn enumerate_bicliques(graph: &ConstrGraph) -> Vec<Biclique>;
```

Field meanings:

- `left_node_ids[i]` indexes `graph.left_nodes`
- `left_coeffs[i]` is the coefficient assigned to `left_node_ids[i]`
- `right_node_ids[j]` indexes `graph.right_nodes`
- `right_coeffs[j]` is the coefficient assigned to `right_node_ids[j]`
- `terms_used` is the union of provenance bits from all selected edges

The node-id vectors and coefficient vectors are aligned by position.

## Legality

A returned `Biclique` is legal when:

- every selected left node is connected to every selected right node
- each selected edge coefficient satisfies:

```text
edge.coeff == left_coeff * right_coeff
```

- selected edge provenance bitsets are pairwise disjoint
- the biclique has sharing:

```rust
left_node_ids.len() >= 2 || right_node_ids.len() >= 2
```

The sharing rule excludes trivial one-edge bicliques.

## Maximality

Enumeration returns inclusion-maximal legal bicliques.

For an emitted biclique, no additional left graph node or right graph node can
be added while preserving the legality rules above.

This module does not rank maximal bicliques by cost, savings, size, or any
other heuristic.

## Output Ordering

The returned `Vec<Biclique>` has no semantic ordering guarantee.

The implementation may return bicliques in traversal order. Tests and callers
should compare by content rather than relying on vector position.

The node order inside each emitted `Biclique` is also traversal-defined. The
module should not sort node ids before emission. In particular, do not include
a `canonicalize_biclique` helper in the first implementation.

Later modules should treat the emitted node order as the template/mask order
for that candidate. If a stronger ordering is needed later, it should be
designed deliberately.

## Algorithm

The implementation should closely port the existing `rustymill` recursive
search.

Keep:

- the side-aware `SearchNode` enum
- the incremental `Delta` record
- the `cand` + frontier recursion structure
- `expand`
- `sift`
- `build_child_frontiers`
- `update_delta`
- `push` / `pop`
- starting from left nodes only

Remove:

- `canonicalize_biclique`
- any output sorting
- any cost/profitability checks

## Internal Types

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SearchNode {
    Left(usize),
    Right(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Delta {
    coeff: Rational,
    terms: u64,
}
```

`Delta` records the coefficient and provenance contribution implied if a
candidate `SearchNode` is added to the current partial biclique.

## Internal Function Pipeline

Recommended private helpers:

```rust
fn all_candidates(graph: &ConstrGraph) -> Vec<SearchNode>;

fn initial_frontier(graph: &ConstrGraph) -> HashMap<SearchNode, Delta>;

fn empty_biclique() -> Biclique;

fn edge_between(
    graph: &ConstrGraph,
    left_id: usize,
    right_id: usize,
) -> Option<&GraphEdge>;

fn expand(
    graph: &ConstrGraph,
    biclique: &mut Biclique,
    frontier: &HashMap<SearchNode, Delta>,
    candidates: &mut Vec<SearchNode>,
    out: &mut Vec<Biclique>,
);

fn sift(
    biclique: &Biclique,
    candidates: &[SearchNode],
    frontier: &HashMap<SearchNode, Delta>,
    child_frontiers: &HashMap<SearchNode, HashMap<SearchNode, Delta>>,
) -> Vec<SearchNode>;

fn build_child_frontiers(
    graph: &ConstrGraph,
    biclique: &Biclique,
    frontier: &HashMap<SearchNode, Delta>,
) -> HashMap<SearchNode, HashMap<SearchNode, Delta>>;

fn update_delta(
    graph: &ConstrGraph,
    biclique: &Biclique,
    chosen: SearchNode,
    chosen_delta: &Delta,
    candidate: SearchNode,
    candidate_delta: &Delta,
) -> Option<Delta>;

fn has_sharing(biclique: &Biclique) -> bool;

fn push(biclique: &mut Biclique, node: SearchNode, delta: &Delta);

fn pop(biclique: &mut Biclique, node: SearchNode, delta: &Delta);
```

Helper responsibilities:

- `all_candidates` returns left nodes followed by right nodes
- `initial_frontier` initializes every search node with coefficient `1` and
  empty provenance
- `empty_biclique` creates an empty mutable search state
- `edge_between` finds an edge by graph-local node ids
- `expand` performs the recursive maximal biclique search
- `sift` chooses branchable nodes using the existing pruning logic
- `build_child_frontiers` computes legal next frontiers for each candidate
- `update_delta` enforces edge existence, coefficient consistency, and
  provenance disjointness
- `has_sharing` enforces the nontrivial-sharing rule
- `push` adds a node and its delta to the mutable biclique
- `pop` removes the most recently added node and delta

## Top-Level Flow

```text
enumerate_bicliques(graph):
  if graph.edges.len() < 2:
    return []

  biclique = empty_biclique()
  candidates = all_candidates(graph)
  frontier = initial_frontier(graph)
  out = []

  expand(graph, biclique, frontier, candidates, out)
  return out
```

Emission rule inside `expand`:

```text
if has_sharing(biclique) && frontier is empty:
  out.push(biclique.clone())
  return
```

Do not sort the emitted biclique before pushing it.

## `update_delta` Semantics

`update_delta` decides whether an existing `candidate` remains legal after
choosing `chosen`.

Same-side case:

- no edge is required
- provenance between `chosen_delta` and `candidate_delta` must be disjoint
- if disjoint, keep `candidate_delta` unchanged

Opposite-side case:

- the graph must contain an edge between the two nodes
- that edge's provenance must be disjoint from:
  - the current biclique provenance
  - `chosen_delta.terms`
  - `candidate_delta.terms`
- coefficient consistency must hold:

```text
expected = edge.coeff / chosen_delta.coeff
```

- if `candidate_delta.terms == 0`, assign
  `candidate_delta.coeff = expected`
- otherwise require `candidate_delta.coeff == expected`
- if all checks pass, OR the edge provenance into `candidate_delta.terms`

This is the core legality check. It should be ported carefully.

## Search Behavior

Initial branching should consider left nodes only. This avoids symmetric
duplicates from bootstrapping on both sides.

After one left node has been selected, right nodes must be selected through
actual incident edges. This bootstraps meaningful bicliques and avoids emitting
isolated same-side sets.

Once both sides are present, recursion follows the existing frontier and sift
logic.

## Relationship To Adjacent Modules

`graph` provides `ConstrGraph`, `GraphEdge`, node IDs, edge coefficients, and
source-term provenance.

`biclique` returns graph-local `Biclique` values only.

`rewrite` will later pair each `Biclique` with its originating `ConstrGraph`
and export candidate templates. `biclique` does not know about that action
layer.

## Testing

Initial tests should cover:

- graphs with fewer than two edges produce no bicliques
- one-edge bicliques are not emitted
- `2x1` and `1x2` sharing cases are emitted
- larger complete bicliques are emitted
- non-factorizable coefficient patterns are rejected
- overlapping provenance is rejected
- only inclusion-maximal legal bicliques are emitted
- starting from left nodes avoids symmetric duplicate bootstraps
- emitted node/coeff order is traversal-defined and not sorted by the module
- output vector order is not asserted

## Acceptance Criteria

The `biclique` module is complete when:

- `enumerate_bicliques` consumes one `ConstrGraph`
- returned `Biclique` values are graph-local records
- coefficient factorization and provenance disjointness are enforced
- trivial one-edge bicliques are excluded
- inclusion-maximal legal bicliques are emitted
- the existing `rustymill` recursive algorithm is preserved
- no `canonicalize_biclique` helper exists
- output order is unspecified
