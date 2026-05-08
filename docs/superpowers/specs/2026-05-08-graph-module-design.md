# `graph` Module Design

## Goal

Define construction-graph building for the rewrite kernel.

`graph` consumes one stream of already-canonicalized splits grouped by source
term and builds bipartite construction graphs. It does not know whether the
stream came from the left-owner or right-owner output of `canon_split`.

It does not enumerate raw splits, canonicalize terms, enumerate bicliques,
rank candidates, or apply rewrites.

## Scope

This module includes:

- `ConstrGraph`
- `GraphEdge`
- graph construction from canonicalized split streams
- graph bucketing by explicit `SplitInterface`
- independent left/right node interning
- edge coefficient/provenance merging

This module excludes:

- side orientation repair
- interface reconstruction from term overlap
- owner-stream handling
- biclique enumeration
- candidate template export
- action masks
- rewrite construction

## Public API

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphEdge {
    pub left_id: usize,
    pub right_id: usize,
    pub coeff: Rational,
    pub terms_used: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstrGraph {
    pub interface: SplitInterface,
    pub left_nodes: Vec<Term>,
    pub right_nodes: Vec<Term>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    SplitTermAlignmentMismatch { expected: usize, got: usize },
    TooManyTerms { len: usize, max: usize },
}

pub fn build_graphs_from_splits(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<Vec<ConstrGraph>, GraphError>;
```

## Input Contract

`splits_by_term` is a stream for one owner orientation.

Rules:

- `splits_by_term.len()` must equal `def.terms.len()`
- `splits_by_term[term_idx]` contains canonicalized splits derived from
  `def.terms[term_idx]`
- the caller is responsible for invoking `graph` once for the left-owner stream
  and once for the right-owner stream, then chaining the graph lists
- every `SplitInterface` is trusted as authoritative

`graph` must not:

- recompute contracted indices from factor overlap
- recompute external indices from factor overlap
- swap split sides
- inspect why an owner stream exists

The `split` and `canon` modules are responsible for producing normalized
left/right split roles.

## Graph Meaning

Each `ConstrGraph` corresponds to one `SplitInterface` bucket.

Within a graph:

- `left_nodes` are canonical left-side terms
- `right_nodes` are canonical right-side terms
- an edge means at least one source term produced that left/right pair under
  the graph's interface

Left and right nodes are interned independently. If the same `Term` value
appears on both sides, it still receives separate left and right node IDs.

## Edge Payload

Each edge stores:

- `left_id`: index into `left_nodes`
- `right_id`: index into `right_nodes`
- `coeff`: summed coefficient contribution for this node pair
- `terms_used`: bitset of source-term indices contributing to this edge

For one split contribution:

```text
edge_coeff = def.terms[term_idx].coeff * split.left.coeff * split.right.coeff
```

Before interning node terms, reset `split.left.coeff` and `split.right.coeff`
to `1`. The side-term coefficients are absorbed into the edge coefficient.

Merge rules:

- if no edge exists for `(left_id, right_id)`, create one
- if an edge exists and its `terms_used` already contains `term_idx`, ignore
  the duplicate contribution
- if an edge exists and `term_idx` is new, add the coefficient and set the bit
- after all insertions, remove edges whose coefficient is zero
- after zero-edge removal, drop graphs with fewer than two edges

The duplicate same-term rule prevents one source term from being counted twice
when it reaches the same canonical edge through multiple split derivations.

## Term Limit

`terms_used` is a `u64`, so graph construction supports at most 64 source terms
in one active definition.

If `def.terms.len() > 64`, return:

```rust
GraphError::TooManyTerms { len, max: 64 }
```

This limit is explicit in the graph API. A future implementation can replace
the bitset representation without changing the high-level graph semantics.

## Output Ordering

The returned `Vec<ConstrGraph>` has no semantic ordering guarantee.

Implementation may return graphs in `HashMap` iteration order. Tests and
callers should compare graph sets by content rather than by vector position.

Within each graph:

- node IDs are assigned by first-seen order within that graph
- edge order is first-seen order within that graph

Those internal orders are deterministic for a fixed insertion order, but they
are implementation details of graph construction, not ranking signals.

## Internal Function Pipeline

The implementation should avoid a separate `PendingGraph` type. It can bucket
directly into public `ConstrGraph` values:

```rust
fn build_graphs_from_splits(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<Vec<ConstrGraph>, GraphError> {
    validate_splits_by_term(def, splits_by_term)?;

    let mut graphs: HashMap<SplitInterface, ConstrGraph> = HashMap::new();

    for (term_idx, splits) in splits_by_term.iter().enumerate() {
        let source_coeff = &def.terms[term_idx].coeff;

        for split in splits {
            let graph = graphs
                .entry(split.interface.clone())
                .or_insert_with(|| empty_graph(split.interface.clone()));
            insert_split(graph, source_coeff, term_idx, split)?;
        }
    }

    Ok(finalize_graphs(graphs))
}
```

Recommended private helpers:

```rust
fn validate_splits_by_term(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<(), GraphError>;

fn empty_graph(interface: SplitInterface) -> ConstrGraph;

fn insert_split(
    graph: &mut ConstrGraph,
    source_coeff: &Rational,
    term_idx: usize,
    split: &Split,
) -> Result<(), GraphError>;

fn normalize_edge_contribution(
    source_coeff: &Rational,
    split: &Split,
) -> (Term, Term, Rational);

fn ensure_node(nodes: &mut Vec<Term>, term: Term) -> usize;

fn merge_or_push_edge(
    edges: &mut Vec<GraphEdge>,
    left_id: usize,
    right_id: usize,
    term_idx: usize,
    coeff: Rational,
) -> Result<(), GraphError>;

fn finalize_graphs(
    graphs: HashMap<SplitInterface, ConstrGraph>,
) -> Vec<ConstrGraph>;
```

Helper responsibilities:

- `validate_splits_by_term` checks alignment and the 64-term limit
- `empty_graph` creates a graph with empty node and edge lists
- `insert_split` normalizes a contribution, interns nodes, and merges the edge
- `normalize_edge_contribution` absorbs side coefficients into the edge and
  returns unit-coefficient node terms
- `ensure_node` linearly scans one node list and returns an existing or new ID
- `merge_or_push_edge` linearly scans edges and applies merge rules
- `finalize_graphs` removes zero edges, drops graphs with fewer than two edges,
  and returns the remaining graphs without sorting

The linear node and edge scans are intentional for the first implementation.
They keep the graph module simple and avoid sidecar lookup structs. If graph
building becomes hot later, maps can be introduced behind the same public API.

## Relationship To Adjacent Modules

`split` chooses normalized public left/right sides using external-index
membership.

`canon` takes each raw split and returns one canonical split for each owner
orientation.

`rewrite` unpacks the `(left_owner, right_owner)` results into two
`Vec<Vec<Split>>` streams and calls:

```rust
build_graphs_from_splits(def, &left_owner_splits_by_term)?;
build_graphs_from_splits(def, &right_owner_splits_by_term)?;
```

`biclique` consumes each returned `ConstrGraph`.

## Testing

Initial tests should cover:

- alignment mismatch returns `SplitTermAlignmentMismatch`
- definitions with more than 64 terms return `TooManyTerms`
- graph bucketing by equal `SplitInterface`
- different interfaces produce different graphs
- left and right nodes are interned independently
- identical left/right node values on opposite sides get separate IDs
- edge coefficients include source term and side term coefficients
- repeated source-term contribution to the same edge is ignored
- distinct source terms contributing to the same edge are summed
- zero-sum edges are removed
- graphs with fewer than two remaining edges are omitted
- output graph order is not asserted

## Acceptance Criteria

The `graph` module is complete when:

- `build_graphs_from_splits` consumes `&[Vec<Split>]`
- graph buckets are keyed by explicit `SplitInterface`
- no owner-orientation logic exists in the module
- no interface data is recomputed from factor overlap
- no `PendingGraph` or equivalent sidecar struct is introduced
- node and edge lookup is linear in the first implementation
- graph construction returns `Result<Vec<ConstrGraph>, GraphError>`
- output graph order is unspecified
