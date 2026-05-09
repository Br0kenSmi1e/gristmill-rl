# graph Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `graph` module that consumes already-canonicalized split streams and emits construction graphs grouped by explicit `SplitInterface`.

**Architecture:** Add a focused `src/graph.rs` module exposed from `src/lib.rs`. The module trusts `SplitInterface` as authoritative, interns left and right nodes independently, absorbs source and side coefficients into graph edges, deduplicates duplicate derivations from the same source term by provenance, and omits zero or single-edge graphs during finalization.

**Tech Stack:** Rust 2024, existing `repr` and `split` modules, `num::rational::Ratio` through `repr::Rational`, standard library `HashMap` and `Vec`.

---

## File Structure

- Create `src/graph.rs`: public `GraphEdge`, `ConstrGraph`, `GraphError`, `build_graphs_from_splits`, and private graph-building helpers.
- Modify `src/lib.rs`: expose `pub mod graph;`.
- Create `tests/graph.rs`: integration tests for public graph behavior only.

---

### Task 1: Add Public API And Validation

**Files:**
- Modify: `src/lib.rs`
- Create: `src/graph.rs`
- Create: `tests/graph.rs`

- [ ] **Step 1: Write the failing tests**

Create `tests/graph.rs`:

```rust
use gristmill_symbolics::graph::{GraphError, build_graphs_from_splits};
use gristmill_symbolics::repr::{Factor, Index, IndexId, RangeId, Rational, TensorDef, TensorId, Term};
use gristmill_symbolics::split::{Split, SplitInterface};

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

fn term(coeff: Rational, factors: Vec<Factor>) -> Term {
    Term {
        coeff,
        sum_indices: vec![],
        factors,
    }
}

fn empty_def_with_terms(len: usize) -> TensorDef {
    TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: (0..len)
            .map(|_| Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            })
            .collect(),
    }
}

#[test]
fn split_term_alignment_mismatch_returns_graph_error() {
    let def = empty_def_with_terms(2);

    assert_eq!(
        build_graphs_from_splits(&def, &[vec![]]),
        Err(GraphError::SplitTermAlignmentMismatch {
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn more_than_64_terms_returns_graph_error() {
    let def = empty_def_with_terms(65);
    let splits_by_term = vec![vec![]; 65];

    assert_eq!(
        build_graphs_from_splits(&def, &splits_by_term),
        Err(GraphError::TooManyTerms { len: 65, max: 64 })
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test graph -- --nocapture`

Expected: compile failure because `gristmill_symbolics::graph` does not exist.

- [ ] **Step 3: Add the public API and validation-only implementation**

Modify `src/lib.rs`:

```rust
pub mod canon;
pub mod graph;
pub mod repr;
pub mod split;
```

Create `src/graph.rs`:

```rust
use crate::repr::{Rational, TensorDef, Term};
use crate::split::{Split, SplitInterface};
use std::collections::HashMap;

const MAX_TERMS: usize = 64;

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
) -> Result<Vec<ConstrGraph>, GraphError> {
    validate_splits_by_term(def, splits_by_term)?;
    Ok(vec![])
}

fn validate_splits_by_term(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<(), GraphError> {
    if splits_by_term.len() != def.terms.len() {
        return Err(GraphError::SplitTermAlignmentMismatch {
            expected: def.terms.len(),
            got: splits_by_term.len(),
        });
    }

    if def.terms.len() > MAX_TERMS {
        return Err(GraphError::TooManyTerms {
            len: def.terms.len(),
            max: MAX_TERMS,
        });
    }

    Ok(())
}

fn empty_graph(interface: SplitInterface) -> ConstrGraph {
    ConstrGraph {
        interface,
        left_nodes: vec![],
        right_nodes: vec![],
        edges: vec![],
    }
}

fn finalize_graphs(_graphs: HashMap<SplitInterface, ConstrGraph>) -> Vec<ConstrGraph> {
    vec![]
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test graph -- --nocapture`

Expected: PASS for the two validation tests. Warnings about unused helper functions are acceptable at this checkpoint.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/graph.rs tests/graph.rs
git commit -m "feat: add graph module API"
```

---

### Task 2: Build Interface Buckets, Intern Nodes, And Absorb Coefficients

**Files:**
- Modify: `src/graph.rs`
- Modify: `tests/graph.rs`

- [ ] **Step 1: Add the failing tests**

Append to `tests/graph.rs`:

```rust
use gristmill_symbolics::graph::{ConstrGraph, GraphEdge};

fn split(left: Term, right: Term, interface: SplitInterface) -> Split {
    Split {
        left,
        right,
        interface,
    }
}

fn iface(left_external: Vec<Index>, right_external: Vec<Index>, contracted: Vec<Index>) -> SplitInterface {
    SplitInterface {
        left_external,
        right_external,
        contracted,
    }
}

#[test]
fn builds_graph_bucket_with_independent_left_and_right_nodes() {
    let a = idx(0, 0);
    let b = idx(1, 0);
    let k = idx(2, 0);
    let interface = iface(vec![a], vec![b], vec![k]);
    let shared_term_value = term(one(), vec![factor(7, &[0, 2])]);
    let source_def = TensorDef {
        base: TensorId(9),
        ext_indices: vec![a, b],
        terms: vec![
            Term {
                coeff: Rational::new(3, 1),
                sum_indices: vec![k],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(5, 1),
                sum_indices: vec![k],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(
            shared_term_value.clone(),
            shared_term_value.clone(),
            interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(8, &[2, 1])]),
            term(one(), vec![factor(10, &[0])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&source_def, &splits_by_term).unwrap();

    assert_eq!(
        graphs,
        vec![ConstrGraph {
            interface,
            left_nodes: vec![
                shared_term_value.clone(),
                term(one(), vec![factor(8, &[2, 1])]),
            ],
            right_nodes: vec![
                shared_term_value,
                term(one(), vec![factor(10, &[0])]),
            ],
            edges: vec![
                GraphEdge {
                    left_id: 0,
                    right_id: 0,
                    coeff: Rational::new(3, 1),
                    terms_used: 1,
                },
                GraphEdge {
                    left_id: 1,
                    right_id: 1,
                    coeff: Rational::new(5, 1),
                    terms_used: 2,
                },
            ],
        }]
    );
}

#[test]
fn edge_coefficients_absorb_source_and_side_coefficients_and_nodes_are_unit_terms() {
    let interface = iface(vec![], vec![], vec![]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(6, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(
            term(Rational::new(2, 1), vec![factor(0, &[0])]),
            term(Rational::new(-3, 1), vec![factor(1, &[1])]),
            interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();
    let graph = &graphs[0];

    assert_eq!(graph.left_nodes[0].coeff, one());
    assert_eq!(graph.right_nodes[0].coeff, one());
    assert_eq!(graph.edges[0].coeff, Rational::new(-36, 1));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test graph -- --nocapture`

Expected: FAIL because `build_graphs_from_splits` still returns no graphs.

- [ ] **Step 3: Implement graph insertion, node interning, and coefficient normalization without merge logic**

Replace `src/graph.rs` with:

```rust
use crate::repr::{Rational, TensorDef, Term};
use crate::split::{Split, SplitInterface};
use std::collections::HashMap;

const MAX_TERMS: usize = 64;

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

fn validate_splits_by_term(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<(), GraphError> {
    if splits_by_term.len() != def.terms.len() {
        return Err(GraphError::SplitTermAlignmentMismatch {
            expected: def.terms.len(),
            got: splits_by_term.len(),
        });
    }

    if def.terms.len() > MAX_TERMS {
        return Err(GraphError::TooManyTerms {
            len: def.terms.len(),
            max: MAX_TERMS,
        });
    }

    Ok(())
}

fn empty_graph(interface: SplitInterface) -> ConstrGraph {
    ConstrGraph {
        interface,
        left_nodes: vec![],
        right_nodes: vec![],
        edges: vec![],
    }
}

fn insert_split(
    graph: &mut ConstrGraph,
    source_coeff: &Rational,
    term_idx: usize,
    split: &Split,
) -> Result<(), GraphError> {
    let (left, right, coeff) = normalize_edge_contribution(source_coeff, split);
    let left_id = ensure_node(&mut graph.left_nodes, left);
    let right_id = ensure_node(&mut graph.right_nodes, right);
    push_edge(&mut graph.edges, left_id, right_id, term_idx, coeff)
}

fn normalize_edge_contribution(source_coeff: &Rational, split: &Split) -> (Term, Term, Rational) {
    let mut left = split.left.clone();
    let mut right = split.right.clone();
    let coeff = source_coeff * &left.coeff * &right.coeff;
    left.coeff = Rational::new(1, 1);
    right.coeff = Rational::new(1, 1);
    (left, right, coeff)
}

fn ensure_node(nodes: &mut Vec<Term>, term: Term) -> usize {
    if let Some(index) = nodes.iter().position(|node| node == &term) {
        index
    } else {
        let index = nodes.len();
        nodes.push(term);
        index
    }
}

fn push_edge(
    edges: &mut Vec<GraphEdge>,
    left_id: usize,
    right_id: usize,
    term_idx: usize,
    coeff: Rational,
) -> Result<(), GraphError> {
    let term_bit = 1_u64 << term_idx;

    edges.push(GraphEdge {
        left_id,
        right_id,
        coeff,
        terms_used: term_bit,
    });
    Ok(())
}

fn finalize_graphs(graphs: HashMap<SplitInterface, ConstrGraph>) -> Vec<ConstrGraph> {
    graphs
        .into_values()
        .filter_map(|mut graph| {
            graph
                .edges
                .retain(|edge| edge.coeff != Rational::new(0, 1));
            (graph.edges.len() >= 2).then_some(graph)
        })
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test graph -- --nocapture`

Expected: PASS for all graph tests added so far.

- [ ] **Step 5: Commit**

```bash
git add src/graph.rs tests/graph.rs
git commit -m "feat: build construction graphs from splits"
```

---

### Task 3: Add Edge Merging And Duplicate Provenance

**Files:**
- Modify: `src/graph.rs`
- Modify: `tests/graph.rs`

- [ ] **Step 1: Add tests for multiple interfaces and merge behavior**

Append to `tests/graph.rs`:

```rust
fn graph_by_interface<'a>(
    graphs: &'a [ConstrGraph],
    interface: &SplitInterface,
) -> &'a ConstrGraph {
    graphs
        .iter()
        .find(|graph| &graph.interface == interface)
        .expect("graph with interface should exist")
}

#[test]
fn equal_interfaces_share_a_bucket_and_different_interfaces_create_separate_graphs() {
    let a = idx(0, 0);
    let b = idx(1, 0);
    let first_interface = iface(vec![a], vec![b], vec![]);
    let second_interface = iface(vec![b], vec![a], vec![]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![a, b],
        terms: vec![
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(
            term(one(), vec![factor(0, &[0])]),
            term(one(), vec![factor(1, &[1])]),
            first_interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(2, &[0])]),
            term(one(), vec![factor(3, &[1])]),
            first_interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(4, &[1])]),
            term(one(), vec![factor(5, &[0])]),
            second_interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(6, &[1])]),
            term(one(), vec![factor(7, &[0])]),
            second_interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();

    assert_eq!(graphs.len(), 2);
    assert_eq!(graph_by_interface(&graphs, &first_interface).edges.len(), 2);
    assert_eq!(graph_by_interface(&graphs, &second_interface).edges.len(), 2);
}

#[test]
fn distinct_source_terms_contributing_to_same_edge_are_summed() {
    let interface = iface(vec![], vec![], vec![]);
    let left = term(one(), vec![factor(0, &[0])]);
    let right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(5, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(left.clone(), right.clone(), interface.clone())],
        vec![split(left.clone(), right.clone(), interface.clone())],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();
    let merged_edge = &graphs[0].edges[0];

    assert_eq!(merged_edge.coeff, Rational::new(7, 1));
    assert_eq!(merged_edge.terms_used, 0b011);
}

#[test]
fn repeated_source_term_contribution_to_same_edge_is_ignored() {
    let interface = iface(vec![], vec![], vec![]);
    let left = term(one(), vec![factor(0, &[0])]);
    let right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![
            split(left.clone(), right.clone(), interface.clone()),
            split(left.clone(), right.clone(), interface.clone()),
        ],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();

    assert_eq!(graphs[0].edges.len(), 2);
    assert_eq!(graphs[0].edges[0].coeff, Rational::new(2, 1));
    assert_eq!(graphs[0].edges[0].terms_used, 0b001);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test graph -- --nocapture`

Expected: FAIL because distinct source-term contributions to the same `(left_id, right_id)` edge are still emitted as separate edges.

- [ ] **Step 3: Replace push-only edge insertion with merge-or-push behavior**

In `src/graph.rs`, change `insert_split` to call `merge_or_push_edge`:

```rust
fn insert_split(
    graph: &mut ConstrGraph,
    source_coeff: &Rational,
    term_idx: usize,
    split: &Split,
) -> Result<(), GraphError> {
    let (left, right, coeff) = normalize_edge_contribution(source_coeff, split);
    let left_id = ensure_node(&mut graph.left_nodes, left);
    let right_id = ensure_node(&mut graph.right_nodes, right);
    merge_or_push_edge(&mut graph.edges, left_id, right_id, term_idx, coeff)
}
```

Replace the `push_edge` helper with:

```rust
fn merge_or_push_edge(
    edges: &mut Vec<GraphEdge>,
    left_id: usize,
    right_id: usize,
    term_idx: usize,
    coeff: Rational,
) -> Result<(), GraphError> {
    let term_bit = 1_u64 << term_idx;

    if let Some(edge) = edges
        .iter_mut()
        .find(|edge| edge.left_id == left_id && edge.right_id == right_id)
    {
        if edge.terms_used & term_bit == 0 {
            edge.coeff += coeff;
            edge.terms_used |= term_bit;
        }
        return Ok(());
    }

    edges.push(GraphEdge {
        left_id,
        right_id,
        coeff,
        terms_used: term_bit,
    });
    Ok(())
}
```

- [ ] **Step 4: Run graph tests**

Run: `cargo test --test graph -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Run the full crate test suite**

Run: `cargo test`

Expected: PASS for existing `repr`, `split`, `canon`, and new `graph` tests.

- [ ] **Step 6: Commit**

```bash
git add src/graph.rs tests/graph.rs
git commit -m "test: cover graph edge merging"
```

---

### Task 4: Cover Finalization And Output-Order Independence

**Files:**
- Modify: `tests/graph.rs`

- [ ] **Step 1: Add finalization tests**

Append to `tests/graph.rs`:

```rust
#[test]
fn zero_sum_edges_are_removed_and_single_edge_graphs_are_omitted() {
    let interface = iface(vec![], vec![], vec![]);
    let left = term(one(), vec![factor(0, &[0])]);
    let right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(-2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(9, 1),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(left.clone(), right.clone(), interface.clone())],
        vec![split(left, right, interface.clone())],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    assert_eq!(build_graphs_from_splits(&def, &splits_by_term).unwrap(), vec![]);
}

#[test]
fn graph_with_two_remaining_edges_survives_after_zero_edge_removal() {
    let interface = iface(vec![], vec![], vec![]);
    let cancel_left = term(one(), vec![factor(0, &[0])]);
    let cancel_right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(-2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(3, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(4, 1),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(cancel_left.clone(), cancel_right.clone(), interface.clone())],
        vec![split(cancel_left, cancel_right, interface.clone())],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(4, &[4])]),
            term(one(), vec![factor(5, &[5])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();

    assert_eq!(graphs.len(), 1);
    assert_eq!(graphs[0].edges.len(), 2);
    assert!(graphs[0].edges.iter().all(|edge| edge.coeff != Rational::new(0, 1)));
}
```

- [ ] **Step 2: Run graph tests**

Run: `cargo test --test graph -- --nocapture`

Expected: PASS.

- [ ] **Step 3: Run formatting and full tests**

Run: `cargo fmt -- --check`

Expected: PASS.

Run: `cargo test`

Expected: PASS.

- [ ] **Step 4: Boundary check**

Run: `rg -n "canon|enumerate_splits|TensorComputation|biclique|rewrite|left_owner|right_owner|overlap" src/graph.rs tests/graph.rs`

Expected: no matches in `src/graph.rs`. Matches in `tests/graph.rs` are acceptable only if they occur inside helper names or comments added by the test author; the first implementation should not add such comments.

- [ ] **Step 5: Commit**

```bash
git add src/graph.rs tests/graph.rs
git commit -m "test: cover graph finalization"
```

---

## Self-Review Checklist

- [ ] `GraphEdge`, `ConstrGraph`, `GraphError`, and `build_graphs_from_splits` match the graph design spec public API.
- [ ] `build_graphs_from_splits` consumes exactly one `&[Vec<Split>]` stream and has no owner-orientation logic.
- [ ] Graph buckets are keyed by `SplitInterface` and no interface data is reconstructed from factor overlap.
- [ ] Left and right node interning is independent, even when the same `Term` value appears on both sides.
- [ ] Edge coefficients include source term coefficients and side term coefficients, while graph node terms are unit coefficient.
- [ ] Duplicate same-source-term derivations to the same edge are ignored by provenance.
- [ ] Contributions from distinct source terms to the same edge are summed and provenance bits are OR-ed.
- [ ] Zero coefficient edges are removed and graphs with fewer than two remaining edges are omitted.
- [ ] Tests do not assert the ordering of multiple output graphs.
- [ ] The implementation uses only linear node and edge scans in the first pass.
- [ ] `cargo fmt -- --check` and `cargo test` pass.
