# biclique Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `biclique` module that enumerates legal inclusion-maximal bicliques from finalized construction graphs.

**Architecture:** Add a focused `src/biclique.rs` module exposed from `src/lib.rs`. The module consumes `graph::ConstrGraph`, preserves `rustymill` recursive search behavior exactly, enforces coefficient factorization and provenance disjointness, and deliberately omits `canonicalize_biclique` and output sorting.

**Tech Stack:** Rust 2024, existing `graph` and `repr` modules, `repr::Rational` backed by `num::rational::Ratio<i64>`, standard library `HashMap` and `Vec`.

---

## File Structure

- Create `src/biclique.rs`: public `Biclique`, public `enumerate_bicliques`, and private search helpers.
- Modify `src/lib.rs`: expose `pub mod biclique;`.
- Create `tests/biclique.rs`: integration tests for public biclique behavior using hand-built finalized `ConstrGraph` values.

---

### Task 1: Public API And Empty Cases

**Files:**
- Create: `src/biclique.rs`
- Modify: `src/lib.rs`
- Create: `tests/biclique.rs`

- [ ] **Step 1: Write the failing API and empty-case tests**

Create `tests/biclique.rs`:

```rust
use gristmill_symbolics::biclique::{Biclique, enumerate_bicliques};
use gristmill_symbolics::graph::{ConstrGraph, GraphEdge};
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorId, Term,
};
use gristmill_symbolics::split::SplitInterface;

fn rat(num: i64, den: i64) -> Rational {
    Rational::new(num, den)
}

fn factor(tensor: u32, indices: &[u32]) -> Factor {
    Factor {
        tensor: TensorId(tensor),
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn index(id: u32, range: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(range),
    }
}

fn term(coeff_num: i64, coeff_den: i64, sum_indices: &[Index], factors: Vec<Factor>) -> Term {
    Term {
        coeff: rat(coeff_num, coeff_den),
        sum_indices: sum_indices.to_vec(),
        factors,
    }
}

fn base_interface() -> SplitInterface {
    SplitInterface {
        left_external: vec![index(0, 0)],
        right_external: vec![index(1, 0)],
        contracted: vec![index(2, 0)],
    }
}

fn graph(
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: &[(usize, usize, Rational, u64)],
) -> ConstrGraph {
    ConstrGraph {
        interface: base_interface(),
        left_nodes,
        right_nodes,
        edges: edges
            .iter()
            .map(|(left_id, right_id, coeff, terms_used)| GraphEdge {
                left_id: *left_id,
                right_id: *right_id,
                coeff: coeff.clone(),
                terms_used: *terms_used,
            })
            .collect(),
    }
}

fn graph_i64(
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: &[(usize, usize, i64, u64)],
) -> ConstrGraph {
    ConstrGraph {
        interface: base_interface(),
        left_nodes,
        right_nodes,
        edges: edges
            .iter()
            .map(|(left_id, right_id, coeff, terms_used)| GraphEdge {
                left_id: *left_id,
                right_id: *right_id,
                coeff: rat(*coeff, 1),
                terms_used: *terms_used,
            })
            .collect(),
    }
}

fn sample_left_nodes() -> Vec<Term> {
    vec![
        term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]),
        term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]),
        term(1, 1, &[index(2, 0)], vec![factor(6, &[0, 2])]),
    ]
}

fn sample_right_nodes() -> Vec<Term> {
    vec![
        term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]),
        term(1, 1, &[index(2, 0)], vec![factor(4, &[2, 1])]),
        term(1, 1, &[index(2, 0)], vec![factor(5, &[2, 1])]),
    ]
}

fn find_biclique<'a>(
    bicliques: &'a [Biclique],
    left_ids: &[usize],
    right_ids: &[usize],
) -> &'a Biclique {
    bicliques
        .iter()
        .find(|biclique| {
            biclique.left_node_ids == left_ids && biclique.right_node_ids == right_ids
        })
        .expect("expected biclique was not returned")
}

#[test]
fn crate_surface_exposes_biclique_enumerator_api() {
    let enumerate_fn: fn(&ConstrGraph) -> Vec<Biclique> = enumerate_bicliques;

    let biclique = Biclique {
        left_node_ids: vec![0],
        right_node_ids: vec![0],
        left_coeffs: vec![rat(1, 1)],
        right_coeffs: vec![rat(2, 1)],
        terms_used: 0b1,
    };

    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert_eq!(biclique.terms_used, 0b1);
    assert!(enumerate_fn(&graph).is_empty());
}

#[test]
fn graphs_with_fewer_than_two_edges_produce_no_bicliques() {
    let empty_graph = graph_i64(sample_left_nodes(), sample_right_nodes(), &[]);
    let one_edge_graph = graph_i64(
        sample_left_nodes()[0..1].to_vec(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert!(enumerate_bicliques(&empty_graph).is_empty());
    assert!(enumerate_bicliques(&one_edge_graph).is_empty());
}

#[test]
fn one_edge_bicliques_are_not_emitted() {
    let graph = graph_i64(
        sample_left_nodes()[0..1].to_vec(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert!(enumerate_bicliques(&graph).is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test biclique -- --nocapture`

Expected: compile failure because `gristmill_symbolics::biclique` does not exist.

- [ ] **Step 3: Add the public module and minimal implementation**

Modify `src/lib.rs`:

```rust
pub mod biclique;
pub mod canon;
pub mod graph;
pub mod repr;
pub mod split;
```

Create `src/biclique.rs`:

```rust
use crate::graph::ConstrGraph;
use crate::repr::Rational;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Biclique {
    pub left_node_ids: Vec<usize>,
    pub right_node_ids: Vec<usize>,
    pub left_coeffs: Vec<Rational>,
    pub right_coeffs: Vec<Rational>,
    pub terms_used: u64,
}

pub fn enumerate_bicliques(graph: &ConstrGraph) -> Vec<Biclique> {
    if graph.edges.len() < 2 {
        return Vec::new();
    }

    Vec::new()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test biclique -- --nocapture`

Expected: all 3 biclique tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/biclique.rs tests/biclique.rs
git commit -m "Add biclique module surface"
```

---

### Task 2: Recursive Search And Sharing Bicliques

**Files:**
- Modify: `src/biclique.rs`
- Modify: `tests/biclique.rs`

- [ ] **Step 1: Add failing tests for `2x1` and `1x2` sharing**

Append to `tests/biclique.rs`:

```rust
#[test]
fn enumerate_bicliques_bootstraps_to_a_2x1_biclique() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b001), (1, 0, 6, 0b010)],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1)]);
    assert_eq!(biclique.terms_used, 0b011);
}

#[test]
fn enumerate_bicliques_bootstraps_to_a_1x2_biclique() {
    let graph = graph_i64(
        sample_left_nodes()[0..1].to_vec(),
        sample_right_nodes(),
        &[(0, 0, 2, 0b001), (0, 1, 4, 0b010)],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1), rat(4, 1)]);
    assert_eq!(biclique.terms_used, 0b011);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test biclique -- --nocapture`

Expected: the two new sharing tests fail because `enumerate_bicliques` still returns no bicliques for two-edge graphs.

- [ ] **Step 3: Replace `src/biclique.rs` with the recursive search skeleton**

Use this implementation. It ports the `rustymill` recursion shape and bootstrap behavior, but leaves strict coefficient equality for Task 3 and exact pivot pruning for Task 4.

```rust
use crate::graph::{ConstrGraph, GraphEdge};
use crate::repr::Rational;
use std::collections::HashMap;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Biclique {
    pub left_node_ids: Vec<usize>,
    pub right_node_ids: Vec<usize>,
    pub left_coeffs: Vec<Rational>,
    pub right_coeffs: Vec<Rational>,
    pub terms_used: u64,
}

pub fn enumerate_bicliques(graph: &ConstrGraph) -> Vec<Biclique> {
    if graph.edges.len() < 2 {
        return Vec::new();
    }

    let mut biclique = empty_biclique();
    let mut candidates = all_candidates(graph);
    let mut out = Vec::new();
    let frontier = initial_frontier(graph);

    expand(graph, &mut biclique, &frontier, &mut candidates, &mut out);
    out
}

fn all_candidates(graph: &ConstrGraph) -> Vec<SearchNode> {
    (0..graph.left_nodes.len())
        .map(SearchNode::Left)
        .chain((0..graph.right_nodes.len()).map(SearchNode::Right))
        .collect()
}

fn initial_frontier(graph: &ConstrGraph) -> HashMap<SearchNode, Delta> {
    all_candidates(graph)
        .into_iter()
        .map(|node| {
            (
                node,
                Delta {
                    coeff: Rational::new(1, 1),
                    terms: 0,
                },
            )
        })
        .collect()
}

fn empty_biclique() -> Biclique {
    Biclique {
        left_node_ids: vec![],
        right_node_ids: vec![],
        left_coeffs: vec![],
        right_coeffs: vec![],
        terms_used: 0,
    }
}

fn edge_between(graph: &ConstrGraph, left_id: usize, right_id: usize) -> Option<&GraphEdge> {
    graph
        .edges
        .iter()
        .find(|edge| edge.left_id == left_id && edge.right_id == right_id)
}

fn expand(
    graph: &ConstrGraph,
    biclique: &mut Biclique,
    frontier: &HashMap<SearchNode, Delta>,
    candidates: &mut Vec<SearchNode>,
    out: &mut Vec<Biclique>,
) {
    if has_sharing(biclique) && frontier.is_empty() {
        out.push(biclique.clone());
        return;
    }

    let child_frontiers = build_child_frontiers(graph, biclique, frontier);
    let current = sift(biclique, candidates, frontier, &child_frontiers);

    for node in current {
        let Some(delta) = frontier.get(&node) else {
            continue;
        };
        let Some(position) = candidates.iter().position(|candidate| *candidate == node) else {
            continue;
        };

        let removed = candidates.remove(position);
        let child_frontier = child_frontiers.get(&removed).cloned().unwrap_or_default();
        let mut child_candidates: Vec<SearchNode> = candidates
            .iter()
            .copied()
            .filter(|candidate| child_frontier.contains_key(candidate))
            .collect();

        push(biclique, removed, delta);
        expand(graph, biclique, &child_frontier, &mut child_candidates, out);
        pop(biclique, removed, delta);
    }
}

fn sift(
    biclique: &Biclique,
    candidates: &[SearchNode],
    frontier: &HashMap<SearchNode, Delta>,
    _child_frontiers: &HashMap<SearchNode, HashMap<SearchNode, Delta>>,
) -> Vec<SearchNode> {
    if biclique.left_node_ids.is_empty() && biclique.right_node_ids.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Left(_)))
            .copied()
            .collect();
    }

    if biclique.left_node_ids.len() == 1 && biclique.right_node_ids.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Right(_)))
            .filter(|node| matches!(frontier.get(node), Some(delta) if delta.terms != 0))
            .copied()
            .collect();
    }

    candidates.to_vec()
}

fn build_child_frontiers(
    graph: &ConstrGraph,
    biclique: &Biclique,
    frontier: &HashMap<SearchNode, Delta>,
) -> HashMap<SearchNode, HashMap<SearchNode, Delta>> {
    let mut out = HashMap::new();

    for (chosen, chosen_delta) in frontier {
        let mut child = HashMap::new();
        for (candidate, candidate_delta) in frontier {
            if chosen == candidate {
                continue;
            }

            if let Some(updated) = update_delta(
                graph,
                biclique,
                *chosen,
                chosen_delta,
                *candidate,
                candidate_delta,
            ) {
                child.insert(*candidate, updated);
            }
        }
        out.insert(*chosen, child);
    }

    out
}

fn update_delta(
    graph: &ConstrGraph,
    biclique: &Biclique,
    chosen: SearchNode,
    chosen_delta: &Delta,
    candidate: SearchNode,
    candidate_delta: &Delta,
) -> Option<Delta> {
    if matches!(
        (chosen, candidate),
        (SearchNode::Left(_), SearchNode::Left(_))
            | (SearchNode::Right(_), SearchNode::Right(_))
    ) {
        if chosen_delta.terms & candidate_delta.terms != 0 {
            return None;
        }
        return Some(candidate_delta.clone());
    }

    let (left_id, right_id) = match (chosen, candidate) {
        (SearchNode::Left(left_id), SearchNode::Right(right_id)) => (left_id, right_id),
        (SearchNode::Right(right_id), SearchNode::Left(left_id)) => (left_id, right_id),
        _ => unreachable!(),
    };

    let edge = edge_between(graph, left_id, right_id)?;

    if chosen_delta.terms & candidate_delta.terms != 0
        || biclique.terms_used & edge.terms_used != 0
        || chosen_delta.terms & edge.terms_used != 0
        || candidate_delta.terms & edge.terms_used != 0
    {
        return None;
    }

    let expected = edge.coeff.clone() / chosen_delta.coeff.clone();
    let mut next = candidate_delta.clone();
    if next.terms == 0 {
        next.coeff = expected;
    }
    next.terms |= edge.terms_used;
    Some(next)
}

fn has_sharing(biclique: &Biclique) -> bool {
    biclique.left_node_ids.len() >= 2 || biclique.right_node_ids.len() >= 2
}

fn push(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    biclique.terms_used |= delta.terms;
    let coeff = delta.coeff.clone();

    match node {
        SearchNode::Left(id) => {
            biclique.left_node_ids.push(id);
            biclique.left_coeffs.push(coeff);
        }
        SearchNode::Right(id) => {
            biclique.right_node_ids.push(id);
            biclique.right_coeffs.push(coeff);
        }
    }
}

fn pop(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    biclique.terms_used ^= delta.terms;

    match node {
        SearchNode::Left(_) => {
            biclique.left_node_ids.pop();
            biclique.left_coeffs.pop();
        }
        SearchNode::Right(_) => {
            biclique.right_node_ids.pop();
            biclique.right_coeffs.pop();
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test biclique -- --nocapture`

Expected: all 5 biclique tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/biclique.rs tests/biclique.rs
git commit -m "Add biclique recursive search"
```

---

### Task 3: Coefficient And Provenance Legality

**Files:**
- Modify: `src/biclique.rs`
- Modify: `tests/biclique.rs`

- [ ] **Step 1: Add failing tests for factorization and provenance rules**

Append to `tests/biclique.rs`:

```rust
#[test]
fn enumerate_bicliques_finds_factorizable_2x2_rectangle() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 2, 0b0001),
            (0, 1, 4, 0b0010),
            (1, 0, 6, 0b0100),
            (1, 1, 12, 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1), rat(4, 1)]);
    assert_eq!(biclique.terms_used, 0b1111);
}

#[test]
fn enumerate_bicliques_rejects_non_factorizable_2x2_rectangle() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 2, 0b0001),
            (0, 1, 4, 0b0010),
            (1, 0, 6, 0b0100),
            (1, 1, 11, 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);

    assert!(
        bicliques
            .iter()
            .all(|biclique| biclique.left_node_ids != [0, 1]
                || biclique.right_node_ids != [0, 1])
    );
}

#[test]
fn enumerate_bicliques_rejects_overlapping_provenance() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b001), (1, 0, 6, 0b001)],
    );

    assert!(enumerate_bicliques(&graph).is_empty());
}

#[test]
fn enumerate_bicliques_supports_negative_rational_coefficients() {
    let graph = graph(
        sample_left_nodes()[0..2].to_vec(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, rat(-1, 2), 0b0001),
            (0, 1, rat(3, 4), 0b0010),
            (1, 0, rat(-1, 1), 0b0100),
            (1, 1, rat(3, 2), 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(2, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(-1, 2), rat(3, 4)]);
    assert_eq!(biclique.terms_used, 0b1111);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test biclique -- --nocapture`

Expected: `enumerate_bicliques_rejects_non_factorizable_2x2_rectangle` fails because the Task 2 implementation does not reject already-assigned coefficient mismatches.

- [ ] **Step 3: Replace `update_delta` with the exact `rustymill` legality check**

Replace the entire `update_delta` function in `src/biclique.rs` with:

```rust
fn update_delta(
    graph: &ConstrGraph,
    biclique: &Biclique,
    chosen: SearchNode,
    chosen_delta: &Delta,
    candidate: SearchNode,
    candidate_delta: &Delta,
) -> Option<Delta> {
    if matches!(
        (chosen, candidate),
        (SearchNode::Left(_), SearchNode::Left(_))
            | (SearchNode::Right(_), SearchNode::Right(_))
    ) {
        if chosen_delta.terms & candidate_delta.terms != 0 {
            return None;
        }
        return Some(candidate_delta.clone());
    }

    let (left_id, right_id) = match (chosen, candidate) {
        (SearchNode::Left(left_id), SearchNode::Right(right_id)) => (left_id, right_id),
        (SearchNode::Right(right_id), SearchNode::Left(left_id)) => (left_id, right_id),
        _ => unreachable!(),
    };

    let edge = edge_between(graph, left_id, right_id)?;

    if chosen_delta.terms & candidate_delta.terms != 0
        || biclique.terms_used & edge.terms_used != 0
        || chosen_delta.terms & edge.terms_used != 0
        || candidate_delta.terms & edge.terms_used != 0
    {
        return None;
    }

    let expected = edge.coeff.clone() / chosen_delta.coeff.clone();
    let mut next = candidate_delta.clone();
    if candidate_delta.terms == 0 {
        next.coeff = expected;
    } else if candidate_delta.coeff != expected {
        return None;
    }
    next.terms |= edge.terms_used;
    Some(next)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test biclique -- --nocapture`

Expected: all 9 biclique tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/biclique.rs tests/biclique.rs
git commit -m "Enforce biclique coefficient legality"
```

---

### Task 4: Exact Maximality Pruning And No Sorting

**Files:**
- Modify: `src/biclique.rs`
- Modify: `tests/biclique.rs`

- [ ] **Step 1: Add failing tests for maximality and `rustymill` pivot behavior**

Append to `tests/biclique.rs`:

```rust
#[test]
fn enumerate_bicliques_emits_only_the_maximal_2x3_rectangle_once() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes(),
        &[
            (0, 0, 2, 0b000001),
            (0, 1, 4, 0b000010),
            (0, 2, 6, 0b000100),
            (1, 0, 6, 0b001000),
            (1, 1, 12, 0b010000),
            (1, 2, 18, 0b100000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);

    assert_eq!(bicliques.len(), 1);

    let biclique = &bicliques[0];
    assert_eq!(biclique.left_node_ids, vec![0, 1]);
    assert_eq!(biclique.right_node_ids, vec![0, 1, 2]);
    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(
        biclique.right_coeffs,
        vec![rat(2, 1), rat(4, 1), rat(6, 1)]
    );
    assert_eq!(biclique.terms_used, 0b111111);
}

#[test]
fn enumerate_bicliques_ignores_non_current_left_pivots_after_bootstrap() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 10, 0b001000),
            (0, 1, 20, 0b010000),
            (1, 0, 2, 0b000001),
            (1, 1, 4, 0b000010),
            (2, 0, 6, 0b000100),
            (2, 1, 12, 0b001000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[1, 2], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1), rat(4, 1)]);
    assert_eq!(biclique.terms_used, 0b001111);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test biclique -- --nocapture`

Expected: the maximality test fails before exact `sift` is ported because Task 2 branches over every remaining candidate after bootstrap.

- [ ] **Step 3: Replace `sift` with the exact `rustymill` pivot behavior**

Replace the entire `sift` function in `src/biclique.rs` with:

```rust
fn sift(
    biclique: &Biclique,
    candidates: &[SearchNode],
    frontier: &HashMap<SearchNode, Delta>,
    child_frontiers: &HashMap<SearchNode, HashMap<SearchNode, Delta>>,
) -> Vec<SearchNode> {
    if biclique.left_node_ids.is_empty() && biclique.right_node_ids.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Left(_)))
            .copied()
            .collect();
    }

    if biclique.left_node_ids.len() == 1 && biclique.right_node_ids.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Right(_)))
            .filter(|node| matches!(frontier.get(node), Some(delta) if delta.terms != 0))
            .copied()
            .collect();
    }

    let current = candidates.to_vec();

    let mut best_forbidden = Vec::new();
    let mut best_score = 0usize;
    for &node in &current {
        let forbidden: Vec<SearchNode> = child_frontiers
            .get(&node)
            .map(|next| next.keys().copied().collect())
            .unwrap_or_default();
        let score = forbidden
            .iter()
            .filter(|candidate| current.contains(candidate))
            .count();
        if score > best_score {
            best_score = score;
            best_forbidden = forbidden;
        }
    }

    current
        .into_iter()
        .filter(|node| !best_forbidden.contains(node))
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test biclique -- --nocapture`

Expected: all 11 biclique tests pass.

- [ ] **Step 5: Verify no sorting helper was introduced**

Run: `rg -n "canonicalize_biclique|sort_by|sort_by_key|\\.sort\\(" src/biclique.rs`

Expected: no output and exit code 1. This confirms the implementation did not restore the old `rustymill` output sorting path.

- [ ] **Step 6: Commit**

```bash
git add src/biclique.rs tests/biclique.rs
git commit -m "Preserve rustymill biclique maximality"
```

---

### Task 5: Full Verification

**Files:**
- Verify: `src/biclique.rs`
- Verify: `tests/biclique.rs`
- Verify: `src/lib.rs`

- [ ] **Step 1: Format the crate**

Run: `cargo fmt`

Expected: command exits 0.

- [ ] **Step 2: Run the biclique test target**

Run: `cargo test --test biclique -- --nocapture`

Expected: all 11 biclique tests pass.

- [ ] **Step 3: Run the full test suite**

Run: `cargo test`

Expected: all crate tests pass.

- [ ] **Step 4: Verify the no-sorting contract**

Run: `rg -n "canonicalize_biclique|sort_by|sort_by_key|\\.sort\\(" src/biclique.rs`

Expected: no output and exit code 1.

- [ ] **Step 5: Check for whitespace errors**

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 6: Commit final verification changes if formatting changed files**

```bash
git status --short
git add src/biclique.rs tests/biclique.rs src/lib.rs
git commit -m "Verify biclique module"
```

Expected: commit only if `cargo fmt` changed tracked files after Task 4.

---

## Self-Review Notes

- Spec coverage: Tasks 1-4 cover public API, finalized graph input behavior, nonempty sharing, coefficient factorization, provenance disjointness, inclusion maximality, left-node bootstrap, `rustymill` pivot behavior, and no output sorting.
- Type consistency: All code uses `graph::ConstrGraph`, `graph::GraphEdge`, and `repr::Rational` from the current crate.
- Scope: The plan stops at graph-local biclique enumeration and does not include rewrite templates, action masks, ranking, graph building, JSON, or mutation.
