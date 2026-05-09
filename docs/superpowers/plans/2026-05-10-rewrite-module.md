# Rewrite Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `rewrite` module that exposes action-space generation, decision validation, rewrite construction, and rewrite application on top of `split`, `canon`, `graph`, and `biclique`.

**Architecture:** Add a thin `src/rewrite.rs` action layer and expose it from `src/lib.rs`. The module keeps visible `Factorization` templates aligned with private `ConstrGraph` and `Biclique` sidecars, trusts `SplitInterface` as the index source of truth, and assumes each `ActionSpace` is used only with the unchanged `TensorComputation` that produced it.

**Tech Stack:** Rust 2024, existing `repr`, `split`, `canon`, `graph`, and `biclique` modules, `repr::Rational` backed by `num::rational::Ratio<i64>`, standard library `Vec` and `HashSet`.

---

## File Structure

- Create `src/rewrite.rs`: public rewrite API, private candidate sidecars, decision validation, candidate orchestration, factorization construction, rewrite application, and module unit tests for private helpers.
- Modify `src/lib.rs`: expose `pub mod rewrite;`.
- Create `tests/rewrite.rs`: public integration tests for action-space generation, error propagation, rewrite construction, and rewrite application.

`src/rewrite.rs` owns no algebraic stage internals. It calls:

- `split::enumerate_splits`
- `canon::build_tensor_symmetry_map`
- `canon::build_index_pool`
- `canon::canon_split`
- `graph::build_graphs_from_splits`
- `biclique::enumerate_bicliques`

Candidate indices are meaningful only within one `ActionSpace`. Do not add sorting for cross-call stability.

---

### Task 1: Add Public API And Decision Validation

**Files:**
- Modify: `src/lib.rs`
- Create: `src/rewrite.rs`

- [ ] **Step 1: Write the failing unit tests**

Modify `src/lib.rs`:

```rust
pub mod biclique;
pub mod canon;
pub mod graph;
pub mod repr;
pub mod rewrite;
pub mod split;
```

Create `src/rewrite.rs` with the validation tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::{Rational, TensorDef, TensorId, Term};

    fn term() -> Term {
        Term {
            coeff: Rational::new(1, 1),
            sum_indices: vec![],
            factors: vec![],
        }
    }

    fn def(base: u32, term_count: usize) -> TensorDef {
        TensorDef {
            base: TensorId(base),
            ext_indices: vec![],
            terms: (0..term_count).map(|_| term()).collect(),
        }
    }

    fn validation_space() -> ActionSpace {
        ActionSpace {
            def_index: 0,
            candidate_templates: vec![Factorization {
                left_definition: def(10, 2),
                right_definition: def(11, 1),
                rewritten_definition: def(0, 1),
            }],
            candidate_graphs: vec![],
            candidate_bicliques: vec![],
        }
    }

    #[test]
    fn validate_decision_rejects_out_of_range_candidate_index() {
        let space = validation_space();
        let decision = Decision {
            candidate_index: 1,
            left_mask: vec![true, true],
            right_mask: vec![true],
        };

        assert_eq!(
            validate_decision(&space, &decision),
            Err(RewriteError::CandidateIndexOutOfRange { index: 1, len: 1 })
        );
    }

    #[test]
    fn validate_decision_rejects_mask_length_mismatches() {
        let space = validation_space();

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![true],
                    right_mask: vec![true],
                },
            ),
            Err(RewriteError::LeftMaskLengthMismatch {
                expected: 2,
                got: 1,
            })
        );

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![true, true],
                    right_mask: vec![true, false],
                },
            ),
            Err(RewriteError::RightMaskLengthMismatch {
                expected: 1,
                got: 2,
            })
        );
    }

    #[test]
    fn validate_decision_rejects_empty_selected_sides() {
        let space = validation_space();

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![false, false],
                    right_mask: vec![true],
                },
            ),
            Err(RewriteError::EmptyLeftMask)
        );

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![true, false],
                    right_mask: vec![false],
                },
            ),
            Err(RewriteError::EmptyRightMask)
        );
    }

    #[test]
    fn validate_decision_accepts_nonempty_masks_with_expected_lengths() {
        let space = validation_space();
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true, false],
            right_mask: vec![true],
        };

        assert_eq!(validate_decision(&space, &decision), Ok(()));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test rewrite::tests::validate_decision -- --nocapture
```

Expected: compile failure because `ActionSpace`, `Factorization`, `Decision`, `RewriteError`, and `validate_decision` are not defined.

- [ ] **Step 3: Add the public API and validation implementation**

Replace the top of `src/rewrite.rs` before the test module with:

```rust
use crate::biclique::Biclique;
use crate::canon::CanonError;
use crate::graph::{ConstrGraph, GraphError};
use crate::repr::{TensorComputation, TensorDef};
use crate::split::SplitError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Factorization {
    pub left_definition: TensorDef,
    pub right_definition: TensorDef,
    pub rewritten_definition: TensorDef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpace {
    pub def_index: usize,
    pub candidate_templates: Vec<Factorization>,
    candidate_graphs: Vec<ConstrGraph>,
    candidate_bicliques: Vec<Biclique>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub candidate_index: usize,
    pub left_mask: Vec<bool>,
    pub right_mask: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizationRewrite {
    pub def_index: usize,
    pub factorization: Factorization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    Split(SplitError),
    Canon(CanonError),
    Graph(GraphError),
    CandidateIndexOutOfRange { index: usize, len: usize },
    LeftMaskLengthMismatch { expected: usize, got: usize },
    RightMaskLengthMismatch { expected: usize, got: usize },
    EmptyLeftMask,
    EmptyRightMask,
    DefinitionIndexOutOfRange { index: usize, len: usize },
}

impl From<SplitError> for RewriteError {
    fn from(error: SplitError) -> Self {
        RewriteError::Split(error)
    }
}

impl From<CanonError> for RewriteError {
    fn from(error: CanonError) -> Self {
        RewriteError::Canon(error)
    }
}

impl From<GraphError> for RewriteError {
    fn from(error: GraphError) -> Self {
        RewriteError::Graph(error)
    }
}

pub fn next_action_space(
    _comp: &TensorComputation,
    _start_from: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    Ok(None)
}

pub fn validate_decision(
    space: &ActionSpace,
    decision: &Decision,
) -> Result<(), RewriteError> {
    let Some(template) = space.candidate_templates.get(decision.candidate_index) else {
        return Err(RewriteError::CandidateIndexOutOfRange {
            index: decision.candidate_index,
            len: space.candidate_templates.len(),
        });
    };

    let expected_left = template.left_definition.terms.len();
    if decision.left_mask.len() != expected_left {
        return Err(RewriteError::LeftMaskLengthMismatch {
            expected: expected_left,
            got: decision.left_mask.len(),
        });
    }

    let expected_right = template.right_definition.terms.len();
    if decision.right_mask.len() != expected_right {
        return Err(RewriteError::RightMaskLengthMismatch {
            expected: expected_right,
            got: decision.right_mask.len(),
        });
    }

    if !decision.left_mask.iter().any(|keep| *keep) {
        return Err(RewriteError::EmptyLeftMask);
    }

    if !decision.right_mask.iter().any(|keep| *keep) {
        return Err(RewriteError::EmptyRightMask);
    }

    Ok(())
}

pub fn build_rewrite(
    _comp: &TensorComputation,
    _space: &ActionSpace,
    _decision: &Decision,
) -> Result<FactorizationRewrite, RewriteError> {
    Err(RewriteError::DefinitionIndexOutOfRange { index: 0, len: 0 })
}

pub fn apply_rewrite(
    _comp: &mut TensorComputation,
    _rewrite: FactorizationRewrite,
) -> Result<(), RewriteError> {
    Ok(())
}
```

- [ ] **Step 4: Run the validation tests**

Run:

```bash
cargo test rewrite::tests::validate_decision -- --nocapture
```

Expected: PASS for the four validation tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/rewrite.rs
git commit -m "feat: add rewrite action API"
```

---

### Task 2: Build Factorization Templates From Graphs And Bicliques

**Files:**
- Modify: `src/rewrite.rs`

- [ ] **Step 1: Add failing unit tests for factorization construction**

Append these helpers and tests inside the existing `#[cfg(test)] mod tests` in `src/rewrite.rs`:

```rust
    use crate::biclique::Biclique;
    use crate::graph::{ConstrGraph, GraphEdge};
    use crate::repr::{Factor, Index, IndexId, RangeId};
    use crate::split::SplitInterface;

    fn rat(value: i64) -> Rational {
        Rational::new(value, 1)
    }

    fn idx(id: u32) -> Index {
        Index {
            id: IndexId(id),
            range: RangeId(0),
        }
    }

    fn factor(tensor: u32, indices: &[u32]) -> Factor {
        Factor {
            tensor: TensorId(tensor),
            indices: indices.iter().copied().map(IndexId).collect(),
        }
    }

    fn term_with_sum(coeff: Rational, sum_indices: Vec<Index>, factors: Vec<Factor>) -> Term {
        Term {
            coeff,
            sum_indices,
            factors,
        }
    }

    fn source_def_for_factorization() -> TensorDef {
        TensorDef {
            base: TensorId(50),
            ext_indices: vec![idx(0), idx(1)],
            terms: vec![
                term_with_sum(rat(1), vec![idx(2)], vec![factor(1, &[0, 2])]),
                term_with_sum(rat(1), vec![idx(3)], vec![factor(2, &[3, 1])]),
                term_with_sum(rat(9), vec![], vec![factor(9, &[0, 1])]),
            ],
        }
    }

    fn graph_and_biclique_for_factorization() -> (ConstrGraph, Biclique) {
        let graph = ConstrGraph {
            interface: SplitInterface {
                left_external: vec![idx(0)],
                right_external: vec![idx(1)],
                contracted: vec![idx(2)],
            },
            left_nodes: vec![term_with_sum(
                rat(1),
                vec![idx(4)],
                vec![factor(10, &[0, 4, 2])],
            )],
            right_nodes: vec![
                term_with_sum(rat(1), vec![], vec![factor(11, &[2, 1])]),
                term_with_sum(rat(1), vec![], vec![factor(12, &[2, 1])]),
            ],
            edges: vec![
                GraphEdge {
                    left_id: 0,
                    right_id: 0,
                    coeff: rat(15),
                    terms_used: 0b001,
                },
                GraphEdge {
                    left_id: 0,
                    right_id: 1,
                    coeff: rat(21),
                    terms_used: 0b010,
                },
            ],
        };

        let biclique = Biclique {
            left_node_ids: vec![0],
            right_node_ids: vec![0, 1],
            left_coeffs: vec![rat(3)],
            right_coeffs: vec![rat(5), rat(7)],
            terms_used: 0b011,
        };

        (graph, biclique)
    }

    #[test]
    fn build_factorization_uses_interface_indices_as_source_of_truth() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization = build_factorization(
            &def,
            &graph,
            &biclique,
            TensorId(60),
            TensorId(61),
        );

        assert_eq!(factorization.left_definition.base, TensorId(60));
        assert_eq!(factorization.right_definition.base, TensorId(61));
        assert_eq!(factorization.left_definition.ext_indices, vec![idx(0), idx(2)]);
        assert_eq!(factorization.right_definition.ext_indices, vec![idx(1), idx(2)]);
        assert_eq!(factorization.rewritten_definition.base, TensorId(50));
        assert_eq!(factorization.rewritten_definition.ext_indices, vec![idx(0), idx(1)]);
    }

    #[test]
    fn build_factorization_preserves_private_sum_indices_and_side_coefficients() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization = build_factorization(
            &def,
            &graph,
            &biclique,
            TensorId(60),
            TensorId(61),
        );

        assert_eq!(factorization.left_definition.terms.len(), 1);
        assert_eq!(factorization.left_definition.terms[0].coeff, rat(3));
        assert_eq!(factorization.left_definition.terms[0].sum_indices, vec![idx(4)]);
        assert_eq!(factorization.right_definition.terms.len(), 2);
        assert_eq!(factorization.right_definition.terms[0].coeff, rat(5));
        assert_eq!(factorization.right_definition.terms[1].coeff, rat(7));
    }

    #[test]
    fn build_factorization_removes_consumed_terms_and_appends_replacement() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization = build_factorization(
            &def,
            &graph,
            &biclique,
            TensorId(60),
            TensorId(61),
        );

        assert_eq!(factorization.rewritten_definition.terms.len(), 2);
        assert_eq!(factorization.rewritten_definition.terms[0], def.terms[2]);
        assert_eq!(
            factorization.rewritten_definition.terms[1],
            Term {
                coeff: rat(1),
                sum_indices: vec![idx(2)],
                factors: vec![
                    Factor {
                        tensor: TensorId(60),
                        indices: vec![IndexId(0), IndexId(2)],
                    },
                    Factor {
                        tensor: TensorId(61),
                        indices: vec![IndexId(1), IndexId(2)],
                    },
                ],
            }
        );
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test rewrite::tests::build_factorization -- --nocapture
```

Expected: compile failure because `build_factorization` is not defined.

- [ ] **Step 3: Implement factorization helpers**

Expand the imports at the top of `src/rewrite.rs`:

```rust
use crate::repr::{Factor, Index, Rational, TensorComputation, TensorDef, TensorId, Term};
```

Add these private helpers above the test module:

```rust
fn build_factorization(
    def: &TensorDef,
    graph: &ConstrGraph,
    biclique: &Biclique,
    left_tid: TensorId,
    right_tid: TensorId,
) -> Factorization {
    let contracted = contracted_indices(graph);
    let (left_external, right_external) = side_external_indices(graph);

    let left_definition = build_side_definition(
        &graph.left_nodes,
        &biclique.left_node_ids,
        &biclique.left_coeffs,
        &left_external,
        &contracted,
        left_tid,
    );
    let right_definition = build_side_definition(
        &graph.right_nodes,
        &biclique.right_node_ids,
        &biclique.right_coeffs,
        &right_external,
        &contracted,
        right_tid,
    );
    let consumed = consumed_term_indices(biclique);
    let rewritten_definition = build_rewritten_definition(
        def,
        &left_definition,
        &right_definition,
        &contracted,
        &consumed,
    );

    Factorization {
        left_definition,
        right_definition,
        rewritten_definition,
    }
}

fn contracted_indices(graph: &ConstrGraph) -> Vec<Index> {
    graph.interface.contracted.clone()
}

fn side_external_indices(graph: &ConstrGraph) -> (Vec<Index>, Vec<Index>) {
    (
        graph.interface.left_external.clone(),
        graph.interface.right_external.clone(),
    )
}

fn consumed_term_indices(biclique: &Biclique) -> Vec<usize> {
    bits_to_vec(biclique.terms_used)
}

fn build_side_definition(
    source_nodes: &[Term],
    node_ids: &[usize],
    coeffs: &[Rational],
    side_external: &[Index],
    contracted: &[Index],
    tensor: TensorId,
) -> TensorDef {
    let mut ext_indices = side_external.to_vec();
    ext_indices.extend_from_slice(contracted);

    TensorDef {
        base: tensor,
        ext_indices,
        terms: node_ids
            .iter()
            .zip(coeffs)
            .map(|(&node_id, coeff)| build_side_term(source_nodes, node_id, coeff))
            .collect(),
    }
}

fn build_side_term(source_nodes: &[Term], node_id: usize, coeff: &Rational) -> Term {
    let mut term = source_nodes[node_id].clone();
    term.coeff *= *coeff;
    term
}

fn build_rewritten_definition(
    def: &TensorDef,
    left_def: &TensorDef,
    right_def: &TensorDef,
    contracted: &[Index],
    consumed: &[usize],
) -> TensorDef {
    let mut terms: Vec<_> = def
        .terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| {
            if consumed.contains(&index) {
                None
            } else {
                Some(term.clone())
            }
        })
        .collect();

    terms.push(Term {
        coeff: Rational::new(1, 1),
        sum_indices: contracted.to_vec(),
        factors: vec![
            Factor {
                tensor: left_def.base,
                indices: left_def.ext_indices.iter().map(|index| index.id).collect(),
            },
            Factor {
                tensor: right_def.base,
                indices: right_def.ext_indices.iter().map(|index| index.id).collect(),
            },
        ],
    });

    TensorDef {
        base: def.base,
        ext_indices: def.ext_indices.clone(),
        terms,
    }
}

fn bits_to_vec(mask: u64) -> Vec<usize> {
    (0..64)
        .filter(|position| mask & (1_u64 << position) != 0)
        .collect()
}
```

- [ ] **Step 4: Run the factorization tests**

Run:

```bash
cargo test rewrite::tests::build_factorization -- --nocapture
```

Expected: PASS for the three factorization tests.

- [ ] **Step 5: Commit**

```bash
git add src/rewrite.rs
git commit -m "feat: build rewrite factorization templates"
```

---

### Task 3: Build Selected Sub-Bicliques And Factorization Rewrites

**Files:**
- Modify: `src/rewrite.rs`

- [ ] **Step 1: Add failing unit tests for subset decisions and rewrite construction**

Append these tests inside the existing `#[cfg(test)] mod tests` in `src/rewrite.rs`:

```rust
    fn comp_with_definition(def: TensorDef) -> TensorComputation {
        let mut comp = TensorComputation::new();
        comp.add_definition(def.base, def.ext_indices.clone(), def.terms.clone());
        comp
    }

    fn action_space_for_factorization(comp: &TensorComputation) -> ActionSpace {
        let def = &comp.definitions()[0];
        let (graph, biclique) = graph_and_biclique_for_factorization();
        let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
        let template = build_factorization(def, &graph, &biclique, left_tid, right_tid);

        ActionSpace {
            def_index: 0,
            candidate_templates: vec![template],
            candidate_graphs: vec![graph],
            candidate_bicliques: vec![biclique],
        }
    }

    #[test]
    fn sub_biclique_from_decision_keeps_selected_terms_and_recomputes_provenance() {
        let (graph, biclique) = graph_and_biclique_for_factorization();
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true],
            right_mask: vec![false, true],
        };

        let sub_biclique = sub_biclique_from_decision(&graph, &biclique, &decision);

        assert_eq!(sub_biclique.left_node_ids, vec![0]);
        assert_eq!(sub_biclique.right_node_ids, vec![1]);
        assert_eq!(sub_biclique.left_coeffs, vec![rat(3)]);
        assert_eq!(sub_biclique.right_coeffs, vec![rat(7)]);
        assert_eq!(sub_biclique.terms_used, 0b010);
    }

    #[test]
    fn build_rewrite_full_biclique_matches_visible_template() {
        let comp = comp_with_definition(source_def_for_factorization());
        let space = action_space_for_factorization(&comp);
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true],
            right_mask: vec![true, true],
        };

        let rewrite = build_rewrite(&comp, &space, &decision).unwrap();

        assert_eq!(rewrite.def_index, 0);
        assert_eq!(rewrite.factorization, space.candidate_templates[0]);
    }

    #[test]
    fn build_rewrite_subset_decision_shrinks_side_definition_and_consumed_terms() {
        let comp = comp_with_definition(source_def_for_factorization());
        let space = action_space_for_factorization(&comp);
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true],
            right_mask: vec![false, true],
        };

        let rewrite = build_rewrite(&comp, &space, &decision).unwrap();

        assert_eq!(rewrite.factorization.left_definition.terms.len(), 1);
        assert_eq!(rewrite.factorization.right_definition.terms.len(), 1);
        assert_eq!(rewrite.factorization.right_definition.terms[0].coeff, rat(7));
        assert_eq!(rewrite.factorization.rewritten_definition.terms.len(), 3);
        assert_eq!(
            rewrite.factorization.rewritten_definition.terms[0],
            comp.definitions()[0].terms[0]
        );
        assert_eq!(
            rewrite.factorization.rewritten_definition.terms[1],
            comp.definitions()[0].terms[2]
        );
    }

    #[test]
    fn build_rewrite_rejects_out_of_range_definition_index() {
        let comp = TensorComputation::new();
        let space = ActionSpace {
            def_index: 0,
            candidate_templates: vec![],
            candidate_graphs: vec![],
            candidate_bicliques: vec![],
        };
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![],
            right_mask: vec![],
        };

        assert_eq!(
            build_rewrite(&comp, &space, &decision),
            Err(RewriteError::DefinitionIndexOutOfRange { index: 0, len: 0 })
        );
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test rewrite::tests -- --nocapture
```

Expected: compile failure because `fresh_rewrite_tensor_ids` and `sub_biclique_from_decision` are not defined, and `build_rewrite` still returns the placeholder error.

- [ ] **Step 3: Implement sub-biclique selection and rewrite construction**

Add this import at the top of `src/rewrite.rs`:

```rust
use std::collections::HashSet;
```

Replace the placeholder `build_rewrite` and add the private helpers:

```rust
pub fn build_rewrite(
    comp: &TensorComputation,
    space: &ActionSpace,
    decision: &Decision,
) -> Result<FactorizationRewrite, RewriteError> {
    let def = comp.definitions().get(space.def_index).ok_or(
        RewriteError::DefinitionIndexOutOfRange {
            index: space.def_index,
            len: comp.definitions().len(),
        },
    )?;

    validate_decision(space, decision)?;

    let graph = space
        .candidate_graphs
        .get(decision.candidate_index)
        .ok_or(RewriteError::CandidateIndexOutOfRange {
            index: decision.candidate_index,
            len: space.candidate_graphs.len(),
        })?;
    let biclique = space
        .candidate_bicliques
        .get(decision.candidate_index)
        .ok_or(RewriteError::CandidateIndexOutOfRange {
            index: decision.candidate_index,
            len: space.candidate_bicliques.len(),
        })?;

    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
    let sub_biclique = sub_biclique_from_decision(graph, biclique, decision);
    let factorization = build_factorization(def, graph, &sub_biclique, left_tid, right_tid);

    Ok(FactorizationRewrite {
        def_index: space.def_index,
        factorization,
    })
}

fn sub_biclique_from_decision(
    graph: &ConstrGraph,
    biclique: &Biclique,
    decision: &Decision,
) -> Biclique {
    let left: Vec<_> = biclique
        .left_node_ids
        .iter()
        .copied()
        .zip(biclique.left_coeffs.iter().copied())
        .zip(decision.left_mask.iter().copied())
        .filter_map(|((node_id, coeff), keep)| keep.then_some((node_id, coeff)))
        .collect();
    let right: Vec<_> = biclique
        .right_node_ids
        .iter()
        .copied()
        .zip(biclique.right_coeffs.iter().copied())
        .zip(decision.right_mask.iter().copied())
        .filter_map(|((node_id, coeff), keep)| keep.then_some((node_id, coeff)))
        .collect();

    let selected_left: HashSet<_> = left.iter().map(|(node_id, _)| *node_id).collect();
    let selected_right: HashSet<_> = right.iter().map(|(node_id, _)| *node_id).collect();
    let terms_used = graph
        .edges
        .iter()
        .filter(|edge| {
            selected_left.contains(&edge.left_id) && selected_right.contains(&edge.right_id)
        })
        .fold(0, |acc, edge| acc | edge.terms_used);

    Biclique {
        left_node_ids: left.iter().map(|(node_id, _)| *node_id).collect(),
        right_node_ids: right.iter().map(|(node_id, _)| *node_id).collect(),
        left_coeffs: left.iter().map(|(_, coeff)| *coeff).collect(),
        right_coeffs: right.iter().map(|(_, coeff)| *coeff).collect(),
        terms_used,
    }
}

fn fresh_rewrite_tensor_ids(comp: &TensorComputation) -> (TensorId, TensorId) {
    let left = comp.next_tensor_id();
    let right = TensorId(left.0 + 1);
    (left, right)
}
```

- [ ] **Step 4: Run the rewrite construction tests**

Run:

```bash
cargo test rewrite::tests -- --nocapture
```

Expected: PASS for the four tests added in this task.

- [ ] **Step 5: Commit**

```bash
git add src/rewrite.rs
git commit -m "feat: build selected factorization rewrites"
```

---

### Task 4: Generate Action Spaces Through The Stage Pipeline

**Files:**
- Modify: `src/rewrite.rs`
- Create: `tests/rewrite.rs`

- [ ] **Step 1: Add failing public integration tests for action-space generation**

Create `tests/rewrite.rs`:

```rust
use gristmill_symbolics::canon::CanonError;
use gristmill_symbolics::graph::GraphError;
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId, Term,
};
use gristmill_symbolics::rewrite::{RewriteError, next_action_space};
use gristmill_symbolics::split::SplitError;

fn one() -> Rational {
    Rational::new(1, 1)
}

fn idx(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(0),
    }
}

fn factor(tensor: TensorId, indices: &[u32]) -> Factor {
    Factor {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn term(sum_indices: Vec<Index>, factors: Vec<Factor>) -> Term {
    Term {
        coeff: one(),
        sum_indices,
        factors,
    }
}

fn comp_with_shared_left_candidate() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let b = comp.add_tensor(vec![]);
    let c = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);

    comp.add_definition(
        out,
        vec![idx(0), idx(1)],
        vec![
            term(vec![idx(2)], vec![factor(a, &[0, 2]), factor(b, &[2, 1])]),
            term(vec![idx(3)], vec![factor(a, &[0, 3]), factor(c, &[3, 1])]),
        ],
    );

    comp
}

#[test]
fn next_action_space_returns_none_when_no_definition_is_actionable() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);

    assert_eq!(next_action_space(&comp, 0), Ok(None));
}

#[test]
fn next_action_space_returns_first_actionable_definition() {
    let mut comp = comp_with_shared_left_candidate();
    let extra_base = comp.add_tensor(vec![]);
    let skipped = gristmill_symbolics::repr::TensorDef {
        base: extra_base,
        ext_indices: vec![idx(0)],
        terms: vec![term(vec![], vec![factor(TensorId(0), &[0])])],
    };
    comp.definitions_mut().insert(0, skipped);

    let space = next_action_space(&comp, 0).unwrap().unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}

#[test]
fn next_action_space_propagates_split_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    let many_factors = (0..65).map(|_| factor(a, &[])).collect();

    comp.add_definition(
        out,
        vec![],
        vec![term(vec![], many_factors), term(vec![], vec![factor(a, &[])])],
    );

    assert_eq!(
        next_action_space(&comp, 0),
        Err(RewriteError::Split(SplitError::TooManyFactors {
            len: 65,
            max: 64,
        }))
    );
}

#[test]
fn next_action_space_propagates_canon_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let out = comp.add_tensor(vec![]);
    let missing = TensorId(99);

    comp.add_definition(
        out,
        vec![],
        vec![
            term(vec![], vec![factor(missing, &[]), factor(missing, &[])]),
            term(vec![], vec![factor(missing, &[]), factor(missing, &[])]),
        ],
    );

    assert_eq!(
        next_action_space(&comp, 0),
        Err(RewriteError::Canon(CanonError::MissingTensorSymmetry {
            tensor: missing,
        }))
    );
}

#[test]
fn next_action_space_propagates_graph_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(128);
    let a = comp.add_tensor(vec![]);
    let b = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    let terms: Vec<_> = (0..65)
        .map(|offset| {
            let sum_id = 2 + offset;
            term(
                vec![idx(sum_id)],
                vec![factor(a, &[0, sum_id]), factor(b, &[sum_id, 1])],
            )
        })
        .collect();

    comp.add_definition(out, vec![idx(0), idx(1)], terms);

    assert_eq!(
        next_action_space(&comp, 0),
        Err(RewriteError::Graph(GraphError::TooManyTerms {
            len: 65,
            max: 64,
        }))
    );
}
```

- [ ] **Step 2: Run the integration tests to verify they fail**

Run:

```bash
cargo test --test rewrite -- --nocapture
```

Expected: at least `next_action_space_returns_first_actionable_definition` fails because `next_action_space` still returns `Ok(None)`.

- [ ] **Step 3: Implement candidate enumeration and action-space generation**

Expand the imports at the top of `src/rewrite.rs`:

```rust
use crate::{biclique, canon, graph, split};
use crate::split::Split;
```

Replace `next_action_space` and add `enumerate_candidates`:

```rust
pub fn next_action_space(
    comp: &TensorComputation,
    start_from: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);

    for (def_index, def) in comp.definitions().iter().enumerate().skip(start_from) {
        if def.terms.len() < 2 {
            continue;
        }

        let (candidate_graphs, candidate_bicliques) = enumerate_candidates(comp, def)?;
        if candidate_bicliques.is_empty() {
            continue;
        }

        let candidate_templates = candidate_graphs
            .iter()
            .zip(&candidate_bicliques)
            .map(|(graph, biclique)| {
                build_factorization(def, graph, biclique, left_tid, right_tid)
            })
            .collect();

        return Ok(Some(ActionSpace {
            def_index,
            candidate_templates,
            candidate_graphs,
            candidate_bicliques,
        }));
    }

    Ok(None)
}

fn enumerate_candidates(
    comp: &TensorComputation,
    def: &TensorDef,
) -> Result<(Vec<ConstrGraph>, Vec<Biclique>), RewriteError> {
    let symmetry = canon::build_tensor_symmetry_map(comp.tensors());
    let pool = canon::build_index_pool(def);
    let mut left_owner_splits_by_term: Vec<Vec<Split>> = vec![vec![]; def.terms.len()];
    let mut right_owner_splits_by_term: Vec<Vec<Split>> = vec![vec![]; def.terms.len()];

    for (term_idx, term) in def.terms.iter().enumerate() {
        for raw_split in split::enumerate_splits(term, def)? {
            let (left_owner, right_owner) =
                canon::canon_split(&raw_split, &symmetry, &pool)?;
            left_owner_splits_by_term[term_idx].push(left_owner);
            right_owner_splits_by_term[term_idx].push(right_owner);
        }
    }

    let mut graphs = Vec::new();
    graphs.extend(graph::build_graphs_from_splits(
        def,
        &left_owner_splits_by_term,
    )?);
    graphs.extend(graph::build_graphs_from_splits(
        def,
        &right_owner_splits_by_term,
    )?);

    let mut candidate_graphs = Vec::new();
    let mut candidate_bicliques = Vec::new();
    for graph in graphs {
        for biclique in biclique::enumerate_bicliques(&graph) {
            candidate_graphs.push(graph.clone());
            candidate_bicliques.push(biclique);
        }
    }

    Ok((candidate_graphs, candidate_bicliques))
}
```

- [ ] **Step 4: Run the rewrite integration tests**

Run:

```bash
cargo test --test rewrite -- --nocapture
```

Expected: PASS for the five integration tests in `tests/rewrite.rs`.

- [ ] **Step 5: Run existing stage tests**

Run:

```bash
cargo test --test split --test canon --test graph --test biclique -- --nocapture
```

Expected: PASS for all existing stage tests.

- [ ] **Step 6: Commit**

```bash
git add src/rewrite.rs tests/rewrite.rs
git commit -m "feat: generate rewrite action spaces"
```

---

### Task 5: Apply Factorization Rewrites

**Files:**
- Modify: `src/rewrite.rs`
- Modify: `tests/rewrite.rs`

- [ ] **Step 1: Add failing public tests for applying rewrites**

Append these tests to `tests/rewrite.rs`:

```rust
use gristmill_symbolics::rewrite::{
    Decision, Factorization, FactorizationRewrite, apply_rewrite, build_rewrite,
};

fn empty_def(base: TensorId) -> gristmill_symbolics::repr::TensorDef {
    gristmill_symbolics::repr::TensorDef {
        base,
        ext_indices: vec![],
        terms: vec![],
    }
}

fn first_full_decision(space: &gristmill_symbolics::rewrite::ActionSpace) -> Decision {
    let template = &space.candidate_templates[0];
    Decision {
        candidate_index: 0,
        left_mask: vec![true; template.left_definition.terms.len()],
        right_mask: vec![true; template.right_definition.terms.len()],
    }
}

#[test]
fn apply_rewrite_registers_tensors_inserts_definitions_and_validates() {
    let mut comp = comp_with_shared_left_candidate();
    let original_tensors = comp.tensors().len();
    let original_definitions = comp.definitions().len();
    let space = next_action_space(&comp, 0).unwrap().unwrap();
    let decision = first_full_decision(&space);
    let rewrite = build_rewrite(&comp, &space, &decision).unwrap();
    let def_index = rewrite.def_index;
    let left_base = rewrite.factorization.left_definition.base;
    let right_base = rewrite.factorization.right_definition.base;
    let rewritten_base = rewrite.factorization.rewritten_definition.base;

    apply_rewrite(&mut comp, rewrite).unwrap();

    assert_eq!(comp.tensors().len(), original_tensors + 2);
    assert_eq!(comp.definitions().len(), original_definitions + 2);
    assert_eq!(comp.definitions()[def_index].base, left_base);
    assert_eq!(comp.definitions()[def_index + 1].base, right_base);
    assert_eq!(comp.definitions()[def_index + 2].base, rewritten_base);
    assert_eq!(comp.validate(), Ok(()));
}

#[test]
fn apply_rewrite_rejects_out_of_range_definition_index_before_mutation() {
    let mut comp = TensorComputation::new();
    let rewrite = FactorizationRewrite {
        def_index: 7,
        factorization: Factorization {
            left_definition: empty_def(TensorId(0)),
            right_definition: empty_def(TensorId(1)),
            rewritten_definition: empty_def(TensorId(2)),
        },
    };

    assert_eq!(
        apply_rewrite(&mut comp, rewrite),
        Err(RewriteError::DefinitionIndexOutOfRange { index: 7, len: 0 })
    );
    assert_eq!(comp.tensors().len(), 0);
    assert_eq!(comp.definitions().len(), 0);
}

#[test]
fn apply_rewrite_only_checks_definition_index_after_rewrite_construction() {
    let mut comp = comp_with_shared_left_candidate();
    let space = next_action_space(&comp, 0).unwrap().unwrap();
    let decision = first_full_decision(&space);
    let rewrite = build_rewrite(&comp, &space, &decision).unwrap();

    comp.definitions_mut()[rewrite.def_index].terms.clear();

    assert_eq!(apply_rewrite(&mut comp, rewrite), Ok(()));
}
```

- [ ] **Step 2: Run the apply tests to verify they fail**

Run:

```bash
cargo test --test rewrite apply_rewrite -- --nocapture
```

Expected: at least `apply_rewrite_rejects_out_of_range_definition_index_before_mutation` fails because `apply_rewrite` still returns `Ok(())`.

- [ ] **Step 3: Implement rewrite application helpers**

Replace the placeholder `apply_rewrite` and add these helpers:

```rust
pub fn apply_rewrite(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) -> Result<(), RewriteError> {
    verify_rewrite_def_index(comp, &rewrite)?;
    register_rewrite_tensors(comp);
    replace_definition_with_factorization(comp, rewrite);
    Ok(())
}

fn verify_rewrite_def_index(
    comp: &TensorComputation,
    rewrite: &FactorizationRewrite,
) -> Result<(), RewriteError> {
    if rewrite.def_index < comp.definitions().len() {
        Ok(())
    } else {
        Err(RewriteError::DefinitionIndexOutOfRange {
            index: rewrite.def_index,
            len: comp.definitions().len(),
        })
    }
}

fn register_rewrite_tensors(comp: &mut TensorComputation) {
    comp.add_tensor(vec![]);
    comp.add_tensor(vec![]);
}

fn replace_definition_with_factorization(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) {
    let def_index = rewrite.def_index;
    let Factorization {
        left_definition,
        right_definition,
        rewritten_definition,
    } = rewrite.factorization;
    let definitions = comp.definitions_mut();

    definitions.remove(def_index);
    definitions.insert(def_index, rewritten_definition);
    definitions.insert(def_index, right_definition);
    definitions.insert(def_index, left_definition);
}
```

- [ ] **Step 4: Run the apply tests**

Run:

```bash
cargo test --test rewrite apply_rewrite -- --nocapture
```

Expected: PASS for the three apply tests.

- [ ] **Step 5: Run all rewrite tests**

Run:

```bash
cargo test --test rewrite -- --nocapture
cargo test rewrite::tests -- --nocapture
```

Expected: PASS for public integration tests and private module tests.

- [ ] **Step 6: Commit**

```bash
git add src/rewrite.rs tests/rewrite.rs
git commit -m "feat: apply factorization rewrites"
```

---

### Task 6: Final Verification And Cleanup

**Files:**
- Modify only files that fail formatting or verification:
  - `src/rewrite.rs`
  - `src/lib.rs`
  - `tests/rewrite.rs`

- [ ] **Step 1: Format the Rust code**

Run:

```bash
cargo fmt
```

Expected: command exits successfully with no output or only rustfmt diagnostics for files it formatted.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
cargo test
```

Expected: all unit tests, integration tests, and doc tests pass.

- [ ] **Step 3: Inspect the diff**

Run:

```bash
git diff -- src/lib.rs src/rewrite.rs tests/rewrite.rs
```

Expected: diff contains only the rewrite module, library export, and rewrite tests.

- [ ] **Step 4: Commit final formatting if needed**

If `cargo fmt` changed files after the Task 5 commit, run:

```bash
git add src/lib.rs src/rewrite.rs tests/rewrite.rs
git commit -m "style: format rewrite module"
```

Expected: a formatting-only commit is created only when the diff is nonempty.

---

## Self-Review

Spec coverage:

- Public action API: Task 1.
- Hidden sidecar alignment and private sidecar tests: Tasks 1 and 3 in `src/rewrite.rs`.
- Decision validation: Task 1.
- Factorization construction from `SplitInterface`: Task 2.
- Sub-biclique construction from masks: Task 3.
- Candidate enumeration through `split`, `canon`, `graph`, and `biclique`: Task 4.
- Upstream error propagation: Task 4.
- Rewrite application and post-apply validation: Task 5.
- Candidate-order scope: preserved by not sorting candidates in Task 4.
- No stale-definition equality check: Task 5 verifies `apply_rewrite` checks only `def_index`.

Type consistency:

- `Factorization`, `ActionSpace`, `Decision`, `FactorizationRewrite`, and `RewriteError` match the rewrite spec.
- `ActionSpace::candidate_graphs` and `ActionSpace::candidate_bicliques` remain private.
- `build_factorization` consumes `ConstrGraph` and `Biclique` records without reconstructing external or contracted indices from node terms.
