use crate::biclique::Biclique;
use crate::canon::CanonError;
use crate::graph::{ConstrGraph, GraphError};
use crate::repr::{Factor, Index, Rational, TensorComputation, TensorDef, TensorId, Term};
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

pub fn validate_decision(space: &ActionSpace, decision: &Decision) -> Result<(), RewriteError> {
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

#[allow(dead_code)]
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

#[allow(dead_code)]
fn contracted_indices(graph: &ConstrGraph) -> Vec<Index> {
    graph.interface.contracted.clone()
}

#[allow(dead_code)]
fn side_external_indices(graph: &ConstrGraph) -> (Vec<Index>, Vec<Index>) {
    (
        graph.interface.left_external.clone(),
        graph.interface.right_external.clone(),
    )
}

#[allow(dead_code)]
fn consumed_term_indices(biclique: &Biclique) -> Vec<usize> {
    bits_to_vec(biclique.terms_used)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn build_side_term(source_nodes: &[Term], node_id: usize, coeff: &Rational) -> Term {
    let mut term = source_nodes[node_id].clone();
    term.coeff *= *coeff;
    term
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn bits_to_vec(mask: u64) -> Vec<usize> {
    (0..64)
        .filter(|position| mask & (1_u64 << position) != 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biclique::Biclique;
    use crate::graph::{ConstrGraph, GraphEdge};
    use crate::repr::{Factor, Index, IndexId, RangeId, Rational, TensorDef, TensorId, Term};
    use crate::split::SplitInterface;

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

        let factorization =
            build_factorization(&def, &graph, &biclique, TensorId(60), TensorId(61));

        assert_eq!(factorization.left_definition.base, TensorId(60));
        assert_eq!(factorization.right_definition.base, TensorId(61));
        assert_eq!(
            factorization.left_definition.ext_indices,
            vec![idx(0), idx(2)]
        );
        assert_eq!(
            factorization.right_definition.ext_indices,
            vec![idx(1), idx(2)]
        );
        assert_eq!(factorization.rewritten_definition.base, TensorId(50));
        assert_eq!(
            factorization.rewritten_definition.ext_indices,
            vec![idx(0), idx(1)]
        );
    }

    #[test]
    fn build_factorization_preserves_private_sum_indices_and_side_coefficients() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization =
            build_factorization(&def, &graph, &biclique, TensorId(60), TensorId(61));

        assert_eq!(factorization.left_definition.terms.len(), 1);
        assert_eq!(factorization.left_definition.terms[0].coeff, rat(3));
        assert_eq!(
            factorization.left_definition.terms[0].sum_indices,
            vec![idx(4)]
        );
        assert_eq!(factorization.right_definition.terms.len(), 2);
        assert_eq!(factorization.right_definition.terms[0].coeff, rat(5));
        assert_eq!(factorization.right_definition.terms[1].coeff, rat(7));
    }

    #[test]
    fn build_factorization_removes_consumed_terms_and_appends_replacement() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization =
            build_factorization(&def, &graph, &biclique, TensorId(60), TensorId(61));

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
}
