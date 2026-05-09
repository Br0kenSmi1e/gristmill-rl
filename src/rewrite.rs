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
