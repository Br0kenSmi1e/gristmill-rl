use crate::repr::{Rational, TensorDef, Term};
use crate::split::{Split, SplitInterface};

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
