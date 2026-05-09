use gristmill_symbolics::graph::{build_graphs_from_splits, GraphError};
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
