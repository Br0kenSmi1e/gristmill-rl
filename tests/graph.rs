use gristmill_symbolics::graph::{build_graphs_from_splits, GraphError};
use gristmill_symbolics::repr::{Rational, TensorDef, TensorId, Term};

fn one() -> Rational {
    Rational::new(1, 1)
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
