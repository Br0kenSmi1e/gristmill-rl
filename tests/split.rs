use gristmill_symbolics::repr::{IndexId, Rational, TensorDef, TensorId, Term};
use gristmill_symbolics::repr::{Factor, Index, RangeId};
use gristmill_symbolics::split::enumerate_splits;
use gristmill_symbolics::split::{Split, SplitInterface};

fn one() -> Rational {
    Rational::new(1, 1)
}

#[test]
fn terms_with_fewer_than_two_factors_produce_no_splits() {
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![],
    };

    let zero_factor = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![],
    };
    assert_eq!(enumerate_splits(&zero_factor, &def).unwrap(), vec![]);

    let one_factor = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![gristmill_symbolics::repr::Factor {
            tensor: TensorId(0),
            indices: vec![IndexId(0)],
        }],
    };
    assert_eq!(enumerate_splits(&one_factor, &def).unwrap(), vec![]);
}

#[test]
fn two_factor_term_produces_one_unit_coefficient_split() {
    let range = RangeId(0);
    let a = Index {
        id: IndexId(0),
        range,
    };
    let b = Index {
        id: IndexId(1),
        range,
    };
    let c = Index {
        id: IndexId(2),
        range,
    };
    let x = TensorId(0);
    let y = TensorId(1);

    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![a, b],
        terms: vec![],
    };
    let term = Term {
        coeff: Rational::new(7, 3),
        sum_indices: vec![c],
        factors: vec![
            Factor {
                tensor: x,
                indices: vec![a.id, c.id],
            },
            Factor {
                tensor: y,
                indices: vec![c.id, b.id],
            },
        ],
    };

    assert_eq!(
        enumerate_splits(&term, &def).unwrap(),
        vec![Split {
            left: Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![Factor {
                    tensor: x,
                    indices: vec![a.id, c.id],
                }],
            },
            right: Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![Factor {
                    tensor: y,
                    indices: vec![c.id, b.id],
                }],
            },
            interface: SplitInterface {
                left_external: vec![a],
                right_external: vec![b],
                contracted: vec![c],
            },
        }]
    );
}
