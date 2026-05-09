use gristmill_symbolics::repr::{IndexId, Rational, TensorDef, TensorId, Term};
use gristmill_symbolics::split::enumerate_splits;

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
