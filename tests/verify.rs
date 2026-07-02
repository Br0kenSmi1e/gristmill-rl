use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId, Term,
};
use gristmill_symbolics::verify::{VerifyError, equivalent_computations};

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

#[test]
fn equivalent_when_outputs_are_directly_equal() {
    let mut lhs = TensorComputation::new();
    lhs.add_range(10);
    let a = lhs.add_tensor(vec![]);
    let r = lhs.add_tensor(vec![]);
    lhs.add_definition(
        r,
        vec![idx(0, 0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(a.0, &[0])],
        }],
    );

    let mut rhs = TensorComputation::new();
    rhs.add_range(10);
    let a = rhs.add_tensor(vec![]);
    let r = rhs.add_tensor(vec![]);
    rhs.add_definition(
        r,
        vec![idx(0, 0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(a.0, &[0])],
        }],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[r]).unwrap());
}

#[test]
fn equivalent_after_external_index_alpha_renaming() {
    let mut lhs = TensorComputation::new();
    lhs.add_range(10);
    let a = lhs.add_tensor(vec![]);
    let r = lhs.add_tensor(vec![]);
    lhs.add_definition(
        r,
        vec![idx(0, 0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(a.0, &[0])],
        }],
    );

    let mut rhs = TensorComputation::new();
    rhs.add_range(10);
    let a = rhs.add_tensor(vec![]);
    let r = rhs.add_tensor(vec![]);
    rhs.add_definition(
        r,
        vec![idx(1, 0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(a.0, &[1])],
        }],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[r]).unwrap());
}

#[test]
fn equivalent_when_simple_intermediate_is_inlined() {
    let mut lhs = TensorComputation::new();
    lhs.add_range(7);
    let b = lhs.add_tensor(vec![]);
    let c = lhs.add_tensor(vec![]);
    let a = lhs.add_tensor(vec![]);
    let tau = lhs.add_tensor(vec![]);
    let r = lhs.add_tensor(vec![]);

    lhs.add_definition(
        tau,
        vec![idx(0, 0)],
        vec![
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![factor(b.0, &[0])],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![factor(c.0, &[0])],
            },
        ],
    );
    lhs.add_definition(
        r,
        vec![idx(0, 0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(a.0, &[0]), factor(tau.0, &[0])],
        }],
    );

    let mut rhs = TensorComputation::new();
    rhs.add_range(7);
    let b = rhs.add_tensor(vec![]);
    let c = rhs.add_tensor(vec![]);
    let a = rhs.add_tensor(vec![]);
    let _tau = rhs.add_tensor(vec![]);
    let r = rhs.add_tensor(vec![]);

    rhs.add_definition(
        r,
        vec![idx(0, 0)],
        vec![
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![factor(a.0, &[0]), factor(b.0, &[0])],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![factor(a.0, &[0]), factor(c.0, &[0])],
            },
        ],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[r]).unwrap());
}

#[test]
fn equivalent_keeps_dummy_instances_distinct_when_inline_expands_twice_same_source() {
    let mut lhs = TensorComputation::new();
    let range0 = lhs.add_range(11);
    let range1 = lhs.add_range(11);
    let b = lhs.add_tensor(vec![]);
    let a = lhs.add_tensor(vec![]);
    let tau = lhs.add_tensor(vec![]);
    let r = lhs.add_tensor(vec![]);

    lhs.add_definition(
        tau,
        vec![idx(0, range0.0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![idx(1, range1.0)],
            factors: vec![factor(b.0, &[0, 1])],
        }],
    );
    lhs.add_definition(
        r,
        vec![idx(2, range0.0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![idx(1, range1.0)],
            factors: vec![factor(a.0, &[2, 1]), factor(tau.0, &[1])],
        }],
    );

    let mut rhs = TensorComputation::new();
    let range0 = rhs.add_range(11);
    let range1 = rhs.add_range(11);
    let b = rhs.add_tensor(vec![]);
    let a = rhs.add_tensor(vec![]);
    let _tau = rhs.add_tensor(vec![]);
    let r = rhs.add_tensor(vec![]);

    rhs.add_definition(
        r,
        vec![idx(2, range0.0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![idx(1, range1.0), idx(3, range1.0)],
            factors: vec![factor(a.0, &[2, 1]), factor(b.0, &[1, 3])],
        }],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[r]).unwrap());
}

#[test]
fn equivalent_requires_distinct_dummy_indices_for_reused_intermediate() {
    let mut lhs = TensorComputation::new();
    let range0 = lhs.add_range(11);
    let range1 = lhs.add_range(11);
    let a = lhs.add_tensor(vec![]);
    let tau = lhs.add_tensor(vec![]);
    let r = lhs.add_tensor(vec![]);

    lhs.add_definition(
        tau,
        vec![idx(0, range0.0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![idx(1, range1.0)],
            factors: vec![factor(a.0, &[0, 1])],
        }],
    );
    lhs.add_definition(
        r,
        vec![idx(2, range0.0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(tau.0, &[2]), factor(tau.0, &[2])],
        }],
    );

    let mut rhs = TensorComputation::new();
    let range0 = rhs.add_range(11);
    let range1 = rhs.add_range(11);
    let a = rhs.add_tensor(vec![]);
    let _tau = rhs.add_tensor(vec![]);
    let r = rhs.add_tensor(vec![]);

    rhs.add_definition(
        r,
        vec![idx(2, range0.0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![idx(1, range1.0), idx(3, range1.0)],
            factors: vec![factor(a.0, &[2, 1]), factor(a.0, &[2, 3])],
        }],
    );

    assert!(equivalent_computations(&lhs, &rhs, &[r]).unwrap());
}

#[test]
fn non_equivalent_when_output_definitions_differ() {
    let mut lhs = TensorComputation::new();
    lhs.add_range(3);
    let a = lhs.add_tensor(vec![]);
    let r = lhs.add_tensor(vec![]);
    lhs.add_definition(
        r,
        vec![idx(0, 0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(a.0, &[0])],
        }],
    );

    let mut rhs = TensorComputation::new();
    rhs.add_range(3);
    let b = rhs.add_tensor(vec![]);
    let r = rhs.add_tensor(vec![]);
    rhs.add_definition(
        r,
        vec![idx(0, 0)],
        vec![Term {
            coeff: Rational::new(2, 1),
            sum_indices: vec![],
            factors: vec![factor(b.0, &[0])],
        }],
    );

    assert!(!equivalent_computations(&lhs, &rhs, &[r]).unwrap());
}

#[test]
fn equivalent_when_both_outputs_are_inputs() {
    let mut lhs = TensorComputation::new();
    let _r = lhs.add_tensor(vec![]);
    lhs.add_tensor(vec![]); // other tensor just to keep IDs nontrivial

    let mut rhs = TensorComputation::new();
    let r = rhs.add_tensor(vec![]);
    rhs.add_tensor(vec![]);

    assert!(equivalent_computations(&lhs, &rhs, &[r]).unwrap());
}

#[test]
fn missing_output_in_one_computation_is_error() {
    let mut lhs = TensorComputation::new();
    let range = lhs.add_range(3);
    let a = lhs.add_tensor(vec![]);
    let r = lhs.add_tensor(vec![]);
    lhs.add_definition(
        r,
        vec![idx(0, range.0)],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![factor(a.0, &[0])],
        }],
    );

    let mut rhs = TensorComputation::new();
    rhs.add_range(3);
    rhs.add_tensor(vec![]); // a
    let r = rhs.add_tensor(vec![]); // output is present only as input

    let err = equivalent_computations(&lhs, &rhs, &[r]).unwrap_err();
    assert_eq!(
        err,
        VerifyError::MissingOutputDefinition {
            tensor: r,
            side: "rhs",
        }
    );
}
