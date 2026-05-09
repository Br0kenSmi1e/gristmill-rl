use gristmill_symbolics::repr::{Factor, Index, RangeId};
use gristmill_symbolics::repr::{IndexId, Rational, TensorDef, TensorId, Term};
use gristmill_symbolics::split::SplitError;
use gristmill_symbolics::split::enumerate_splits;
use gristmill_symbolics::split::{Split, SplitInterface};

fn one() -> Rational {
    Rational::new(1, 1)
}

fn factor_with_index(index: IndexId) -> Factor {
    Factor {
        tensor: TensorId(0),
        indices: vec![index],
    }
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

#[test]
fn three_factor_chain_emits_public_unordered_bipartitions() {
    let range = RangeId(0);
    let a = Index {
        id: IndexId(10),
        range,
    };
    let b = Index {
        id: IndexId(0),
        range,
    };
    let c = Index {
        id: IndexId(30),
        range,
    };
    let d = Index {
        id: IndexId(20),
        range,
    };
    let x = TensorId(0);
    let y = TensorId(1);
    let z = TensorId(2);

    let def = TensorDef {
        base: TensorId(3),
        ext_indices: vec![a, b],
        terms: vec![],
    };
    let term = Term {
        coeff: Rational::new(-5, 2),
        sum_indices: vec![d, c],
        factors: vec![
            Factor {
                tensor: x,
                indices: vec![a.id, c.id],
            },
            Factor {
                tensor: y,
                indices: vec![c.id, d.id],
            },
            Factor {
                tensor: z,
                indices: vec![d.id, b.id],
            },
        ],
    };

    let splits = enumerate_splits(&term, &def).unwrap();
    assert_eq!(splits.len(), 3);

    assert_eq!(splits[0].interface.left_external, vec![a]);
    assert_eq!(splits[0].interface.right_external, vec![b]);
    assert_eq!(splits[0].interface.contracted, vec![c]);
    assert_eq!(splits[0].left.factors, vec![term.factors[0].clone()]);
    assert_eq!(
        splits[0].right.factors,
        vec![term.factors[1].clone(), term.factors[2].clone()]
    );
    assert_eq!(splits[0].left.sum_indices, vec![]);
    assert_eq!(splits[0].right.sum_indices, vec![d]);

    assert_eq!(splits[1].interface.left_external, vec![]);
    assert_eq!(splits[1].interface.right_external, vec![b, a]);
    assert_eq!(splits[1].interface.contracted, vec![d, c]);
    assert_eq!(splits[1].left.factors, vec![term.factors[1].clone()]);
    assert_eq!(
        splits[1].right.factors,
        vec![term.factors[0].clone(), term.factors[2].clone()]
    );
    assert_eq!(splits[1].left.sum_indices, vec![]);
    assert_eq!(splits[1].right.sum_indices, vec![]);

    assert_eq!(splits[2].interface.left_external, vec![a]);
    assert_eq!(splits[2].interface.right_external, vec![b]);
    assert_eq!(splits[2].interface.contracted, vec![d]);
    assert_eq!(
        splits[2].left.factors,
        vec![term.factors[0].clone(), term.factors[1].clone()]
    );
    assert_eq!(splits[2].right.factors, vec![term.factors[2].clone()]);
    assert_eq!(splits[2].left.sum_indices, vec![c]);
    assert_eq!(splits[2].right.sum_indices, vec![]);

    for split in splits {
        assert_eq!(split.left.coeff, one());
        assert_eq!(split.right.coeff, one());
    }
}

#[test]
fn too_many_factors_returns_split_error() {
    let range = RangeId(0);
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: vec![Index {
            id: IndexId(0),
            range,
        }],
        terms: vec![],
    };
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: (0..65).map(|_| factor_with_index(IndexId(0))).collect(),
    };

    assert_eq!(
        enumerate_splits(&term, &def),
        Err(SplitError::TooManyFactors { len: 65, max: 64 })
    );
}

#[test]
fn too_many_sum_indices_returns_split_error() {
    let range = RangeId(0);
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: vec![],
        terms: vec![],
    };
    let term = Term {
        coeff: one(),
        sum_indices: (0..65)
            .map(|id| Index {
                id: IndexId(id),
                range,
            })
            .collect(),
        factors: vec![factor_with_index(IndexId(0)), factor_with_index(IndexId(1))],
    };

    assert_eq!(
        enumerate_splits(&term, &def),
        Err(SplitError::TooManySumIndices { len: 65, max: 64 })
    );
}

#[test]
fn too_many_external_indices_returns_split_error() {
    let range = RangeId(0);
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: (0..65)
            .map(|id| Index {
                id: IndexId(id),
                range,
            })
            .collect(),
        terms: vec![],
    };
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor_with_index(IndexId(0)), factor_with_index(IndexId(1))],
    };

    assert_eq!(
        enumerate_splits(&term, &def),
        Err(SplitError::TooManyExternalIndices { len: 65, max: 64 })
    );
}
