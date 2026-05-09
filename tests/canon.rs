use gristmill_symbolics::canon::{
    CanonError, build_index_pool, build_tensor_symmetry_map, canon_term,
};
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, SymAction, SymGenerator, TensorDef, TensorId,
    TensorInfo, Term,
};

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
fn build_index_pool_groups_sorts_and_deduplicates_sum_indices() {
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![idx(0, 0)],
        terms: vec![
            Term {
                coeff: one(),
                sum_indices: vec![idx(5, 1), idx(2, 0)],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![idx(2, 0), idx(4, 1), idx(0, 0), idx(3, 0)],
                factors: vec![],
            },
        ],
    };

    let pool = build_index_pool(&def);

    assert_eq!(
        pool.get(&RangeId(0)).unwrap(),
        &vec![IndexId(2), IndexId(3)]
    );
    assert_eq!(
        pool.get(&RangeId(1)).unwrap(),
        &vec![IndexId(4), IndexId(5)]
    );
    assert!(!pool.values().any(|ids| ids.contains(&IndexId(0))));
}

#[test]
fn build_tensor_symmetry_map_indexes_by_tensor_id_and_preserves_order() {
    let first = SymGenerator {
        perm: vec![1, 0],
        action: SymAction::Negate,
    };
    let second = SymGenerator {
        perm: vec![0, 1],
        action: SymAction::Identity,
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(7),
            symmetry: vec![first.clone(), second.clone()],
        },
        TensorInfo {
            id: TensorId(3),
            symmetry: vec![],
        },
    ];

    let symmetry = build_tensor_symmetry_map(&tensors);

    assert_eq!(symmetry.get(&TensorId(7)).unwrap(), &vec![first, second]);
    assert_eq!(
        symmetry.get(&TensorId(3)).unwrap(),
        &Vec::<SymGenerator>::new()
    );
}

#[test]
fn canon_term_reports_missing_tensor_symmetry() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(9, &[0])],
    };

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&[]),
            &build_index_pool(&TensorDef {
                base: TensorId(0),
                ext_indices: vec![],
                terms: vec![term.clone()],
            })
        ),
        Err(CanonError::MissingTensorSymmetry {
            tensor: TensorId(9),
        })
    );
}

#[test]
fn canon_term_applies_factor_symmetry_and_negates_coefficient() {
    let term = Term {
        coeff: Rational::new(3, 1),
        sum_indices: vec![],
        factors: vec![factor(0, &[2, 1])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Negate,
        }],
    }];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&TensorDef {
            base: TensorId(0),
            ext_indices: vec![idx(1, 0), idx(2, 0)],
            terms: vec![term.clone()],
        }),
    )
    .unwrap();

    assert_eq!(
        canonical,
        Term {
            coeff: Rational::new(-3, 1),
            sum_indices: vec![],
            factors: vec![factor(0, &[1, 2])],
        }
    );
}

#[test]
fn canon_term_reports_symmetry_arity_mismatch() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(0, &[0])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Identity,
        }],
    }];

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&tensors),
            &build_index_pool(&TensorDef {
                base: TensorId(0),
                ext_indices: vec![idx(0, 0)],
                terms: vec![term.clone()],
            }),
        ),
        Err(CanonError::SymmetryArityMismatch {
            tensor: TensorId(0),
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn canon_term_reports_invalid_symmetry_permutation() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(0, &[0, 1])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![0, 0],
            action: SymAction::Identity,
        }],
    }];

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&tensors),
            &build_index_pool(&TensorDef {
                base: TensorId(0),
                ext_indices: vec![idx(0, 0), idx(1, 0)],
                terms: vec![term.clone()],
            }),
        ),
        Err(CanonError::InvalidSymmetryPermutation {
            tensor: TensorId(0),
            perm: vec![0, 0],
        })
    );
}

#[test]
fn canon_term_normalizes_dummy_names_and_sum_index_order() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![idx(8, 0), idx(4, 0)],
        factors: vec![factor(0, &[8, 4]), factor(1, &[4, 8])],
    };
    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![],
        terms: vec![term.clone()],
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(0),
            symmetry: vec![],
        },
        TensorInfo {
            id: TensorId(1),
            symmetry: vec![],
        },
    ];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&def),
    )
    .unwrap();

    assert_eq!(
        canonical,
        Term {
            coeff: one(),
            sum_indices: vec![idx(4, 0), idx(8, 0)],
            factors: vec![factor(0, &[4, 8]), factor(1, &[8, 4])],
        }
    );
}

#[test]
fn canon_term_orders_factors_but_preserves_external_id_distinctions() {
    let a = idx(1, 0);
    let b = idx(2, 0);
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(1, &[b.id.0]), factor(0, &[a.id.0])],
    };
    let def = TensorDef {
        base: TensorId(2),
        ext_indices: vec![a, b],
        terms: vec![term.clone()],
    };
    let tensors = vec![
        TensorInfo {
            id: TensorId(0),
            symmetry: vec![],
        },
        TensorInfo {
            id: TensorId(1),
            symmetry: vec![],
        },
    ];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&def),
    )
    .unwrap();

    assert_eq!(
        canonical.factors,
        vec![factor(0, &[a.id.0]), factor(1, &[b.id.0])]
    );
}

#[test]
fn canon_term_is_deterministic_for_tied_factor_groups() {
    let term_a = Term {
        coeff: one(),
        sum_indices: vec![idx(10, 0), idx(11, 0), idx(12, 0)],
        factors: vec![
            factor(0, &[11, 10]),
            factor(0, &[12, 11]),
            factor(0, &[10, 12]),
        ],
    };
    let term_b = Term {
        coeff: one(),
        sum_indices: vec![idx(12, 0), idx(10, 0), idx(11, 0)],
        factors: vec![
            factor(0, &[10, 12]),
            factor(0, &[11, 10]),
            factor(0, &[12, 11]),
        ],
    };
    let def = TensorDef {
        base: TensorId(1),
        ext_indices: vec![],
        terms: vec![term_a.clone(), term_b.clone()],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![],
    }];
    let symmetry = build_tensor_symmetry_map(&tensors);
    let pool = build_index_pool(&def);

    assert_eq!(
        canon_term(&term_a, &symmetry, &pool).unwrap(),
        canon_term(&term_b, &symmetry, &pool).unwrap()
    );
}

#[test]
fn canon_term_reports_missing_index_pool() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![idx(10, 4)],
        factors: vec![factor(0, &[10])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![],
    }];

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&tensors),
            &Default::default()
        ),
        Err(CanonError::MissingIndexPool { range: RangeId(4) })
    );
}

#[test]
fn canon_term_reports_missing_index_pool_for_unused_dummy() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![idx(10, 4)],
        factors: vec![],
    };

    assert_eq!(
        canon_term(&term, &build_tensor_symmetry_map(&[]), &Default::default()),
        Err(CanonError::MissingIndexPool { range: RangeId(4) })
    );
}

#[test]
fn canon_term_selects_representative_by_structure_not_coefficient() {
    let term = Term {
        coeff: Rational::new(5, 1),
        sum_indices: vec![],
        factors: vec![factor(0, &[1, 2])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Negate,
        }],
    }];

    let canonical = canon_term(
        &term,
        &build_tensor_symmetry_map(&tensors),
        &build_index_pool(&TensorDef {
            base: TensorId(0),
            ext_indices: vec![idx(1, 0), idx(2, 0)],
            terms: vec![term.clone()],
        }),
    )
    .unwrap();

    assert_eq!(canonical.coeff, Rational::new(5, 1));
    assert_eq!(canonical.factors, vec![factor(0, &[1, 2])]);
}

#[test]
fn canon_term_reports_inconsistent_symmetry_coefficient() {
    let term = Term {
        coeff: one(),
        sum_indices: vec![],
        factors: vec![factor(0, &[1, 1])],
    };
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Negate,
        }],
    }];

    assert_eq!(
        canon_term(
            &term,
            &build_tensor_symmetry_map(&tensors),
            &build_index_pool(&TensorDef {
                base: TensorId(0),
                ext_indices: vec![idx(1, 0)],
                terms: vec![term.clone()],
            }),
        ),
        Err(CanonError::InconsistentSymmetryCoefficient)
    );
}
