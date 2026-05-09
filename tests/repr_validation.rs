use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, ReprError, SymAction, SymGenerator, TensorComputation,
    TensorId, Term,
};
use serde_json::json;

fn one() -> num::rational::Ratio<i64> {
    num::rational::Ratio::new(1, 1)
}

fn well_formed_computation() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![SymGenerator {
        perm: vec![0],
        action: SymAction::Identity,
    }]);

    comp.add_definition(
        tensor_id,
        vec![Index {
            id: IndexId(0),
            range: range_id,
        }],
        vec![Term {
            coeff: one(),
            sum_indices: vec![],
            factors: vec![Factor {
                tensor: tensor_id,
                indices: vec![IndexId(0)],
            }],
        }],
    );

    comp
}

#[test]
fn validate_accepts_a_well_formed_computation() {
    well_formed_computation().validate().unwrap();
}

#[test]
fn validate_rejects_id_position_mismatches() {
    let range_bad: TensorComputation = serde_json::from_value(json!({
        "ranges": [{ "id": 7, "size": 3 }],
        "tensors": [],
        "definitions": []
    }))
    .unwrap();
    assert_eq!(
        range_bad.validate(),
        Err(ReprError::RangeIdMismatch {
            position: 0,
            found: RangeId(7),
        })
    );

    let tensor_bad: TensorComputation = serde_json::from_value(json!({
        "ranges": [],
        "tensors": [{ "id": 2, "symmetry": [] }],
        "definitions": []
    }))
    .unwrap();
    assert_eq!(
        tensor_bad.validate(),
        Err(ReprError::TensorIdMismatch {
            position: 0,
            found: TensorId(2),
        })
    );
}

#[test]
fn validate_rejects_unknown_references() {
    let mut unknown_range = well_formed_computation();
    unknown_range.definitions_mut()[0].ext_indices[0].range = RangeId(99);
    assert_eq!(
        unknown_range.validate(),
        Err(ReprError::UnknownRange { range: RangeId(99) })
    );

    let mut unknown_base = well_formed_computation();
    unknown_base.definitions_mut()[0].base = TensorId(99);
    assert_eq!(
        unknown_base.validate(),
        Err(ReprError::UnknownTensor {
            tensor: TensorId(99),
        })
    );

    let mut unknown_factor_tensor = well_formed_computation();
    unknown_factor_tensor.definitions_mut()[0].terms[0].factors[0].tensor = TensorId(99);
    assert_eq!(
        unknown_factor_tensor.validate(),
        Err(ReprError::UnknownTensor {
            tensor: TensorId(99),
        })
    );

    let mut unknown_index = well_formed_computation();
    unknown_index.definitions_mut()[0].terms[0].factors[0].indices = vec![IndexId(99)];
    assert_eq!(
        unknown_index.validate(),
        Err(ReprError::UnknownIndex {
            def_index: 0,
            term_index: 0,
            index: IndexId(99),
        })
    );
}

#[test]
fn validate_rejects_factor_index_declared_only_in_another_term() {
    let mut comp = TensorComputation::new();
    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![]);
    comp.add_definition(
        tensor_id,
        vec![],
        vec![
            Term {
                coeff: one(),
                sum_indices: vec![Index {
                    id: IndexId(1),
                    range: range_id,
                }],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![Factor {
                    tensor: tensor_id,
                    indices: vec![IndexId(1)],
                }],
            },
        ],
    );

    assert_eq!(
        comp.validate(),
        Err(ReprError::UnknownIndex {
            def_index: 0,
            term_index: 1,
            index: IndexId(1),
        })
    );
}

#[test]
fn validate_rejects_inconsistent_index_ranges() {
    let mut comp = TensorComputation::new();
    let range_0 = comp.add_range(3);
    let range_1 = comp.add_range(5);
    let tensor_id = comp.add_tensor(vec![]);
    comp.add_definition(
        tensor_id,
        vec![Index {
            id: IndexId(0),
            range: range_0,
        }],
        vec![Term {
            coeff: one(),
            sum_indices: vec![Index {
                id: IndexId(0),
                range: range_1,
            }],
            factors: vec![],
        }],
    );

    assert_eq!(
        comp.validate(),
        Err(ReprError::InconsistentIndexRange {
            def_index: 0,
            index: IndexId(0),
            first: range_0,
            second: range_1,
        })
    );
}

#[test]
fn validate_rejects_sum_index_range_mismatches_across_terms() {
    let mut comp = TensorComputation::new();
    let range_0 = comp.add_range(3);
    let range_1 = comp.add_range(5);
    let tensor_id = comp.add_tensor(vec![]);
    comp.add_definition(
        tensor_id,
        vec![],
        vec![
            Term {
                coeff: one(),
                sum_indices: vec![Index {
                    id: IndexId(1),
                    range: range_0,
                }],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![Index {
                    id: IndexId(1),
                    range: range_1,
                }],
                factors: vec![],
            },
        ],
    );

    assert_eq!(
        comp.validate(),
        Err(ReprError::InconsistentIndexRange {
            def_index: 0,
            index: IndexId(1),
            first: range_0,
            second: range_1,
        })
    );
}

#[test]
fn validate_rejects_duplicate_index_declarations() {
    let mut duplicate_external = well_formed_computation();
    duplicate_external.definitions_mut()[0]
        .ext_indices
        .push(Index {
            id: IndexId(0),
            range: RangeId(0),
        });
    assert_eq!(
        duplicate_external.validate(),
        Err(ReprError::DuplicateExternalIndex {
            def_index: 0,
            index: IndexId(0),
        })
    );

    let mut duplicate_sum = well_formed_computation();
    duplicate_sum.definitions_mut()[0].terms[0].sum_indices = vec![
        Index {
            id: IndexId(1),
            range: RangeId(0),
        },
        Index {
            id: IndexId(1),
            range: RangeId(0),
        },
    ];
    assert_eq!(
        duplicate_sum.validate(),
        Err(ReprError::DuplicateSumIndex {
            def_index: 0,
            term_index: 0,
            index: IndexId(1),
        })
    );
}

#[test]
fn validate_rejects_external_and_sum_overlap() {
    let mut comp = well_formed_computation();
    let def = &mut comp.definitions_mut()[0];
    def.terms[0].sum_indices = vec![Index {
        id: IndexId(0),
        range: RangeId(0),
    }];

    assert_eq!(
        comp.validate(),
        Err(ReprError::ExternalAndSumIndexOverlap {
            def_index: 0,
            index: IndexId(0),
        })
    );
}

#[test]
fn validate_rejects_invalid_symmetry_permutation() {
    let mut comp = TensorComputation::new();
    comp.add_tensor(vec![SymGenerator {
        perm: vec![0, 0],
        action: SymAction::Identity,
    }]);

    assert_eq!(
        comp.validate(),
        Err(ReprError::InvalidPermutation { perm: vec![0, 0] })
    );
}
