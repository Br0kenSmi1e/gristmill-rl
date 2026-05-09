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
