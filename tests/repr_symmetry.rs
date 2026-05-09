use gristmill_symbolics::repr::{ReprError, SymAction, SymGenerator};

#[test]
fn sym_action_combines_signs() {
    assert_eq!(
        SymAction::Identity.combine(SymAction::Identity),
        SymAction::Identity
    );
    assert_eq!(
        SymAction::Identity.combine(SymAction::Negate),
        SymAction::Negate
    );
    assert_eq!(
        SymAction::Negate.combine(SymAction::Identity),
        SymAction::Negate
    );
    assert_eq!(
        SymAction::Negate.combine(SymAction::Negate),
        SymAction::Identity
    );
}

#[test]
fn sym_generator_applies_permutation_and_action() {
    let generator = SymGenerator {
        perm: vec![1, 0],
        action: SymAction::Negate,
    };

    let (indices, action) = generator.apply(&[10, 20]).unwrap();
    assert_eq!(indices, vec![20, 10]);
    assert_eq!(action, SymAction::Negate);
}

#[test]
fn sym_generator_apply_rejects_arity_mismatch() {
    let generator = SymGenerator {
        perm: vec![0, 1],
        action: SymAction::Identity,
    };

    assert_eq!(
        generator.apply(&[7]),
        Err(ReprError::SymmetryArityMismatch {
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn sym_generator_apply_rejects_invalid_permutation() {
    let generator = SymGenerator {
        perm: vec![0, 0],
        action: SymAction::Identity,
    };

    assert_eq!(
        generator.apply(&[7, 8]),
        Err(ReprError::InvalidPermutation { perm: vec![0, 0] })
    );
}
