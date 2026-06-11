use gristmill_symbolics::canon::CanonError;
use gristmill_symbolics::graph::GraphError;
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId, Term,
};
use gristmill_symbolics::rewrite::{Decision, RewriteError, RewriteState, validate_decision};
use gristmill_symbolics::split::SplitError;

fn one() -> Rational {
    Rational::new(1, 1)
}

fn idx(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(0),
    }
}

fn factor(tensor: TensorId, indices: &[u32]) -> Factor {
    Factor {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn term(sum_indices: Vec<Index>, factors: Vec<Factor>) -> Term {
    Term {
        coeff: one(),
        sum_indices,
        factors,
    }
}

fn comp_with_shared_left_candidate() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let b = comp.add_tensor(vec![]);
    let c = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);

    comp.add_definition(
        out,
        vec![idx(0), idx(1)],
        vec![
            term(vec![idx(2)], vec![factor(a, &[0, 2]), factor(b, &[2, 1])]),
            term(vec![idx(3)], vec![factor(a, &[0, 3]), factor(c, &[3, 1])]),
        ],
    );

    comp
}

fn comp_with_two_unsplittable_terms() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);

    comp.add_definition(
        out,
        vec![idx(0)],
        vec![
            term(vec![], vec![factor(a, &[0])]),
            term(vec![], vec![factor(a, &[0])]),
        ],
    );

    comp
}

fn comp_with_unsplittable_then_actionable_definition() -> TensorComputation {
    let mut comp = comp_with_shared_left_candidate();
    let extra_base = comp.add_tensor(vec![]);
    comp.definitions_mut().insert(
        0,
        gristmill_symbolics::repr::TensorDef {
            base: extra_base,
            ext_indices: vec![idx(0)],
            terms: vec![
                term(vec![], vec![factor(TensorId(0), &[0])]),
                term(vec![], vec![factor(TensorId(0), &[0])]),
            ],
        },
    );
    comp
}

fn first_full_decision(space: &gristmill_symbolics::rewrite::ActionSpace) -> Decision {
    let template = &space.candidate_templates[0];
    Decision {
        candidate_index: 0,
        left_mask: vec![true; template.left_definition.terms.len()],
        right_mask: vec![true; template.right_definition.terms.len()],
    }
}

#[test]
fn public_validate_decision_rejects_the_same_decision_errors() {
    let mut state = RewriteState::new(comp_with_shared_left_candidate());
    let space = state.action_space_for_def(0).unwrap().unwrap();
    let template = &space.candidate_templates[0];

    assert_eq!(
        validate_decision(
            &space,
            &Decision {
                candidate_index: space.candidate_templates.len(),
                left_mask: vec![true; template.left_definition.terms.len()],
                right_mask: vec![true; template.right_definition.terms.len()],
            },
        ),
        Err(RewriteError::CandidateIndexOutOfRange {
            index: space.candidate_templates.len(),
            len: space.candidate_templates.len(),
        })
    );

    assert_eq!(
        validate_decision(
            &space,
            &Decision {
                candidate_index: 0,
                left_mask: vec![],
                right_mask: vec![true; template.right_definition.terms.len()],
            },
        ),
        Err(RewriteError::LeftMaskLengthMismatch {
            expected: template.left_definition.terms.len(),
            got: 0,
        })
    );
}

#[test]
fn apply_validated_decision_preserves_caller_owned_action_space_provenance() {
    let comp = comp_with_shared_left_candidate();
    let mut source_state = RewriteState::new(comp.clone());
    let space = source_state.action_space_for_def(0).unwrap().unwrap();
    let decision = first_full_decision(&space);
    validate_decision(&space, &decision).unwrap();

    let mut left = RewriteState::new(comp.clone());
    let mut right = RewriteState::new(comp);

    left.apply_validated_decision(&space, &decision).unwrap();
    right.apply_validated_decision(&space, &decision).unwrap();

    assert_eq!(left.computation(), right.computation());
    assert_eq!(left.definition_mask(), right.definition_mask());
}

#[test]
fn validate_decision_rejects_invalid_decision_before_state_mutation() {
    let mut state = RewriteState::new(comp_with_shared_left_candidate());
    let space = state.action_space_for_def(0).unwrap().unwrap();
    let before = state.computation().clone();
    let bad_decision = Decision {
        candidate_index: 0,
        left_mask: vec![],
        right_mask: vec![true],
    };

    assert!(validate_decision(&space, &bad_decision).is_err());
    assert_eq!(state.computation(), &before);
}

#[test]
fn rewrite_state_initializes_definition_mask_from_term_count() {
    let basic = {
        let mut comp = TensorComputation::new();
        comp.add_range(8);
        let a = comp.add_tensor(vec![]);
        let out = comp.add_tensor(vec![]);
        comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);
        comp
    };
    let exact_empty = comp_with_two_unsplittable_terms();
    let actionable = comp_with_shared_left_candidate();

    assert_eq!(RewriteState::new(basic).definition_mask(), &[false]);
    assert_eq!(RewriteState::new(exact_empty).definition_mask(), &[true]);
    assert_eq!(RewriteState::new(actionable).definition_mask(), &[true]);
}

#[test]
fn action_space_for_def_returns_none_for_false_mask_entry() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);
    let mut state = RewriteState::new(comp);

    assert_eq!(state.definition_mask(), &[false]);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    assert_eq!(state.definition_mask(), &[false]);
}

#[test]
fn action_space_for_def_refines_exact_empty_definition_to_false() {
    let mut state = RewriteState::new(comp_with_two_unsplittable_terms());

    assert_eq!(state.definition_mask(), &[true]);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    assert_eq!(state.definition_mask(), &[false]);
}

#[test]
fn action_space_for_def_returns_requested_definition_without_scanning() {
    let mut state = RewriteState::new(comp_with_unsplittable_then_actionable_definition());

    assert_eq!(state.definition_mask(), &[true, true]);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    assert_eq!(state.definition_mask(), &[false, true]);

    let space = state.action_space_for_def(1).unwrap().unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}

#[test]
fn action_space_for_def_rejects_out_of_range_definition_index() {
    let mut state = RewriteState::new(comp_with_shared_left_candidate());

    assert_eq!(
        state.action_space_for_def(7),
        Err(RewriteError::DefinitionIndexOutOfRange { index: 7, len: 1 })
    );
}

#[test]
fn rewrite_state_apply_validated_decision_mutates_computation_and_updates_mask() {
    let original = comp_with_unsplittable_then_actionable_definition();
    let original_tensors = original.tensors().len();
    let original_definitions = original.definitions().len();
    let mut state = RewriteState::new(original);
    assert_eq!(state.action_space_for_def(0), Ok(None));
    let space = state.action_space_for_def(1).unwrap().unwrap();
    let decision = first_full_decision(&space);

    validate_decision(&space, &decision).unwrap();
    state.apply_validated_decision(&space, &decision).unwrap();

    assert_eq!(state.computation().tensors().len(), original_tensors + 2);
    assert_eq!(
        state.computation().definitions().len(),
        original_definitions + 2
    );
    assert_eq!(state.computation().validate(), Ok(()));
    assert_eq!(
        state.definition_mask().len(),
        state.computation().definitions().len()
    );
    assert!(!state.definition_mask()[0]);
}

#[test]
fn rewrite_state_returns_none_when_no_definition_is_actionable() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);
    let mut state = RewriteState::new(comp);

    assert_eq!(state.action_space_for_def(0), Ok(None));
}

#[test]
fn rewrite_state_returns_action_space_for_selected_definition() {
    let mut state = RewriteState::new(comp_with_unsplittable_then_actionable_definition());

    assert_eq!(state.action_space_for_def(0), Ok(None));
    let space = state.action_space_for_def(1).unwrap().unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}

#[test]
fn action_space_for_def_propagates_split_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    let many_factors = (0..65).map(|_| factor(a, &[])).collect();

    comp.add_definition(
        out,
        vec![],
        vec![
            term(vec![], many_factors),
            term(vec![], vec![factor(a, &[])]),
        ],
    );
    let mut state = RewriteState::new(comp);

    assert_eq!(
        state.action_space_for_def(0),
        Err(RewriteError::Split(SplitError::TooManyFactors {
            len: 65,
            max: 64,
        }))
    );
}

#[test]
fn action_space_for_def_propagates_canon_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let out = comp.add_tensor(vec![]);
    let missing = TensorId(99);

    comp.add_definition(
        out,
        vec![],
        vec![
            term(vec![], vec![factor(missing, &[]), factor(missing, &[])]),
            term(vec![], vec![factor(missing, &[]), factor(missing, &[])]),
        ],
    );
    let mut state = RewriteState::new(comp);

    assert_eq!(
        state.action_space_for_def(0),
        Err(RewriteError::Canon(CanonError::MissingTensorSymmetry {
            tensor: missing,
        }))
    );
}

#[test]
fn action_space_for_def_propagates_graph_errors() {
    let mut comp = TensorComputation::new();
    comp.add_range(128);
    let a = comp.add_tensor(vec![]);
    let b = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    let terms: Vec<_> = (0..65)
        .map(|offset| {
            let sum_id = 2 + offset;
            term(
                vec![idx(sum_id)],
                vec![factor(a, &[0, sum_id]), factor(b, &[sum_id, 1])],
            )
        })
        .collect();

    comp.add_definition(out, vec![idx(0), idx(1)], terms);
    let mut state = RewriteState::new(comp);

    assert_eq!(
        state.action_space_for_def(0),
        Err(RewriteError::Graph(GraphError::TooManyTerms {
            len: 65,
            max: 64,
        }))
    );
}
