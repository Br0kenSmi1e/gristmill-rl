use gristmill_symbolics::canon::CanonError;
use gristmill_symbolics::graph::GraphError;
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId, Term,
};
use gristmill_symbolics::rewrite::{
    ActionSpaceEntry, Decision, RewriteError, RewriteState, RewriteStateRow, validate_decision,
};
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

fn comp_with_split_error() -> TensorComputation {
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

fn computation_snapshots(row: &RewriteStateRow) -> Vec<TensorComputation> {
    row.states()
        .iter()
        .map(|state| state.computation().clone())
        .collect()
}

#[test]
fn rewrite_state_row_preserves_length_masks_and_sample_order() {
    let left = RewriteState::new(comp_with_shared_left_candidate());
    let middle = RewriteState::new(comp_with_two_unsplittable_terms());
    let right = RewriteState::new(comp_with_shared_left_candidate());

    let row = RewriteStateRow::from_states(vec![left.clone(), middle.clone(), right.clone()]);

    assert_eq!(row.len(), 3);
    assert!(!row.is_empty());
    assert_eq!(
        row.definition_masks(),
        vec![
            left.definition_mask().to_vec(),
            middle.definition_mask().to_vec(),
            right.definition_mask().to_vec(),
        ]
    );
    assert_eq!(row.states()[0].computation(), left.computation());
    assert_eq!(row.states()[1].computation(), middle.computation());
    assert_eq!(row.states()[2].computation(), right.computation());
}

#[test]
fn row_query_width_one_matches_scalar_action_space_for_def() {
    let comp = comp_with_shared_left_candidate();
    let mut scalar = RewriteState::new(comp.clone());
    let scalar_space = scalar.action_space_for_def(0).unwrap().unwrap();
    let mut row = RewriteStateRow::from_states(vec![RewriteState::new(comp)]);

    let spaces = row.query_action_spaces_for_row(&[0], &[true]).unwrap();

    assert_eq!(spaces.len(), 1);
    assert_eq!(spaces.entry_kinds(), vec!["non_empty"]);
    assert_eq!(
        spaces.entries(),
        &[ActionSpaceEntry::NonEmpty(scalar_space)]
    );
}

#[test]
fn row_query_skips_stop_and_inactive_entries_without_mutating_them() {
    let stop = RewriteState::new(comp_with_shared_left_candidate());
    let inactive = RewriteState::new(comp_with_two_unsplittable_terms());
    let active = RewriteState::new(comp_with_shared_left_candidate());
    let mut row = RewriteStateRow::from_states(vec![stop, inactive, active]);
    let before = computation_snapshots(&row);

    let spaces = row
        .query_action_spaces_for_row(&[-1, 0, 0], &[true, false, true])
        .unwrap();

    assert_eq!(
        spaces.entry_kinds(),
        vec!["skipped", "skipped", "non_empty"]
    );
    assert_eq!(row.states()[0].computation(), &before[0]);
    assert_eq!(row.states()[1].computation(), &before[1]);
}

#[test]
fn row_query_exact_empty_refines_only_owning_state_mask() {
    let exact_empty = RewriteState::new(comp_with_two_unsplittable_terms());
    let actionable = RewriteState::new(comp_with_shared_left_candidate());
    let mut row = RewriteStateRow::from_states(vec![exact_empty, actionable]);

    let spaces = row
        .query_action_spaces_for_row(&[0, 0], &[true, true])
        .unwrap();

    assert_eq!(spaces.entry_kinds(), vec!["exact_empty", "non_empty"]);
    assert_eq!(row.definition_masks()[0], vec![false]);
    assert_eq!(row.definition_masks()[1], vec![true]);
}

#[test]
fn row_query_wraps_scalar_errors_and_leaves_row_state_unchanged() {
    let mut row = RewriteStateRow::from_states(vec![
        RewriteState::new(comp_with_two_unsplittable_terms()),
        RewriteState::new(comp_with_split_error()),
    ]);
    let before_masks = row.definition_masks();
    let before_computations = computation_snapshots(&row);

    assert_eq!(
        row.query_action_spaces_for_row(&[0, 0], &[true, true]),
        Err(RewriteError::RowQueryFailed {
            sample: 1,
            source: Box::new(RewriteError::Split(SplitError::TooManyFactors {
                len: 65,
                max: 64,
            })),
        })
    );
    assert_eq!(row.definition_masks(), before_masks);
    assert_eq!(computation_snapshots(&row), before_computations);
}

#[test]
fn row_query_rejects_length_mismatches_and_masked_definitions() {
    let mut row = RewriteStateRow::from_states(vec![
        RewriteState::new(comp_with_shared_left_candidate()),
        RewriteState::new(comp_with_shared_left_candidate()),
    ]);

    assert_eq!(
        row.query_action_spaces_for_row(&[0], &[true, true]),
        Err(RewriteError::RowLengthMismatch {
            operation: "query_action_spaces_for_row",
            field: "target_choices",
            expected: 2,
            got: 1,
        })
    );

    assert_eq!(
        row.query_action_spaces_for_row(&[0, 0], &[true]),
        Err(RewriteError::RowLengthMismatch {
            operation: "query_action_spaces_for_row",
            field: "active_mask",
            expected: 2,
            got: 1,
        })
    );

    let mut masked = RewriteStateRow::from_states(vec![RewriteState::new({
        let mut comp = TensorComputation::new();
        comp.add_range(8);
        let a = comp.add_tensor(vec![]);
        let out = comp.add_tensor(vec![]);
        comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);
        comp
    })]);

    assert_eq!(
        masked.query_action_spaces_for_row(&[0], &[true]),
        Err(RewriteError::TargetDefinitionMasked {
            sample: 0,
            index: 0,
        })
    );
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
