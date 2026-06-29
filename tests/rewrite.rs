use gristmill_symbolics::canon::CanonError;
use gristmill_symbolics::graph::GraphError;
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId,
    Term,
};
use gristmill_symbolics::rewrite::{
    action_space_for_def, action_spaces_for_batch, apply_decision,
    apply_decisions_for_batch, validate_decision, validate_decisions_for_batch,
    BatchField, BatchRewriteError, Decision, DecisionSide, RewriteError,
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

fn comp_with_one_term() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);

    comp.add_definition(
        out,
        vec![idx(0)],
        vec![term(vec![], vec![factor(a, &[0])])],
    );
    comp
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

fn comp_with_canon_error() -> TensorComputation {
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
    comp
}

fn comp_with_graph_error() -> TensorComputation {
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
    comp
}

fn first_full_decision(
    space: &gristmill_symbolics::rewrite::ActionSpace,
) -> Decision {
    let template = &space.candidate_templates[0];
    Decision {
        candidate_index: 0,
        left_mask: vec![true; template.left_definition.terms.len()],
        right_mask: vec![true; template.right_definition.terms.len()],
    }
}

#[test]
fn action_space_for_def_returns_none_for_non_actionable_definitions() {
    assert_eq!(action_space_for_def(&comp_with_one_term(), 0), Ok(None));
    assert_eq!(
        action_space_for_def(&comp_with_two_unsplittable_terms(), 0),
        Ok(None)
    );
}

#[test]
fn action_space_for_def_returns_requested_definition_without_scanning() {
    let comp = comp_with_unsplittable_then_actionable_definition();

    assert_eq!(action_space_for_def(&comp, 0), Ok(None));
    let space = action_space_for_def(&comp, 1).unwrap().unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}

#[test]
fn action_space_for_def_rejects_out_of_range_definition_index() {
    let comp = comp_with_shared_left_candidate();

    assert_eq!(
        action_space_for_def(&comp, 7),
        Err(RewriteError::DefinitionIndexOutOfRange { index: 7, len: 1 })
    );
}

#[test]
fn action_space_for_def_propagates_split_errors() {
    assert_eq!(
        action_space_for_def(&comp_with_split_error(), 0),
        Err(RewriteError::Split(SplitError::TooManyFactors {
            len: 65,
            max: 64,
        }))
    );
}

#[test]
fn action_space_for_def_propagates_canon_errors() {
    assert_eq!(
        action_space_for_def(&comp_with_canon_error(), 0),
        Err(RewriteError::Canon(CanonError::MissingTensorSymmetry {
            tensor: TensorId(99),
        }))
    );
}

#[test]
fn action_space_for_def_propagates_graph_errors() {
    assert_eq!(
        action_space_for_def(&comp_with_graph_error(), 0),
        Err(RewriteError::Graph(GraphError::TooManyTerms {
            len: 65,
            max: 64,
        }))
    );
}

#[test]
fn validate_decision_rejects_candidate_index_out_of_range() {
    let space = action_space_for_def(&comp_with_shared_left_candidate(), 0)
        .unwrap()
        .unwrap();
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
}

#[test]
fn validate_decision_rejects_mask_length_mismatch() {
    let space = action_space_for_def(&comp_with_shared_left_candidate(), 0)
        .unwrap()
        .unwrap();
    let template = &space.candidate_templates[0];

    assert_eq!(
        validate_decision(
            &space,
            &Decision {
                candidate_index: 0,
                left_mask: vec![],
                right_mask: vec![true; template.right_definition.terms.len()],
            },
        ),
        Err(RewriteError::MaskLengthMismatch {
            side: DecisionSide::Left,
            expected: template.left_definition.terms.len(),
            got: 0,
        })
    );
}

#[test]
fn validate_decision_rejects_empty_masks() {
    let space = action_space_for_def(&comp_with_shared_left_candidate(), 0)
        .unwrap()
        .unwrap();
    let template = &space.candidate_templates[0];

    assert_eq!(
        validate_decision(
            &space,
            &Decision {
                candidate_index: 0,
                left_mask: vec![true; template.left_definition.terms.len()],
                right_mask: vec![false; template.right_definition.terms.len()],
            },
        ),
        Err(RewriteError::EmptyMask {
            side: DecisionSide::Right,
        })
    );
}

#[test]
fn apply_decision_mutates_computation_with_valid_decision() {
    let mut comp = comp_with_unsplittable_then_actionable_definition();
    let original_tensors = comp.tensors().len();
    let original_definitions = comp.definitions().len();
    let space = action_space_for_def(&comp, 1).unwrap().unwrap();
    let decision = first_full_decision(&space);

    validate_decision(&space, &decision).unwrap();
    apply_decision(&mut comp, &space, &decision).unwrap();

    assert_eq!(comp.tensors().len(), original_tensors + 2);
    assert_eq!(comp.definitions().len(), original_definitions + 2);
    assert_eq!(comp.validate(), Ok(()));
}

#[test]
fn apply_decision_is_deterministic_for_cloned_computations() {
    let comp = comp_with_shared_left_candidate();
    let space = action_space_for_def(&comp, 0).unwrap().unwrap();
    let decision = first_full_decision(&space);
    let mut left = comp.clone();
    let mut right = comp;

    validate_decision(&space, &decision).unwrap();
    apply_decision(&mut left, &space, &decision).unwrap();
    apply_decision(&mut right, &space, &decision).unwrap();

    assert_eq!(left, right);
}

#[test]
fn action_spaces_for_batch_matches_single_for_width_one() {
    let comp = comp_with_shared_left_candidate();
    let expected = action_space_for_def(&comp, 0).unwrap();
    let spaces = action_spaces_for_batch(&[comp], &[Some(0)]).unwrap();

    assert_eq!(spaces, vec![expected]);
}

#[test]
fn action_spaces_for_batch_skips_none_targets() {
    let comps = vec![
        comp_with_shared_left_candidate(),
        comp_with_two_unsplittable_terms(),
    ];

    let spaces = action_spaces_for_batch(&comps, &[None, Some(0)]).unwrap();

    assert_eq!(spaces, vec![None, None]);
}

#[test]
fn action_spaces_for_batch_reports_length_mismatch() {
    let comps = vec![
        comp_with_shared_left_candidate(),
        comp_with_shared_left_candidate(),
    ];

    assert_eq!(
        action_spaces_for_batch(&comps, &[Some(0)]),
        Err(BatchRewriteError::LengthMismatch {
            field: BatchField::Targets,
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn action_spaces_for_batch_wraps_single_errors_with_sample() {
    let comps =
        vec![comp_with_two_unsplittable_terms(), comp_with_split_error()];

    assert_eq!(
        action_spaces_for_batch(&comps, &[Some(0), Some(0)]),
        Err(BatchRewriteError::Single {
            sample: 1,
            source: RewriteError::Split(SplitError::TooManyFactors {
                len: 65,
                max: 64,
            }),
        })
    );
}

#[test]
fn validate_decisions_for_batch_accepts_valid_and_skipped_decisions() {
    let comp = comp_with_shared_left_candidate();
    let space = action_space_for_def(&comp, 0).unwrap().unwrap();
    let decision = first_full_decision(&space);
    let spaces = vec![Some(space), None];
    let decisions = vec![Some(decision), None];

    assert_eq!(validate_decisions_for_batch(&spaces, &decisions), Ok(()));
}

#[test]
fn validate_decisions_for_batch_rejects_decision_without_space() {
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![true],
        right_mask: vec![true],
    };

    assert_eq!(
        validate_decisions_for_batch(&[None], &[Some(decision)]),
        Err(BatchRewriteError::DecisionWithoutActionSpace { sample: 0 })
    );
}

#[test]
fn validate_decisions_for_batch_wraps_single_errors_with_sample() {
    let comp = comp_with_shared_left_candidate();
    let space = action_space_for_def(&comp, 0).unwrap().unwrap();
    let len = space.candidate_templates.len();
    let bad = Decision {
        candidate_index: len,
        left_mask: vec![true],
        right_mask: vec![true],
    };

    assert_eq!(
        validate_decisions_for_batch(&[Some(space)], &[Some(bad)]),
        Err(BatchRewriteError::Single {
            sample: 0,
            source: RewriteError::CandidateIndexOutOfRange { index: len, len },
        })
    );
}

#[test]
fn validate_decisions_for_batch_reports_length_mismatch() {
    assert_eq!(
        validate_decisions_for_batch(&[None, None], &[None]),
        Err(BatchRewriteError::LengthMismatch {
            field: BatchField::Decisions,
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn apply_decisions_for_batch_mutates_only_decided_samples() {
    let mut comps = vec![
        comp_with_shared_left_candidate(),
        comp_with_shared_left_candidate(),
        comp_with_two_unsplittable_terms(),
    ];
    let spaces =
        action_spaces_for_batch(&comps, &[Some(0), Some(0), Some(0)]).unwrap();
    let decision = first_full_decision(spaces[0].as_ref().unwrap());
    let before = comps.clone();

    let applied = apply_decisions_for_batch(
        &mut comps,
        &spaces,
        &[Some(decision), None, None],
    )
    .unwrap();

    assert_eq!(applied, vec![true, false, false]);
    assert_ne!(comps[0], before[0]);
    assert_eq!(comps[1], before[1]);
    assert_eq!(comps[2], before[2]);
    assert_eq!(comps[0].validate(), Ok(()));
}

#[test]
fn apply_decisions_for_batch_rejects_decision_without_space() {
    let mut comps = vec![comp_with_shared_left_candidate()];
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![true],
        right_mask: vec![true],
    };

    assert_eq!(
        apply_decisions_for_batch(&mut comps, &[None], &[Some(decision)]),
        Err(BatchRewriteError::DecisionWithoutActionSpace { sample: 0 })
    );
}

#[test]
fn apply_decisions_for_batch_reports_length_mismatches() {
    let mut comps = vec![
        comp_with_shared_left_candidate(),
        comp_with_shared_left_candidate(),
    ];

    assert_eq!(
        apply_decisions_for_batch(&mut comps, &[None], &[None, None]),
        Err(BatchRewriteError::LengthMismatch {
            field: BatchField::Spaces,
            expected: 2,
            got: 1,
        })
    );
    assert_eq!(
        apply_decisions_for_batch(&mut comps, &[None, None], &[None]),
        Err(BatchRewriteError::LengthMismatch {
            field: BatchField::Decisions,
            expected: 2,
            got: 1,
        })
    );
}
