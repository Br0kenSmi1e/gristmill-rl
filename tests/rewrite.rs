use gristmill_symbolics::canon::CanonError;
use gristmill_symbolics::graph::GraphError;
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId, Term,
};
use gristmill_symbolics::rewrite::{
    Decision, Factorization, FactorizationRewrite, RewriteError, apply_rewrite, build_rewrite,
    next_action_space,
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

fn empty_def(base: TensorId) -> gristmill_symbolics::repr::TensorDef {
    gristmill_symbolics::repr::TensorDef {
        base,
        ext_indices: vec![],
        terms: vec![],
    }
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
fn next_action_space_returns_none_when_no_definition_is_actionable() {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);
    comp.add_definition(out, vec![idx(0)], vec![term(vec![], vec![factor(a, &[0])])]);

    assert_eq!(next_action_space(&comp, 0), Ok(None));
}

#[test]
fn next_action_space_returns_first_actionable_definition() {
    let mut comp = comp_with_shared_left_candidate();
    let extra_base = comp.add_tensor(vec![]);
    let skipped = gristmill_symbolics::repr::TensorDef {
        base: extra_base,
        ext_indices: vec![idx(0)],
        terms: vec![term(vec![], vec![factor(TensorId(0), &[0])])],
    };
    comp.definitions_mut().insert(0, skipped);

    let space = next_action_space(&comp, 0).unwrap().unwrap();

    assert_eq!(space.def_index, 1);
    assert!(!space.candidate_templates.is_empty());
}

#[test]
fn next_action_space_propagates_split_errors() {
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

    assert_eq!(
        next_action_space(&comp, 0),
        Err(RewriteError::Split(SplitError::TooManyFactors {
            len: 65,
            max: 64,
        }))
    );
}

#[test]
fn next_action_space_propagates_canon_errors() {
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

    assert_eq!(
        next_action_space(&comp, 0),
        Err(RewriteError::Canon(CanonError::MissingTensorSymmetry {
            tensor: missing,
        }))
    );
}

#[test]
fn next_action_space_propagates_graph_errors() {
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

    assert_eq!(
        next_action_space(&comp, 0),
        Err(RewriteError::Graph(GraphError::TooManyTerms {
            len: 65,
            max: 64,
        }))
    );
}

#[test]
fn apply_rewrite_registers_tensors_inserts_definitions_and_validates() {
    let mut comp = comp_with_shared_left_candidate();
    let original_tensors = comp.tensors().len();
    let original_definitions = comp.definitions().len();
    let space = next_action_space(&comp, 0).unwrap().unwrap();
    let decision = first_full_decision(&space);
    let rewrite = build_rewrite(&comp, &space, &decision).unwrap();
    let def_index = rewrite.def_index;
    let left_base = rewrite.factorization.left_definition.base;
    let right_base = rewrite.factorization.right_definition.base;
    let rewritten_base = rewrite.factorization.rewritten_definition.base;

    apply_rewrite(&mut comp, rewrite).unwrap();

    assert_eq!(comp.tensors().len(), original_tensors + 2);
    assert_eq!(comp.definitions().len(), original_definitions + 2);
    assert_eq!(comp.definitions()[def_index].base, left_base);
    assert_eq!(comp.definitions()[def_index + 1].base, right_base);
    assert_eq!(comp.definitions()[def_index + 2].base, rewritten_base);
    assert_eq!(comp.validate(), Ok(()));
}

#[test]
fn apply_rewrite_rejects_out_of_range_definition_index_before_mutation() {
    let mut comp = TensorComputation::new();
    let rewrite = FactorizationRewrite {
        def_index: 7,
        factorization: Factorization {
            left_definition: empty_def(TensorId(0)),
            right_definition: empty_def(TensorId(1)),
            rewritten_definition: empty_def(TensorId(2)),
        },
    };

    assert_eq!(
        apply_rewrite(&mut comp, rewrite),
        Err(RewriteError::DefinitionIndexOutOfRange { index: 7, len: 0 })
    );
    assert_eq!(comp.tensors().len(), 0);
    assert_eq!(comp.definitions().len(), 0);
}

#[test]
fn apply_rewrite_only_checks_definition_index_after_rewrite_construction() {
    let mut comp = comp_with_shared_left_candidate();
    let space = next_action_space(&comp, 0).unwrap().unwrap();
    let decision = first_full_decision(&space);
    let rewrite = build_rewrite(&comp, &space, &decision).unwrap();

    comp.definitions_mut()[rewrite.def_index].terms.clear();

    assert_eq!(apply_rewrite(&mut comp, rewrite), Ok(()));
}
