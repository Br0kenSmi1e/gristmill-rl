use crate::biclique::Biclique;
use crate::canon::CanonError;
use crate::graph::{ConstrGraph, GraphError};
use crate::repr::{
    Factor, Index, Rational, TensorComputation, TensorDef, TensorId, Term,
};
use crate::split::{Split, SplitError};
use crate::{biclique, canon, graph, split};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Factorization {
    pub left_definition: TensorDef,
    pub right_definition: TensorDef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpace {
    pub def_index: usize,
    pub candidate_templates: Vec<Factorization>,
    candidate_graphs: Vec<ConstrGraph>,
    candidate_bicliques: Vec<Biclique>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub candidate_index: usize,
    pub left_mask: Vec<bool>,
    pub right_mask: Vec<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionSide {
    Left,
    Right,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    Split(SplitError),
    Canon(CanonError),
    Graph(GraphError),
    CandidateIndexOutOfRange {
        index: usize,
        len: usize,
    },
    MaskLengthMismatch {
        side: DecisionSide,
        expected: usize,
        got: usize,
    },
    EmptyMask {
        side: DecisionSide,
    },
    DefinitionIndexOutOfRange {
        index: usize,
        len: usize,
    },
}

impl From<SplitError> for RewriteError {
    fn from(error: SplitError) -> Self {
        RewriteError::Split(error)
    }
}

impl From<CanonError> for RewriteError {
    fn from(error: CanonError) -> Self {
        RewriteError::Canon(error)
    }
}

impl From<GraphError> for RewriteError {
    fn from(error: GraphError) -> Self {
        RewriteError::Graph(error)
    }
}

pub fn action_space_for_def(
    comp: &TensorComputation,
    def_index: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    let definition = source_definition(comp, def_index)?;
    if definition.terms.len() < 2 {
        return Ok(None);
    }

    let (candidate_graphs, candidate_bicliques) =
        enumerate_candidates(comp, definition)?;
    if candidate_bicliques.is_empty() {
        return Ok(None);
    }

    Ok(Some(build_action_space(
        def_index,
        comp,
        candidate_graphs,
        candidate_bicliques,
    )))
}

fn source_definition(
    comp: &TensorComputation,
    def_index: usize,
) -> Result<&TensorDef, RewriteError> {
    comp.definitions().get(def_index).ok_or(
        RewriteError::DefinitionIndexOutOfRange {
            index: def_index,
            len: comp.definitions().len(),
        },
    )
}

fn build_action_space(
    def_index: usize,
    comp: &TensorComputation,
    raw_graphs: Vec<ConstrGraph>,
    raw_bicliques: Vec<Biclique>,
) -> ActionSpace {
    let (left_tensor, right_tensor) = fresh_rewrite_tensor_ids(comp);
    let mut seen_templates = HashSet::new();
    let mut candidate_templates = Vec::new();
    let mut candidate_graphs = Vec::new();
    let mut candidate_bicliques = Vec::new();

    for (graph, biclique) in raw_graphs.into_iter().zip(raw_bicliques) {
        let template = factorization_template(
            &graph,
            &biclique,
            left_tensor,
            right_tensor,
        );
        if seen_templates.insert(template.clone()) {
            candidate_templates.push(template);
            candidate_graphs.push(graph);
            candidate_bicliques.push(biclique);
        }
    }

    ActionSpace {
        def_index,
        candidate_templates,
        candidate_graphs,
        candidate_bicliques,
    }
}

fn enumerate_candidates(
    comp: &TensorComputation,
    definition: &TensorDef,
) -> Result<(Vec<ConstrGraph>, Vec<Biclique>), RewriteError> {
    let symmetry = canon::build_tensor_symmetry_map(comp.tensors());
    let index_pool = canon::build_index_pool(definition);
    let mut left_owner_splits_by_term = empty_split_buckets(definition);
    let mut right_owner_splits_by_term = empty_split_buckets(definition);

    for (term_index, term) in definition.terms.iter().enumerate() {
        for raw_split in split::enumerate_splits(term, definition)? {
            let (left_owner, right_owner) =
                canon::canon_split(&raw_split, &symmetry, &index_pool)?;
            left_owner_splits_by_term[term_index].push(left_owner);
            right_owner_splits_by_term[term_index].push(right_owner);
        }
    }

    let mut graphs = Vec::new();
    graphs.extend(graph::build_graphs_from_splits(
        definition,
        &left_owner_splits_by_term,
    )?);
    graphs.extend(graph::build_graphs_from_splits(
        definition,
        &right_owner_splits_by_term,
    )?);

    let mut candidate_graphs = Vec::new();
    let mut candidate_bicliques = Vec::new();
    for graph in graphs {
        for biclique in biclique::enumerate_bicliques(&graph) {
            candidate_graphs.push(graph.clone());
            candidate_bicliques.push(biclique);
        }
    }

    Ok((candidate_graphs, candidate_bicliques))
}

fn empty_split_buckets(definition: &TensorDef) -> Vec<Vec<Split>> {
    vec![vec![]; definition.terms.len()]
}

fn factorization_template(
    graph: &ConstrGraph,
    biclique: &Biclique,
    left_tensor: TensorId,
    right_tensor: TensorId,
) -> Factorization {
    Factorization {
        left_definition: side_definition(
            &graph.left_nodes,
            &biclique.left_node_ids,
            &biclique.left_coeffs,
            &graph.interface.left_external,
            &graph.interface.contracted,
            left_tensor,
        ),
        right_definition: side_definition(
            &graph.right_nodes,
            &biclique.right_node_ids,
            &biclique.right_coeffs,
            &graph.interface.right_external,
            &graph.interface.contracted,
            right_tensor,
        ),
    }
}

fn side_definition(
    source_terms: &[Term],
    term_ids: &[usize],
    coefficients: &[Rational],
    external_indices: &[Index],
    contracted_indices: &[Index],
    tensor: TensorId,
) -> TensorDef {
    let mut ext_indices = external_indices.to_vec();
    ext_indices.extend_from_slice(contracted_indices);

    TensorDef {
        base: tensor,
        ext_indices,
        terms: term_ids
            .iter()
            .zip(coefficients)
            .map(|(&term_id, coefficient)| {
                let mut term = source_terms[term_id].clone();
                term.coeff *= *coefficient;
                term
            })
            .collect(),
    }
}

fn fresh_rewrite_tensor_ids(comp: &TensorComputation) -> (TensorId, TensorId) {
    let left = comp.next_tensor_id();
    let right = TensorId(left.0 + 1);
    (left, right)
}

pub fn validate_decision(
    space: &ActionSpace,
    decision: &Decision,
) -> Result<(), RewriteError> {
    let Some(template) =
        space.candidate_templates.get(decision.candidate_index)
    else {
        return Err(RewriteError::CandidateIndexOutOfRange {
            index: decision.candidate_index,
            len: space.candidate_templates.len(),
        });
    };

    validate_mask(
        DecisionSide::Left,
        &decision.left_mask,
        template.left_definition.terms.len(),
    )?;
    validate_mask(
        DecisionSide::Right,
        &decision.right_mask,
        template.right_definition.terms.len(),
    )?;
    Ok(())
}

fn validate_mask(
    side: DecisionSide,
    mask: &[bool],
    expected: usize,
) -> Result<(), RewriteError> {
    if mask.len() != expected {
        return Err(RewriteError::MaskLengthMismatch {
            side,
            expected,
            got: mask.len(),
        });
    }
    if !mask.iter().any(|keep| *keep) {
        return Err(RewriteError::EmptyMask { side });
    }
    Ok(())
}

pub fn apply_decision(
    comp: &mut TensorComputation,
    space: &ActionSpace,
    decision: &Decision,
) -> Result<(), RewriteError> {
    let source = source_definition(comp, space.def_index)?;
    let template = &space.candidate_templates[decision.candidate_index];
    let graph = &space.candidate_graphs[decision.candidate_index];
    let biclique = &space.candidate_bicliques[decision.candidate_index];
    let selected = selected_biclique(graph, biclique, decision);
    let left = side_definition(
        &graph.left_nodes,
        &selected.left_node_ids,
        &selected.left_coeffs,
        &graph.interface.left_external,
        &graph.interface.contracted,
        template.left_definition.base,
    );
    let right = side_definition(
        &graph.right_nodes,
        &selected.right_node_ids,
        &selected.right_coeffs,
        &graph.interface.right_external,
        &graph.interface.contracted,
        template.right_definition.base,
    );
    let rewritten = rewritten_definition(
        source,
        &left,
        &right,
        &graph.interface.contracted,
        &selected,
    );
    replace_definition(comp, space.def_index, left, right, rewritten);
    Ok(())
}

fn selected_biclique(
    graph: &ConstrGraph,
    biclique: &Biclique,
    decision: &Decision,
) -> Biclique {
    let left = selected_side(
        &biclique.left_node_ids,
        &biclique.left_coeffs,
        &decision.left_mask,
    );
    let right = selected_side(
        &biclique.right_node_ids,
        &biclique.right_coeffs,
        &decision.right_mask,
    );
    let terms_used = selected_terms_used(graph, &left, &right);
    Biclique {
        left_node_ids: left.iter().map(|(node_id, _)| *node_id).collect(),
        right_node_ids: right.iter().map(|(node_id, _)| *node_id).collect(),
        left_coeffs: left.iter().map(|(_, coeff)| *coeff).collect(),
        right_coeffs: right.iter().map(|(_, coeff)| *coeff).collect(),
        terms_used,
    }
}

fn selected_side(
    node_ids: &[usize],
    coeffs: &[Rational],
    mask: &[bool],
) -> Vec<(usize, Rational)> {
    node_ids
        .iter()
        .copied()
        .zip(coeffs.iter().copied())
        .zip(mask.iter().copied())
        .filter_map(|((node_id, coeff), keep)| keep.then_some((node_id, coeff)))
        .collect()
}

fn selected_terms_used(
    graph: &ConstrGraph,
    left: &[(usize, Rational)],
    right: &[(usize, Rational)],
) -> u64 {
    graph
        .edges
        .iter()
        .filter(|edge| side_has_node(left, edge.left_id))
        .filter(|edge| side_has_node(right, edge.right_id))
        .fold(0, |acc, edge| acc | edge.terms_used)
}

fn side_has_node(side: &[(usize, Rational)], node_id: usize) -> bool {
    side.iter()
        .any(|(side_node_id, _)| *side_node_id == node_id)
}

fn rewritten_definition(
    source: &TensorDef,
    left: &TensorDef,
    right: &TensorDef,
    contracted: &[Index],
    selected: &Biclique,
) -> TensorDef {
    let mut terms: Vec<_> = source
        .terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| {
            let consumed = selected.terms_used & (1_u64 << index) != 0;
            (!consumed).then(|| term.clone())
        })
        .collect();
    let replacement_factor = |definition: &TensorDef| Factor {
        tensor: definition.base,
        indices: definition
            .ext_indices
            .iter()
            .map(|index| index.id)
            .collect(),
    };
    terms.push(Term {
        coeff: Rational::new(1, 1),
        sum_indices: contracted.to_vec(),
        factors: vec![replacement_factor(left), replacement_factor(right)],
    });
    TensorDef {
        base: source.base,
        ext_indices: source.ext_indices.clone(),
        terms,
    }
}

fn replace_definition(
    comp: &mut TensorComputation,
    def_index: usize,
    left: TensorDef,
    right: TensorDef,
    rewritten: TensorDef,
) {
    comp.add_tensor(vec![]);
    comp.add_tensor(vec![]);
    let definitions = comp.definitions_mut();
    definitions.remove(def_index);
    definitions.insert(def_index, rewritten);
    definitions.insert(def_index, right);
    definitions.insert(def_index, left);
}
