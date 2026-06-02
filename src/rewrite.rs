use crate::biclique::Biclique;
use crate::canon::CanonError;
use crate::graph::{ConstrGraph, GraphError};
use crate::repr::{Factor, Index, Rational, TensorComputation, TensorDef, TensorId, Term};
use crate::split::Split;
use crate::split::SplitError;
use crate::{biclique, canon, graph, split};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Factorization {
    pub left_definition: TensorDef,
    pub right_definition: TensorDef,
    pub rewritten_definition: TensorDef,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewriteState {
    comp: TensorComputation,
    def_mask: Vec<bool>,
}

impl RewriteState {
    pub fn new(comp: TensorComputation) -> Self {
        let def_mask = comp.definitions().iter().map(cheap_possible).collect();
        Self { comp, def_mask }
    }

    pub fn computation(&self) -> &TensorComputation {
        &self.comp
    }

    pub fn into_computation(self) -> TensorComputation {
        self.comp
    }

    pub fn definition_mask(&self) -> &[bool] {
        &self.def_mask
    }

    pub fn action_space_for_def(
        &mut self,
        def_index: usize,
    ) -> Result<Option<ActionSpace>, RewriteError> {
        if def_index >= self.def_mask.len() {
            return Err(RewriteError::DefinitionIndexOutOfRange {
                index: def_index,
                len: self.def_mask.len(),
            });
        }
        if !self.def_mask[def_index] {
            return Ok(None);
        }

        let Some(space) = action_space_for_definition(&self.comp, def_index)? else {
            self.def_mask[def_index] = false;
            return Ok(None);
        };
        Ok(Some(space))
    }

    pub fn step_with_space(
        &mut self,
        space: &ActionSpace,
        decision: &Decision,
    ) -> Result<(), RewriteError> {
        let rewrite = build_rewrite(&self.comp, space, decision)?;
        let def_index = rewrite.def_index;
        apply_rewrite(&mut self.comp, rewrite)?;
        self.refresh_mask_after_rewrite(def_index);
        Ok(())
    }

    fn refresh_mask_after_rewrite(&mut self, def_index: usize) {
        let replacement_mask: Vec<bool> = self.comp.definitions()[def_index..def_index + 3]
            .iter()
            .map(cheap_possible)
            .collect();
        self.def_mask.remove(def_index);
        for (offset, mask_value) in replacement_mask.into_iter().enumerate() {
            self.def_mask.insert(def_index + offset, mask_value);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizationRewrite {
    pub def_index: usize,
    pub factorization: Factorization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    Split(SplitError),
    Canon(CanonError),
    Graph(GraphError),
    CandidateIndexOutOfRange { index: usize, len: usize },
    LeftMaskLengthMismatch { expected: usize, got: usize },
    RightMaskLengthMismatch { expected: usize, got: usize },
    EmptyLeftMask,
    EmptyRightMask,
    DefinitionIndexOutOfRange { index: usize, len: usize },
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

pub fn next_action_space(
    comp: &TensorComputation,
    start_from: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    for def_index in start_from..comp.definitions().len() {
        if let Some(space) = action_space_for_definition(comp, def_index)? {
            return Ok(Some(space));
        }
    }
    Ok(None)
}

fn action_space_for_definition(
    comp: &TensorComputation,
    def_index: usize,
) -> Result<Option<ActionSpace>, RewriteError> {
    let Some(def) = comp.definitions().get(def_index) else {
        return Err(RewriteError::DefinitionIndexOutOfRange {
            index: def_index,
            len: comp.definitions().len(),
        });
    };
    if !cheap_possible(def) {
        return Ok(None);
    }

    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
    let (candidate_graphs, candidate_bicliques) = enumerate_candidates(comp, def)?;
    if candidate_bicliques.is_empty() {
        return Ok(None);
    }

    let candidate_templates = candidate_graphs
        .iter()
        .zip(&candidate_bicliques)
        .map(|(graph, biclique)| build_factorization(def, graph, biclique, left_tid, right_tid))
        .collect();

    Ok(Some(ActionSpace {
        def_index,
        candidate_templates,
        candidate_graphs,
        candidate_bicliques,
    }))
}

fn cheap_possible(def: &TensorDef) -> bool {
    def.terms.len() >= 2
}

fn enumerate_candidates(
    comp: &TensorComputation,
    def: &TensorDef,
) -> Result<(Vec<ConstrGraph>, Vec<Biclique>), RewriteError> {
    let symmetry = canon::build_tensor_symmetry_map(comp.tensors());
    let pool = canon::build_index_pool(def);
    let mut left_owner_splits_by_term: Vec<Vec<Split>> = vec![vec![]; def.terms.len()];
    let mut right_owner_splits_by_term: Vec<Vec<Split>> = vec![vec![]; def.terms.len()];

    for (term_idx, term) in def.terms.iter().enumerate() {
        for raw_split in split::enumerate_splits(term, def)? {
            let (left_owner, right_owner) = canon::canon_split(&raw_split, &symmetry, &pool)?;
            left_owner_splits_by_term[term_idx].push(left_owner);
            right_owner_splits_by_term[term_idx].push(right_owner);
        }
    }

    let mut graphs = Vec::new();
    graphs.extend(graph::build_graphs_from_splits(
        def,
        &left_owner_splits_by_term,
    )?);
    graphs.extend(graph::build_graphs_from_splits(
        def,
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

pub fn validate_decision(space: &ActionSpace, decision: &Decision) -> Result<(), RewriteError> {
    let Some(template) = space.candidate_templates.get(decision.candidate_index) else {
        return Err(RewriteError::CandidateIndexOutOfRange {
            index: decision.candidate_index,
            len: space.candidate_templates.len(),
        });
    };

    let expected_left = template.left_definition.terms.len();
    if decision.left_mask.len() != expected_left {
        return Err(RewriteError::LeftMaskLengthMismatch {
            expected: expected_left,
            got: decision.left_mask.len(),
        });
    }

    let expected_right = template.right_definition.terms.len();
    if decision.right_mask.len() != expected_right {
        return Err(RewriteError::RightMaskLengthMismatch {
            expected: expected_right,
            got: decision.right_mask.len(),
        });
    }

    if !decision.left_mask.iter().any(|keep| *keep) {
        return Err(RewriteError::EmptyLeftMask);
    }

    if !decision.right_mask.iter().any(|keep| *keep) {
        return Err(RewriteError::EmptyRightMask);
    }

    Ok(())
}

pub fn build_rewrite(
    comp: &TensorComputation,
    space: &ActionSpace,
    decision: &Decision,
) -> Result<FactorizationRewrite, RewriteError> {
    let def =
        comp.definitions()
            .get(space.def_index)
            .ok_or(RewriteError::DefinitionIndexOutOfRange {
                index: space.def_index,
                len: comp.definitions().len(),
            })?;

    validate_decision(space, decision)?;

    let graph = space.candidate_graphs.get(decision.candidate_index).ok_or(
        RewriteError::CandidateIndexOutOfRange {
            index: decision.candidate_index,
            len: space.candidate_graphs.len(),
        },
    )?;
    let biclique = space
        .candidate_bicliques
        .get(decision.candidate_index)
        .ok_or(RewriteError::CandidateIndexOutOfRange {
            index: decision.candidate_index,
            len: space.candidate_bicliques.len(),
        })?;

    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
    let sub_biclique = sub_biclique_from_decision(graph, biclique, decision);
    let factorization = build_factorization(def, graph, &sub_biclique, left_tid, right_tid);

    Ok(FactorizationRewrite {
        def_index: space.def_index,
        factorization,
    })
}

pub fn apply_rewrite(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) -> Result<(), RewriteError> {
    verify_rewrite_def_index(comp, &rewrite)?;
    register_rewrite_tensors(comp);
    replace_definition_with_factorization(comp, rewrite);
    Ok(())
}

fn verify_rewrite_def_index(
    comp: &TensorComputation,
    rewrite: &FactorizationRewrite,
) -> Result<(), RewriteError> {
    if rewrite.def_index < comp.definitions().len() {
        Ok(())
    } else {
        Err(RewriteError::DefinitionIndexOutOfRange {
            index: rewrite.def_index,
            len: comp.definitions().len(),
        })
    }
}

fn register_rewrite_tensors(comp: &mut TensorComputation) {
    comp.add_tensor(vec![]);
    comp.add_tensor(vec![]);
}

fn replace_definition_with_factorization(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) {
    let def_index = rewrite.def_index;
    let Factorization {
        left_definition,
        right_definition,
        rewritten_definition,
    } = rewrite.factorization;
    let definitions = comp.definitions_mut();

    definitions.remove(def_index);
    definitions.insert(def_index, rewritten_definition);
    definitions.insert(def_index, right_definition);
    definitions.insert(def_index, left_definition);
}

#[allow(dead_code)]
fn build_factorization(
    def: &TensorDef,
    graph: &ConstrGraph,
    biclique: &Biclique,
    left_tid: TensorId,
    right_tid: TensorId,
) -> Factorization {
    let contracted = contracted_indices(graph);
    let (left_external, right_external) = side_external_indices(graph);

    let left_definition = build_side_definition(
        &graph.left_nodes,
        &biclique.left_node_ids,
        &biclique.left_coeffs,
        &left_external,
        &contracted,
        left_tid,
    );
    let right_definition = build_side_definition(
        &graph.right_nodes,
        &biclique.right_node_ids,
        &biclique.right_coeffs,
        &right_external,
        &contracted,
        right_tid,
    );
    let consumed = consumed_term_indices(biclique);
    let rewritten_definition = build_rewritten_definition(
        def,
        &left_definition,
        &right_definition,
        &contracted,
        &consumed,
    );

    Factorization {
        left_definition,
        right_definition,
        rewritten_definition,
    }
}

#[allow(dead_code)]
fn contracted_indices(graph: &ConstrGraph) -> Vec<Index> {
    graph.interface.contracted.clone()
}

#[allow(dead_code)]
fn side_external_indices(graph: &ConstrGraph) -> (Vec<Index>, Vec<Index>) {
    (
        graph.interface.left_external.clone(),
        graph.interface.right_external.clone(),
    )
}

#[allow(dead_code)]
fn consumed_term_indices(biclique: &Biclique) -> Vec<usize> {
    bits_to_vec(biclique.terms_used)
}

#[allow(dead_code)]
fn build_side_definition(
    source_nodes: &[Term],
    node_ids: &[usize],
    coeffs: &[Rational],
    side_external: &[Index],
    contracted: &[Index],
    tensor: TensorId,
) -> TensorDef {
    let mut ext_indices = side_external.to_vec();
    ext_indices.extend_from_slice(contracted);

    TensorDef {
        base: tensor,
        ext_indices,
        terms: node_ids
            .iter()
            .zip(coeffs)
            .map(|(&node_id, coeff)| build_side_term(source_nodes, node_id, coeff))
            .collect(),
    }
}

#[allow(dead_code)]
fn build_side_term(source_nodes: &[Term], node_id: usize, coeff: &Rational) -> Term {
    let mut term = source_nodes[node_id].clone();
    term.coeff *= *coeff;
    term
}

#[allow(dead_code)]
fn build_rewritten_definition(
    def: &TensorDef,
    left_def: &TensorDef,
    right_def: &TensorDef,
    contracted: &[Index],
    consumed: &[usize],
) -> TensorDef {
    let mut terms: Vec<_> = def
        .terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| {
            if consumed.contains(&index) {
                None
            } else {
                Some(term.clone())
            }
        })
        .collect();

    terms.push(Term {
        coeff: Rational::new(1, 1),
        sum_indices: contracted.to_vec(),
        factors: vec![
            Factor {
                tensor: left_def.base,
                indices: left_def.ext_indices.iter().map(|index| index.id).collect(),
            },
            Factor {
                tensor: right_def.base,
                indices: right_def.ext_indices.iter().map(|index| index.id).collect(),
            },
        ],
    });

    TensorDef {
        base: def.base,
        ext_indices: def.ext_indices.clone(),
        terms,
    }
}

#[allow(dead_code)]
fn bits_to_vec(mask: u64) -> Vec<usize> {
    (0..64)
        .filter(|position| mask & (1_u64 << position) != 0)
        .collect()
}

#[allow(dead_code)]
fn sub_biclique_from_decision(
    graph: &ConstrGraph,
    biclique: &Biclique,
    decision: &Decision,
) -> Biclique {
    let left: Vec<_> = biclique
        .left_node_ids
        .iter()
        .copied()
        .zip(biclique.left_coeffs.iter().copied())
        .zip(decision.left_mask.iter().copied())
        .filter_map(|((node_id, coeff), keep)| keep.then_some((node_id, coeff)))
        .collect();
    let right: Vec<_> = biclique
        .right_node_ids
        .iter()
        .copied()
        .zip(biclique.right_coeffs.iter().copied())
        .zip(decision.right_mask.iter().copied())
        .filter_map(|((node_id, coeff), keep)| keep.then_some((node_id, coeff)))
        .collect();

    let selected_left: HashSet<_> = left.iter().map(|(node_id, _)| *node_id).collect();
    let selected_right: HashSet<_> = right.iter().map(|(node_id, _)| *node_id).collect();
    let terms_used = graph
        .edges
        .iter()
        .filter(|edge| {
            selected_left.contains(&edge.left_id) && selected_right.contains(&edge.right_id)
        })
        .fold(0, |acc, edge| acc | edge.terms_used);

    Biclique {
        left_node_ids: left.iter().map(|(node_id, _)| *node_id).collect(),
        right_node_ids: right.iter().map(|(node_id, _)| *node_id).collect(),
        left_coeffs: left.iter().map(|(_, coeff)| *coeff).collect(),
        right_coeffs: right.iter().map(|(_, coeff)| *coeff).collect(),
        terms_used,
    }
}

fn fresh_rewrite_tensor_ids(comp: &TensorComputation) -> (TensorId, TensorId) {
    let left = comp.next_tensor_id();
    let right = TensorId(left.0 + 1);
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biclique::Biclique;
    use crate::graph::{ConstrGraph, GraphEdge};
    use crate::repr::{Factor, Index, IndexId, RangeId, Rational, TensorDef, TensorId, Term};
    use crate::split::SplitInterface;

    fn term() -> Term {
        Term {
            coeff: Rational::new(1, 1),
            sum_indices: vec![],
            factors: vec![],
        }
    }

    fn def(base: u32, term_count: usize) -> TensorDef {
        TensorDef {
            base: TensorId(base),
            ext_indices: vec![],
            terms: (0..term_count).map(|_| term()).collect(),
        }
    }

    fn validation_space() -> ActionSpace {
        ActionSpace {
            def_index: 0,
            candidate_templates: vec![Factorization {
                left_definition: def(10, 2),
                right_definition: def(11, 1),
                rewritten_definition: def(0, 1),
            }],
            candidate_graphs: vec![],
            candidate_bicliques: vec![],
        }
    }

    #[test]
    fn validate_decision_rejects_out_of_range_candidate_index() {
        let space = validation_space();
        let decision = Decision {
            candidate_index: 1,
            left_mask: vec![true, true],
            right_mask: vec![true],
        };

        assert_eq!(
            validate_decision(&space, &decision),
            Err(RewriteError::CandidateIndexOutOfRange { index: 1, len: 1 })
        );
    }

    #[test]
    fn validate_decision_rejects_mask_length_mismatches() {
        let space = validation_space();

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![true],
                    right_mask: vec![true],
                },
            ),
            Err(RewriteError::LeftMaskLengthMismatch {
                expected: 2,
                got: 1,
            })
        );

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![true, true],
                    right_mask: vec![true, false],
                },
            ),
            Err(RewriteError::RightMaskLengthMismatch {
                expected: 1,
                got: 2,
            })
        );
    }

    #[test]
    fn validate_decision_rejects_empty_selected_sides() {
        let space = validation_space();

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![false, false],
                    right_mask: vec![true],
                },
            ),
            Err(RewriteError::EmptyLeftMask)
        );

        assert_eq!(
            validate_decision(
                &space,
                &Decision {
                    candidate_index: 0,
                    left_mask: vec![true, false],
                    right_mask: vec![false],
                },
            ),
            Err(RewriteError::EmptyRightMask)
        );
    }

    #[test]
    fn validate_decision_accepts_nonempty_masks_with_expected_lengths() {
        let space = validation_space();
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true, false],
            right_mask: vec![true],
        };

        assert_eq!(validate_decision(&space, &decision), Ok(()));
    }

    fn rat(value: i64) -> Rational {
        Rational::new(value, 1)
    }

    fn idx(id: u32) -> Index {
        Index {
            id: IndexId(id),
            range: RangeId(0),
        }
    }

    fn factor(tensor: u32, indices: &[u32]) -> Factor {
        Factor {
            tensor: TensorId(tensor),
            indices: indices.iter().copied().map(IndexId).collect(),
        }
    }

    fn term_with_sum(coeff: Rational, sum_indices: Vec<Index>, factors: Vec<Factor>) -> Term {
        Term {
            coeff,
            sum_indices,
            factors,
        }
    }

    fn source_def_for_factorization() -> TensorDef {
        TensorDef {
            base: TensorId(50),
            ext_indices: vec![idx(0), idx(1)],
            terms: vec![
                term_with_sum(rat(1), vec![idx(2)], vec![factor(1, &[0, 2])]),
                term_with_sum(rat(1), vec![idx(3)], vec![factor(2, &[3, 1])]),
                term_with_sum(rat(9), vec![], vec![factor(9, &[0, 1])]),
            ],
        }
    }

    fn graph_and_biclique_for_factorization() -> (ConstrGraph, Biclique) {
        let graph = ConstrGraph {
            interface: SplitInterface {
                left_external: vec![idx(0)],
                right_external: vec![idx(1)],
                contracted: vec![idx(2)],
            },
            left_nodes: vec![term_with_sum(
                rat(1),
                vec![idx(4)],
                vec![factor(10, &[0, 4, 2])],
            )],
            right_nodes: vec![
                term_with_sum(rat(1), vec![], vec![factor(11, &[2, 1])]),
                term_with_sum(rat(1), vec![], vec![factor(12, &[2, 1])]),
            ],
            edges: vec![
                GraphEdge {
                    left_id: 0,
                    right_id: 0,
                    coeff: rat(15),
                    terms_used: 0b001,
                },
                GraphEdge {
                    left_id: 0,
                    right_id: 1,
                    coeff: rat(21),
                    terms_used: 0b010,
                },
            ],
        };

        let biclique = Biclique {
            left_node_ids: vec![0],
            right_node_ids: vec![0, 1],
            left_coeffs: vec![rat(3)],
            right_coeffs: vec![rat(5), rat(7)],
            terms_used: 0b011,
        };

        (graph, biclique)
    }

    #[test]
    fn build_factorization_uses_interface_indices_as_source_of_truth() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization =
            build_factorization(&def, &graph, &biclique, TensorId(60), TensorId(61));

        assert_eq!(factorization.left_definition.base, TensorId(60));
        assert_eq!(factorization.right_definition.base, TensorId(61));
        assert_eq!(
            factorization.left_definition.ext_indices,
            vec![idx(0), idx(2)]
        );
        assert_eq!(
            factorization.right_definition.ext_indices,
            vec![idx(1), idx(2)]
        );
        assert_eq!(factorization.rewritten_definition.base, TensorId(50));
        assert_eq!(
            factorization.rewritten_definition.ext_indices,
            vec![idx(0), idx(1)]
        );
    }

    #[test]
    fn build_factorization_preserves_private_sum_indices_and_side_coefficients() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization =
            build_factorization(&def, &graph, &biclique, TensorId(60), TensorId(61));

        assert_eq!(factorization.left_definition.terms.len(), 1);
        assert_eq!(factorization.left_definition.terms[0].coeff, rat(3));
        assert_eq!(
            factorization.left_definition.terms[0].sum_indices,
            vec![idx(4)]
        );
        assert_eq!(factorization.right_definition.terms.len(), 2);
        assert_eq!(factorization.right_definition.terms[0].coeff, rat(5));
        assert_eq!(factorization.right_definition.terms[1].coeff, rat(7));
    }

    #[test]
    fn build_factorization_removes_consumed_terms_and_appends_replacement() {
        let def = source_def_for_factorization();
        let (graph, biclique) = graph_and_biclique_for_factorization();

        let factorization =
            build_factorization(&def, &graph, &biclique, TensorId(60), TensorId(61));

        assert_eq!(factorization.rewritten_definition.terms.len(), 2);
        assert_eq!(factorization.rewritten_definition.terms[0], def.terms[2]);
        assert_eq!(
            factorization.rewritten_definition.terms[1],
            Term {
                coeff: rat(1),
                sum_indices: vec![idx(2)],
                factors: vec![
                    Factor {
                        tensor: TensorId(60),
                        indices: vec![IndexId(0), IndexId(2)],
                    },
                    Factor {
                        tensor: TensorId(61),
                        indices: vec![IndexId(1), IndexId(2)],
                    },
                ],
            }
        );
    }

    fn comp_with_definition(def: TensorDef) -> TensorComputation {
        let mut comp = TensorComputation::new();
        comp.add_definition(def.base, def.ext_indices.clone(), def.terms.clone());
        comp
    }

    fn action_space_for_factorization(comp: &TensorComputation) -> ActionSpace {
        let def = &comp.definitions()[0];
        let (graph, biclique) = graph_and_biclique_for_factorization();
        let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
        let template = build_factorization(def, &graph, &biclique, left_tid, right_tid);

        ActionSpace {
            def_index: 0,
            candidate_templates: vec![template],
            candidate_graphs: vec![graph],
            candidate_bicliques: vec![biclique],
        }
    }

    #[test]
    fn sub_biclique_from_decision_keeps_selected_terms_and_recomputes_provenance() {
        let (graph, biclique) = graph_and_biclique_for_factorization();
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true],
            right_mask: vec![false, true],
        };

        let sub_biclique = sub_biclique_from_decision(&graph, &biclique, &decision);

        assert_eq!(sub_biclique.left_node_ids, vec![0]);
        assert_eq!(sub_biclique.right_node_ids, vec![1]);
        assert_eq!(sub_biclique.left_coeffs, vec![rat(3)]);
        assert_eq!(sub_biclique.right_coeffs, vec![rat(7)]);
        assert_eq!(sub_biclique.terms_used, 0b010);
    }

    #[test]
    fn build_rewrite_full_biclique_matches_visible_template() {
        let comp = comp_with_definition(source_def_for_factorization());
        let space = action_space_for_factorization(&comp);
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true],
            right_mask: vec![true, true],
        };

        let rewrite = build_rewrite(&comp, &space, &decision).unwrap();

        assert_eq!(rewrite.def_index, 0);
        assert_eq!(rewrite.factorization, space.candidate_templates[0]);
    }

    #[test]
    fn build_rewrite_subset_decision_shrinks_side_definition_and_consumed_terms() {
        let comp = comp_with_definition(source_def_for_factorization());
        let space = action_space_for_factorization(&comp);
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true],
            right_mask: vec![false, true],
        };

        let rewrite = build_rewrite(&comp, &space, &decision).unwrap();

        assert_eq!(rewrite.factorization.left_definition.terms.len(), 1);
        assert_eq!(rewrite.factorization.right_definition.terms.len(), 1);
        assert_eq!(
            rewrite.factorization.right_definition.terms[0].coeff,
            rat(7)
        );
        assert_eq!(rewrite.factorization.rewritten_definition.terms.len(), 3);
        assert_eq!(
            rewrite.factorization.rewritten_definition.terms[0],
            comp.definitions()[0].terms[0]
        );
        assert_eq!(
            rewrite.factorization.rewritten_definition.terms[1],
            comp.definitions()[0].terms[2]
        );
    }

    #[test]
    fn build_rewrite_rejects_out_of_range_definition_index() {
        let comp = TensorComputation::new();
        let space = ActionSpace {
            def_index: 0,
            candidate_templates: vec![],
            candidate_graphs: vec![],
            candidate_bicliques: vec![],
        };
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![],
            right_mask: vec![],
        };

        assert_eq!(
            build_rewrite(&comp, &space, &decision),
            Err(RewriteError::DefinitionIndexOutOfRange { index: 0, len: 0 })
        );
    }
}
