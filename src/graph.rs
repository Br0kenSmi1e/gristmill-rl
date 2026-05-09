use crate::repr::{Rational, TensorDef, Term};
use crate::split::{Split, SplitInterface};
use std::collections::HashMap;

const MAX_TERMS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphEdge {
    pub left_id: usize,
    pub right_id: usize,
    pub coeff: Rational,
    pub terms_used: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstrGraph {
    pub interface: SplitInterface,
    pub left_nodes: Vec<Term>,
    pub right_nodes: Vec<Term>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    SplitTermAlignmentMismatch { expected: usize, got: usize },
    TooManyTerms { len: usize, max: usize },
}

pub fn build_graphs_from_splits(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<Vec<ConstrGraph>, GraphError> {
    validate_splits_by_term(def, splits_by_term)?;

    let mut graphs: HashMap<SplitInterface, ConstrGraph> = HashMap::new();

    for (term_idx, splits) in splits_by_term.iter().enumerate() {
        let source_coeff = &def.terms[term_idx].coeff;

        for split in splits {
            let graph = graphs
                .entry(split.interface.clone())
                .or_insert_with(|| empty_graph(split.interface.clone()));
            insert_split(graph, source_coeff, term_idx, split);
        }
    }

    Ok(finalize_graphs(graphs))
}

fn empty_graph(interface: SplitInterface) -> ConstrGraph {
    ConstrGraph {
        interface,
        left_nodes: vec![],
        right_nodes: vec![],
        edges: vec![],
    }
}

fn insert_split(graph: &mut ConstrGraph, source_coeff: &Rational, term_idx: usize, split: &Split) {
    let (left, right, coeff) = normalize_edge_contribution(source_coeff, split);
    let left_id = ensure_node(&mut graph.left_nodes, left);
    let right_id = ensure_node(&mut graph.right_nodes, right);
    merge_or_push_edge(&mut graph.edges, left_id, right_id, term_idx, coeff);
}

fn normalize_edge_contribution(source_coeff: &Rational, split: &Split) -> (Term, Term, Rational) {
    let mut left = split.left.clone();
    let mut right = split.right.clone();
    let coeff = *source_coeff * left.coeff * right.coeff;
    left.coeff = Rational::new(1, 1);
    right.coeff = Rational::new(1, 1);
    (left, right, coeff)
}

fn ensure_node(nodes: &mut Vec<Term>, term: Term) -> usize {
    if let Some(index) = nodes.iter().position(|node| node == &term) {
        index
    } else {
        let index = nodes.len();
        nodes.push(term);
        index
    }
}

fn merge_or_push_edge(
    edges: &mut Vec<GraphEdge>,
    left_id: usize,
    right_id: usize,
    term_idx: usize,
    coeff: Rational,
) {
    let term_bit = 1_u64 << term_idx;

    if let Some(edge) = edges
        .iter_mut()
        .find(|edge| edge.left_id == left_id && edge.right_id == right_id)
    {
        if edge.terms_used & term_bit == 0 {
            edge.coeff += coeff;
            edge.terms_used |= term_bit;
        }
        return;
    }

    edges.push(GraphEdge {
        left_id,
        right_id,
        coeff,
        terms_used: term_bit,
    });
}

fn finalize_graphs(graphs: HashMap<SplitInterface, ConstrGraph>) -> Vec<ConstrGraph> {
    graphs
        .into_values()
        .filter_map(|mut graph| {
            graph.edges.retain(|edge| edge.coeff != Rational::new(0, 1));
            (graph.edges.len() >= 2).then_some(graph)
        })
        .collect()
}

fn validate_splits_by_term(
    def: &TensorDef,
    splits_by_term: &[Vec<Split>],
) -> Result<(), GraphError> {
    if splits_by_term.len() != def.terms.len() {
        return Err(GraphError::SplitTermAlignmentMismatch {
            expected: def.terms.len(),
            got: splits_by_term.len(),
        });
    }

    if def.terms.len() > MAX_TERMS {
        return Err(GraphError::TooManyTerms {
            len: def.terms.len(),
            max: MAX_TERMS,
        });
    }

    Ok(())
}
