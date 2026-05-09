use crate::graph::{ConstrGraph, GraphEdge};
use crate::repr::Rational;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SearchNode {
    Left(usize),
    Right(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Delta {
    coeff: Rational,
    terms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Biclique {
    pub left_node_ids: Vec<usize>,
    pub right_node_ids: Vec<usize>,
    pub left_coeffs: Vec<Rational>,
    pub right_coeffs: Vec<Rational>,
    pub terms_used: u64,
}

pub fn enumerate_bicliques(graph: &ConstrGraph) -> Vec<Biclique> {
    if graph.edges.len() < 2 {
        return Vec::new();
    }

    let mut biclique = empty_biclique();
    let mut candidates = all_candidates(graph);
    let mut out = Vec::new();
    let frontier = initial_frontier(graph);

    expand(graph, &mut biclique, &frontier, &mut candidates, &mut out);
    out
}

fn all_candidates(graph: &ConstrGraph) -> Vec<SearchNode> {
    (0..graph.left_nodes.len())
        .map(SearchNode::Left)
        .chain((0..graph.right_nodes.len()).map(SearchNode::Right))
        .collect()
}

fn initial_frontier(graph: &ConstrGraph) -> HashMap<SearchNode, Delta> {
    all_candidates(graph)
        .into_iter()
        .map(|node| {
            (
                node,
                Delta {
                    coeff: Rational::new(1, 1),
                    terms: 0,
                },
            )
        })
        .collect()
}

fn empty_biclique() -> Biclique {
    Biclique {
        left_node_ids: vec![],
        right_node_ids: vec![],
        left_coeffs: vec![],
        right_coeffs: vec![],
        terms_used: 0,
    }
}

fn edge_between(graph: &ConstrGraph, left_id: usize, right_id: usize) -> Option<&GraphEdge> {
    graph
        .edges
        .iter()
        .find(|edge| edge.left_id == left_id && edge.right_id == right_id)
}

fn expand(
    graph: &ConstrGraph,
    biclique: &mut Biclique,
    frontier: &HashMap<SearchNode, Delta>,
    candidates: &mut Vec<SearchNode>,
    out: &mut Vec<Biclique>,
) {
    if has_sharing(biclique) && frontier.is_empty() {
        out.push(biclique.clone());
        return;
    }

    let child_frontiers = build_child_frontiers(graph, biclique, frontier);
    let current = sift(biclique, candidates, frontier, &child_frontiers);

    for node in current {
        let Some(delta) = frontier.get(&node) else {
            continue;
        };
        let Some(position) = candidates.iter().position(|candidate| *candidate == node) else {
            continue;
        };

        let removed = candidates.remove(position);
        let child_frontier = child_frontiers.get(&removed).cloned().unwrap_or_default();
        let mut child_candidates: Vec<SearchNode> = candidates
            .iter()
            .copied()
            .filter(|candidate| child_frontier.contains_key(candidate))
            .collect();

        push(biclique, removed, delta);
        expand(graph, biclique, &child_frontier, &mut child_candidates, out);
        pop(biclique, removed, delta);
    }
}

fn sift(
    biclique: &Biclique,
    candidates: &[SearchNode],
    frontier: &HashMap<SearchNode, Delta>,
    child_frontiers: &HashMap<SearchNode, HashMap<SearchNode, Delta>>,
) -> Vec<SearchNode> {
    if biclique.left_node_ids.is_empty() && biclique.right_node_ids.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Left(_)))
            .copied()
            .collect();
    }

    if biclique.left_node_ids.len() == 1 && biclique.right_node_ids.is_empty() {
        return candidates
            .iter()
            .filter(|node| matches!(node, SearchNode::Right(_)))
            .filter(|node| matches!(frontier.get(node), Some(delta) if delta.terms != 0))
            .copied()
            .collect();
    }

    let current = candidates.to_vec();

    let mut best_forbidden = Vec::new();
    let mut best_score = 0usize;
    for &node in &current {
        let forbidden: Vec<SearchNode> = child_frontiers
            .get(&node)
            .map(|next| next.keys().copied().collect())
            .unwrap_or_default();
        let score = forbidden
            .iter()
            .filter(|candidate| current.contains(candidate))
            .count();
        if score > best_score {
            best_score = score;
            best_forbidden = forbidden;
        }
    }

    current
        .into_iter()
        .filter(|node| !best_forbidden.contains(node))
        .collect()
}

fn build_child_frontiers(
    graph: &ConstrGraph,
    biclique: &Biclique,
    frontier: &HashMap<SearchNode, Delta>,
) -> HashMap<SearchNode, HashMap<SearchNode, Delta>> {
    let mut out = HashMap::new();

    for (chosen, chosen_delta) in frontier {
        let mut child = HashMap::new();
        for (candidate, candidate_delta) in frontier {
            if chosen == candidate {
                continue;
            }

            if let Some(updated) = update_delta(
                graph,
                biclique,
                *chosen,
                chosen_delta,
                *candidate,
                candidate_delta,
            ) {
                child.insert(*candidate, updated);
            }
        }
        out.insert(*chosen, child);
    }

    out
}

fn update_delta(
    graph: &ConstrGraph,
    biclique: &Biclique,
    chosen: SearchNode,
    chosen_delta: &Delta,
    candidate: SearchNode,
    candidate_delta: &Delta,
) -> Option<Delta> {
    if matches!(
        (chosen, candidate),
        (SearchNode::Left(_), SearchNode::Left(_)) | (SearchNode::Right(_), SearchNode::Right(_))
    ) {
        if chosen_delta.terms & candidate_delta.terms != 0 {
            return None;
        }
        return Some(candidate_delta.clone());
    }

    let (left_id, right_id) = match (chosen, candidate) {
        (SearchNode::Left(left_id), SearchNode::Right(right_id)) => (left_id, right_id),
        (SearchNode::Right(right_id), SearchNode::Left(left_id)) => (left_id, right_id),
        _ => unreachable!(),
    };

    let edge = edge_between(graph, left_id, right_id)?;

    if chosen_delta.terms & candidate_delta.terms != 0
        || biclique.terms_used & edge.terms_used != 0
        || chosen_delta.terms & edge.terms_used != 0
        || candidate_delta.terms & edge.terms_used != 0
    {
        return None;
    }

    let expected = edge.coeff / chosen_delta.coeff;
    let mut next = candidate_delta.clone();
    if candidate_delta.terms == 0 {
        next.coeff = expected;
    } else if candidate_delta.coeff != expected {
        return None;
    }
    next.terms |= edge.terms_used;
    Some(next)
}

fn has_sharing(biclique: &Biclique) -> bool {
    biclique.left_node_ids.len() >= 2 || biclique.right_node_ids.len() >= 2
}

fn push(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    biclique.terms_used |= delta.terms;
    let coeff = delta.coeff;

    match node {
        SearchNode::Left(id) => {
            biclique.left_node_ids.push(id);
            biclique.left_coeffs.push(coeff);
        }
        SearchNode::Right(id) => {
            biclique.right_node_ids.push(id);
            biclique.right_coeffs.push(coeff);
        }
    }
}

fn pop(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    biclique.terms_used ^= delta.terms;

    match node {
        SearchNode::Left(_) => {
            biclique.left_node_ids.pop();
            biclique.left_coeffs.pop();
        }
        SearchNode::Right(_) => {
            biclique.right_node_ids.pop();
            biclique.right_coeffs.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta() -> Delta {
        Delta {
            coeff: Rational::new(1, 1),
            terms: 1,
        }
    }

    #[test]
    fn sift_prunes_candidates_for_best_post_bootstrap_pivot() {
        let biclique = Biclique {
            left_node_ids: vec![0],
            right_node_ids: vec![0],
            left_coeffs: vec![Rational::new(1, 1)],
            right_coeffs: vec![Rational::new(1, 1)],
            terms_used: 1,
        };
        let candidates = vec![
            SearchNode::Left(1),
            SearchNode::Right(1),
            SearchNode::Left(2),
        ];
        let frontier = candidates
            .iter()
            .copied()
            .map(|node| (node, delta()))
            .collect();
        let child_frontiers = HashMap::from([
            (
                SearchNode::Left(1),
                HashMap::from([
                    (SearchNode::Right(1), delta()),
                    (SearchNode::Left(2), delta()),
                ]),
            ),
            (
                SearchNode::Right(1),
                HashMap::from([(SearchNode::Left(2), delta())]),
            ),
        ]);

        let sifted = sift(&biclique, &candidates, &frontier, &child_frontiers);

        assert_eq!(sifted, vec![SearchNode::Left(1)]);
    }
}
