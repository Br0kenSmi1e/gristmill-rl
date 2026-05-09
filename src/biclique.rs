use crate::graph::ConstrGraph;
use crate::repr::Rational;

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

    Vec::new()
}
