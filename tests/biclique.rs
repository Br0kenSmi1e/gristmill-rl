use gristmill_symbolics::biclique::{Biclique, enumerate_bicliques};
use gristmill_symbolics::graph::{ConstrGraph, GraphEdge};
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorId, Term,
};
use gristmill_symbolics::split::SplitInterface;

fn rat(num: i64, den: i64) -> Rational {
    Rational::new(num, den)
}

fn factor(tensor: u32, indices: &[u32]) -> Factor {
    Factor {
        tensor: TensorId(tensor),
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn index(id: u32, range: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(range),
    }
}

fn term(coeff_num: i64, coeff_den: i64, sum_indices: &[Index], factors: Vec<Factor>) -> Term {
    Term {
        coeff: rat(coeff_num, coeff_den),
        sum_indices: sum_indices.to_vec(),
        factors,
    }
}

fn base_interface() -> SplitInterface {
    SplitInterface {
        left_external: vec![index(0, 0)],
        right_external: vec![index(1, 0)],
        contracted: vec![index(2, 0)],
    }
}

fn graph(
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: &[(usize, usize, Rational, u64)],
) -> ConstrGraph {
    ConstrGraph {
        interface: base_interface(),
        left_nodes,
        right_nodes,
        edges: edges
            .iter()
            .map(|(left_id, right_id, coeff, terms_used)| GraphEdge {
                left_id: *left_id,
                right_id: *right_id,
                coeff: coeff.clone(),
                terms_used: *terms_used,
            })
            .collect(),
    }
}

fn graph_i64(
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: &[(usize, usize, i64, u64)],
) -> ConstrGraph {
    ConstrGraph {
        interface: base_interface(),
        left_nodes,
        right_nodes,
        edges: edges
            .iter()
            .map(|(left_id, right_id, coeff, terms_used)| GraphEdge {
                left_id: *left_id,
                right_id: *right_id,
                coeff: rat(*coeff, 1),
                terms_used: *terms_used,
            })
            .collect(),
    }
}

fn sample_left_nodes() -> Vec<Term> {
    vec![
        term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]),
        term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]),
        term(1, 1, &[index(2, 0)], vec![factor(6, &[0, 2])]),
    ]
}

fn sample_right_nodes() -> Vec<Term> {
    vec![
        term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]),
        term(1, 1, &[index(2, 0)], vec![factor(4, &[2, 1])]),
        term(1, 1, &[index(2, 0)], vec![factor(5, &[2, 1])]),
    ]
}

fn find_biclique<'a>(
    bicliques: &'a [Biclique],
    left_ids: &[usize],
    right_ids: &[usize],
) -> &'a Biclique {
    bicliques
        .iter()
        .find(|biclique| {
            biclique.left_node_ids == left_ids && biclique.right_node_ids == right_ids
        })
        .expect("expected biclique was not returned")
}

#[test]
fn crate_surface_exposes_biclique_enumerator_api() {
    let enumerate_fn: fn(&ConstrGraph) -> Vec<Biclique> = enumerate_bicliques;

    let biclique = Biclique {
        left_node_ids: vec![0],
        right_node_ids: vec![0],
        left_coeffs: vec![rat(1, 1)],
        right_coeffs: vec![rat(2, 1)],
        terms_used: 0b1,
    };

    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert_eq!(biclique.terms_used, 0b1);
    assert!(enumerate_fn(&graph).is_empty());
}

#[test]
fn graphs_with_fewer_than_two_edges_produce_no_bicliques() {
    let empty_graph = graph_i64(sample_left_nodes(), sample_right_nodes(), &[]);
    let one_edge_graph = graph_i64(
        sample_left_nodes()[0..1].to_vec(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert!(enumerate_bicliques(&empty_graph).is_empty());
    assert!(enumerate_bicliques(&one_edge_graph).is_empty());
}

#[test]
fn one_edge_bicliques_are_not_emitted() {
    let graph = graph_i64(
        sample_left_nodes()[0..1].to_vec(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert!(enumerate_bicliques(&graph).is_empty());
}
