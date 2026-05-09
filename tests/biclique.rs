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

#[test]
fn enumerate_bicliques_bootstraps_to_a_2x1_biclique() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b001), (1, 0, 6, 0b010)],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1)]);
    assert_eq!(biclique.terms_used, 0b011);
}

#[test]
fn enumerate_bicliques_bootstraps_to_a_1x2_biclique() {
    let graph = graph_i64(
        sample_left_nodes()[0..1].to_vec(),
        sample_right_nodes(),
        &[(0, 0, 2, 0b001), (0, 1, 4, 0b010)],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1), rat(4, 1)]);
    assert_eq!(biclique.terms_used, 0b011);
}

#[test]
fn enumerate_bicliques_finds_factorizable_2x2_rectangle() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 2, 0b0001),
            (0, 1, 4, 0b0010),
            (1, 0, 6, 0b0100),
            (1, 1, 12, 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1), rat(4, 1)]);
    assert_eq!(biclique.terms_used, 0b1111);
}

#[test]
fn enumerate_bicliques_rejects_non_factorizable_2x2_rectangle() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 2, 0b0001),
            (0, 1, 4, 0b0010),
            (1, 0, 6, 0b0100),
            (1, 1, 11, 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);

    assert!(
        bicliques.iter().all(|biclique| biclique.left_node_ids != [0, 1]
            || biclique.right_node_ids != [0, 1])
    );
}

#[test]
fn enumerate_bicliques_rejects_overlapping_provenance() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b001), (1, 0, 6, 0b001)],
    );

    assert!(enumerate_bicliques(&graph).is_empty());
}

#[test]
fn enumerate_bicliques_supports_negative_rational_coefficients() {
    let graph = graph(
        sample_left_nodes()[0..2].to_vec(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, rat(-1, 2), 0b0001),
            (0, 1, rat(3, 4), 0b0010),
            (1, 0, rat(-1, 1), 0b0100),
            (1, 1, rat(3, 2), 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(2, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(-1, 2), rat(3, 4)]);
    assert_eq!(biclique.terms_used, 0b1111);
}

#[test]
fn enumerate_bicliques_emits_only_the_maximal_2x3_rectangle_once() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes(),
        &[
            (0, 0, 2, 0b000001),
            (0, 1, 4, 0b000010),
            (0, 2, 6, 0b000100),
            (1, 0, 6, 0b001000),
            (1, 1, 12, 0b010000),
            (1, 2, 18, 0b100000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);

    assert_eq!(bicliques.len(), 1);

    let biclique = &bicliques[0];
    assert_eq!(biclique.left_node_ids, vec![0, 1]);
    assert_eq!(biclique.right_node_ids, vec![0, 1, 2]);
    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(
        biclique.right_coeffs,
        vec![rat(2, 1), rat(4, 1), rat(6, 1)]
    );
    assert_eq!(biclique.terms_used, 0b111111);
}

#[test]
fn enumerate_bicliques_ignores_non_current_left_pivots_after_bootstrap() {
    let graph = graph_i64(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 10, 0b001000),
            (0, 1, 20, 0b010000),
            (1, 0, 2, 0b000001),
            (1, 1, 4, 0b000010),
            (2, 0, 6, 0b000100),
            (2, 1, 12, 0b001000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[1, 2], &[0, 1]);

    assert_eq!(biclique.left_coeffs, vec![rat(1, 1), rat(3, 1)]);
    assert_eq!(biclique.right_coeffs, vec![rat(2, 1), rat(4, 1)]);
    assert_eq!(biclique.terms_used, 0b001111);
}
