use gristmill_symbolics::graph::{ConstrGraph, GraphEdge, GraphError, build_graphs_from_splits};
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorDef, TensorId, Term,
};
use gristmill_symbolics::split::{Split, SplitInterface};

fn one() -> Rational {
    Rational::new(1, 1)
}

fn empty_def_with_terms(len: usize) -> TensorDef {
    TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: (0..len)
            .map(|_| Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            })
            .collect(),
    }
}

fn idx(id: u32, range: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(range),
    }
}

fn factor(tensor: u32, indices: &[u32]) -> Factor {
    Factor {
        tensor: TensorId(tensor),
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn term(coeff: Rational, factors: Vec<Factor>) -> Term {
    Term {
        coeff,
        sum_indices: vec![],
        factors,
    }
}

fn split(left: Term, right: Term, interface: SplitInterface) -> Split {
    Split {
        left,
        right,
        interface,
    }
}

fn iface(
    left_external: Vec<Index>,
    right_external: Vec<Index>,
    contracted: Vec<Index>,
) -> SplitInterface {
    SplitInterface {
        left_external,
        right_external,
        contracted,
    }
}

#[test]
fn split_term_alignment_mismatch_returns_graph_error() {
    let def = empty_def_with_terms(2);

    assert_eq!(
        build_graphs_from_splits(&def, &[vec![]]),
        Err(GraphError::SplitTermAlignmentMismatch {
            expected: 2,
            got: 1,
        })
    );
}

#[test]
fn more_than_64_terms_returns_graph_error() {
    let def = empty_def_with_terms(65);
    let splits_by_term = vec![vec![]; 65];

    assert_eq!(
        build_graphs_from_splits(&def, &splits_by_term),
        Err(GraphError::TooManyTerms { len: 65, max: 64 })
    );
}

#[test]
fn builds_graph_bucket_with_independent_left_and_right_nodes() {
    let a = idx(0, 0);
    let b = idx(1, 0);
    let k = idx(2, 0);
    let interface = iface(vec![a], vec![b], vec![k]);
    let shared_term_value = term(one(), vec![factor(7, &[0, 2])]);
    let source_def = TensorDef {
        base: TensorId(9),
        ext_indices: vec![a, b],
        terms: vec![
            Term {
                coeff: Rational::new(3, 1),
                sum_indices: vec![k],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(5, 1),
                sum_indices: vec![k],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(
            shared_term_value.clone(),
            shared_term_value.clone(),
            interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(8, &[2, 1])]),
            term(one(), vec![factor(10, &[0])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&source_def, &splits_by_term).unwrap();

    assert_eq!(
        graphs,
        vec![ConstrGraph {
            interface,
            left_nodes: vec![
                shared_term_value.clone(),
                term(one(), vec![factor(8, &[2, 1])]),
            ],
            right_nodes: vec![shared_term_value, term(one(), vec![factor(10, &[0])]),],
            edges: vec![
                GraphEdge {
                    left_id: 0,
                    right_id: 0,
                    coeff: Rational::new(3, 1),
                    terms_used: 1,
                },
                GraphEdge {
                    left_id: 1,
                    right_id: 1,
                    coeff: Rational::new(5, 1),
                    terms_used: 2,
                },
            ],
        }]
    );
}

#[test]
fn edge_coefficients_absorb_source_and_side_coefficients_and_nodes_are_unit_terms() {
    let interface = iface(vec![], vec![], vec![]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(6, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(
            term(Rational::new(2, 1), vec![factor(0, &[0])]),
            term(Rational::new(-3, 1), vec![factor(1, &[1])]),
            interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();
    let graph = &graphs[0];

    assert_eq!(graph.left_nodes[0].coeff, one());
    assert_eq!(graph.right_nodes[0].coeff, one());
    assert_eq!(graph.edges[0].coeff, Rational::new(-36, 1));
}

fn graph_by_interface<'a>(
    graphs: &'a [ConstrGraph],
    interface: &SplitInterface,
) -> &'a ConstrGraph {
    graphs
        .iter()
        .find(|graph| &graph.interface == interface)
        .expect("graph with interface should exist")
}

#[test]
fn equal_interfaces_share_a_bucket_and_different_interfaces_create_separate_graphs() {
    let a = idx(0, 0);
    let b = idx(1, 0);
    let first_interface = iface(vec![a], vec![b], vec![]);
    let second_interface = iface(vec![b], vec![a], vec![]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![a, b],
        terms: vec![
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(
            term(one(), vec![factor(0, &[0])]),
            term(one(), vec![factor(1, &[1])]),
            first_interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(2, &[0])]),
            term(one(), vec![factor(3, &[1])]),
            first_interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(4, &[1])]),
            term(one(), vec![factor(5, &[0])]),
            second_interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(6, &[1])]),
            term(one(), vec![factor(7, &[0])]),
            second_interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();

    assert_eq!(graphs.len(), 2);
    assert_eq!(graph_by_interface(&graphs, &first_interface).edges.len(), 2);
    assert_eq!(
        graph_by_interface(&graphs, &second_interface).edges.len(),
        2
    );
}

#[test]
fn distinct_source_terms_contributing_to_same_edge_are_summed() {
    let interface = iface(vec![], vec![], vec![]);
    let left = term(one(), vec![factor(0, &[0])]);
    let right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(5, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(left.clone(), right.clone(), interface.clone())],
        vec![split(left.clone(), right.clone(), interface.clone())],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();
    let merged_edge = &graphs[0].edges[0];

    assert_eq!(merged_edge.coeff, Rational::new(7, 1));
    assert_eq!(merged_edge.terms_used, 0b011);
}

#[test]
fn repeated_source_term_contribution_to_same_edge_is_ignored() {
    let interface = iface(vec![], vec![], vec![]);
    let left = term(one(), vec![factor(0, &[0])]);
    let right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: one(),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![
            split(left.clone(), right.clone(), interface.clone()),
            split(left.clone(), right.clone(), interface.clone()),
        ],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();

    assert_eq!(graphs[0].edges.len(), 2);
    assert_eq!(graphs[0].edges[0].coeff, Rational::new(2, 1));
    assert_eq!(graphs[0].edges[0].terms_used, 0b001);
}

#[test]
fn zero_sum_edges_are_removed_and_single_edge_graphs_are_omitted() {
    let interface = iface(vec![], vec![], vec![]);
    let left = term(one(), vec![factor(0, &[0])]);
    let right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(-2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(9, 1),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(left.clone(), right.clone(), interface.clone())],
        vec![split(left, right, interface.clone())],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
    ];

    assert_eq!(
        build_graphs_from_splits(&def, &splits_by_term).unwrap(),
        vec![]
    );
}

#[test]
fn graph_with_two_remaining_edges_survives_after_zero_edge_removal() {
    let interface = iface(vec![], vec![], vec![]);
    let cancel_left = term(one(), vec![factor(0, &[0])]);
    let cancel_right = term(one(), vec![factor(1, &[1])]);
    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![],
        terms: vec![
            Term {
                coeff: Rational::new(2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(-2, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(3, 1),
                sum_indices: vec![],
                factors: vec![],
            },
            Term {
                coeff: Rational::new(4, 1),
                sum_indices: vec![],
                factors: vec![],
            },
        ],
    };
    let splits_by_term = vec![
        vec![split(
            cancel_left.clone(),
            cancel_right.clone(),
            interface.clone(),
        )],
        vec![split(cancel_left, cancel_right, interface.clone())],
        vec![split(
            term(one(), vec![factor(2, &[2])]),
            term(one(), vec![factor(3, &[3])]),
            interface.clone(),
        )],
        vec![split(
            term(one(), vec![factor(4, &[4])]),
            term(one(), vec![factor(5, &[5])]),
            interface.clone(),
        )],
    ];

    let graphs = build_graphs_from_splits(&def, &splits_by_term).unwrap();

    assert_eq!(graphs.len(), 1);
    assert_eq!(graphs[0].edges.len(), 2);
    assert!(
        graphs[0]
            .edges
            .iter()
            .all(|edge| edge.coeff != Rational::new(0, 1))
    );
}
