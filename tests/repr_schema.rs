use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, SymAction, SymGenerator, TensorComputation, TensorId, Term,
};

#[test]
fn builders_assign_ids_and_accessors_work() {
    let mut comp = TensorComputation::new();
    assert_eq!(TensorComputation::default(), TensorComputation::new());
    assert_eq!(comp.ranges(), &[]);
    assert_eq!(comp.tensors(), &[]);
    assert_eq!(comp.definitions(), &[]);
    assert_eq!(comp.next_tensor_id(), TensorId(0));

    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![]);
    comp.add_definition(tensor_id, vec![], vec![]);

    assert_eq!(range_id, RangeId(0));
    assert_eq!(tensor_id, TensorId(0));
    assert_eq!(comp.ranges()[0].size, 3);
    assert_eq!(comp.tensors()[0].id, tensor_id);
    assert_eq!(comp.definitions()[0].base, tensor_id);
    assert_eq!(comp.next_tensor_id(), TensorId(1));
}

#[test]
fn serde_keeps_compatibility_fields() {
    let mut comp = TensorComputation::new();
    let range_id = comp.add_range(3);
    let tensor_id = comp.add_tensor(vec![SymGenerator {
        perm: vec![0],
        action: SymAction::Identity,
    }]);

    comp.add_definition(
        tensor_id,
        vec![Index {
            id: IndexId(0),
            range: range_id,
        }],
        vec![Term {
            coeff: num::rational::Ratio::new(1, 1),
            sum_indices: vec![],
            factors: vec![Factor {
                tensor: tensor_id,
                indices: vec![IndexId(0)],
            }],
        }],
    );

    let json = serde_json::to_string(&comp).unwrap();
    for field in [
        "ranges",
        "tensors",
        "definitions",
        "id",
        "size",
        "symmetry",
        "perm",
        "action",
        "range",
        "base",
        "ext_indices",
        "terms",
        "coeff",
        "sum_indices",
        "factors",
        "tensor",
        "indices",
    ] {
        assert!(json.contains(&format!("\"{field}\"")));
    }
}
