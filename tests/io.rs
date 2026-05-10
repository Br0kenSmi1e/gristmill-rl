use gristmill_symbolics::io::{from_json, to_json};
use gristmill_symbolics::repr::TensorComputation;

#[test]
fn from_json_parses_existing_repr_fixture() {
    let json = include_str!("fixtures/repr/basic.json");

    let comp = from_json(json).unwrap();

    assert_eq!(comp.ranges().len(), 1);
    assert_eq!(comp.tensors().len(), 1);
    assert_eq!(comp.definitions().len(), 1);
}

#[test]
fn to_json_emits_pretty_json_that_round_trips() {
    let json = include_str!("fixtures/repr/basic.json");
    let comp = from_json(json).unwrap();

    let encoded = to_json(&comp).unwrap();
    let reparsed: TensorComputation = serde_json::from_str(&encoded).unwrap();

    assert_eq!(reparsed, comp);
    assert!(encoded.contains('\n'));
    assert!(encoded.contains("  \"ranges\""));
    assert!(!encoded.ends_with('\n'));
}

#[test]
fn from_json_rejects_unsupported_legacy_symmetry_actions() {
    let json = include_str!("fixtures/repr/legacy_conjugate.json");

    assert!(from_json(json).is_err());
}
