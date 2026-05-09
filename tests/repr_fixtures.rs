use gristmill_symbolics::repr::TensorComputation;

#[test]
fn compatible_fixture_round_trips() {
    let json = include_str!("fixtures/repr/basic.json");
    let comp: TensorComputation = serde_json::from_str(json).unwrap();
    let round_trip = serde_json::to_string(&comp).unwrap();
    let reparsed: TensorComputation = serde_json::from_str(&round_trip).unwrap();
    assert_eq!(comp, reparsed);
}

#[test]
fn legacy_conjugate_actions_are_rejected() {
    let json = include_str!("fixtures/repr/legacy_conjugate.json");
    assert!(serde_json::from_str::<TensorComputation>(json).is_err());
}
