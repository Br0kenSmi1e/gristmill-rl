use gristmill_symbolics::io::{IoJsonError, from_json, read_json, to_json, write_json};
use gristmill_symbolics::repr::TensorComputation;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "gristmill-symbolics-{name}-{}-{nanos}.json",
        std::process::id()
    ))
}

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

#[test]
fn write_json_then_read_json_round_trips_through_a_file() {
    let path = unique_temp_path("round-trip");
    let comp = from_json(include_str!("fixtures/repr/basic.json")).unwrap();

    write_json(&path, &comp).unwrap();
    let reparsed = read_json(&path).unwrap();

    assert_eq!(reparsed, comp);

    fs::remove_file(path).ok();
}

#[test]
fn read_json_reports_malformed_json_as_json_error() {
    let path = unique_temp_path("malformed");
    fs::write(&path, "{ not json").unwrap();

    let err = read_json(&path).unwrap_err();

    match err {
        IoJsonError::Json(source) => {
            assert!(source.to_string().contains("key must be a string"));
        }
        IoJsonError::Io(source) => panic!("expected JSON error, got IO error: {source}"),
    }

    fs::remove_file(path).ok();
}

#[test]
fn read_json_reports_missing_file_as_io_error() {
    let path = unique_temp_path("missing");
    fs::remove_file(&path).ok();

    let err = read_json(&path).unwrap_err();

    match err {
        IoJsonError::Io(source) => {
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        IoJsonError::Json(source) => panic!("expected IO error, got JSON error: {source}"),
    }
}

#[test]
fn io_json_error_display_and_source_delegate_to_wrapped_error() {
    let io_error = IoJsonError::from(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        "blocked",
    ));

    assert!(io_error.to_string().contains("blocked"));
    assert!(std::error::Error::source(&io_error).is_some());

    let json_error = IoJsonError::from(from_json("{").unwrap_err());

    assert!(!json_error.to_string().is_empty());
    assert!(std::error::Error::source(&json_error).is_some());
}
