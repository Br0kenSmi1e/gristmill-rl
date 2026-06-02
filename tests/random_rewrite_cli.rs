use gristmill_symbolics::io::{from_json, read_json, write_json};
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId, Term,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "gristmill-symbolics-random-rewrite-cli-{name}-{}-{nanos}",
        std::process::id()
    ))
}

struct TempCase {
    root: PathBuf,
}

impl TempCase {
    fn new(name: &str) -> Self {
        let root = unique_temp_path(name);
        fs::create_dir(&root).unwrap();
        Self { root }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

impl Drop for TempCase {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).ok();
    }
}

fn one() -> Rational {
    Rational::new(1, 1)
}

fn idx(id: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(0),
    }
}

fn factor(tensor: TensorId, indices: &[u32]) -> Factor {
    Factor {
        tensor,
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn term(sum_indices: Vec<Index>, factors: Vec<Factor>) -> Term {
    Term {
        coeff: one(),
        sum_indices,
        factors,
    }
}

fn comp_with_shared_left_candidate() -> TensorComputation {
    let mut comp = TensorComputation::new();
    comp.add_range(8);
    let a = comp.add_tensor(vec![]);
    let b = comp.add_tensor(vec![]);
    let c = comp.add_tensor(vec![]);
    let out = comp.add_tensor(vec![]);

    comp.add_definition(
        out,
        vec![idx(0), idx(1)],
        vec![
            term(vec![idx(2)], vec![factor(a, &[0, 2]), factor(b, &[2, 1])]),
            term(vec![idx(3)], vec![factor(a, &[0, 3]), factor(c, &[3, 1])]),
        ],
    );

    comp
}

fn basic_fixture() -> TensorComputation {
    from_json(include_str!("fixtures/repr/basic.json")).unwrap()
}

fn run_random_rewrite_with_options(options: &[&str], paths: &[&Path]) {
    let output = Command::new(env!("CARGO_BIN_EXE_random-rewrite"))
        .args(options)
        .args(paths)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "random-rewrite failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn steps_zero_writes_unchanged_output() {
    let case = TempCase::new("steps-zero");
    let input = case.path("input.json");
    let output = case.path("output.json");
    let comp = comp_with_shared_left_candidate();
    write_json(&input, &comp).unwrap();

    run_random_rewrite_with_options(&["--steps", "0"], &[&input, &output]);

    assert_eq!(read_json(&output).unwrap(), comp);
}

#[test]
fn early_stop_when_no_action_space_exists_succeeds() {
    let case = TempCase::new("early-stop");
    let input = case.path("input.json");
    let output = case.path("output.json");
    let snapshot_dir = case.path("snapshots");
    let comp = basic_fixture();
    write_json(&input, &comp).unwrap();

    run_random_rewrite_with_options(
        &["--steps", "3", "--snapshot-dir"],
        &[&snapshot_dir, &input, &output],
    );

    assert_eq!(read_json(&output).unwrap(), comp);
    assert_eq!(read_json(snapshot_dir.join("step-000.json")).unwrap(), comp);
    assert!(!snapshot_dir.join("step-001.json").exists());
}

#[test]
fn smoke_run_on_actionable_computation_writes_valid_json() {
    let case = TempCase::new("smoke-actionable");
    let input = case.path("input.json");
    let output = case.path("output.json");
    let comp = comp_with_shared_left_candidate();
    let original_tensors = comp.tensors().len();
    let original_definitions = comp.definitions().len();
    write_json(&input, &comp).unwrap();

    run_random_rewrite_with_options(&["--seed", "42", "--steps", "1"], &[&input, &output]);

    let rewritten = read_json(&output).unwrap();
    assert_eq!(rewritten.validate(), Ok(()));
    assert_eq!(rewritten.tensors().len(), original_tensors + 2);
    assert_eq!(rewritten.definitions().len(), original_definitions + 2);
}

#[test]
fn snapshots_record_initial_and_successful_rewrite_states() {
    let case = TempCase::new("snapshots");
    let input = case.path("input.json");
    let output = case.path("output.json");
    let snapshot_dir = case.path("snapshots");
    let comp = comp_with_shared_left_candidate();
    write_json(&input, &comp).unwrap();

    run_random_rewrite_with_options(&["--snapshot-dir"], &[&snapshot_dir, &input, &output]);

    let final_output = read_json(&output).unwrap();
    let step_000 = read_json(snapshot_dir.join("step-000.json")).unwrap();
    let step_001 = read_json(snapshot_dir.join("step-001.json")).unwrap();

    assert_eq!(step_000, comp);
    assert_eq!(step_001.validate(), Ok(()));
    assert_eq!(step_001, final_output);
    assert!(!snapshot_dir.join("step-002.json").exists());
}

#[test]
fn random_subsets_run_writes_valid_json() {
    let case = TempCase::new("random-subsets");
    let input = case.path("input.json");
    let output = case.path("output.json");
    let comp = comp_with_shared_left_candidate();
    write_json(&input, &comp).unwrap();

    run_random_rewrite_with_options(
        &["--random-subsets", "--seed", "5", "--steps", "1"],
        &[&input, &output],
    );

    let rewritten = read_json(&output).unwrap();
    assert_eq!(rewritten.validate(), Ok(()));
}

#[test]
fn seeded_runs_are_deterministic_under_random_definition_policy() {
    let case = TempCase::new("seeded-determinism");
    let input = case.path("input.json");
    let left_output = case.path("left-output.json");
    let right_output = case.path("right-output.json");
    let comp = comp_with_shared_left_candidate();
    write_json(&input, &comp).unwrap();

    run_random_rewrite_with_options(
        &["--random-subsets", "--seed", "17", "--steps", "2"],
        &[&input, &left_output],
    );
    run_random_rewrite_with_options(
        &["--random-subsets", "--seed", "17", "--steps", "2"],
        &[&input, &right_output],
    );

    assert_eq!(
        read_json(&left_output).unwrap(),
        read_json(&right_output).unwrap()
    );
}
