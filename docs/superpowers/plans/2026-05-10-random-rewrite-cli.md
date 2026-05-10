# Random Rewrite CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the named `random-rewrite` CLI that reads a computation JSON file, applies up to a seeded number of random rewrites, and writes the resulting computation JSON file.

**Architecture:** Add a named Cargo binary at `src/bin/random-rewrite.rs` and remove the unused package-name binary stub at `src/main.rs`. The binary composes existing library APIs: `io::read_json`, `rewrite::next_action_space`, random `Decision` construction, `rewrite::build_rewrite`, `rewrite::apply_rewrite`, optional snapshot writes, and `io::write_json`.

**Tech Stack:** Rust 2024, `clap` derive for argument parsing, `rand::rngs::StdRng` seeded with `SeedableRng::seed_from_u64`, existing `gristmill_symbolics::{io, rewrite}` APIs, standard library filesystem and process testing.

---

## File Structure

- Modify `Cargo.toml`: add runtime dependencies `clap` and `rand`.
- Delete `src/main.rs`: remove the package-name default binary stub.
- Create `src/bin/random-rewrite.rs`: named CLI binary, parser, random decision helpers, rewrite loop, snapshots, status reporting, and binary-local unit tests.
- Create `tests/random_rewrite_cli.rs`: process-level integration tests for output files, early stop, snapshots, and random subsets.

No library API changes are needed. The CLI remains a consumer of the existing public kernel.

---

### Task 1: Add The Named Binary And Argument Parser

**Files:**
- Modify: `Cargo.toml`
- Delete: `src/main.rs`
- Create: `src/bin/random-rewrite.rs`

- [ ] **Step 1: Write the failing parser tests**

Create `src/bin/random-rewrite.rs` with parser tests first:

```rust
use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Parser)]
#[command(name = "random-rewrite")]
struct Args {
    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 1)]
    steps: usize,

    #[arg(long)]
    random_subsets: bool,

    #[arg(long)]
    snapshot_dir: Option<PathBuf>,

    input: PathBuf,
    output: PathBuf,
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_for_seed_steps_and_optional_flags() {
        let args = Args::try_parse_from(["random-rewrite", "input.json", "output.json"]).unwrap();

        assert_eq!(args.seed, 42);
        assert_eq!(args.steps, 1);
        assert!(!args.random_subsets);
        assert_eq!(args.snapshot_dir, None);
        assert_eq!(args.input, PathBuf::from("input.json"));
        assert_eq!(args.output, PathBuf::from("output.json"));
    }

    #[test]
    fn parse_explicit_options() {
        let args = Args::try_parse_from([
            "random-rewrite",
            "--seed",
            "7",
            "--steps",
            "12",
            "--random-subsets",
            "--snapshot-dir",
            "snapshots",
            "input.json",
            "output.json",
        ])
        .unwrap();

        assert_eq!(args.seed, 7);
        assert_eq!(args.steps, 12);
        assert!(args.random_subsets);
        assert_eq!(args.snapshot_dir, Some(PathBuf::from("snapshots")));
        assert_eq!(args.input, PathBuf::from("input.json"));
        assert_eq!(args.output, PathBuf::from("output.json"));
    }

    #[test]
    fn parse_requires_input_and_output_paths() {
        let err = Args::try_parse_from(["random-rewrite", "input.json"]).unwrap_err();

        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }
}
```

- [ ] **Step 2: Run the parser tests to verify they fail**

Run:

```bash
cargo test --bin random-rewrite -- --nocapture
```

Expected: compile failure because `clap` is not yet in `Cargo.toml`.

- [ ] **Step 3: Add CLI dependencies and remove the default binary stub**

Modify `Cargo.toml` to contain:

```toml
[package]
name = "gristmill-symbolics"
version = "0.1.0"
edition = "2024"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
num = { version = "0.4", features = ["serde"] }
clap = { version = "4", features = ["derive"] }
rand = "0.8"
```

Delete `src/main.rs`.

- [ ] **Step 4: Run the parser tests to verify they pass**

Run:

```bash
cargo test --bin random-rewrite -- --nocapture
```

Expected: 3 tests pass. `Cargo.lock` is updated with `clap`, `rand`, and their transitive dependencies.

- [ ] **Step 5: Commit the parser target**

Run:

```bash
git add Cargo.toml Cargo.lock src/bin/random-rewrite.rs src/main.rs
git commit -m "feat: add random rewrite CLI parser"
```

Expected: commit succeeds. The staged deletion of `src/main.rs` is included.

---

### Task 2: Add Random Decision Helpers

**Files:**
- Modify: `src/bin/random-rewrite.rs`

- [ ] **Step 1: Add decision construction helpers and unit tests**

Replace `src/bin/random-rewrite.rs` with:

```rust
use clap::Parser;
use gristmill_symbolics::repr::{Rational, TensorDef, TensorId, Term};
use gristmill_symbolics::rewrite::{Decision, Factorization};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Parser)]
#[command(name = "random-rewrite")]
struct Args {
    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 1)]
    steps: usize,

    #[arg(long)]
    random_subsets: bool,

    #[arg(long)]
    snapshot_dir: Option<PathBuf>,

    input: PathBuf,
    output: PathBuf,
}

fn choose_candidate_index(candidate_count: usize, rng: &mut impl Rng) -> usize {
    debug_assert!(candidate_count > 0);
    rng.gen_range(0..candidate_count)
}

fn full_mask(len: usize) -> Vec<bool> {
    vec![true; len]
}

fn random_nonempty_mask(len: usize, rng: &mut impl Rng) -> Vec<bool> {
    if len == 0 {
        return vec![];
    }

    let mut mask: Vec<bool> = (0..len).map(|_| rng.gen_bool(0.5)).collect();
    if !mask.iter().any(|keep| *keep) {
        let forced_index = rng.gen_range(0..len);
        mask[forced_index] = true;
    }
    mask
}

fn decision_for_template(
    candidate_index: usize,
    template: &Factorization,
    random_subsets: bool,
    rng: &mut impl Rng,
) -> Decision {
    let left_len = template.left_definition.terms.len();
    let right_len = template.right_definition.terms.len();

    let left_mask = if random_subsets {
        random_nonempty_mask(left_len, rng)
    } else {
        full_mask(left_len)
    };
    let right_mask = if random_subsets {
        random_nonempty_mask(right_len, rng)
    } else {
        full_mask(right_len)
    };

    Decision {
        candidate_index,
        left_mask,
        right_mask,
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn term() -> Term {
        Term {
            coeff: Rational::new(1, 1),
            sum_indices: vec![],
            factors: vec![],
        }
    }

    fn def(base: u32, term_count: usize) -> TensorDef {
        TensorDef {
            base: TensorId(base),
            ext_indices: vec![],
            terms: (0..term_count).map(|_| term()).collect(),
        }
    }

    fn factorization(left_terms: usize, right_terms: usize) -> Factorization {
        Factorization {
            left_definition: def(10, left_terms),
            right_definition: def(11, right_terms),
            rewritten_definition: def(0, 1),
        }
    }

    #[test]
    fn parse_defaults_for_seed_steps_and_optional_flags() {
        let args = Args::try_parse_from(["random-rewrite", "input.json", "output.json"]).unwrap();

        assert_eq!(args.seed, 42);
        assert_eq!(args.steps, 1);
        assert!(!args.random_subsets);
        assert_eq!(args.snapshot_dir, None);
        assert_eq!(args.input, PathBuf::from("input.json"));
        assert_eq!(args.output, PathBuf::from("output.json"));
    }

    #[test]
    fn parse_explicit_options() {
        let args = Args::try_parse_from([
            "random-rewrite",
            "--seed",
            "7",
            "--steps",
            "12",
            "--random-subsets",
            "--snapshot-dir",
            "snapshots",
            "input.json",
            "output.json",
        ])
        .unwrap();

        assert_eq!(args.seed, 7);
        assert_eq!(args.steps, 12);
        assert!(args.random_subsets);
        assert_eq!(args.snapshot_dir, Some(PathBuf::from("snapshots")));
        assert_eq!(args.input, PathBuf::from("input.json"));
        assert_eq!(args.output, PathBuf::from("output.json"));
    }

    #[test]
    fn parse_requires_input_and_output_paths() {
        let err = Args::try_parse_from(["random-rewrite", "input.json"]).unwrap_err();

        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn choose_candidate_index_uses_the_available_range() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let index = choose_candidate_index(5, &mut rng);
            assert!(index < 5);
        }
    }

    #[test]
    fn decision_uses_full_masks_by_default() {
        let template = factorization(2, 3);
        let mut rng = StdRng::seed_from_u64(42);

        let decision = decision_for_template(4, &template, false, &mut rng);

        assert_eq!(
            decision,
            Decision {
                candidate_index: 4,
                left_mask: vec![true, true],
                right_mask: vec![true, true, true],
            }
        );
    }

    #[test]
    fn random_nonempty_mask_forces_at_least_one_selected_term() {
        for seed in 0..128 {
            let mut rng = StdRng::seed_from_u64(seed);

            let mask = random_nonempty_mask(4, &mut rng);

            assert_eq!(mask.len(), 4);
            assert!(mask.iter().any(|keep| *keep));
        }
    }

    #[test]
    fn decision_uses_nonempty_random_masks_when_enabled() {
        let template = factorization(3, 2);
        let mut rng = StdRng::seed_from_u64(3);

        let decision = decision_for_template(1, &template, true, &mut rng);

        assert_eq!(decision.candidate_index, 1);
        assert_eq!(decision.left_mask.len(), 3);
        assert_eq!(decision.right_mask.len(), 2);
        assert!(decision.left_mask.iter().any(|keep| *keep));
        assert!(decision.right_mask.iter().any(|keep| *keep));
    }
}
```

- [ ] **Step 2: Run the focused tests**

Run:

```bash
cargo test --bin random-rewrite -- --nocapture
```

Expected: 7 tests pass. If compilation fails, the failure points to an import or API mismatch in the helper code.

- [ ] **Step 3: Commit the decision helper work**

Run:

```bash
git add src/bin/random-rewrite.rs
git commit -m "feat: add random rewrite decision helpers"
```

Expected: commit succeeds with only `src/bin/random-rewrite.rs` staged.

---

### Task 3: Implement The Rewrite Loop, Snapshots, And Status

**Files:**
- Modify: `src/bin/random-rewrite.rs`

- [ ] **Step 1: Replace the binary with the full CLI implementation**

Replace `src/bin/random-rewrite.rs` with:

```rust
use clap::Parser;
use gristmill_symbolics::io::{self, IoJsonError};
use gristmill_symbolics::repr::{Rational, TensorComputation, TensorDef, TensorId, Term};
use gristmill_symbolics::rewrite::{self, ActionSpace, Decision, Factorization, RewriteError};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Parser)]
#[command(name = "random-rewrite")]
struct Args {
    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 1)]
    steps: usize,

    #[arg(long)]
    random_subsets: bool,

    #[arg(long)]
    snapshot_dir: Option<PathBuf>,

    input: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    ReachedStepLimit,
    NoActionSpace,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::ReachedStepLimit => write!(f, "reached step limit"),
            StopReason::NoActionSpace => write!(f, "no action space remained"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RunSummary {
    seed: u64,
    requested_steps: usize,
    applied_rewrites: usize,
    stop_reason: StopReason,
}

#[derive(Debug)]
enum CliError {
    ReadInput { path: PathBuf, source: IoJsonError },
    WriteOutput { path: PathBuf, source: IoJsonError },
    CreateSnapshotDir { path: PathBuf, source: std::io::Error },
    WriteSnapshot { path: PathBuf, source: IoJsonError },
    Rewrite { step: usize, source: RewriteError },
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::ReadInput { path, source } => {
                write!(f, "error reading {}: {source}", path.display())
            }
            CliError::WriteOutput { path, source } => {
                write!(f, "error writing {}: {source}", path.display())
            }
            CliError::CreateSnapshotDir { path, source } => {
                write!(
                    f,
                    "error creating snapshot directory {}: {source}",
                    path.display()
                )
            }
            CliError::WriteSnapshot { path, source } => {
                write!(f, "error writing snapshot {}: {source}", path.display())
            }
            CliError::Rewrite { step, source } => {
                write!(f, "rewrite error at step {step}: {source:?}")
            }
        }
    }
}

impl Error for CliError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CliError::ReadInput { source, .. } => Some(source),
            CliError::WriteOutput { source, .. } => Some(source),
            CliError::CreateSnapshotDir { source, .. } => Some(source),
            CliError::WriteSnapshot { source, .. } => Some(source),
            CliError::Rewrite { .. } => None,
        }
    }
}

fn choose_candidate_index(candidate_count: usize, rng: &mut impl Rng) -> usize {
    debug_assert!(candidate_count > 0);
    rng.gen_range(0..candidate_count)
}

fn full_mask(len: usize) -> Vec<bool> {
    vec![true; len]
}

fn random_nonempty_mask(len: usize, rng: &mut impl Rng) -> Vec<bool> {
    if len == 0 {
        return vec![];
    }

    let mut mask: Vec<bool> = (0..len).map(|_| rng.gen_bool(0.5)).collect();
    if !mask.iter().any(|keep| *keep) {
        let forced_index = rng.gen_range(0..len);
        mask[forced_index] = true;
    }
    mask
}

fn decision_for_template(
    candidate_index: usize,
    template: &Factorization,
    random_subsets: bool,
    rng: &mut impl Rng,
) -> Decision {
    let left_len = template.left_definition.terms.len();
    let right_len = template.right_definition.terms.len();

    let left_mask = if random_subsets {
        random_nonempty_mask(left_len, rng)
    } else {
        full_mask(left_len)
    };
    let right_mask = if random_subsets {
        random_nonempty_mask(right_len, rng)
    } else {
        full_mask(right_len)
    };

    Decision {
        candidate_index,
        left_mask,
        right_mask,
    }
}

fn random_decision(
    space: &ActionSpace,
    random_subsets: bool,
    rng: &mut impl Rng,
) -> Decision {
    let candidate_index = choose_candidate_index(space.candidate_templates.len(), rng);
    decision_for_template(
        candidate_index,
        &space.candidate_templates[candidate_index],
        random_subsets,
        rng,
    )
}

fn snapshot_path(dir: &Path, step: usize) -> PathBuf {
    dir.join(format!("step-{step:03}.json"))
}

fn write_snapshot(dir: &Path, step: usize, comp: &TensorComputation) -> Result<(), CliError> {
    let path = snapshot_path(dir, step);
    io::write_json(&path, comp).map_err(|source| CliError::WriteSnapshot { path, source })
}

fn run(args: Args) -> Result<RunSummary, CliError> {
    let mut comp = io::read_json(&args.input).map_err(|source| CliError::ReadInput {
        path: args.input.clone(),
        source,
    })?;

    if let Some(snapshot_dir) = &args.snapshot_dir {
        fs::create_dir_all(snapshot_dir).map_err(|source| CliError::CreateSnapshotDir {
            path: snapshot_dir.clone(),
            source,
        })?;
        write_snapshot(snapshot_dir, 0, &comp)?;
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut start_from = 0;
    let mut applied_rewrites = 0;
    let mut stop_reason = StopReason::ReachedStepLimit;

    for step in 0..args.steps {
        let Some(space) = rewrite::next_action_space(&comp, start_from)
            .map_err(|source| CliError::Rewrite { step, source })?
        else {
            stop_reason = StopReason::NoActionSpace;
            break;
        };

        let decision = random_decision(&space, args.random_subsets, &mut rng);
        let factorization_rewrite = rewrite::build_rewrite(&comp, &space, &decision)
            .map_err(|source| CliError::Rewrite { step, source })?;
        let next_start_from = space.def_index;

        rewrite::apply_rewrite(&mut comp, factorization_rewrite)
            .map_err(|source| CliError::Rewrite { step, source })?;

        applied_rewrites += 1;
        start_from = next_start_from;

        if let Some(snapshot_dir) = &args.snapshot_dir {
            write_snapshot(snapshot_dir, applied_rewrites, &comp)?;
        }
    }

    io::write_json(&args.output, &comp).map_err(|source| CliError::WriteOutput {
        path: args.output.clone(),
        source,
    })?;

    Ok(RunSummary {
        seed: args.seed,
        requested_steps: args.steps,
        applied_rewrites,
        stop_reason,
    })
}

fn main() {
    let args = Args::parse();
    match run(args) {
        Ok(summary) => {
            eprintln!(
                "random-rewrite: seed={} steps={} applied={} stop={}",
                summary.seed,
                summary.requested_steps,
                summary.applied_rewrites,
                summary.stop_reason
            );
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term() -> Term {
        Term {
            coeff: Rational::new(1, 1),
            sum_indices: vec![],
            factors: vec![],
        }
    }

    fn def(base: u32, term_count: usize) -> TensorDef {
        TensorDef {
            base: TensorId(base),
            ext_indices: vec![],
            terms: (0..term_count).map(|_| term()).collect(),
        }
    }

    fn factorization(left_terms: usize, right_terms: usize) -> Factorization {
        Factorization {
            left_definition: def(10, left_terms),
            right_definition: def(11, right_terms),
            rewritten_definition: def(0, 1),
        }
    }

    #[test]
    fn parse_defaults_for_seed_steps_and_optional_flags() {
        let args = Args::try_parse_from(["random-rewrite", "input.json", "output.json"]).unwrap();

        assert_eq!(args.seed, 42);
        assert_eq!(args.steps, 1);
        assert!(!args.random_subsets);
        assert_eq!(args.snapshot_dir, None);
        assert_eq!(args.input, PathBuf::from("input.json"));
        assert_eq!(args.output, PathBuf::from("output.json"));
    }

    #[test]
    fn parse_explicit_options() {
        let args = Args::try_parse_from([
            "random-rewrite",
            "--seed",
            "7",
            "--steps",
            "12",
            "--random-subsets",
            "--snapshot-dir",
            "snapshots",
            "input.json",
            "output.json",
        ])
        .unwrap();

        assert_eq!(args.seed, 7);
        assert_eq!(args.steps, 12);
        assert!(args.random_subsets);
        assert_eq!(args.snapshot_dir, Some(PathBuf::from("snapshots")));
        assert_eq!(args.input, PathBuf::from("input.json"));
        assert_eq!(args.output, PathBuf::from("output.json"));
    }

    #[test]
    fn parse_requires_input_and_output_paths() {
        let err = Args::try_parse_from(["random-rewrite", "input.json"]).unwrap_err();

        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn choose_candidate_index_uses_the_available_range() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let index = choose_candidate_index(5, &mut rng);
            assert!(index < 5);
        }
    }

    #[test]
    fn decision_uses_full_masks_by_default() {
        let template = factorization(2, 3);
        let mut rng = StdRng::seed_from_u64(42);

        let decision = decision_for_template(4, &template, false, &mut rng);

        assert_eq!(
            decision,
            Decision {
                candidate_index: 4,
                left_mask: vec![true, true],
                right_mask: vec![true, true, true],
            }
        );
    }

    #[test]
    fn random_nonempty_mask_forces_at_least_one_selected_term() {
        for seed in 0..128 {
            let mut rng = StdRng::seed_from_u64(seed);

            let mask = random_nonempty_mask(4, &mut rng);

            assert_eq!(mask.len(), 4);
            assert!(mask.iter().any(|keep| *keep));
        }
    }

    #[test]
    fn decision_uses_nonempty_random_masks_when_enabled() {
        let template = factorization(3, 2);
        let mut rng = StdRng::seed_from_u64(3);

        let decision = decision_for_template(1, &template, true, &mut rng);

        assert_eq!(decision.candidate_index, 1);
        assert_eq!(decision.left_mask.len(), 3);
        assert_eq!(decision.right_mask.len(), 2);
        assert!(decision.left_mask.iter().any(|keep| *keep));
        assert!(decision.right_mask.iter().any(|keep| *keep));
    }

    #[test]
    fn snapshot_path_uses_zero_padded_step_numbers() {
        assert_eq!(
            snapshot_path(Path::new("snapshots"), 7),
            PathBuf::from("snapshots").join("step-007.json")
        );
        assert_eq!(
            snapshot_path(Path::new("snapshots"), 1234),
            PathBuf::from("snapshots").join("step-1234.json")
        );
    }
}
```

- [ ] **Step 2: Run the binary unit tests**

Run:

```bash
cargo test --bin random-rewrite -- --nocapture
```

Expected: 8 tests pass.

- [ ] **Step 3: Run formatting**

Run:

```bash
cargo fmt
```

Expected: command exits successfully and keeps `src/bin/random-rewrite.rs` formatted.

- [ ] **Step 4: Run the binary unit tests after formatting**

Run:

```bash
cargo test --bin random-rewrite -- --nocapture
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit the run loop**

Run:

```bash
git add src/bin/random-rewrite.rs
git commit -m "feat: implement random rewrite CLI loop"
```

Expected: commit succeeds with only `src/bin/random-rewrite.rs` staged.

---

### Task 4: Add Process-Level CLI Tests

**Files:**
- Create: `tests/random_rewrite_cli.rs`

- [ ] **Step 1: Add process-level tests**

Create `tests/random_rewrite_cli.rs`:

```rust
use gristmill_symbolics::io::{from_json, read_json, write_json};
use gristmill_symbolics::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorId, Term,
};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "gristmill-symbolics-random-rewrite-{name}-{}-{nanos}",
        std::process::id()
    ))
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "status: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_random_rewrite() -> Command {
    Command::new(env!("CARGO_BIN_EXE_random-rewrite"))
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

fn write_input(name: &str, comp: &TensorComputation) -> PathBuf {
    let path = unique_path(name).with_extension("json");
    write_json(&path, comp).unwrap();
    path
}

#[test]
fn steps_zero_writes_unchanged_output() {
    let comp = basic_fixture();
    let input = write_input("steps-zero-input", &comp);
    let output = unique_path("steps-zero-output").with_extension("json");

    let process_output = run_random_rewrite()
        .arg("--steps")
        .arg("0")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();

    assert_success(&process_output);
    assert_eq!(read_json(&output).unwrap(), comp);

    fs::remove_file(input).ok();
    fs::remove_file(output).ok();
}

#[test]
fn early_stop_when_no_action_space_exists_succeeds() {
    let comp = basic_fixture();
    let input = write_input("early-stop-input", &comp);
    let output = unique_path("early-stop-output").with_extension("json");
    let snapshot_dir = unique_path("early-stop-snapshots");

    let process_output = run_random_rewrite()
        .arg("--steps")
        .arg("3")
        .arg("--snapshot-dir")
        .arg(&snapshot_dir)
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();

    assert_success(&process_output);
    assert_eq!(read_json(&output).unwrap(), comp);
    assert_eq!(read_json(snapshot_dir.join("step-000.json")).unwrap(), comp);
    assert!(!snapshot_dir.join("step-001.json").exists());

    fs::remove_file(input).ok();
    fs::remove_file(output).ok();
    fs::remove_dir_all(snapshot_dir).ok();
}

#[test]
fn smoke_run_on_actionable_computation_writes_valid_json() {
    let comp = comp_with_shared_left_candidate();
    let input = write_input("smoke-input", &comp);
    let output = unique_path("smoke-output").with_extension("json");

    let process_output = run_random_rewrite()
        .arg("--seed")
        .arg("42")
        .arg("--steps")
        .arg("1")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();

    assert_success(&process_output);
    let rewritten = read_json(&output).unwrap();
    assert_eq!(rewritten.validate(), Ok(()));
    assert_eq!(rewritten.tensors().len(), comp.tensors().len() + 2);
    assert_eq!(rewritten.definitions().len(), comp.definitions().len() + 2);

    fs::remove_file(input).ok();
    fs::remove_file(output).ok();
}

#[test]
fn snapshots_record_initial_and_successful_rewrite_states() {
    let comp = comp_with_shared_left_candidate();
    let input = write_input("snapshot-input", &comp);
    let output = unique_path("snapshot-output").with_extension("json");
    let snapshot_dir = unique_path("snapshot-dir");

    let process_output = run_random_rewrite()
        .arg("--seed")
        .arg("42")
        .arg("--steps")
        .arg("1")
        .arg("--snapshot-dir")
        .arg(&snapshot_dir)
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();

    assert_success(&process_output);
    assert_eq!(read_json(snapshot_dir.join("step-000.json")).unwrap(), comp);
    let after_one = read_json(snapshot_dir.join("step-001.json")).unwrap();
    assert_eq!(after_one.validate(), Ok(()));
    assert_eq!(read_json(&output).unwrap(), after_one);
    assert!(!snapshot_dir.join("step-002.json").exists());

    fs::remove_file(input).ok();
    fs::remove_file(output).ok();
    fs::remove_dir_all(snapshot_dir).ok();
}

#[test]
fn random_subsets_run_writes_valid_json() {
    let comp = comp_with_shared_left_candidate();
    let input = write_input("random-subsets-input", &comp);
    let output = unique_path("random-subsets-output").with_extension("json");

    let process_output = run_random_rewrite()
        .arg("--seed")
        .arg("5")
        .arg("--steps")
        .arg("1")
        .arg("--random-subsets")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();

    assert_success(&process_output);
    let rewritten = read_json(&output).unwrap();
    assert_eq!(rewritten.validate(), Ok(()));

    fs::remove_file(input).ok();
    fs::remove_file(output).ok();
}
```

- [ ] **Step 2: Run the new integration tests**

Run:

```bash
cargo test --test random_rewrite_cli -- --nocapture
```

Expected: 5 tests pass. If any process exits nonzero, the test failure prints stdout and stderr from the CLI.

- [ ] **Step 3: Run all tests**

Run:

```bash
cargo test
```

Expected: all existing tests plus the new binary and integration tests pass.

- [ ] **Step 4: Run formatting**

Run:

```bash
cargo fmt
```

Expected: command exits successfully.

- [ ] **Step 5: Run all tests after formatting**

Run:

```bash
cargo test
```

Expected: all tests pass.

- [ ] **Step 6: Commit process-level coverage**

Run:

```bash
git add tests/random_rewrite_cli.rs src/bin/random-rewrite.rs
git commit -m "test: cover random rewrite CLI behavior"
```

Expected: commit succeeds. Include `src/bin/random-rewrite.rs` only if `cargo fmt` changed it during this task.

---

### Task 5: Final Verification

**Files:**
- Verify: `Cargo.toml`
- Verify: `Cargo.lock`
- Verify: `src/bin/random-rewrite.rs`
- Verify: `tests/random_rewrite_cli.rs`

- [ ] **Step 1: Confirm the named binary works from Cargo**

Run:

```bash
cargo run --bin random-rewrite -- --help
```

Expected: command exits successfully and help output includes these options:

```text
--seed
--steps
--random-subsets
--snapshot-dir
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
cargo test
```

Expected: all tests pass.

- [ ] **Step 3: Check the final diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: no unstaged formatting drift. Any remaining changes are exactly the intended CLI files if earlier task commits were skipped:

```text
Cargo.toml
Cargo.lock
src/main.rs
src/bin/random-rewrite.rs
tests/random_rewrite_cli.rs
```

- [ ] **Step 4: Commit any remaining verification changes**

If `git status --short` shows remaining intended changes, run:

```bash
git add Cargo.toml Cargo.lock src/main.rs src/bin/random-rewrite.rs tests/random_rewrite_cli.rs
git commit -m "feat: add random rewrite CLI"
```

Expected: commit succeeds, or there are no changes left to commit because earlier task commits already captured everything.
