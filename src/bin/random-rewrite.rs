use clap::Parser;
use gristmill_symbolics::io::{self, IoJsonError};
use gristmill_symbolics::repr::TensorComputation;
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
    ReadInput {
        path: PathBuf,
        source: IoJsonError,
    },
    WriteOutput {
        path: PathBuf,
        source: IoJsonError,
    },
    CreateSnapshotDir {
        path: PathBuf,
        source: std::io::Error,
    },
    WriteSnapshot {
        path: PathBuf,
        source: IoJsonError,
    },
    Rewrite {
        step: usize,
        source: RewriteError,
    },
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

fn random_decision(space: &ActionSpace, random_subsets: bool, rng: &mut impl Rng) -> Decision {
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
    use gristmill_symbolics::repr::{Rational, TensorDef, TensorId, Term};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
