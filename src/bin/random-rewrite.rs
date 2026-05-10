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
