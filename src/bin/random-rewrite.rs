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
