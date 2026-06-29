use super::single::{
    action_space_for_def, apply_decision, validate_decision, ActionSpace,
    Decision, RewriteError,
};
use crate::repr::TensorComputation;
use rayon::prelude::*;

pub type ActionSpaceBatch = Vec<Option<ActionSpace>>;
pub type DecisionBatch = Vec<Option<Decision>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchField {
    Targets,
    Spaces,
    Decisions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchRewriteError {
    Single {
        sample: usize,
        source: RewriteError,
    },
    LengthMismatch {
        field: BatchField,
        expected: usize,
        got: usize,
    },
    DecisionWithoutActionSpace {
        sample: usize,
    },
}

pub fn action_spaces_for_batch(
    comps: &[TensorComputation],
    targets: &[Option<usize>],
) -> Result<ActionSpaceBatch, BatchRewriteError> {
    check_len(BatchField::Targets, comps.len(), targets.len())?;

    let results: Vec<_> = comps
        .par_iter()
        .zip(targets.par_iter())
        .enumerate()
        .map(|(sample, (comp, target))| {
            let Some(def_index) = *target else {
                return Ok(None);
            };
            action_space_for_def(comp, def_index)
                .map_err(|source| BatchRewriteError::Single { sample, source })
        })
        .collect();

    results.into_iter().collect()
}

pub fn validate_decisions_for_batch(
    spaces: &[Option<ActionSpace>],
    decisions: &[Option<Decision>],
) -> Result<(), BatchRewriteError> {
    check_len(BatchField::Decisions, spaces.len(), decisions.len())?;

    let results: Vec<_> = spaces
        .par_iter()
        .zip(decisions.par_iter())
        .enumerate()
        .map(|(sample, (space, decision))| {
            let Some(decision) = decision else {
                return Ok(());
            };
            let Some(space) = space else {
                return Err(BatchRewriteError::DecisionWithoutActionSpace {
                    sample,
                });
            };
            validate_decision(space, decision)
                .map_err(|source| BatchRewriteError::Single { sample, source })
        })
        .collect();

    results.into_iter().collect()
}

pub fn apply_decisions_for_batch(
    comps: &mut [TensorComputation],
    spaces: &[Option<ActionSpace>],
    decisions: &[Option<Decision>],
) -> Result<Vec<bool>, BatchRewriteError> {
    check_len(BatchField::Spaces, comps.len(), spaces.len())?;
    check_len(BatchField::Decisions, comps.len(), decisions.len())?;

    let results: Vec<_> = comps
        .par_iter_mut()
        .zip(spaces.par_iter())
        .zip(decisions.par_iter())
        .enumerate()
        .map(|(sample, ((comp, space), decision))| {
            let Some(decision) = decision else {
                return Ok(false);
            };
            let Some(space) = space else {
                return Err(BatchRewriteError::DecisionWithoutActionSpace {
                    sample,
                });
            };
            apply_decision(comp, space, decision).map_err(|source| {
                BatchRewriteError::Single { sample, source }
            })?;
            Ok(true)
        })
        .collect();

    results.into_iter().collect()
}

fn check_len(
    field: BatchField,
    expected: usize,
    got: usize,
) -> Result<(), BatchRewriteError> {
    if got == expected {
        return Ok(());
    }

    Err(BatchRewriteError::LengthMismatch {
        field,
        expected,
        got,
    })
}
