use super::scalar::{ActionSpace, Decision, RewriteError, RewriteState};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewriteStateRow {
    states: Vec<RewriteState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpaceRow {
    entries: Vec<ActionSpaceEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionSpaceEntry {
    Skipped,
    ExactEmpty,
    NonEmpty(ActionSpace),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedActionRow {
    entries: Vec<ValidatedActionEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidatedActionEntry {
    Skipped,
    Valid {
        space: ActionSpace,
        decision: Decision,
    },
}

impl RewriteStateRow {
    pub fn from_states(states: Vec<RewriteState>) -> Self {
        Self { states }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn states(&self) -> &[RewriteState] {
        &self.states
    }

    pub fn definition_masks(&self) -> Vec<Vec<bool>> {
        self.states
            .iter()
            .map(|state| state.definition_mask().to_vec())
            .collect()
    }

    pub fn query_action_spaces_for_row(
        &mut self,
        target_choices: &[isize],
        active_mask: &[bool],
    ) -> Result<ActionSpaceRow, RewriteError> {
        check_row_len(
            "query_action_spaces_for_row",
            "target_choices",
            self.states.len(),
            target_choices.len(),
        )?;
        check_row_len(
            "query_action_spaces_for_row",
            "active_mask",
            self.states.len(),
            active_mask.len(),
        )?;

        let results = self
            .states
            .par_iter()
            .cloned()
            .enumerate()
            .map(|(sample, mut state)| {
                let entry = query_action_space_entry(
                    &mut state,
                    sample,
                    target_choices[sample],
                    active_mask[sample],
                )?;
                Ok((state, entry))
            })
            .collect::<Result<Vec<_>, RewriteError>>()?;

        let (states, entries): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        self.states = states;

        Ok(ActionSpaceRow { entries })
    }
}

impl ActionSpaceRow {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[ActionSpaceEntry] {
        &self.entries
    }

    pub fn entry_kinds(&self) -> Vec<&'static str> {
        self.entries.iter().map(ActionSpaceEntry::kind).collect()
    }
}

impl ActionSpaceEntry {
    pub fn kind(&self) -> &'static str {
        match self {
            ActionSpaceEntry::Skipped => "skipped",
            ActionSpaceEntry::ExactEmpty => "exact_empty",
            ActionSpaceEntry::NonEmpty(_) => "non_empty",
        }
    }
}

impl ValidatedActionRow {
    pub fn entries(&self) -> &[ValidatedActionEntry] {
        &self.entries
    }
}

fn query_action_space_entry(
    state: &mut RewriteState,
    sample: usize,
    target: isize,
    active: bool,
) -> Result<ActionSpaceEntry, RewriteError> {
    if !active || target == -1 {
        return Ok(ActionSpaceEntry::Skipped);
    }
    if target < -1 {
        return Err(RewriteError::InvalidTargetChoice { sample, target });
    }

    let index =
        usize::try_from(target).expect("non-negative target choices should always fit into usize");
    let definition_mask = state.definition_mask();
    if index >= definition_mask.len() {
        return Err(RewriteError::TargetDefinitionIndexOutOfRange {
            sample,
            index,
            len: definition_mask.len(),
        });
    }
    if !definition_mask[index] {
        return Err(RewriteError::TargetDefinitionMasked { sample, index });
    }

    match state
        .action_space_for_def(index)
        .map_err(|source| RewriteError::RowQueryFailed {
            sample,
            source: Box::new(source),
        })? {
        Some(space) => Ok(ActionSpaceEntry::NonEmpty(space)),
        None => Ok(ActionSpaceEntry::ExactEmpty),
    }
}

fn check_row_len(
    operation: &'static str,
    field: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), RewriteError> {
    if got == expected {
        return Ok(());
    }

    Err(RewriteError::RowLengthMismatch {
        operation,
        field,
        expected,
        got,
    })
}
