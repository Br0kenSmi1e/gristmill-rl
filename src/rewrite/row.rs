use super::scalar::{ActionSpace, Decision, RewriteError, RewriteState, validate_decision};
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

    pub fn validate_actions_for_row(
        &self,
        action_spaces: &ActionSpaceRow,
        decisions: &[Option<Decision>],
        action_score_mask: &[bool],
    ) -> Result<ValidatedActionRow, RewriteError> {
        check_row_len(
            "validate_actions_for_row",
            "action_space_row",
            self.states.len(),
            action_spaces.len(),
        )?;
        check_row_len(
            "validate_actions_for_row",
            "decisions",
            self.states.len(),
            decisions.len(),
        )?;
        check_row_len(
            "validate_actions_for_row",
            "action_score_mask",
            self.states.len(),
            action_score_mask.len(),
        )?;

        let results: Vec<Result<ValidatedActionEntry, RewriteError>> = action_spaces
            .entries
            .par_iter()
            .zip(decisions.par_iter())
            .zip(action_score_mask.par_iter())
            .enumerate()
            .map(|(sample, ((entry, decision), scored))| {
                validate_action_entry(sample, entry, decision.as_ref(), *scored)
            })
            .collect();
        let entries = results.into_iter().collect::<Result<Vec<_>, _>>()?;

        Ok(ValidatedActionRow { entries })
    }

    pub fn apply_validated_actions_for_row(
        &mut self,
        validated: &ValidatedActionRow,
    ) -> Result<Vec<bool>, RewriteError> {
        check_row_len(
            "apply_validated_actions_for_row",
            "validated_action_row",
            self.states.len(),
            validated.len(),
        )?;

        let results: Vec<Result<bool, RewriteError>> = self
            .states
            .par_iter_mut()
            .zip(validated.entries.par_iter())
            .enumerate()
            .map(|(sample, (state, entry))| apply_validated_action_entry(state, sample, entry))
            .collect();

        results.into_iter().collect()
    }
}

impl ActionSpaceRow {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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

impl ValidatedActionEntry {
    pub fn kind(&self) -> &'static str {
        match self {
            ValidatedActionEntry::Skipped => "skipped",
            ValidatedActionEntry::Valid { .. } => "valid",
        }
    }
}

impl ValidatedActionRow {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ValidatedActionEntry] {
        &self.entries
    }

    pub fn entry_kinds(&self) -> Vec<&'static str> {
        self.entries
            .iter()
            .map(ValidatedActionEntry::kind)
            .collect()
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

fn validate_action_entry(
    sample: usize,
    entry: &ActionSpaceEntry,
    decision: Option<&Decision>,
    scored: bool,
) -> Result<ValidatedActionEntry, RewriteError> {
    if !scored {
        return Ok(ValidatedActionEntry::Skipped);
    }

    let ActionSpaceEntry::NonEmpty(space) = entry else {
        return Err(RewriteError::ActionSpaceEntryNotActionable {
            sample,
            entry_kind: entry.kind(),
        });
    };
    let decision = decision.ok_or(RewriteError::MissingScoredDecision { sample })?;

    validate_decision(space, decision).map_err(|source| RewriteError::RowValidationFailed {
        sample,
        source: Box::new(source),
    })?;

    Ok(ValidatedActionEntry::Valid {
        space: space.clone(),
        decision: decision.clone(),
    })
}

fn apply_validated_action_entry(
    state: &mut RewriteState,
    sample: usize,
    entry: &ValidatedActionEntry,
) -> Result<bool, RewriteError> {
    let ValidatedActionEntry::Valid { space, decision } = entry else {
        return Ok(false);
    };

    state
        .apply_validated_decision(space, decision)
        .map_err(|source| RewriteError::RowApplyFailed {
            sample,
            source: Box::new(source),
        })?;
    Ok(true)
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
