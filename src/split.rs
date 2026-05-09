use crate::repr::{Index, TensorDef, Term};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SplitInterface {
    pub left_external: Vec<Index>,
    pub right_external: Vec<Index>,
    pub contracted: Vec<Index>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub left: Term,
    pub right: Term,
    pub interface: SplitInterface,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SplitError {
    TooManyFactors { len: usize, max: usize },
    TooManySumIndices { len: usize, max: usize },
    TooManyExternalIndices { len: usize, max: usize },
}

pub fn enumerate_splits(term: &Term, _def: &TensorDef) -> Result<Vec<Split>, SplitError> {
    if term.factors.len() < 2 {
        return Ok(vec![]);
    }

    Ok(vec![])
}
