use num::rational::Ratio;
use serde::{Deserialize, Serialize};

pub type Rational = Ratio<i64>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RangeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymAction {
    Identity,
    Negate,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymGenerator {
    pub perm: Vec<usize>,
    pub action: SymAction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReprError {
    RangeIdMismatch {
        position: usize,
        found: RangeId,
    },
    TensorIdMismatch {
        position: usize,
        found: TensorId,
    },
    UnknownRange {
        range: RangeId,
    },
    UnknownTensor {
        tensor: TensorId,
    },
    UnknownIndex {
        def_index: usize,
        term_index: usize,
        index: IndexId,
    },
    InconsistentIndexRange {
        def_index: usize,
        index: IndexId,
        first: RangeId,
        second: RangeId,
    },
    DuplicateExternalIndex {
        def_index: usize,
        index: IndexId,
    },
    ExternalAndSumIndexOverlap {
        def_index: usize,
        index: IndexId,
    },
    DuplicateSumIndex {
        def_index: usize,
        term_index: usize,
        index: IndexId,
    },
    InvalidPermutation {
        perm: Vec<usize>,
    },
    SymmetryArityMismatch {
        expected: usize,
        got: usize,
    },
}

impl SymAction {
    pub fn combine(self, other: SymAction) -> SymAction {
        match (self, other) {
            (SymAction::Identity, rhs) => rhs,
            (SymAction::Negate, SymAction::Identity) => SymAction::Negate,
            (SymAction::Negate, SymAction::Negate) => SymAction::Identity,
        }
    }
}

impl SymGenerator {
    pub fn apply<T: Copy>(&self, indices: &[T]) -> Result<(Vec<T>, SymAction), ReprError> {
        if self.perm.len() != indices.len() {
            return Err(ReprError::SymmetryArityMismatch {
                expected: self.perm.len(),
                got: indices.len(),
            });
        }

        let mut seen = vec![false; self.perm.len()];
        for &position in &self.perm {
            if position >= self.perm.len() || seen[position] {
                return Err(ReprError::InvalidPermutation {
                    perm: self.perm.clone(),
                });
            }
            seen[position] = true;
        }

        Ok((
            self.perm
                .iter()
                .map(|&position| indices[position])
                .collect(),
            self.action,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Range {
    pub id: RangeId,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Index {
    pub id: IndexId,
    pub range: RangeId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub id: TensorId,
    pub symmetry: Vec<SymGenerator>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Factor {
    pub tensor: TensorId,
    pub indices: Vec<IndexId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Term {
    pub coeff: Rational,
    pub sum_indices: Vec<Index>,
    pub factors: Vec<Factor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDef {
    pub base: TensorId,
    pub ext_indices: Vec<Index>,
    pub terms: Vec<Term>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorComputation {
    ranges: Vec<Range>,
    tensors: Vec<TensorInfo>,
    definitions: Vec<TensorDef>,
}

impl TensorComputation {
    pub fn new() -> Self {
        Self {
            ranges: vec![],
            tensors: vec![],
            definitions: vec![],
        }
    }

    pub fn add_range(&mut self, size: u64) -> RangeId {
        let id = RangeId(self.ranges.len() as u32);
        self.ranges.push(Range { id, size });
        id
    }

    pub fn add_tensor(&mut self, symmetry: Vec<SymGenerator>) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorInfo { id, symmetry });
        id
    }

    pub fn add_definition(
        &mut self,
        base: TensorId,
        ext_indices: Vec<Index>,
        terms: Vec<Term>,
    ) {
        self.definitions.push(TensorDef {
            base,
            ext_indices,
            terms,
        });
    }

    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn definitions(&self) -> &[TensorDef] {
        &self.definitions
    }

    pub fn definitions_mut(&mut self) -> &mut Vec<TensorDef> {
        &mut self.definitions
    }

    pub fn next_tensor_id(&self) -> TensorId {
        TensorId(self.tensors.len() as u32)
    }
}

impl Default for TensorComputation {
    fn default() -> Self {
        Self::new()
    }
}
