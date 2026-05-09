use num::rational::Ratio;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

    pub fn add_definition(&mut self, base: TensorId, ext_indices: Vec<Index>, terms: Vec<Term>) {
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

    pub fn validate(&self) -> Result<(), ReprError> {
        for (position, range) in self.ranges.iter().enumerate() {
            let expected = RangeId(position as u32);
            if range.id != expected {
                return Err(ReprError::RangeIdMismatch {
                    position,
                    found: range.id,
                });
            }
        }

        for (position, tensor) in self.tensors.iter().enumerate() {
            let expected = TensorId(position as u32);
            if tensor.id != expected {
                return Err(ReprError::TensorIdMismatch {
                    position,
                    found: tensor.id,
                });
            }

            for generator in &tensor.symmetry {
                validate_permutation(&generator.perm)?;
            }
        }

        for (def_index, definition) in self.definitions.iter().enumerate() {
            self.ensure_tensor_exists(definition.base)?;

            let mut external_ranges = HashMap::new();
            for index in &definition.ext_indices {
                self.ensure_range_exists(index.range)?;
                if external_ranges.insert(index.id, index.range).is_some() {
                    return Err(ReprError::DuplicateExternalIndex {
                        def_index,
                        index: index.id,
                    });
                }
            }

            let mut definition_index_ranges = external_ranges.clone();
            for (term_index, term) in definition.terms.iter().enumerate() {
                let mut sum_index_ids = HashSet::new();
                let mut visible_ranges = external_ranges.clone();

                for index in &term.sum_indices {
                    self.ensure_range_exists(index.range)?;

                    if !sum_index_ids.insert(index.id) {
                        return Err(ReprError::DuplicateSumIndex {
                            def_index,
                            term_index,
                            index: index.id,
                        });
                    }

                    if let Some(&external_range) = external_ranges.get(&index.id) {
                        if external_range != index.range {
                            return Err(ReprError::InconsistentIndexRange {
                                def_index,
                                index: index.id,
                                first: external_range,
                                second: index.range,
                            });
                        }

                        return Err(ReprError::ExternalAndSumIndexOverlap {
                            def_index,
                            index: index.id,
                        });
                    }

                    if let Some(&first_range) = definition_index_ranges.get(&index.id) {
                        if first_range != index.range {
                            return Err(ReprError::InconsistentIndexRange {
                                def_index,
                                index: index.id,
                                first: first_range,
                                second: index.range,
                            });
                        }
                    } else {
                        definition_index_ranges.insert(index.id, index.range);
                    }

                    visible_ranges.insert(index.id, index.range);
                }

                for factor in &term.factors {
                    self.ensure_tensor_exists(factor.tensor)?;

                    for &index in &factor.indices {
                        if !visible_ranges.contains_key(&index) {
                            return Err(ReprError::UnknownIndex {
                                def_index,
                                term_index,
                                index,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn ensure_range_exists(&self, range: RangeId) -> Result<(), ReprError> {
        if (range.0 as usize) < self.ranges.len() {
            Ok(())
        } else {
            Err(ReprError::UnknownRange { range })
        }
    }

    fn ensure_tensor_exists(&self, tensor: TensorId) -> Result<(), ReprError> {
        if (tensor.0 as usize) < self.tensors.len() {
            Ok(())
        } else {
            Err(ReprError::UnknownTensor { tensor })
        }
    }
}

impl Default for TensorComputation {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_permutation(perm: &[usize]) -> Result<(), ReprError> {
    let mut seen = vec![false; perm.len()];
    for &position in perm {
        if position >= perm.len() || seen[position] {
            return Err(ReprError::InvalidPermutation {
                perm: perm.to_vec(),
            });
        }
        seen[position] = true;
    }

    Ok(())
}
