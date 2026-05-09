use crate::repr::{IndexId, RangeId, SymGenerator, TensorDef, TensorId, TensorInfo, Term};
use crate::split::Split;
use std::collections::{HashMap, HashSet};

pub type IndexPool = HashMap<RangeId, Vec<IndexId>>;
pub type TensorSymmetryMap = HashMap<TensorId, Vec<SymGenerator>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonError {
    MissingTensorSymmetry {
        tensor: TensorId,
    },
    SymmetryArityMismatch {
        tensor: TensorId,
        expected: usize,
        got: usize,
    },
    InvalidSymmetryPermutation {
        tensor: TensorId,
        perm: Vec<usize>,
    },
    MissingIndexPool {
        range: RangeId,
    },
    ExhaustedIndexPool {
        range: RangeId,
    },
    EmptyCanonicalCandidates,
    InconsistentSymmetryCoefficient,
}

pub fn build_index_pool(def: &TensorDef) -> IndexPool {
    let ext_indices: HashSet<_> = def.ext_indices.iter().map(|index| index.id).collect();
    let mut pool: IndexPool = HashMap::new();

    for term in &def.terms {
        for index in &term.sum_indices {
            if !ext_indices.contains(&index.id) {
                pool.entry(index.range).or_default().push(index.id);
            }
        }
    }

    for ids in pool.values_mut() {
        ids.sort();
        ids.dedup();
    }

    pool
}

pub fn build_tensor_symmetry_map(tensors: &[TensorInfo]) -> TensorSymmetryMap {
    tensors
        .iter()
        .map(|tensor| (tensor.id, tensor.symmetry.clone()))
        .collect()
}

pub fn canon_term(
    term: &Term,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<Term, CanonError> {
    ensure_factor_symmetries(term, symmetry)?;
    let _ = pool;
    Ok(term.clone())
}

pub fn canon_split(
    split: &Split,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<(Split, Split), CanonError> {
    let _ = (split, symmetry, pool);
    Err(CanonError::EmptyCanonicalCandidates)
}

fn ensure_factor_symmetries(term: &Term, symmetry: &TensorSymmetryMap) -> Result<(), CanonError> {
    for factor in &term.factors {
        if !symmetry.contains_key(&factor.tensor) {
            return Err(CanonError::MissingTensorSymmetry {
                tensor: factor.tensor,
            });
        }
    }

    Ok(())
}
