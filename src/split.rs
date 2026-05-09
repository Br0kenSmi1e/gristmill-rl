use crate::repr::{Index, IndexId, Rational, TensorDef, Term};
use std::collections::{HashMap, HashSet};

const MAX_MASK_BITS: usize = 64;

type FactorSubset = u64;

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

struct TermIndexInfo {
    factor_sum_bits: Vec<u64>,
    factor_external_bits: Vec<u64>,
}

pub fn enumerate_splits(term: &Term, def: &TensorDef) -> Result<Vec<Split>, SplitError> {
    if term.factors.len() < 2 {
        return Ok(vec![]);
    }

    validate_mask_len(term.factors.len(), SplitLimit::Factors)?;
    let info = build_term_index_info(term, def)?;
    let full = full_factor_mask(term.factors.len());
    let mut out = Vec::new();

    for mut left in 1..full {
        let mut right = full ^ left;
        if left >= right {
            continue;
        }

        if subset_external_bits(&info, left) > subset_external_bits(&info, right) {
            std::mem::swap(&mut left, &mut right);
        }

        out.push(make_split(term, def, &info, left, right));
    }

    Ok(out)
}

enum SplitLimit {
    Factors,
    SumIndices,
    ExternalIndices,
}

fn validate_mask_len(len: usize, limit: SplitLimit) -> Result<(), SplitError> {
    if len <= MAX_MASK_BITS {
        return Ok(());
    }

    match limit {
        SplitLimit::Factors => Err(SplitError::TooManyFactors {
            len,
            max: MAX_MASK_BITS,
        }),
        SplitLimit::SumIndices => Err(SplitError::TooManySumIndices {
            len,
            max: MAX_MASK_BITS,
        }),
        SplitLimit::ExternalIndices => Err(SplitError::TooManyExternalIndices {
            len,
            max: MAX_MASK_BITS,
        }),
    }
}

fn build_term_index_info(term: &Term, def: &TensorDef) -> Result<TermIndexInfo, SplitError> {
    validate_mask_len(term.sum_indices.len(), SplitLimit::SumIndices)?;
    validate_mask_len(def.ext_indices.len(), SplitLimit::ExternalIndices)?;

    let sum_positions = index_positions(&term.sum_indices);
    let external_positions = index_positions(&def.ext_indices);
    let mut factor_sum_bits = Vec::with_capacity(term.factors.len());
    let mut factor_external_bits = Vec::with_capacity(term.factors.len());

    for factor in &term.factors {
        let mut sum_bits = 0;
        let mut external_bits = 0;

        for index in &factor.indices {
            if let Some(position) = sum_positions.get(index) {
                sum_bits |= bit(*position);
            }
            if let Some(position) = external_positions.get(index) {
                external_bits |= bit(*position);
            }
        }

        factor_sum_bits.push(sum_bits);
        factor_external_bits.push(external_bits);
    }

    Ok(TermIndexInfo {
        factor_sum_bits,
        factor_external_bits,
    })
}

fn index_positions(indices: &[Index]) -> HashMap<IndexId, usize> {
    indices
        .iter()
        .enumerate()
        .map(|(position, index)| (index.id, position))
        .collect()
}

fn bit(position: usize) -> u64 {
    1_u64 << position
}

fn full_factor_mask(len: usize) -> FactorSubset {
    if len == MAX_MASK_BITS {
        u64::MAX
    } else {
        (1_u64 << len) - 1
    }
}

fn subset_sum_bits(info: &TermIndexInfo, subset: FactorSubset) -> u64 {
    let mut out = 0;
    for (position, bits) in info.factor_sum_bits.iter().enumerate() {
        if subset & bit(position) != 0 {
            out |= bits;
        }
    }
    out
}

fn subset_external_bits(info: &TermIndexInfo, subset: FactorSubset) -> u64 {
    let mut out = 0;
    for (position, bits) in info.factor_external_bits.iter().enumerate() {
        if subset & bit(position) != 0 {
            out |= bits;
        }
    }
    out
}

fn contracted_sum_bits(info: &TermIndexInfo, left: FactorSubset, right: FactorSubset) -> u64 {
    subset_sum_bits(info, left) & subset_sum_bits(info, right)
}

fn indices_from_mask(source: &[Index], mask: u64) -> Vec<Index> {
    let mut out: Vec<_> = source
        .iter()
        .enumerate()
        .filter_map(|(position, index)| {
            if mask & bit(position) == 0 {
                None
            } else {
                Some(*index)
            }
        })
        .collect();
    out.sort_by_key(|index| index.id);
    out
}

fn make_subterm(term: &Term, subset: FactorSubset, contracted_sum_bits: u64) -> Term {
    let factors: Vec<_> = term
        .factors
        .iter()
        .enumerate()
        .filter_map(|(position, factor)| {
            if subset & bit(position) == 0 {
                None
            } else {
                Some(factor.clone())
            }
        })
        .collect();

    let selected_indices: HashSet<_> = factors
        .iter()
        .flat_map(|factor| factor.indices.iter().copied())
        .collect();
    let sum_indices = term
        .sum_indices
        .iter()
        .enumerate()
        .filter_map(|(position, index)| {
            if selected_indices.contains(&index.id) && contracted_sum_bits & bit(position) == 0 {
                Some(*index)
            } else {
                None
            }
        })
        .collect();

    Term {
        coeff: Rational::new(1, 1),
        sum_indices,
        factors,
    }
}

fn make_interface(
    term: &Term,
    def: &TensorDef,
    info: &TermIndexInfo,
    left: FactorSubset,
    right: FactorSubset,
) -> SplitInterface {
    SplitInterface {
        left_external: indices_from_mask(&def.ext_indices, subset_external_bits(info, left)),
        right_external: indices_from_mask(&def.ext_indices, subset_external_bits(info, right)),
        contracted: indices_from_mask(&term.sum_indices, contracted_sum_bits(info, left, right)),
    }
}

fn make_split(
    term: &Term,
    def: &TensorDef,
    info: &TermIndexInfo,
    left: FactorSubset,
    right: FactorSubset,
) -> Split {
    let contracted = contracted_sum_bits(info, left, right);

    Split {
        left: make_subterm(term, left, contracted),
        right: make_subterm(term, right, contracted),
        interface: make_interface(term, def, info, left, right),
    }
}
