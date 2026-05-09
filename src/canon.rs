use crate::repr::{
    Factor, Index, IndexId, RangeId, Rational, SymAction, SymGenerator, TensorDef, TensorId,
    TensorInfo, Term,
};
use crate::split::Split;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

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
    let _ = pool;
    let candidates = enumerate_symmetry_terms(term, symmetry)?;
    let candidate = candidates
        .into_iter()
        .min_by(compare_terms)
        .ok_or(CanonError::EmptyCanonicalCandidates)?;
    Ok(candidate)
}

pub fn canon_split(
    split: &Split,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<(Split, Split), CanonError> {
    let _ = (split, symmetry, pool);
    Err(CanonError::EmptyCanonicalCandidates)
}

fn enumerate_sym_group(
    tensor: TensorId,
    generators: &[SymGenerator],
    arity: usize,
) -> Result<Vec<(Vec<usize>, SymAction)>, CanonError> {
    for generator in generators {
        validate_generator(tensor, generator, arity)?;
    }

    let identity_perm: Vec<_> = (0..arity).collect();
    let mut group = vec![(identity_perm.clone(), SymAction::Identity)];
    let mut queue = VecDeque::from([(identity_perm, SymAction::Identity)]);

    while let Some((perm, action)) = queue.pop_front() {
        for generator in generators {
            let next_perm = compose_perm(&generator.perm, &perm);
            let next_action = action.combine(generator.action);

            if !group.iter().any(|(seen_perm, seen_action)| {
                seen_perm == &next_perm && seen_action == &next_action
            }) {
                group.push((next_perm.clone(), next_action));
                queue.push_back((next_perm, next_action));
            }
        }
    }

    Ok(group)
}

fn validate_generator(
    tensor: TensorId,
    generator: &SymGenerator,
    arity: usize,
) -> Result<(), CanonError> {
    if generator.perm.len() != arity {
        return Err(CanonError::SymmetryArityMismatch {
            tensor,
            expected: generator.perm.len(),
            got: arity,
        });
    }

    let mut seen = vec![false; generator.perm.len()];
    for &position in &generator.perm {
        if position >= generator.perm.len() || seen[position] {
            return Err(CanonError::InvalidSymmetryPermutation {
                tensor,
                perm: generator.perm.clone(),
            });
        }
        seen[position] = true;
    }

    Ok(())
}

fn compose_perm(left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter().map(|&position| right[position]).collect()
}

fn enumerate_factor_variants(
    tensor: TensorId,
    indices: &[IndexId],
    generators: &[SymGenerator],
) -> Result<Vec<(Vec<IndexId>, SymAction)>, CanonError> {
    enumerate_sym_group(tensor, generators, indices.len()).map(|group| {
        group
            .into_iter()
            .map(|(perm, action)| {
                (
                    perm.into_iter().map(|position| indices[position]).collect(),
                    action,
                )
            })
            .collect()
    })
}

fn enumerate_symmetry_terms(
    term: &Term,
    symmetry: &TensorSymmetryMap,
) -> Result<Vec<Term>, CanonError> {
    let mut factor_variants = Vec::with_capacity(term.factors.len());

    for factor in &term.factors {
        let generators = symmetry
            .get(&factor.tensor)
            .ok_or(CanonError::MissingTensorSymmetry {
                tensor: factor.tensor,
            })?;
        factor_variants.push(enumerate_factor_variants(
            factor.tensor,
            &factor.indices,
            generators,
        )?);
    }

    let mut out = Vec::new();
    enumerate_symmetry_product(
        term,
        &factor_variants,
        0,
        Vec::with_capacity(term.factors.len()),
        SymAction::Identity,
        &mut out,
    );
    Ok(out)
}

fn enumerate_symmetry_product(
    term: &Term,
    factor_variants: &[Vec<(Vec<IndexId>, SymAction)>],
    position: usize,
    factors: Vec<Factor>,
    action: SymAction,
    out: &mut Vec<Term>,
) {
    if position == term.factors.len() {
        out.push(Term {
            coeff: apply_action_to_coeff(term.coeff.clone(), action),
            sum_indices: term.sum_indices.clone(),
            factors,
        });
        return;
    }

    for (indices, factor_action) in &factor_variants[position] {
        let mut next_factors = factors.clone();
        next_factors.push(Factor {
            tensor: term.factors[position].tensor,
            indices: indices.clone(),
        });
        enumerate_symmetry_product(
            term,
            factor_variants,
            position + 1,
            next_factors,
            action.combine(*factor_action),
            out,
        );
    }
}

fn apply_action_to_coeff(coeff: Rational, action: SymAction) -> Rational {
    match action {
        SymAction::Identity => coeff,
        SymAction::Negate => -coeff,
    }
}

fn compare_terms(left: &Term, right: &Term) -> Ordering {
    compare_factors(&left.factors, &right.factors)
        .then_with(|| compare_indices(&left.sum_indices, &right.sum_indices))
        .then_with(|| left.coeff.cmp(&right.coeff))
}

fn compare_factors(left: &[Factor], right: &[Factor]) -> Ordering {
    left.iter()
        .map(factor_key)
        .cmp(right.iter().map(factor_key))
}

fn factor_key(factor: &Factor) -> (TensorId, &[IndexId]) {
    (factor.tensor, &factor.indices)
}

fn compare_indices(left: &[Index], right: &[Index]) -> Ordering {
    left.iter().map(index_key).cmp(right.iter().map(index_key))
}

fn index_key(index: &Index) -> (IndexId, RangeId) {
    (index.id, index.range)
}
