use crate::repr::{
    Factor, Index, IndexId, RangeId, Rational, SymAction, SymGenerator, TensorDef, TensorId,
    TensorInfo, Term,
};
use crate::split::Split;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

pub type IndexPool = HashMap<RangeId, Vec<IndexId>>;
pub type TensorSymmetryMap = HashMap<TensorId, Vec<SymGenerator>>;
type DummyRange = HashMap<IndexId, RangeId>;

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
    let dummy_range = build_term_dummy_range(term);
    let mut candidates = Vec::new();

    for sym_term in enumerate_symmetry_terms(term, symmetry)? {
        for ordered in enumerate_ordered_terms(&sym_term, &dummy_range) {
            candidates.push(rename_standalone_term(&ordered, &dummy_range, pool)?);
        }
    }

    let index = choose_min_term_index(&candidates)?;
    Ok(candidates[index].clone())
}

pub fn canon_split(
    split: &Split,
    symmetry: &TensorSymmetryMap,
    pool: &IndexPool,
) -> Result<(Split, Split), CanonError> {
    let _ = (split, symmetry, pool);
    Err(CanonError::EmptyCanonicalCandidates)
}

fn build_term_dummy_range(term: &Term) -> DummyRange {
    term.sum_indices
        .iter()
        .map(|index| (index.id, index.range))
        .collect()
}

#[derive(Clone, Copy)]
enum IndexSlot {
    Dummy(RangeId),
    External(IndexId),
}

struct PoolAllocator<'a> {
    pool: &'a IndexPool,
    used: HashMap<RangeId, HashSet<usize>>,
}

impl<'a> PoolAllocator<'a> {
    fn new(pool: &'a IndexPool) -> Self {
        Self {
            pool,
            used: HashMap::new(),
        }
    }

    fn alloc_low(&mut self, range: RangeId) -> Result<IndexId, CanonError> {
        let ids = self
            .pool
            .get(&range)
            .ok_or(CanonError::MissingIndexPool { range })?;
        let used = self.used.entry(range).or_default();

        for (position, &id) in ids.iter().enumerate() {
            if used.insert(position) {
                return Ok(id);
            }
        }

        Err(CanonError::ExhaustedIndexPool { range })
    }

    #[allow(dead_code)]
    fn alloc_high(&mut self, range: RangeId) -> Result<IndexId, CanonError> {
        let ids = self
            .pool
            .get(&range)
            .ok_or(CanonError::MissingIndexPool { range })?;
        let used = self.used.entry(range).or_default();

        for (position, &id) in ids.iter().enumerate().rev() {
            if used.insert(position) {
                return Ok(id);
            }
        }

        Err(CanonError::ExhaustedIndexPool { range })
    }
}

#[allow(dead_code)]
fn range_for_new_id(pool: &IndexPool, id: IndexId) -> Option<RangeId> {
    pool.iter()
        .find_map(|(&range, ids)| ids.contains(&id).then_some(range))
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

fn compare_factors_by_structure(
    left: &Factor,
    right: &Factor,
    dummy_range: &DummyRange,
) -> Ordering {
    left.tensor
        .cmp(&right.tensor)
        .then_with(|| compare_index_slots(&left.indices, &right.indices, dummy_range))
}

fn compare_index_slots(left: &[IndexId], right: &[IndexId], dummy_range: &DummyRange) -> Ordering {
    for (left_id, right_id) in left.iter().zip(right) {
        let ordering = index_slot(*left_id, dummy_range).cmp(&index_slot(*right_id, dummy_range));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }

    left.len().cmp(&right.len())
}

fn index_slot(index: IndexId, dummy_range: &DummyRange) -> IndexSlot {
    if let Some(&range) = dummy_range.get(&index) {
        IndexSlot::Dummy(range)
    } else {
        IndexSlot::External(index)
    }
}

impl PartialEq for IndexSlot {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for IndexSlot {}

impl PartialOrd for IndexSlot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexSlot {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (IndexSlot::Dummy(left), IndexSlot::Dummy(right)) => left.cmp(right),
            (IndexSlot::Dummy(_), IndexSlot::External(_)) => Ordering::Less,
            (IndexSlot::External(_), IndexSlot::Dummy(_)) => Ordering::Greater,
            (IndexSlot::External(left), IndexSlot::External(right)) => left.cmp(right),
        }
    }
}

fn enumerate_ordered_terms(term: &Term, dummy_range: &DummyRange) -> Vec<Term> {
    let mut factors = term.factors.clone();
    factors.sort_by(|left, right| compare_factors_by_structure(left, right, dummy_range));
    let groups = tied_groups(&factors, dummy_range);
    let mut factor_orders = Vec::new();
    enumerate_group_permutations(&factors, &groups, 0, &mut factor_orders);

    factor_orders
        .into_iter()
        .map(|factors| Term {
            coeff: term.coeff.clone(),
            sum_indices: term.sum_indices.clone(),
            factors,
        })
        .collect()
}

fn tied_groups(factors: &[Factor], dummy_range: &DummyRange) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut start = 0;

    while start < factors.len() {
        let mut end = start + 1;
        while end < factors.len()
            && compare_factors_by_structure(&factors[start], &factors[end], dummy_range)
                == Ordering::Equal
        {
            end += 1;
        }
        groups.push((start, end));
        start = end;
    }

    groups
}

fn enumerate_group_permutations(
    factors: &[Factor],
    groups: &[(usize, usize)],
    group_index: usize,
    out: &mut Vec<Vec<Factor>>,
) {
    if group_index == groups.len() {
        out.push(factors.to_vec());
        return;
    }

    let (start, end) = groups[group_index];
    for permutation in permutations(&factors[start..end]) {
        let mut next = factors.to_vec();
        next.splice(start..end, permutation);
        enumerate_group_permutations(&next, groups, group_index + 1, out);
    }
}

fn permutations(items: &[Factor]) -> Vec<Vec<Factor>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }

    let mut out = Vec::new();
    for index in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(index);
        for mut tail in permutations(&rest) {
            let mut permutation = vec![head.clone()];
            permutation.append(&mut tail);
            out.push(permutation);
        }
    }

    out
}

fn build_rename_map<F>(
    term: &Term,
    rename_ids: &HashSet<IndexId>,
    dummy_range: &DummyRange,
    base_map: &HashMap<IndexId, IndexId>,
    allocator: &mut PoolAllocator<'_>,
    mut allocate: F,
) -> Result<HashMap<IndexId, IndexId>, CanonError>
where
    F: FnMut(&mut PoolAllocator<'_>, RangeId) -> Result<IndexId, CanonError>,
{
    let mut remap = base_map.clone();

    for factor in &term.factors {
        for &index_id in &factor.indices {
            if !rename_ids.contains(&index_id) || remap.contains_key(&index_id) {
                continue;
            }
            let range = *dummy_range
                .get(&index_id)
                .expect("rename_ids must be present in dummy_range");
            remap.insert(index_id, allocate(allocator, range)?);
        }
    }

    for index in &term.sum_indices {
        if !rename_ids.contains(&index.id) || remap.contains_key(&index.id) {
            continue;
        }
        let range = *dummy_range
            .get(&index.id)
            .expect("rename_ids must be present in dummy_range");
        remap.insert(index.id, allocate(allocator, range)?);
    }

    Ok(remap)
}

fn apply_rename_map(term: &Term, remap: &HashMap<IndexId, IndexId>) -> Term {
    let mut sum_indices: Vec<_> = term
        .sum_indices
        .iter()
        .map(|index| Index {
            id: remap.get(&index.id).copied().unwrap_or(index.id),
            range: index.range,
        })
        .collect();
    sum_indices.sort_by_key(|index| index.id);

    Term {
        coeff: term.coeff.clone(),
        sum_indices,
        factors: term
            .factors
            .iter()
            .map(|factor| Factor {
                tensor: factor.tensor,
                indices: factor
                    .indices
                    .iter()
                    .map(|index| remap.get(index).copied().unwrap_or(*index))
                    .collect(),
            })
            .collect(),
    }
}

fn rename_standalone_term(
    term: &Term,
    dummy_range: &DummyRange,
    pool: &IndexPool,
) -> Result<Term, CanonError> {
    let rename_ids: HashSet<_> = term.sum_indices.iter().map(|index| index.id).collect();
    let mut allocator = PoolAllocator::new(pool);
    let remap = build_rename_map(
        term,
        &rename_ids,
        dummy_range,
        &HashMap::new(),
        &mut allocator,
        |allocator, range| allocator.alloc_low(range),
    )?;

    Ok(apply_rename_map(term, &remap))
}

fn choose_min_term_index(candidates: &[Term]) -> Result<usize, CanonError> {
    if candidates.is_empty() {
        return Err(CanonError::EmptyCanonicalCandidates);
    }

    for left in 0..candidates.len() {
        for right in (left + 1)..candidates.len() {
            if compare_term_structure(&candidates[left], &candidates[right]) == Ordering::Equal
                && candidates[left].coeff != candidates[right].coeff
            {
                return Err(CanonError::InconsistentSymmetryCoefficient);
            }
        }
    }

    let mut best = 0;
    for index in 1..candidates.len() {
        if compare_term_structure(&candidates[index], &candidates[best]) == Ordering::Less {
            best = index;
        }
    }

    Ok(best)
}

fn compare_term_structure(left: &Term, right: &Term) -> Ordering {
    compare_indices(&left.sum_indices, &right.sum_indices)
        .then_with(|| compare_factors(&left.factors, &right.factors))
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

fn index_key(index: &Index) -> (RangeId, IndexId) {
    (index.range, index.id)
}
