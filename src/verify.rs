use crate::{
    canon::{self, CanonError, TensorSymmetryMap, build_index_pool, build_tensor_symmetry_map},
    repr::{
        Factor, Index, IndexId, RangeId, Rational, ReprError, TensorComputation, TensorDef,
        TensorId, Term,
    },
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    CanonicalizationError(CanonError),
    MissingOutputDefinition {
        tensor: TensorId,
        side: &'static str,
    },
    DuplicateOutputDefinition {
        tensor: TensorId,
        side: &'static str,
    },
    OutputArityMismatch {
        tensor: TensorId,
        lhs: usize,
        rhs: usize,
    },
    OutputDefinitionMismatch {
        tensor: TensorId,
    },
    OutputRangeMismatch {
        tensor: TensorId,
        position: usize,
        lhs: RangeId,
        rhs: RangeId,
    },
    SourceInstantiationMismatch {
        source: TensorId,
        factor_arity: usize,
        source_arity: usize,
    },
    MissingRangeForIndex {
        index: IndexId,
        side: &'static str,
    },
    TensorSymmetryMismatch {
        tensor: TensorId,
    },
    ReprValidationError(ReprError),
}

impl From<CanonError> for VerifyError {
    fn from(err: CanonError) -> Self {
        Self::CanonicalizationError(err)
    }
}

impl From<ReprError> for VerifyError {
    fn from(err: ReprError) -> Self {
        Self::ReprValidationError(err)
    }
}

#[derive(Hash, Eq, PartialEq)]
struct TermStructure {
    sum_indices: Vec<Index>,
    factors: Vec<Factor>,
}

impl From<&Term> for TermStructure {
    fn from(term: &Term) -> Self {
        Self {
            sum_indices: term.sum_indices.clone(),
            factors: term.factors.clone(),
        }
    }
}

pub fn equivalent_computations(
    lhs: &TensorComputation,
    rhs: &TensorComputation,
    outputs: &[TensorId],
) -> Result<bool, VerifyError> {
    lhs.validate()?;
    rhs.validate()?;

    let output_set = deduplicate_outputs(outputs)?;
    let symmetry = merged_tensor_symmetry(lhs, rhs)?;

    let start_fresh = max_index_id_union(lhs, rhs);
    let mut lhs_fresh = start_fresh;
    let mut rhs_fresh = start_fresh;

    let lhs_inlined = inline_all_intermediates(lhs, &output_set, &mut lhs_fresh)?;
    let rhs_inlined = inline_all_intermediates(rhs, &output_set, &mut rhs_fresh)?;

    for output in &output_set {
        let lhs_output = output_definition(&lhs_inlined, *output)?;
        let rhs_output = output_definition(&rhs_inlined, *output)?;
        if lhs_output.is_none() && rhs_output.is_none() {
            continue;
        }

        let (lhs_output, mut rhs_output) = match (lhs_output, rhs_output) {
            (Some(lhs_output), Some(rhs_output)) => (lhs_output.clone(), rhs_output.clone()),
            (None, Some(_)) => {
                return Err(VerifyError::MissingOutputDefinition {
                    tensor: *output,
                    side: "lhs",
                });
            }
            (Some(_), None) => {
                return Err(VerifyError::MissingOutputDefinition {
                    tensor: *output,
                    side: "rhs",
                });
            }
            _ => unreachable!(),
        };

        if lhs_output.base != rhs_output.base {
            return Err(VerifyError::OutputDefinitionMismatch { tensor: *output });
        }
        align_external_indices(&lhs_output, &mut rhs_output)?;

        let lhs_indices = all_indices(&lhs_output);
        disambiguate_rhs_dummies(&mut rhs_output, &lhs_indices, &mut rhs_fresh);
        let diff = subtract_definitions(&lhs_output, &rhs_output);
        let normal = canonicalize_and_merge_terms(&diff, &symmetry)?;
        if !normal.is_empty() {
            return Ok(false);
        }
    }

    Ok(true)
}

fn deduplicate_outputs(outputs: &[TensorId]) -> Result<HashSet<TensorId>, VerifyError> {
    let mut set = HashSet::new();
    for &output in outputs {
        if !set.insert(output) {
            return Err(VerifyError::DuplicateOutputDefinition {
                tensor: output,
                side: "outputs",
            });
        }
    }
    Ok(set)
}

fn merged_tensor_symmetry(
    lhs: &TensorComputation,
    rhs: &TensorComputation,
) -> Result<TensorSymmetryMap, VerifyError> {
    let mut symmetry = build_tensor_symmetry_map(lhs.tensors());
    for tensor in rhs.tensors() {
        if let Some(existing) = symmetry.get(&tensor.id) {
            if *existing != tensor.symmetry {
                return Err(VerifyError::TensorSymmetryMismatch { tensor: tensor.id });
            }
            continue;
        }
        symmetry.insert(tensor.id, tensor.symmetry.clone());
    }
    Ok(symmetry)
}

fn inline_all_intermediates(
    comp: &TensorComputation,
    outputs: &HashSet<TensorId>,
    fresh_id: &mut IndexId,
) -> Result<Vec<TensorDef>, VerifyError> {
    let mut defs = comp.definitions().to_vec();

    while let Some(source_idx) = defs
        .iter()
        .rposition(|definition| !outputs.contains(&definition.base))
    {
        let source = defs[source_idx].clone();
        for target_idx in 0..defs.len() {
            if target_idx == source_idx {
                continue;
            }
            let target = defs[target_idx].clone();
            defs[target_idx] = inline_source_into_target(&target, &source, fresh_id)?;
        }
        defs.remove(source_idx);
    }

    Ok(defs
        .into_iter()
        .filter(|definition| outputs.contains(&definition.base))
        .collect())
}

fn inline_source_into_target(
    target: &TensorDef,
    source: &TensorDef,
    fresh_id: &mut IndexId,
) -> Result<TensorDef, VerifyError> {
    let mut terms = Vec::new();
    let visible_index_ranges = index_ranges_for_term_context(target);
    for term in &target.terms {
        terms.extend(inline_source_into_term(
            term,
            source,
            &visible_index_ranges,
            fresh_id,
        )?);
    }

    Ok(TensorDef {
        base: target.base,
        ext_indices: target.ext_indices.clone(),
        terms,
    })
}

fn inline_source_into_term(
    term: &Term,
    source: &TensorDef,
    visible_index_ranges: &HashMap<IndexId, RangeId>,
    fresh_id: &mut IndexId,
) -> Result<Vec<Term>, VerifyError> {
    let mut expanded_factors = Vec::with_capacity(term.factors.len());
    for factor in &term.factors {
        if factor.tensor == source.base {
            expanded_factors.push(instantiate_source_at_factor(
                factor,
                source,
                visible_index_ranges,
                fresh_id,
            )?);
        } else {
            expanded_factors.push(vec![Term {
                coeff: Rational::new(1, 1),
                sum_indices: vec![],
                factors: vec![factor.clone()],
            }]);
        }
    }

    let mut products: Vec<Vec<Term>> = vec![vec![]];
    for factor_expansion in expanded_factors {
        let mut next_products = Vec::with_capacity(products.len() * factor_expansion.len().max(1));
        for product in &products {
            for rhs_term in &factor_expansion {
                let mut next = product.clone();
                next.push(rhs_term.clone());
                next_products.push(next);
            }
        }
        products = next_products;
    }

    let mut output_terms = Vec::with_capacity(products.len());
    for product in products {
        let mut product_term = Term {
            coeff: term.coeff,
            sum_indices: term.sum_indices.clone(),
            factors: Vec::new(),
        };
        for part in product {
            product_term.coeff *= part.coeff;
            product_term.sum_indices.extend(part.sum_indices);
            product_term.factors.extend(part.factors);
        }
        output_terms.push(product_term);
    }

    Ok(output_terms)
}

fn instantiate_source_at_factor(
    factor: &Factor,
    source: &TensorDef,
    visible_index_ranges: &HashMap<IndexId, RangeId>,
    fresh_id: &mut IndexId,
) -> Result<Vec<Term>, VerifyError> {
    if factor.indices.len() != source.ext_indices.len() {
        return Err(VerifyError::SourceInstantiationMismatch {
            source: source.base,
            factor_arity: factor.indices.len(),
            source_arity: source.ext_indices.len(),
        });
    }

    let mut source_ext_to_factor = HashMap::new();
    for (source_index, factor_index) in source.ext_indices.iter().zip(factor.indices.iter()) {
        if !visible_index_ranges.contains_key(factor_index) {
            return Err(VerifyError::MissingRangeForIndex {
                index: *factor_index,
                side: "rhs",
            });
        }
        source_ext_to_factor.insert(source_index.id, *factor_index);
    }

    let mut results = Vec::with_capacity(source.terms.len());
    let mut next_id = *fresh_id;
    for source_term in &source.terms {
        let mut local_dummy_map = HashMap::new();
        let mut mapped_sum_indices = Vec::with_capacity(source_term.sum_indices.len());

        for source_sum in &source_term.sum_indices {
            let mapped = if let Some(mapped) = local_dummy_map.get(&source_sum.id) {
                *mapped
            } else {
                let mapped = next_id;
                next_id = IndexId(next_id.0 + 1);
                local_dummy_map.insert(source_sum.id, mapped);
                mapped
            };
            mapped_sum_indices.push(Index {
                id: mapped,
                range: source_sum.range,
            });
        }

        let mut mapped_factors = Vec::with_capacity(source_term.factors.len());
        for source_factor in &source_term.factors {
            let mut mapped_indices = Vec::with_capacity(source_factor.indices.len());
            for source_index in &source_factor.indices {
                if let Some(mapped) = source_ext_to_factor.get(source_index) {
                    mapped_indices.push(*mapped);
                    continue;
                }

                let Some(mapped) = local_dummy_map.get(source_index).copied() else {
                    return Err(VerifyError::MissingRangeForIndex {
                        index: *source_index,
                        side: "source-term",
                    });
                };
                mapped_indices.push(mapped);
            }

            mapped_factors.push(Factor {
                tensor: source_factor.tensor,
                indices: mapped_indices,
            });
        }

        results.push(Term {
            coeff: source_term.coeff,
            sum_indices: mapped_sum_indices,
            factors: mapped_factors,
        });
    }

    *fresh_id = next_id;
    Ok(results)
}

fn align_external_indices(lhs: &TensorDef, rhs: &mut TensorDef) -> Result<(), VerifyError> {
    if lhs.ext_indices.len() != rhs.ext_indices.len() {
        return Err(VerifyError::OutputArityMismatch {
            tensor: lhs.base,
            lhs: lhs.ext_indices.len(),
            rhs: rhs.ext_indices.len(),
        });
    }

    let mut external_mapping = HashMap::new();
    for (position, (lhs_index, rhs_index)) in lhs
        .ext_indices
        .iter()
        .zip(rhs.ext_indices.iter())
        .enumerate()
    {
        if lhs_index.range != rhs_index.range {
            return Err(VerifyError::OutputRangeMismatch {
                tensor: lhs.base,
                position,
                lhs: lhs_index.range,
                rhs: rhs_index.range,
            });
        }
        external_mapping.insert(rhs_index.id, lhs_index.id);
    }

    for index in &mut rhs.ext_indices {
        if let Some(aligned) = external_mapping.get(&index.id) {
            index.id = *aligned;
        }
    }

    for term in &mut rhs.terms {
        for factor in &mut term.factors {
            for index in &mut factor.indices {
                if let Some(aligned) = external_mapping.get(index) {
                    *index = *aligned;
                }
            }
        }
    }

    Ok(())
}

fn disambiguate_rhs_dummies(
    rhs: &mut TensorDef,
    lhs_indices: &HashSet<IndexId>,
    fresh_id: &mut IndexId,
) {
    let mut rhs_external = HashSet::new();
    for index in &rhs.ext_indices {
        rhs_external.insert(index.id);
    }

    let mut rhs_to_fresh = HashMap::new();
    for term in &rhs.terms {
        for sum_index in &term.sum_indices {
            if rhs_external.contains(&sum_index.id) {
                continue;
            }
            if lhs_indices.contains(&sum_index.id) && !rhs_to_fresh.contains_key(&sum_index.id) {
                rhs_to_fresh.insert(sum_index.id, *fresh_id);
                *fresh_id = IndexId(fresh_id.0 + 1);
            }
        }
        for factor in &term.factors {
            for index in &factor.indices {
                if rhs_external.contains(index) {
                    continue;
                }
                if lhs_indices.contains(index) && !rhs_to_fresh.contains_key(index) {
                    rhs_to_fresh.insert(*index, *fresh_id);
                    *fresh_id = IndexId(fresh_id.0 + 1);
                }
            }
        }
    }

    if rhs_to_fresh.is_empty() {
        return;
    }

    for term in &mut rhs.terms {
        for sum_index in &mut term.sum_indices {
            if let Some(renamed) = rhs_to_fresh.get(&sum_index.id) {
                sum_index.id = *renamed;
            }
        }
        for factor in &mut term.factors {
            for index in &mut factor.indices {
                if let Some(renamed) = rhs_to_fresh.get(index) {
                    *index = *renamed;
                }
            }
        }
    }
}

fn subtract_definitions(lhs: &TensorDef, rhs: &TensorDef) -> TensorDef {
    let mut terms = lhs.terms.clone();
    for term in &rhs.terms {
        let mut negated = term.clone();
        negated.coeff = -negated.coeff;
        terms.push(negated);
    }

    TensorDef {
        base: lhs.base,
        ext_indices: lhs.ext_indices.clone(),
        terms,
    }
}

fn canonicalize_and_merge_terms(
    def: &TensorDef,
    symmetry: &TensorSymmetryMap,
) -> Result<HashMap<TermStructure, Rational>, VerifyError> {
    let pool = build_index_pool(def);
    let mut merged = HashMap::new();

    for term in &def.terms {
        let canon_term = canon::canon_term(term, symmetry, &pool)?;
        if canon_term.coeff == Rational::new(0, 1) {
            continue;
        }

        let key = TermStructure::from(&canon_term);
        let entry = merged.entry(key).or_insert(Rational::new(0, 1));
        *entry += canon_term.coeff;
        if *entry == Rational::new(0, 1) {
            let key = TermStructure::from(&canon_term);
            merged.remove(&key);
        }
    }

    Ok(merged)
}

fn output_definition<'a>(
    defs: &'a [TensorDef],
    output: TensorId,
) -> Result<Option<&'a TensorDef>, VerifyError> {
    let mut matches = defs.iter().filter(|def| def.base == output);
    let first = match matches.next() {
        Some(definition) => definition,
        None => return Ok(None),
    };

    if matches.next().is_some() {
        return Err(VerifyError::DuplicateOutputDefinition {
            tensor: output,
            side: "outputs",
        });
    }

    Ok(Some(first))
}

fn all_indices(def: &TensorDef) -> HashSet<IndexId> {
    let mut ids = HashSet::new();
    for index in &def.ext_indices {
        ids.insert(index.id);
    }
    for term in &def.terms {
        for index in &term.sum_indices {
            ids.insert(index.id);
        }
        for factor in &term.factors {
            for index in &factor.indices {
                ids.insert(*index);
            }
        }
    }
    ids
}

fn index_ranges_for_term_context(def: &TensorDef) -> HashMap<IndexId, RangeId> {
    let mut ranges = HashMap::new();
    for index in &def.ext_indices {
        ranges.insert(index.id, index.range);
    }
    for term in &def.terms {
        for index in &term.sum_indices {
            ranges.insert(index.id, index.range);
        }
    }
    ranges
}

fn max_index_id_from_comp(comp: &TensorComputation) -> Option<IndexId> {
    let mut max = None;
    for definition in comp.definitions() {
        for index in &definition.ext_indices {
            let value = index.id.0;
            max = Some(max.map_or(value, |current: u32| current.max(value)));
        }
        for term in &definition.terms {
            for index in &term.sum_indices {
                let value = index.id.0;
                max = Some(max.map_or(value, |current: u32| current.max(value)));
            }
            for factor in &term.factors {
                for index in &factor.indices {
                    let value = index.0;
                    max = Some(max.map_or(value, |current: u32| current.max(value)));
                }
            }
        }
    }
    max.map(IndexId)
}

fn max_index_id_union(lhs: &TensorComputation, rhs: &TensorComputation) -> IndexId {
    let max = max_index_id_from_comp(lhs)
        .into_iter()
        .chain(max_index_id_from_comp(rhs))
        .map(|id| id.0)
        .max()
        .unwrap_or(0);
    if max == 0 && max_index_id_from_comp(lhs).is_none() && max_index_id_from_comp(rhs).is_none() {
        IndexId(0)
    } else {
        IndexId(max + 1)
    }
}
