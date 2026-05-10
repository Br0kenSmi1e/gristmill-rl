use crate::repr::{Index, Range, RangeId, TensorComputation, TensorDef, Term};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CostError {
    ZeroRangeSize { range: RangeId },
    ZeroTotalFlops,
}

pub fn log_total_flops(comp: &TensorComputation) -> Result<f64, CostError> {
    let mut total = None;

    for def in comp.definitions() {
        for term in &def.terms {
            let log_term = log_term_flops(term, def, comp.ranges());
            total = Some(match total {
                None => log_term,
                Some(acc) => logaddexp(acc, log_term),
            });
        }
    }

    total.ok_or(CostError::ZeroTotalFlops)
}

fn log_term_flops(term: &Term, def: &TensorDef, ranges: &[Range]) -> f64 {
    let log_ext_size = log_size_product(&def.ext_indices, ranges);
    let log_sum_size = log_size_product(&term.sum_indices, ranges);

    if term.sum_indices.is_empty() {
        std::f64::consts::LN_2 + log_ext_size
    } else {
        logaddexp(
            std::f64::consts::LN_2 + log_ext_size + log_sum_size,
            log_ext_size,
        )
    }
}

fn log_size_product(indices: &[Index], ranges: &[Range]) -> f64 {
    indices
        .iter()
        .map(|index| (ranges[index.range.0 as usize].size as f64).ln())
        .sum()
}

fn logaddexp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::{Factor, IndexId, Rational, TensorId};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-10,
            "actual {actual}, expected {expected}"
        );
    }

    fn assert_log_close(actual: Result<f64, CostError>, expected: f64) {
        assert_close(actual.unwrap(), expected);
    }

    fn idx(id: u32, range: RangeId) -> Index {
        Index {
            id: IndexId(id),
            range,
        }
    }

    fn factor(tensor: TensorId, indices: &[u32]) -> Factor {
        Factor {
            tensor,
            indices: indices.iter().copied().map(IndexId).collect(),
        }
    }

    fn term(sum_indices: Vec<Index>, factors: Vec<Factor>) -> Term {
        Term {
            coeff: Rational::new(1, 1),
            sum_indices,
            factors,
        }
    }

    #[test]
    fn empty_computation_returns_zero_total_flops() {
        let comp = TensorComputation::new();

        assert_eq!(log_total_flops(&comp), Err(CostError::ZeroTotalFlops));
    }

    #[test]
    fn logaddexp_combines_equal_values() {
        let actual = logaddexp(2.0, 2.0);

        assert_close(actual, 2.0 + std::f64::consts::LN_2);
    }

    #[test]
    fn logaddexp_keeps_larger_strongly_separated_value() {
        let actual = logaddexp(1000.0, 0.0);

        assert_close(actual, 1000.0);
    }

    #[test]
    fn simple_contraction_log_flops_matches_raw_formula() {
        let mut comp = TensorComputation::new();
        let range = comp.add_range(10);
        let a = comp.add_tensor(vec![]);
        let b = comp.add_tensor(vec![]);
        let out = comp.add_tensor(vec![]);

        comp.add_definition(
            out,
            vec![idx(0, range), idx(1, range)],
            vec![term(
                vec![idx(2, range)],
                vec![factor(a, &[0, 2]), factor(b, &[2, 1])],
            )],
        );

        assert_log_close(log_total_flops(&comp), 2100.0_f64.ln());
    }

    #[test]
    fn no_summation_log_flops_matches_raw_formula() {
        let mut comp = TensorComputation::new();
        let range = comp.add_range(10);
        let a = comp.add_tensor(vec![]);
        let out = comp.add_tensor(vec![]);

        comp.add_definition(
            out,
            vec![idx(0, range), idx(1, range)],
            vec![term(vec![], vec![factor(a, &[0, 1])])],
        );

        assert_log_close(log_total_flops(&comp), 200.0_f64.ln());
    }

    #[test]
    fn scalar_output_log_flops_matches_raw_formula() {
        let mut comp = TensorComputation::new();
        let range = comp.add_range(10);
        let a = comp.add_tensor(vec![]);
        let b = comp.add_tensor(vec![]);
        let out = comp.add_tensor(vec![]);

        comp.add_definition(
            out,
            vec![],
            vec![term(
                vec![idx(0, range), idx(1, range)],
                vec![factor(a, &[0, 1]), factor(b, &[0, 1])],
            )],
        );

        assert_log_close(log_total_flops(&comp), 201.0_f64.ln());
    }

    #[test]
    fn multiple_definitions_fold_with_logaddexp() {
        let mut comp = TensorComputation::new();
        let range = comp.add_range(10);
        let a = comp.add_tensor(vec![]);
        let b = comp.add_tensor(vec![]);
        let out_1 = comp.add_tensor(vec![]);
        let out_2 = comp.add_tensor(vec![]);
        let ext = vec![idx(0, range), idx(1, range)];
        let contraction = term(
            vec![idx(2, range)],
            vec![factor(a, &[0, 2]), factor(b, &[2, 1])],
        );

        comp.add_definition(out_1, ext.clone(), vec![contraction.clone()]);
        comp.add_definition(out_2, ext, vec![contraction]);

        assert_log_close(log_total_flops(&comp), 4200.0_f64.ln());
    }
}
