use crate::repr::{RangeId, TensorComputation};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CostError {
    ZeroRangeSize { range: RangeId },
    ZeroTotalFlops,
}

pub fn log_total_flops(_comp: &TensorComputation) -> Result<f64, CostError> {
    Err(CostError::ZeroTotalFlops)
}

fn logaddexp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "actual {actual}, expected {expected}"
        );
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
}
