use crate::{py_gristmill_error, PyTensorComputation};
use ::gristmill_symbolics::repr::TensorId;
use ::gristmill_symbolics::verify::equivalent_computations;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyList};

fn parse_output_ids(outputs: &Bound<'_, PyAny>) -> PyResult<Vec<TensorId>> {
    let list = outputs
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("outputs must be a list of tensor ids"))?;

    let mut parsed = Vec::with_capacity(list.len());
    for (sample, value) in list.iter().enumerate() {
        if value.is_exact_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "outputs[{sample}] must be a non-negative integer tensor id, not bool"
            )));
        }

        let id = value.extract::<u32>().map_err(|_| {
            PyTypeError::new_err(format!(
                "outputs[{sample}] must be a non-negative integer tensor id"
            ))
        })?;
        parsed.push(TensorId(id));
    }

    Ok(parsed)
}

#[pyfunction(name = "equivalent_computations")]
fn py_equivalent_computations(
    lhs: &PyTensorComputation,
    rhs: &PyTensorComputation,
    outputs: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let outputs = parse_output_ids(outputs)?;
    equivalent_computations(&lhs.inner, &rhs.inner, &outputs).map_err(py_gristmill_error)
}

pub(crate) fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    module.add_function(wrap_pyfunction!(py_equivalent_computations, module)?)?;
    Ok(())
}
