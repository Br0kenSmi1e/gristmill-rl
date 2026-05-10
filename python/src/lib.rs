use pyo3::prelude::*;

pyo3::create_exception!(
    gristmill_symbolics,
    GristmillSymbolicsError,
    pyo3::exceptions::PyException
);

#[pyclass(name = "TensorComputation")]
struct PyTensorComputation;

#[pymethods]
impl PyTensorComputation {}

#[pyclass(name = "ActionSpace")]
struct PyActionSpace;

#[pymethods]
impl PyActionSpace {}

#[pymodule]
fn gristmill_symbolics(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensorComputation>()?;
    module.add_class::<PyActionSpace>()?;
    module.add(
        "GristmillSymbolicsError",
        py.get_type::<GristmillSymbolicsError>(),
    )?;
    Ok(())
}
