mod rewrite_bindings;
mod verify_bindings;

use ::gristmill_symbolics::cost;
use ::gristmill_symbolics::io;
use ::gristmill_symbolics::repr::{
    Factor, Index, Range, Rational, SymAction, SymGenerator,
    TensorComputation as RustTensorComputation, TensorDef, TensorInfo, Term,
};
use pyo3::prelude::*;
use pythonize::pythonize;
use serde_json::{Value, json};
use std::fmt;
use std::path::PathBuf;

pyo3::create_exception!(
    gristmill_symbolics,
    GristmillSymbolicsError,
    pyo3::exceptions::PyException
);

pub(crate) fn py_gristmill_error(error: impl fmt::Debug) -> PyErr {
    GristmillSymbolicsError::new_err(format!("{error:?}"))
}

pub(crate) fn py_gristmill_display_error(error: impl fmt::Display) -> PyErr {
    GristmillSymbolicsError::new_err(error.to_string())
}

fn validate_loaded(comp: RustTensorComputation) -> PyResult<PyTensorComputation> {
    comp.validate().map_err(py_gristmill_error)?;
    Ok(PyTensorComputation { inner: comp })
}

fn range_value(range: &Range) -> Value {
    json!({
        "id": range.id.0,
        "size": range.size,
    })
}

fn tensor_value(tensor: &TensorInfo) -> Value {
    json!({
        "id": tensor.id.0,
        "symmetry": tensor.symmetry.iter().map(sym_generator_value).collect::<Vec<_>>(),
    })
}

fn sym_generator_value(generator: &SymGenerator) -> Value {
    json!({
        "perm": generator.perm,
        "action": sym_action_name(generator.action),
    })
}

fn sym_action_name(action: SymAction) -> &'static str {
    match action {
        SymAction::Identity => "Identity",
        SymAction::Negate => "Negate",
    }
}

pub(crate) fn tensor_def_value(definition: &TensorDef) -> Value {
    json!({
        "base": definition.base.0,
        "ext_indices": definition.ext_indices.iter().map(index_value).collect::<Vec<_>>(),
        "terms": definition.terms.iter().map(term_value).collect::<Vec<_>>(),
    })
}

fn term_value(term: &Term) -> Value {
    json!({
        "coeff": rational_value(&term.coeff),
        "sum_indices": term.sum_indices.iter().map(index_value).collect::<Vec<_>>(),
        "factors": term.factors.iter().map(factor_value).collect::<Vec<_>>(),
    })
}

fn rational_value(rational: &Rational) -> Value {
    json!({
        "numer": *rational.numer(),
        "denom": *rational.denom(),
    })
}

fn index_value(index: &Index) -> Value {
    json!({
        "id": index.id.0,
        "range": index.range.0,
    })
}

fn factor_value(factor: &Factor) -> Value {
    json!({
        "tensor": factor.tensor.0,
        "indices": factor.indices.iter().map(|index| index.0).collect::<Vec<_>>(),
    })
}

pub(crate) fn computation_value(comp: &RustTensorComputation) -> Value {
    json!({
        "ranges": comp.ranges().iter().map(range_value).collect::<Vec<_>>(),
        "tensors": comp.tensors().iter().map(tensor_value).collect::<Vec<_>>(),
        "definitions": comp.definitions().iter().map(tensor_def_value).collect::<Vec<_>>(),
    })
}

#[pyclass(name = "TensorComputation")]
pub(crate) struct PyTensorComputation {
    pub(crate) inner: RustTensorComputation,
}

#[pymethods]
impl PyTensorComputation {
    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        let comp = io::read_json(path).map_err(py_gristmill_display_error)?;
        validate_loaded(comp)
    }

    #[staticmethod]
    fn from_json_string(text: &str) -> PyResult<Self> {
        let comp = io::from_json(text).map_err(py_gristmill_display_error)?;
        validate_loaded(comp)
    }

    fn to_json_string(&self) -> PyResult<String> {
        io::to_json(&self.inner).map_err(py_gristmill_display_error)
    }

    fn write_json(&self, path: PathBuf) -> PyResult<()> {
        io::write_json(path, &self.inner).map_err(py_gristmill_display_error)
    }

    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &computation_value(&self.inner)).map_err(py_gristmill_display_error)
    }

    fn log_total_flops(&self) -> PyResult<f64> {
        cost::log_total_flops(&self.inner).map_err(py_gristmill_error)
    }
}

#[pymodule(name = "_core")]
fn gristmill_symbolics(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensorComputation>()?;
    rewrite_bindings::register(py, module)?;
    verify_bindings::register(py, module)?;
    module.add(
        "GristmillSymbolicsError",
        py.get_type::<GristmillSymbolicsError>(),
    )?;
    Ok(())
}
