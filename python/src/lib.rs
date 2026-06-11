use ::gristmill_symbolics::cost;
use ::gristmill_symbolics::io;
use ::gristmill_symbolics::repr::{
    Factor, Index, Range, Rational, SymAction, SymGenerator,
    TensorComputation as RustTensorComputation, TensorDef, TensorInfo, Term,
};
use ::gristmill_symbolics::rewrite::{
    validate_decision as rust_validate_decision, ActionSpace as RustActionSpace, Decision,
    Factorization, RewriteState as RustRewriteState,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use pythonize::pythonize;
use serde_json::{json, Value};
use std::fmt;
use std::path::PathBuf;

pyo3::create_exception!(
    gristmill_symbolics,
    GristmillSymbolicsError,
    pyo3::exceptions::PyException
);

fn py_gristmill_error(error: impl fmt::Debug) -> PyErr {
    GristmillSymbolicsError::new_err(format!("{error:?}"))
}

fn py_gristmill_display_error(error: impl fmt::Display) -> PyErr {
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

fn tensor_def_value(definition: &TensorDef) -> Value {
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

fn computation_value(comp: &RustTensorComputation) -> Value {
    json!({
        "ranges": comp.ranges().iter().map(range_value).collect::<Vec<_>>(),
        "tensors": comp.tensors().iter().map(tensor_value).collect::<Vec<_>>(),
        "definitions": comp.definitions().iter().map(tensor_def_value).collect::<Vec<_>>(),
    })
}

fn factorization_value(factorization: &Factorization) -> Value {
    json!({
        "left_definition": tensor_def_value(&factorization.left_definition),
        "right_definition": tensor_def_value(&factorization.right_definition),
        "rewritten_definition": tensor_def_value(&factorization.rewritten_definition),
    })
}

fn action_space_value(space: &RustActionSpace) -> Value {
    json!({
        "def_index": space.def_index,
        "candidate_templates": space
            .candidate_templates
            .iter()
            .map(factorization_value)
            .collect::<Vec<_>>(),
    })
}

fn required_dict_item<'py>(dict: &Bound<'py, PyDict>, field: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(field)?
        .ok_or_else(|| PyValueError::new_err(format!("missing decision field '{field}'")))
}

fn parse_bool_mask(value: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<bool>> {
    let list = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err(format!("decision field '{field}' must be a list")))?;

    let mut mask = Vec::with_capacity(list.len());
    for item in list.iter() {
        if !item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "decision field '{field}' must contain only bool values"
            )));
        }
        mask.push(item.extract::<bool>()?);
    }
    Ok(mask)
}

fn parse_candidate_index(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_exact_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "decision field 'candidate_index' must be an integer, not bool",
        ));
    }

    value.extract::<usize>().map_err(|_| {
        PyValueError::new_err("decision field 'candidate_index' must be a non-negative integer")
    })
}

fn parse_decision(value: &Bound<'_, PyAny>) -> PyResult<Decision> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("decision must be a dict"))?;

    let candidate_index = parse_candidate_index(&required_dict_item(dict, "candidate_index")?)?;
    let left_mask = parse_bool_mask(&required_dict_item(dict, "left_mask")?, "left_mask")?;
    let right_mask = parse_bool_mask(&required_dict_item(dict, "right_mask")?, "right_mask")?;

    Ok(Decision {
        candidate_index,
        left_mask,
        right_mask,
    })
}

#[pyfunction(name = "validate_decision")]
fn py_validate_decision(space: &PyActionSpace, decision: &Bound<'_, PyAny>) -> PyResult<()> {
    let decision = parse_decision(decision)?;
    rust_validate_decision(&space.inner, &decision).map_err(py_gristmill_error)
}

#[pyclass(name = "TensorComputation")]
struct PyTensorComputation {
    inner: RustTensorComputation,
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

#[pyclass(name = "RewriteState")]
struct PyRewriteState {
    inner: RustRewriteState,
}

#[pymethods]
impl PyRewriteState {
    #[staticmethod]
    fn from_computation(comp: &PyTensorComputation) -> Self {
        Self {
            inner: RustRewriteState::new(comp.inner.clone()),
        }
    }

    fn definition_mask(&self) -> Vec<bool> {
        self.inner.definition_mask().to_vec()
    }

    fn action_space_for_def(&mut self, def_index: usize) -> PyResult<Option<PyActionSpace>> {
        self.inner
            .action_space_for_def(def_index)
            .map(|space| space.map(|inner| PyActionSpace { inner }))
            .map_err(py_gristmill_error)
    }

    fn apply_validated_decision(
        &mut self,
        space: &PyActionSpace,
        decision: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let decision = parse_decision(decision)?;
        self.inner
            .apply_validated_decision(&space.inner, &decision)
            .map_err(py_gristmill_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &computation_value(self.inner.computation()))
            .map_err(py_gristmill_display_error)
    }

    fn log_total_flops(&self) -> PyResult<f64> {
        cost::log_total_flops(self.inner.computation()).map_err(py_gristmill_error)
    }

    fn to_json_string(&self) -> PyResult<String> {
        io::to_json(self.inner.computation()).map_err(py_gristmill_display_error)
    }

    fn write_json(&self, path: PathBuf) -> PyResult<()> {
        io::write_json(path, self.inner.computation()).map_err(py_gristmill_display_error)
    }
}

#[pyclass(name = "ActionSpace")]
struct PyActionSpace {
    inner: RustActionSpace,
}

#[pymethods]
impl PyActionSpace {
    #[getter]
    fn def_index(&self) -> usize {
        self.inner.def_index
    }

    #[getter]
    fn candidate_count(&self) -> usize {
        self.inner.candidate_templates.len()
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &action_space_value(&self.inner)).map_err(py_gristmill_display_error)
    }
}

#[pymodule]
fn gristmill_symbolics(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensorComputation>()?;
    module.add_class::<PyRewriteState>()?;
    module.add_class::<PyActionSpace>()?;
    module.add_function(wrap_pyfunction!(py_validate_decision, module)?)?;
    module.add(
        "GristmillSymbolicsError",
        py.get_type::<GristmillSymbolicsError>(),
    )?;
    Ok(())
}
