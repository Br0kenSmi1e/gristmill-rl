use crate::{
    PyTensorComputation, computation_value, py_gristmill_display_error, py_gristmill_error,
    tensor_def_value,
};
use ::gristmill_symbolics::cost;
use ::gristmill_symbolics::io;
use ::gristmill_symbolics::rewrite::{
    ActionSpace as RustActionSpace, ActionSpaceEntry as RustActionSpaceEntry,
    ActionSpaceRow as RustActionSpaceRow, Decision, Factorization,
    RewriteState as RustRewriteState, RewriteStateRow as RustRewriteStateRow,
    ValidatedActionRow as RustValidatedActionRow, validate_decision as rust_validate_decision,
};
use pyo3::PyRef;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use pythonize::pythonize;
use serde_json::{Value, json};
use std::path::PathBuf;

trait PythonAllowThreadsCompat {
    fn allow_threads<T, F>(self, f: F) -> T
    where
        F: Ungil + FnOnce() -> T,
        T: Ungil;
}

impl PythonAllowThreadsCompat for Python<'_> {
    fn allow_threads<T, F>(self, f: F) -> T
    where
        F: Ungil + FnOnce() -> T,
        T: Ungil,
    {
        self.detach(f)
    }
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

fn parse_target_choices(value: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    let list = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("target_choices must be a list"))?;

    let mut choices = Vec::with_capacity(list.len());
    for (sample, item) in list.iter().enumerate() {
        if item.is_exact_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "target_choices[{sample}] must be an integer, not bool"
            )));
        }
        choices.push(item.extract::<isize>().map_err(|_| {
            PyValueError::new_err(format!(
                "target_choices[{sample}] must be an integer target choice"
            ))
        })?);
    }
    Ok(choices)
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

#[allow(dead_code)]
#[pyclass(name = "RewriteStateRow")]
struct PyRewriteStateRow {
    inner: RustRewriteStateRow,
}

#[pymethods]
impl PyRewriteStateRow {
    #[staticmethod]
    fn from_states(states: &Bound<'_, PyAny>) -> PyResult<Self> {
        let list = states
            .cast::<PyList>()
            .map_err(|_| PyTypeError::new_err("states must be a list of RewriteState objects"))?;

        let mut rust_states = Vec::with_capacity(list.len());
        for item in list.iter() {
            let state = item.extract::<PyRef<'_, PyRewriteState>>()?;
            rust_states.push(state.inner.clone());
        }

        Ok(Self {
            inner: RustRewriteStateRow::from_states(rust_states),
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn definition_masks(&self) -> Vec<Vec<bool>> {
        self.inner.definition_masks()
    }

    fn snapshots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let snapshots = self
            .inner
            .states()
            .iter()
            .map(|state| computation_value(state.computation()))
            .collect::<Vec<_>>();
        pythonize(py, &snapshots).map_err(py_gristmill_display_error)
    }

    fn query_action_spaces_for_row(
        &mut self,
        py: Python<'_>,
        target_choices: &Bound<'_, PyAny>,
        active_mask: Vec<bool>,
    ) -> PyResult<PyActionSpaceRow> {
        let target_choices = parse_target_choices(target_choices)?;
        let inner = py
            .allow_threads(|| {
                self.inner
                    .query_action_spaces_for_row(&target_choices, &active_mask)
            })
            .map_err(py_gristmill_error)?;
        Ok(PyActionSpaceRow { inner })
    }
}

#[allow(dead_code)]
#[pyclass(name = "ActionSpaceRow")]
struct PyActionSpaceRow {
    inner: RustActionSpaceRow,
}

#[pymethods]
impl PyActionSpaceRow {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn entry_kinds(&self) -> Vec<&'static str> {
        self.inner.entry_kinds()
    }

    fn snapshots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let snapshots = self
            .inner
            .entries()
            .iter()
            .map(|entry| match entry {
                RustActionSpaceEntry::NonEmpty(space) => Some(action_space_value(space)),
                RustActionSpaceEntry::Skipped | RustActionSpaceEntry::ExactEmpty => None,
            })
            .collect::<Vec<Option<Value>>>();
        pythonize(py, &snapshots).map_err(py_gristmill_display_error)
    }
}

#[allow(dead_code)]
#[pyclass(name = "ValidatedActionRow")]
struct PyValidatedActionRow {
    inner: RustValidatedActionRow,
}

pub(crate) fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    module.add_class::<PyRewriteState>()?;
    module.add_class::<PyActionSpace>()?;
    module.add_class::<PyRewriteStateRow>()?;
    module.add_class::<PyActionSpaceRow>()?;
    module.add_class::<PyValidatedActionRow>()?;
    module.add_function(wrap_pyfunction!(py_validate_decision, module)?)?;
    Ok(())
}
