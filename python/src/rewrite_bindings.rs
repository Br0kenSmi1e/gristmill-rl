use crate::{
    py_gristmill_display_error, py_gristmill_error, tensor_def_value,
    PyTensorComputation,
};
use gristmill_symbolics::repr::TensorComputation;
use gristmill_symbolics::rewrite::{
    action_space_for_def as rust_action_space_for_def,
    action_spaces_for_batch as rust_action_spaces_for_batch,
    apply_decision as rust_apply_decision,
    apply_decisions_for_batch as rust_apply_decisions_for_batch,
    validate_decision as rust_validate_decision,
    validate_decisions_for_batch as rust_validate_decisions_for_batch,
    ActionSpace as RustActionSpace, Decision, Factorization,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use pyo3::{PyRef, PyRefMut};
use pythonize::pythonize;
use serde_json::{json, Value};

fn factorization_value(factorization: &Factorization) -> Value {
    json!({
        "left_definition": tensor_def_value(&factorization.left_definition),
        "right_definition": tensor_def_value(&factorization.right_definition),
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

fn required_dict_item<'py>(
    dict: &Bound<'py, PyDict>,
    field: &str,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(field)?.ok_or_else(|| {
        PyValueError::new_err(format!("missing decision field '{field}'"))
    })
}

fn parse_bool_mask(
    value: &Bound<'_, PyAny>,
    field: &str,
) -> PyResult<Vec<bool>> {
    let list = value.cast::<PyList>().map_err(|_| {
        PyTypeError::new_err(format!("decision field '{field}' must be a list"))
    })?;

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

fn parse_usize(value: &Bound<'_, PyAny>, field: &str) -> PyResult<usize> {
    if value.is_exact_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{field} must be an integer, not bool"
        )));
    }

    value.extract::<usize>().map_err(|_| {
        PyValueError::new_err(format!("{field} must be a non-negative integer"))
    })
}

fn parse_decision(value: &Bound<'_, PyAny>) -> PyResult<Decision> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("decision must be a dict"))?;

    Ok(Decision {
        candidate_index: parse_usize(
            &required_dict_item(dict, "candidate_index")?,
            "decision field 'candidate_index'",
        )?,
        left_mask: parse_bool_mask(
            &required_dict_item(dict, "left_mask")?,
            "left_mask",
        )?,
        right_mask: parse_bool_mask(
            &required_dict_item(dict, "right_mask")?,
            "right_mask",
        )?,
    })
}

fn parse_targets(value: &Bound<'_, PyAny>) -> PyResult<Vec<Option<usize>>> {
    let list = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("targets must be a list"))?;

    list.iter()
        .enumerate()
        .map(|(sample, item)| {
            if item.is_none() {
                Ok(None)
            } else {
                parse_usize(&item, &format!("targets[{sample}]")).map(Some)
            }
        })
        .collect()
}

fn parse_comp_batch(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<TensorComputation>> {
    let list = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("comps must be a list"))?;

    list.iter()
        .map(|item| {
            let comp = item.extract::<PyRef<'_, PyTensorComputation>>()?;
            Ok(comp.inner.clone())
        })
        .collect()
}

fn parse_space_batch(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<RustActionSpace>>> {
    let list = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("spaces must be a list"))?;

    list.iter()
        .map(|item| {
            if item.is_none() {
                Ok(None)
            } else {
                let space = item.extract::<PyRef<'_, PyActionSpace>>()?;
                Ok(Some(space.inner.clone()))
            }
        })
        .collect()
}

fn parse_decision_batch(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<Decision>>> {
    let list = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("decisions must be a list"))?;

    list.iter()
        .map(|item| {
            if item.is_none() {
                Ok(None)
            } else {
                parse_decision(&item).map(Some)
            }
        })
        .collect()
}

fn py_action_space_batch<'py>(
    py: Python<'py>,
    spaces: Vec<Option<RustActionSpace>>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for space in spaces {
        match space {
            Some(inner) => out.append(PyActionSpace { inner })?,
            None => out.append(py.None())?,
        }
    }
    Ok(out)
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
        pythonize(py, &action_space_value(&self.inner))
            .map_err(py_gristmill_display_error)
    }
}

#[pyfunction(name = "action_space_for_def")]
fn py_action_space_for_def(
    comp: &PyTensorComputation,
    def_index: usize,
) -> PyResult<Option<PyActionSpace>> {
    rust_action_space_for_def(&comp.inner, def_index)
        .map(|space| space.map(|inner| PyActionSpace { inner }))
        .map_err(py_gristmill_error)
}

#[pyfunction(name = "validate_decision")]
fn py_validate_decision(
    space: &PyActionSpace,
    decision: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let decision = parse_decision(decision)?;
    rust_validate_decision(&space.inner, &decision).map_err(py_gristmill_error)
}

#[pyfunction(name = "apply_decision")]
fn py_apply_decision(
    mut comp: PyRefMut<'_, PyTensorComputation>,
    space: &PyActionSpace,
    decision: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let decision = parse_decision(decision)?;
    rust_apply_decision(&mut comp.inner, &space.inner, &decision)
        .map_err(py_gristmill_error)
}

#[pyfunction(name = "action_spaces_for_batch")]
fn py_action_spaces_for_batch<'py>(
    py: Python<'py>,
    comps: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let comps = parse_comp_batch(comps)?;
    let targets = parse_targets(targets)?;
    let spaces = rust_action_spaces_for_batch(&comps, &targets)
        .map_err(py_gristmill_error)?;
    py_action_space_batch(py, spaces)
}

#[pyfunction(name = "validate_decisions_for_batch")]
fn py_validate_decisions_for_batch(
    spaces: &Bound<'_, PyAny>,
    decisions: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let spaces = parse_space_batch(spaces)?;
    let decisions = parse_decision_batch(decisions)?;
    rust_validate_decisions_for_batch(&spaces, &decisions)
        .map_err(py_gristmill_error)
}

#[pyfunction(name = "apply_decisions_for_batch")]
fn py_apply_decisions_for_batch(
    comps: &Bound<'_, PyAny>,
    spaces: &Bound<'_, PyAny>,
    decisions: &Bound<'_, PyAny>,
) -> PyResult<Vec<bool>> {
    let comp_list = comps
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("comps must be a list"))?;
    let mut rust_comps = parse_comp_batch(comps)?;
    let spaces = parse_space_batch(spaces)?;
    let decisions = parse_decision_batch(decisions)?;
    let applied =
        rust_apply_decisions_for_batch(&mut rust_comps, &spaces, &decisions)
            .map_err(py_gristmill_error)?;

    for (item, rust_comp) in comp_list.iter().zip(rust_comps) {
        let mut comp = item.extract::<PyRefMut<'_, PyTensorComputation>>()?;
        comp.inner = rust_comp;
    }
    Ok(applied)
}

pub(crate) fn register(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let _ = py;
    module.add_class::<PyActionSpace>()?;
    module.add_function(wrap_pyfunction!(py_action_space_for_def, module)?)?;
    module.add_function(wrap_pyfunction!(py_validate_decision, module)?)?;
    module.add_function(wrap_pyfunction!(py_apply_decision, module)?)?;
    module
        .add_function(wrap_pyfunction!(py_action_spaces_for_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_validate_decisions_for_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_apply_decisions_for_batch,
        module
    )?)?;
    Ok(())
}
