use crate::repr::TensorComputation;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub enum IoJsonError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl From<std::io::Error> for IoJsonError {
    fn from(error: std::io::Error) -> Self {
        IoJsonError::Io(error)
    }
}

impl From<serde_json::Error> for IoJsonError {
    fn from(error: serde_json::Error) -> Self {
        IoJsonError::Json(error)
    }
}

impl fmt::Display for IoJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoJsonError::Io(error) => write!(f, "I/O error: {error}"),
            IoJsonError::Json(error) => write!(f, "JSON error: {error}"),
        }
    }
}

impl Error for IoJsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            IoJsonError::Io(error) => Some(error),
            IoJsonError::Json(error) => Some(error),
        }
    }
}

pub fn read_json(path: impl AsRef<Path>) -> Result<TensorComputation, IoJsonError> {
    let input = fs::read_to_string(path)?;
    Ok(from_json(&input)?)
}

pub fn write_json(path: impl AsRef<Path>, comp: &TensorComputation) -> Result<(), IoJsonError> {
    let output = to_json(comp)?;
    Ok(fs::write(path, output)?)
}

pub fn from_json(input: &str) -> Result<TensorComputation, serde_json::Error> {
    serde_json::from_str(input)
}

pub fn to_json(comp: &TensorComputation) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(comp)
}
