use crate::repr::TensorComputation;

pub fn from_json(input: &str) -> Result<TensorComputation, serde_json::Error> {
    serde_json::from_str(input)
}

pub fn to_json(comp: &TensorComputation) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(comp)
}
