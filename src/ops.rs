use mohu_ops::matmul::matmul as rust_matmul;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::array::PyArray;

#[pyfunction]
pub fn matmul(lhs: &PyArray, rhs: &PyArray) -> PyResult<PyArray> {
    Ok(PyArray::from_buffer(
        rust_matmul(&lhs.buffer, &rhs.buffer)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
