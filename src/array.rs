use mohu_core::{mohu_buffer::Buffer, mohu_error::MohuError};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyList, PySequence, PyTuple},
};

#[pyclass(name = "Array")]
pub struct PyArray {
    pub(crate) buffer: Buffer,
}

impl PyArray {
    pub(crate) fn from_buffer(buffer: Buffer) -> Self {
        Self { buffer }
    }

    fn values(&self) -> PyResult<Vec<f64>> {
        self.buffer.to_vec::<f64>().map_err(value_error)
    }
}

fn value_error(error: MohuError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn parse_f64_array(value: &Bound<'_, PyAny>) -> PyResult<Buffer> {
    let sequence = value
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("mu.array() expects a 1-D or 2-D sequence"))?;
    let length = sequence.len()?;
    if length == 0 {
        return Err(PyValueError::new_err(
            "mu.array() cannot infer dtype from an empty sequence",
        ));
    }

    let first = sequence.get_item(0)?;
    if first.downcast::<PySequence>().is_err() {
        let mut values = Vec::with_capacity(length);
        for index in 0..length {
            let item = sequence.get_item(index)?;
            values
                .push(item.extract::<f64>().map_err(|_| {
                    PyTypeError::new_err("mu.array() supports numeric values only")
                })?);
        }
        return Buffer::from_vec(values).map_err(value_error);
    }

    let mut rows = Vec::with_capacity(length);
    let mut columns = None;
    for row_index in 0..length {
        let row_value = sequence.get_item(row_index)?;
        let row = row_value.downcast::<PySequence>()?;
        let row_length = row.len()?;
        if row_length == 0 {
            return Err(PyValueError::new_err(
                "mu.array() rejects empty nested rows",
            ));
        }
        if let Some(expected) = columns {
            if row_length != expected {
                return Err(PyValueError::new_err(
                    "mu.array() rejects ragged nested sequences",
                ));
            }
        } else {
            columns = Some(row_length);
        }
        let mut values = Vec::with_capacity(row_length);
        for column_index in 0..row_length {
            values.push(
                row.get_item(column_index)?
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("mu.array() supports numeric values only"))?,
            );
        }
        rows.push(values);
    }
    let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    Buffer::from_slice_2d(&row_refs).map_err(value_error)
}

fn nested_values<'py>(
    py: Python<'py>,
    shape: &[usize],
    values: &[f64],
) -> PyResult<Bound<'py, PyAny>> {
    match shape {
        [_] => Ok(PyList::new(py, values)?.into_any()),
        [rows, columns] => {
            let nested = values
                .chunks_exact(*columns)
                .take(*rows)
                .map(|row| PyList::new(py, row))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, nested)?.into_any())
        }
        _ => Err(PyValueError::new_err(
            "only 1-D and 2-D arrays are supported",
        )),
    }
}

#[pyfunction]
pub fn array(value: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    Ok(PyArray::from_buffer(parse_f64_array(value)?))
}

#[pymethods]
impl PyArray {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.buffer.shape())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.buffer.ndim()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.buffer.dtype().to_string()
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.buffer.strides())
    }

    #[getter]
    fn size(&self) -> usize {
        self.buffer.len()
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        nested_values(py, self.buffer.shape(), &self.values()?)
    }

    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = shape
            .extract::<Vec<usize>>()
            .map_err(|_| PyTypeError::new_err("reshape() expects a sequence of dimensions"))?;
        if shape.is_empty() || shape.len() > 2 {
            return Err(PyValueError::new_err(
                "reshape() supports 1-D and 2-D shapes",
            ));
        }
        Ok(Self::from_buffer(
            self.buffer.reshape(&shape).map_err(value_error)?,
        ))
    }

    fn transpose(&self) -> Self {
        Self::from_buffer(self.buffer.transpose())
    }

    #[getter(T)]
    fn t(&self) -> Self {
        self.transpose()
    }

    fn __matmul__(&self, rhs: &PyArray) -> PyResult<Self> {
        super::ops::matmul(self, rhs)
    }

    fn __repr__(&self) -> PyResult<String> {
        let values = self.values()?;
        let rendered = match self.buffer.shape() {
            [_] => format!("{values:?}"),
            [rows, columns] => values
                .chunks_exact(*columns)
                .take(*rows)
                .map(|row| format!("{row:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            _ => {
                return Err(PyValueError::new_err(
                    "only 1-D and 2-D arrays are supported",
                ));
            }
        };
        let rendered = if self.buffer.ndim() == 2 {
            format!("[{rendered}]")
        } else {
            rendered
        };
        Ok(format!("mohu.Array({rendered}, dtype={})", self.dtype()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    Ok(())
}
