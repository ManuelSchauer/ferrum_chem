use std::sync::OnceLock;

use ferrum_core::desc::biosynfoni::{compute_biosynfoni_fingerprint, get_biosynfoni_feature_names};
use ferrum_core::desc::calculator::DescriptorCalculator;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::mol::molecule::Molecule;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// SMILES processed per chunk: parsed molecules of one chunk are dropped before
/// the next is parsed, so memory beyond the output arrays stays bounded.
const CHUNK: usize = 4096;

fn to_numpy_2d<T: numpy::Element>(py: Python<'_>, data: Vec<T>, rows: usize, cols: usize) -> PyResult<PyObject> {
    let array = numpy::PyArray1::from_vec(py, data).reshape([rows, cols]).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to reshape matrix: {:?}", e))
    })?;
    Ok(array.into_any().unbind())
}

/// Runs `calc` over `smiles` chunk by chunk without the GIL and converts the
/// result to `(names, scalars, fingerprints, valid)`.
fn compute(
    py: Python<'_>,
    calc: &DescriptorCalculator,
    smiles: &[String],
) -> PyResult<(Vec<String>, PyObject, PyObject, PyObject)> {
    let n_rows = smiles.len();
    let layout = calc.layout();
    let ns = layout.n_scalars();
    let shapes = layout.fingerprint_shapes();

    let (scalars, fp_bytes, valid) = py.allow_threads(|| {
        let mut scalars = Vec::with_capacity(n_rows * ns);
        let mut fp_bytes: Vec<Vec<u8>> =
            shapes.iter().map(|&(_, bits)| Vec::with_capacity(n_rows * bits.div_ceil(8))).collect();
        let mut valid = Vec::with_capacity(n_rows);
        for chunk in smiles.chunks(CHUNK) {
            let molecules: Vec<Option<Molecule>> = chunk.par_iter().map(|s| parse_smiles(s).ok()).collect();
            let table = calc.calculate(&molecules);
            scalars.extend_from_slice(&table.scalars);
            for (out, block) in fp_bytes.iter_mut().zip(&table.fingerprints) {
                out.extend(block.to_bytes());
            }
            valid.extend_from_slice(&table.valid);
        }
        (scalars, fp_bytes, valid)
    });

    let fps = PyDict::new(py);
    for ((name, bits), bytes) in shapes.iter().zip(fp_bytes) {
        fps.set_item(*name, to_numpy_2d(py, bytes, n_rows, bits.div_ceil(8))?)?;
    }
    let valid = numpy::PyArray1::from_vec(py, valid).into_any().unbind();
    Ok((layout.scalar_names().to_vec(), to_numpy_2d(py, scalars, n_rows, ns)?, fps.into_any().unbind(), valid))
}

fn fingerprint_dict(py: Python<'_>, calc: &DescriptorCalculator) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    for (name, bits) in calc.layout().fingerprint_shapes() {
        out.set_item(name, bits)?;
    }
    Ok(out.into_any().unbind())
}

/// Shared default calculators (built once per process).
fn default_calculator(fingerprints: bool) -> &'static DescriptorCalculator {
    static ALL: OnceLock<DescriptorCalculator> = OnceLock::new();
    static SCALARS: OnceLock<DescriptorCalculator> = OnceLock::new();
    if fingerprints {
        ALL.get_or_init(DescriptorCalculator::all)
    } else {
        SCALARS.get_or_init(DescriptorCalculator::scalars_only)
    }
}

/// One descriptor calculator, built once and reused for any number of calls
/// (thread-safe; the calculation itself runs in parallel without the GIL).
///
/// >>> calc = ferrum_py.Calculator()                  # or Calculator(fingerprints=False)
/// >>> names, X, fps, valid = calc.compute(smiles)
#[pyclass(frozen, module = "ferrum_py")]
struct Calculator {
    inner: DescriptorCalculator,
}

#[pymethods]
impl Calculator {
    #[new]
    #[pyo3(signature = (fingerprints = true))]
    fn new(py: Python<'_>, fingerprints: bool) -> Self {
        let inner = py.allow_threads(|| {
            if fingerprints {
                DescriptorCalculator::all()
            } else {
                DescriptorCalculator::scalars_only()
            }
        });
        Self { inner }
    }

    /// Computes all descriptors; returns `(names, scalars, fingerprints, valid)`
    /// (see `compute_descriptors`).
    fn compute(&self, py: Python<'_>, smiles_list: Vec<String>) -> PyResult<(Vec<String>, PyObject, PyObject, PyObject)> {
        compute(py, &self.inner, &smiles_list)
    }

    /// Names of the scalar columns.
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.inner.scalar_names().to_vec()
    }

    /// `{fingerprint name: number of bits}`.
    #[getter]
    fn fingerprint_bits(&self, py: Python<'_>) -> PyResult<PyObject> {
        fingerprint_dict(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Calculator({} scalar columns, fingerprints: {:?})",
            self.inner.layout().n_scalars(),
            self.inner.layout().fingerprint_shapes().iter().map(|(n, _)| *n).collect::<Vec<_>>()
        )
    }
}

/// Computes the complete FerrumChem descriptor bundle for a list of SMILES
/// (using a shared module-level `Calculator`).
///
/// Returns `(names, scalars, fingerprints, valid)`:
/// - `names`: list of the scalar column names;
/// - `scalars`: float64 array `(n, len(names))`; rows of SMILES that could not
///   be parsed are NaN;
/// - `fingerprints`: dict name -> uint8 array `(n, ceil(bits / 8))`, bit-packed
///   with little-endian bit order; unpack with
///   `numpy.unpackbits(a, axis=1, bitorder="little", count=fingerprint_bits()[name])`;
/// - `valid`: bool array `(n,)`, False where the SMILES could not be parsed.
#[pyfunction]
#[pyo3(signature = (smiles_list, fingerprints = true))]
fn compute_descriptors(
    py: Python<'_>,
    smiles_list: Vec<String>,
    fingerprints: bool,
) -> PyResult<(Vec<String>, PyObject, PyObject, PyObject)> {
    let calc = py.allow_threads(|| default_calculator(fingerprints));
    compute(py, calc, &smiles_list)
}

/// Number of bits of each fingerprint returned by `compute_descriptors`.
#[pyfunction]
fn fingerprint_bits(py: Python<'_>) -> PyResult<PyObject> {
    let calc = py.allow_threads(|| default_calculator(true));
    fingerprint_dict(py, calc)
}

/// Computes the Biosynfoni fingerprint (39 counts, version `full_1103`) for a list of SMILES.
///
/// Returns `(feature_names, counts)`: float64 array `(n, 39)`; rows of SMILES
/// that could not be parsed are NaN.
#[pyfunction]
fn compute_biosynfoni(py: Python<'_>, smiles_list: Vec<String>) -> PyResult<(Vec<String>, PyObject)> {
    let feature_names = get_biosynfoni_feature_names();
    let n_rows = smiles_list.len();
    let cols = feature_names.len();

    let data = py.allow_threads(|| {
        smiles_list
            .par_iter()
            .flat_map_iter(|s| match parse_smiles(s) {
                Ok(mol) => compute_biosynfoni_fingerprint(&mol).into_iter().map(|c| c as f64).collect::<Vec<_>>(),
                Err(_) => vec![f64::NAN; cols],
            })
            .collect::<Vec<f64>>()
    });

    Ok((feature_names, to_numpy_2d(py, data, n_rows, cols)?))
}

/// A Python module implemented in Rust.
#[pymodule]
fn ferrum_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Calculator>()?;
    m.add_function(wrap_pyfunction!(compute_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint_bits, m)?)?;
    m.add_function(wrap_pyfunction!(compute_biosynfoni, m)?)?;
    Ok(())
}
