use std::sync::OnceLock;

use ferrum_core::desc::biosynfoni::{compute_biosynfoni_fingerprint, get_biosynfoni_feature_names};
use ferrum_core::desc::calculator::DescriptorCalculator;
use ferrum_core::desc::table::CountFingerprintFormat;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::mol::molecule::Molecule;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rayon::prelude::*;

/// SMILES processed per chunk: parsed molecules of one chunk are dropped before
/// the next is parsed, and the scalar values are written straight into the
/// output matrix, so memory beyond the output arrays stays bounded.
const CHUNK: usize = 4096;

fn to_numpy_2d<T: numpy::Element>(py: Python<'_>, data: Vec<T>, rows: usize, cols: usize) -> PyResult<PyObject> {
    let array = numpy::PyArray1::from_vec(py, data).reshape([rows, cols]).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to reshape matrix: {:?}", e))
    })?;
    Ok(array.into_any().unbind())
}

/// Runs `calc` over `smiles` chunk by chunk without the GIL and converts the
/// result to `(names, scalars, fingerprints, valid)`, or with sparse count
/// fingerprints `(names, scalars, fingerprints, counts, valid)`.
fn compute(py: Python<'_>, calc: &DescriptorCalculator, smiles: &[String]) -> PyResult<PyObject> {
    let n_rows = smiles.len();
    let layout = calc.layout();
    let ns = layout.n_scalars();
    let shapes = layout.fingerprint_shapes();
    let count_shapes = layout.count_fingerprint_shapes();

    let (scalars, fp_bytes, counts, valid) = py.allow_threads(|| {
        // zero-initialised (pages are mapped as the chunks fill them)
        let mut scalars = vec![0.0f64; n_rows * ns];
        let mut fp_bytes: Vec<Vec<u8>> =
            shapes.iter().map(|&(_, bits)| Vec::with_capacity(n_rows * bits.div_ceil(8))).collect();
        let mut counts: Vec<Vec<u32>> = count_shapes.iter().map(|&(_, keys)| Vec::with_capacity(n_rows * keys)).collect();
        let mut valid = Vec::with_capacity(n_rows);
        for (c, chunk) in smiles.chunks(CHUNK).enumerate() {
            let molecules: Vec<Option<Molecule>> = chunk.par_iter().map(|s| parse_smiles(s).ok()).collect();
            let rows = c * CHUNK..c * CHUNK + chunk.len();
            let table = calc.calculate_into(&molecules, &mut scalars[rows.start * ns..rows.end * ns]);
            drop(molecules);
            for (out, block) in fp_bytes.iter_mut().zip(&table.fingerprints) {
                // `FingerprintBlock::to_bytes` without the temporary copy
                for r in 0..table.rows {
                    out.extend(block.row(r).iter().flat_map(|w| w.to_le_bytes()).take(block.bytes_per_row()));
                }
            }
            for (out, block) in counts.iter_mut().zip(&table.counts) {
                // `CountBlock::to_dense` without the temporary copy
                let start = out.len();
                out.resize(start + table.rows * block.n_keys, 0);
                for r in 0..table.rows {
                    let (keys, values) = block.row(r);
                    for (&k, &c) in keys.iter().zip(values) {
                        out[start + r * block.n_keys + k as usize] = c;
                    }
                }
            }
            valid.extend_from_slice(&table.valid);
        }
        (scalars, fp_bytes, counts, valid)
    });

    let fps = PyDict::new(py);
    for ((name, bits), bytes) in shapes.iter().zip(fp_bytes) {
        fps.set_item(*name, to_numpy_2d(py, bytes, n_rows, bits.div_ceil(8))?)?;
    }
    let names = layout.scalar_names().to_vec().into_pyobject(py)?.into_any().unbind();
    let scalars = to_numpy_2d(py, scalars, n_rows, ns)?;
    let valid = numpy::PyArray1::from_vec(py, valid).into_any().unbind();
    let fps = fps.into_any().unbind();
    if layout.count_format() == CountFingerprintFormat::Columns {
        return Ok(PyTuple::new(py, [names, scalars, fps, valid])?.into_any().unbind());
    }
    let cps = PyDict::new(py);
    for ((name, keys), data) in count_shapes.iter().zip(counts) {
        cps.set_item(*name, to_numpy_2d(py, data, n_rows, *keys)?)?;
    }
    Ok(PyTuple::new(py, [names, scalars, fps, cps.into_any().unbind(), valid])?.into_any().unbind())
}

/// Parses the `count_fingerprints` argument.
fn count_format(value: &str) -> PyResult<CountFingerprintFormat> {
    match value {
        "columns" => Ok(CountFingerprintFormat::Columns),
        "sparse" => Ok(CountFingerprintFormat::Sparse),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "count_fingerprints must be \"columns\" or \"sparse\", got {other:?}"
        ))),
    }
}

fn count_dict(py: Python<'_>, calc: &DescriptorCalculator) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    for (name, keys) in calc.layout().count_fingerprint_shapes() {
        out.set_item(name, keys)?;
    }
    Ok(out.into_any().unbind())
}

fn fingerprint_dict(py: Python<'_>, calc: &DescriptorCalculator) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    for (name, bits) in calc.layout().fingerprint_shapes() {
        out.set_item(name, bits)?;
    }
    Ok(out.into_any().unbind())
}

/// Shared default calculators (built once per process).
fn default_calculator(fingerprints: bool, counts: CountFingerprintFormat) -> &'static DescriptorCalculator {
    static CALCS: [OnceLock<DescriptorCalculator>; 4] = [const { OnceLock::new() }; 4];
    let sparse = counts == CountFingerprintFormat::Sparse;
    CALCS[usize::from(fingerprints) * 2 + usize::from(sparse)].get_or_init(|| new_calculator(fingerprints, counts))
}

fn new_calculator(fingerprints: bool, counts: CountFingerprintFormat) -> DescriptorCalculator {
    let calc = if fingerprints { DescriptorCalculator::all() } else { DescriptorCalculator::scalars_only() };
    calc.with_count_format(counts)
}

/// One descriptor calculator, built once and reused for any number of calls
/// (thread-safe; the calculation itself runs in parallel without the GIL).
///
/// >>> calc = ferrum_py.Calculator()                  # or Calculator(fingerprints=False)
/// >>> names, X, fps, valid = calc.compute(smiles)
///
/// Count fingerprints (Klekota-Roth `KRFPC1..`, CDK substructure `SubFPC1..`)
/// are scalar columns by default; with `count_fingerprints="sparse"` they are
/// returned separately and `compute` gives
/// `(names, scalars, fingerprints, counts, valid)` with `counts` a dict
/// name -> uint32 array `(n, keys)` (`"KlekotaRothCount"`, `"CDKSubstructureCount"`).
#[pyclass(frozen, module = "ferrum_py")]
struct Calculator {
    inner: DescriptorCalculator,
}

#[pymethods]
impl Calculator {
    #[new]
    #[pyo3(signature = (fingerprints = true, count_fingerprints = "columns"))]
    fn new(py: Python<'_>, fingerprints: bool, count_fingerprints: &str) -> PyResult<Self> {
        let counts = count_format(count_fingerprints)?;
        let inner = py.allow_threads(|| new_calculator(fingerprints, counts));
        Ok(Self { inner })
    }

    /// Computes all descriptors; returns `(names, scalars, fingerprints, valid)`,
    /// with `count_fingerprints="sparse"` `(names, scalars, fingerprints, counts, valid)`
    /// (see `compute_descriptors`).
    fn compute(&self, py: Python<'_>, smiles_list: Vec<String>) -> PyResult<PyObject> {
        compute(py, &self.inner, &smiles_list)
    }

    /// `{count fingerprint name: number of keys}` (empty unless `count_fingerprints="sparse"`).
    #[getter]
    fn count_fingerprint_keys(&self, py: Python<'_>) -> PyResult<PyObject> {
        count_dict(py, &self.inner)
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
            "Calculator({} scalar columns, fingerprints: {:?}, count fingerprints: {:?})",
            self.inner.layout().n_scalars(),
            self.inner.layout().fingerprint_shapes().iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            self.inner.layout().count_fingerprint_shapes().iter().map(|(n, _)| *n).collect::<Vec<_>>()
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
///
/// Count fingerprints (`KRFPC1..`, `SubFPC1..`) are scalar columns unless
/// `count_fingerprints="sparse"`; then the result is
/// `(names, scalars, fingerprints, counts, valid)` with `counts` a dict
/// name -> uint32 array `(n, keys)`.
#[pyfunction]
#[pyo3(signature = (smiles_list, fingerprints = true, count_fingerprints = "columns"))]
fn compute_descriptors(
    py: Python<'_>,
    smiles_list: Vec<String>,
    fingerprints: bool,
    count_fingerprints: &str,
) -> PyResult<PyObject> {
    let counts = count_format(count_fingerprints)?;
    let calc = py.allow_threads(|| default_calculator(fingerprints, counts));
    compute(py, calc, &smiles_list)
}

/// Number of bits of each fingerprint returned by `compute_descriptors`.
#[pyfunction]
fn fingerprint_bits(py: Python<'_>) -> PyResult<PyObject> {
    let calc = py.allow_threads(|| default_calculator(true, CountFingerprintFormat::Columns));
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
