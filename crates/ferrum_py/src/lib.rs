use ferrum_core::desc::batch::BatchDescriptor;
use ferrum_core::desc::biosynfoni::{compute_biosynfoni_fingerprint, get_biosynfoni_feature_names};
use ferrum_core::desc::table::{calculate_table, TableLayout};
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::mol::molecule::Molecule;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// SMILES processed per chunk: parsed molecules of one chunk are dropped before
/// the next is parsed, so memory beyond the output arrays stays bounded.
const CHUNK: usize = 4096;

fn parse_chunk(smiles: &[String]) -> Vec<Option<Molecule>> {
    smiles.par_iter().map(|s| parse_smiles(s).ok()).collect()
}

fn to_numpy_2d<T: numpy::Element>(py: Python<'_>, data: Vec<T>, rows: usize, cols: usize) -> PyResult<PyObject> {
    let array = numpy::PyArray1::from_vec(py, data).reshape([rows, cols]).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to reshape matrix: {:?}", e))
    })?;
    Ok(array.into_any().unbind())
}

/// Computes the complete FerrumChem descriptor bundle for a list of SMILES.
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
    let descriptors: Vec<BatchDescriptor> = BatchDescriptor::all()
        .iter()
        .copied()
        .filter(|d| fingerprints || !d.is_fingerprint())
        .collect();
    let layout = TableLayout::new(&descriptors);
    let n_rows = smiles_list.len();
    let ns = layout.n_scalars();
    let shapes = layout.fingerprint_shapes();

    // Parse and compute chunk by chunk without holding the GIL.
    let (scalars, fp_bytes, valid) = py.allow_threads(|| {
        let mut scalars = Vec::with_capacity(n_rows * ns);
        let mut fp_bytes: Vec<Vec<u8>> =
            shapes.iter().map(|&(_, bits)| Vec::with_capacity(n_rows * bits.div_ceil(8))).collect();
        let mut valid = Vec::with_capacity(n_rows);
        for chunk in smiles_list.chunks(CHUNK) {
            let mut molecules = parse_chunk(chunk);
            let table = calculate_table(&mut molecules, &layout);
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
    Ok((
        layout.scalar_names().to_vec(),
        to_numpy_2d(py, scalars, n_rows, ns)?,
        fps.into_any().unbind(),
        valid,
    ))
}

/// Number of bits of each fingerprint returned by `compute_descriptors`.
#[pyfunction]
fn fingerprint_bits(py: Python<'_>) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    for (name, bits) in TableLayout::all().fingerprint_shapes() {
        out.set_item(name, bits)?;
    }
    Ok(out.into_any().unbind())
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
    m.add_function(wrap_pyfunction!(compute_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint_bits, m)?)?;
    m.add_function(wrap_pyfunction!(compute_biosynfoni, m)?)?;
    Ok(())
}
