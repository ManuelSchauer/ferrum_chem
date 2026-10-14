//! Batch output layout: `f64` scalars plus bit-packed fingerprints.
//!
//! A [`DescriptorTable`] holds one row per input record:
//! - every non-fingerprint descriptor value as `f64` in a row-major matrix
//!   (`rows x n_scalars`); counts are exact, rows of records that could not be
//!   parsed are NaN and flagged in `valid`;
//! - one [`FingerprintBlock`] per fingerprint, bit-packed into `u64` words:
//!   bit `i` of a row is bit `i % 64` of word `i / 64`. Read as little-endian
//!   bytes, bit `i` is bit `i % 8` (LSB first) of byte `i / 8`, which is the
//!   layout of `numpy.packbits(bits, bitorder="little")`;
//! - count fingerprints (one integer per key, e.g. `KRFPC1..KRFPC4860`) are
//!   ordinary scalar columns by default ([`CountFingerprintFormat::Columns`]);
//!   with [`CountFingerprintFormat::Sparse`] each one is a [`CountBlock`]
//!   instead: only the non-zero `(key index, count)` pairs of each row, in
//!   compressed sparse row form (a few hundred bytes per molecule instead of
//!   8 bytes per key).
//!
//! Tables are computed by [`super::calculator::DescriptorCalculator`]; callers
//! stream large inputs chunk by chunk (see `io::stream`) so memory stays
//! bounded by the chunk size.

use std::io::{self, Write};

use rayon::prelude::*;

use super::batch::BatchDescriptor;
use super::typed::DescriptorValue;
use crate::mol::molecule::Molecule;

/// How count fingerprints ([`BatchDescriptor::is_count_fingerprint`]) are output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CountFingerprintFormat {
    /// One scalar column per key (`KRFPC1`, `KRFPC2`, ...), like any other
    /// count descriptor.
    #[default]
    Columns,
    /// One [`CountBlock`] per count fingerprint with the non-zero
    /// `(key index, count)` pairs of each row (0-based key indices); in CSV one
    /// column per count fingerprint holding space-separated `index:count` pairs.
    Sparse,
}

/// Column layout for a descriptor selection: which descriptors are scalars and
/// which are fingerprints, plus the scalar column names.
#[derive(Debug, Clone)]
pub struct TableLayout {
    descriptors: Vec<BatchDescriptor>,
    scalars: Vec<BatchDescriptor>,
    fingerprints: Vec<BatchDescriptor>,
    count_fingerprints: Vec<BatchDescriptor>,
    count_format: CountFingerprintFormat,
    scalar_names: Vec<String>,
    n_scalars: usize,
}

impl TableLayout {
    /// Splits `descriptors` into scalar and fingerprint descriptors (order is kept
    /// within each group); count fingerprints are scalar columns.
    pub fn new(descriptors: &[BatchDescriptor]) -> Self {
        Self::with_count_format(descriptors, CountFingerprintFormat::Columns)
    }

    /// Like [`Self::new`], with count fingerprints output as `count_format`.
    pub fn with_count_format(descriptors: &[BatchDescriptor], count_format: CountFingerprintFormat) -> Self {
        let sparse = count_format == CountFingerprintFormat::Sparse;
        let mut scalars = Vec::new();
        let mut fingerprints = Vec::new();
        let mut count_fingerprints = Vec::new();
        for &d in descriptors {
            if d.is_fingerprint() {
                fingerprints.push(d);
            } else if sparse && d.is_count_fingerprint() {
                assert!(d.feature_len() <= 1 << 16, "count fingerprint {} has too many keys", d.name());
                count_fingerprints.push(d);
            } else {
                scalars.push(d);
            }
        }
        let scalar_names = super::batch::feature_names(&scalars);
        let n_scalars = scalar_names.len();
        Self {
            descriptors: descriptors.to_vec(),
            scalars,
            fingerprints,
            count_fingerprints,
            count_format,
            scalar_names,
            n_scalars,
        }
    }

    /// The descriptor selection this layout was made for.
    pub fn descriptors(&self) -> &[BatchDescriptor] {
        &self.descriptors
    }

    /// How count fingerprints are output.
    pub fn count_format(&self) -> CountFingerprintFormat {
        self.count_format
    }

    /// Count fingerprints output as [`CountBlock`]s (empty with
    /// [`CountFingerprintFormat::Columns`]).
    pub fn count_fingerprint_descriptors(&self) -> &[BatchDescriptor] {
        &self.count_fingerprints
    }

    /// `(name, number of keys)` of every [`CountBlock`], in output order.
    pub fn count_fingerprint_shapes(&self) -> Vec<(&'static str, usize)> {
        self.count_fingerprints
            .iter()
            .map(|d| (d.count_fingerprint_name().unwrap_or_default(), d.feature_len()))
            .collect()
    }

    /// Layout of the full default bundle ([`BatchDescriptor::all`]).
    pub fn all() -> Self {
        Self::new(BatchDescriptor::all())
    }

    pub fn scalar_descriptors(&self) -> &[BatchDescriptor] {
        &self.scalars
    }

    pub fn fingerprint_descriptors(&self) -> &[BatchDescriptor] {
        &self.fingerprints
    }

    pub fn scalar_names(&self) -> &[String] {
        &self.scalar_names
    }

    pub fn n_scalars(&self) -> usize {
        self.n_scalars
    }

    /// `(short name, bit count)` of every fingerprint, in output order.
    pub fn fingerprint_shapes(&self) -> Vec<(&'static str, usize)> {
        self.fingerprints
            .iter()
            .map(|d| (d.fingerprint_name().unwrap_or_default(), d.feature_len()))
            .collect()
    }

    fn fingerprint_words(&self) -> Vec<usize> {
        self.fingerprints.iter().map(|d| d.feature_len().div_ceil(64)).collect()
    }
}

/// One bit-packed fingerprint for all rows of a table.
#[derive(Debug, Clone, PartialEq)]
pub struct FingerprintBlock {
    pub descriptor: BatchDescriptor,
    /// Short name (`"ECFP4"`, `"MACCS"`, ...).
    pub name: &'static str,
    pub n_bits: usize,
    pub words_per_row: usize,
    /// `rows x words_per_row` words, row-major.
    pub data: Vec<u64>,
}

impl FingerprintBlock {
    /// Packed words of one row.
    pub fn row(&self, row: usize) -> &[u64] {
        &self.data[row * self.words_per_row..(row + 1) * self.words_per_row]
    }

    /// Whether `bit` is set in `row`.
    pub fn get(&self, row: usize, bit: usize) -> bool {
        assert!(bit < self.n_bits, "bit index out of range");
        self.row(row)[bit / 64] >> (bit % 64) & 1 == 1
    }

    /// Indices of the set bits of `row`, ascending.
    pub fn on_bits(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        self.row(row).iter().enumerate().flat_map(|(w, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                (rest != 0).then(|| {
                    let b = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    w * 64 + b
                })
            })
        })
    }

    /// Number of set bits of `row`.
    pub fn count_ones(&self, row: usize) -> u32 {
        self.row(row).iter().map(|w| w.count_ones()).sum()
    }

    /// Bytes per row in the byte view (`ceil(n_bits / 8)`).
    pub fn bytes_per_row(&self) -> usize {
        self.n_bits.div_ceil(8)
    }

    /// The whole block as little-endian bytes, `rows x bytes_per_row`
    /// (`numpy.packbits(..., bitorder="little")` layout).
    pub fn to_bytes(&self) -> Vec<u8> {
        let bpr = self.bytes_per_row();
        let rows = self.data.len().checked_div(self.words_per_row).unwrap_or(0);
        let mut out = Vec::with_capacity(rows * bpr);
        for r in 0..rows {
            out.extend(self.row(r).iter().flat_map(|w| w.to_le_bytes()).take(bpr));
        }
        out
    }
}

/// One count fingerprint for all rows of a table, in compressed sparse row
/// form: row `r` holds the keys `indices[indptr[r]..indptr[r + 1]]`
/// (ascending, 0-based) with the (non-zero) `counts` at the same positions.
#[derive(Debug, Clone, PartialEq)]
pub struct CountBlock {
    pub descriptor: BatchDescriptor,
    /// Name (`"KlekotaRothCount"`, ...).
    pub name: &'static str,
    /// Number of keys (columns).
    pub n_keys: usize,
    /// `rows + 1` offsets into `indices` / `counts`.
    pub indptr: Vec<u32>,
    pub indices: Vec<u16>,
    pub counts: Vec<u32>,
}

impl CountBlock {
    /// Non-zero `(key indices, counts)` of `row`.
    pub fn row(&self, row: usize) -> (&[u16], &[u32]) {
        let (lo, hi) = (self.indptr[row] as usize, self.indptr[row + 1] as usize);
        (&self.indices[lo..hi], &self.counts[lo..hi])
    }

    /// Count of `key` in `row` (0 when absent).
    pub fn get(&self, row: usize, key: usize) -> u32 {
        assert!(key < self.n_keys, "key index out of range");
        let (keys, counts) = self.row(row);
        keys.binary_search(&(key as u16)).map_or(0, |i| counts[i])
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// The whole block as a dense row-major `rows x n_keys` matrix.
    pub fn to_dense(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.rows() * self.n_keys];
        for r in 0..self.rows() {
            let (keys, counts) = self.row(r);
            for (&k, &c) in keys.iter().zip(counts) {
                out[r * self.n_keys + k as usize] = c;
            }
        }
        out
    }
}

/// Descriptor values for a chunk of records.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorTable {
    pub rows: usize,
    pub scalar_names: Vec<String>,
    /// `rows x scalar_names.len()` values, row-major.
    pub scalars: Vec<f64>,
    pub fingerprints: Vec<FingerprintBlock>,
    /// Count fingerprints with [`CountFingerprintFormat::Sparse`] (empty otherwise).
    pub counts: Vec<CountBlock>,
    /// `false` for records whose molecule could not be parsed (NaN scalars,
    /// empty fingerprints).
    pub valid: Vec<bool>,
}

impl DescriptorTable {
    pub fn n_scalars(&self) -> usize {
        self.scalar_names.len()
    }

    /// Scalar values of one row.
    pub fn scalar_row(&self, row: usize) -> &[f64] {
        let n = self.n_scalars();
        &self.scalars[row * n..(row + 1) * n]
    }

    /// Fingerprint block by short name (`"ECFP4"`, ...).
    pub fn fingerprint(&self, name: &str) -> Option<&FingerprintBlock> {
        self.fingerprints.iter().find(|f| f.name == name)
    }

    /// Count block by name (`"KlekotaRothCount"`, ...), with
    /// [`CountFingerprintFormat::Sparse`].
    pub fn count_fingerprint(&self, name: &str) -> Option<&CountBlock> {
        self.counts.iter().find(|c| c.name == name)
    }
}

/// Returned by a cancellable calculation when it was cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

/// A molecule slot of an input batch: a molecule, or `None` for a record that
/// could not be parsed (gives an invalid row).
pub trait MoleculeSlot: Sync {
    fn molecule(&self) -> Option<&Molecule>;
}

impl MoleculeSlot for Molecule {
    fn molecule(&self) -> Option<&Molecule> {
        Some(self)
    }
}

impl MoleculeSlot for &Molecule {
    fn molecule(&self) -> Option<&Molecule> {
        Some(self)
    }
}

impl MoleculeSlot for Option<Molecule> {
    fn molecule(&self) -> Option<&Molecule> {
        self.as_ref()
    }
}

impl MoleculeSlot for Option<&Molecule> {
    fn molecule(&self) -> Option<&Molecule> {
        *self
    }
}

/// Computes all descriptors of `layout` for a batch in parallel (molecules
/// are only read). Use [`super::calculator::DescriptorCalculator`].
pub(crate) fn compute_table<M, F>(
    molecules: &[M],
    layout: &TableLayout,
    should_cancel: &F,
) -> Result<DescriptorTable, Cancelled>
where
    M: MoleculeSlot,
    F: Fn() -> bool + Sync,
{
    let mut scalars = vec![f64::NAN; molecules.len() * layout.n_scalars];
    let mut table = compute_table_into(molecules, layout, should_cancel, &mut scalars)?;
    table.scalars = scalars;
    Ok(table)
}

/// [`compute_table`] with the scalar values written to `scalars`
/// (`molecules.len() x n_scalars`, row-major; overwritten completely) instead
/// of the returned table, whose `scalars` stays empty.
pub(crate) fn compute_table_into<M, F>(
    molecules: &[M],
    layout: &TableLayout,
    should_cancel: &F,
    scalars: &mut [f64],
) -> Result<DescriptorTable, Cancelled>
where
    M: MoleculeSlot,
    F: Fn() -> bool + Sync,
{
    let rows = molecules.len();
    let ns = layout.n_scalars;
    assert_eq!(scalars.len(), rows * ns, "scalar buffer has the wrong size");
    scalars.fill(f64::NAN);
    let fp_words = layout.fingerprint_words();
    let total_words: usize = fp_words.iter().sum();

    // all fingerprints of a row side by side; split into blocks afterwards
    let mut fp_all = vec![0u64; rows * total_words];
    let mut valid = vec![false; rows];
    // sparse count fingerprints: the non-zero (key, count) pairs of each row
    let mut count_rows: Vec<Vec<Vec<(u16, u32)>>> = vec![Vec::new(); rows];

    // `par_chunks_mut(0)` panics, so empty segments use a dummy chunk size
    let scalar_chunks = scalars.par_chunks_mut(ns.max(1));
    let fp_chunks = fp_all.par_chunks_mut(total_words.max(1));
    let work = molecules.par_iter().zip(valid.par_iter_mut().zip(count_rows.par_iter_mut()));
    let run = |(((slot, (ok, crow)), srow), frow): (((&M, (&mut bool, &mut Vec<Vec<(u16, u32)>>)), &mut [f64]), &mut [u64])| {
        if should_cancel() {
            return Err(Cancelled);
        }
        let Some(mol) = slot.molecule() else { return Ok(()) };
        *ok = true;
        compute_row(layout, &fp_words, mol, srow, frow, crow);
        Ok(())
    };

    if ns > 0 && total_words > 0 {
        work.zip(scalar_chunks).zip(fp_chunks).try_for_each(run)?;
    } else if ns > 0 {
        work.zip(scalar_chunks).try_for_each(|(w, s)| run(((w, s), &mut [])))?;
    } else if total_words > 0 {
        work.zip(fp_chunks).try_for_each(|(w, f)| run(((w, &mut []), f)))?;
    } else {
        // only sparse count fingerprints (or nothing): still mark valid rows
        work.try_for_each(|w| run(((w, &mut []), &mut [])))?;
    }

    let mut fingerprints = Vec::with_capacity(layout.fingerprints.len());
    let mut word = 0;
    for (&d, &nw) in layout.fingerprints.iter().zip(&fp_words) {
        let mut data = Vec::with_capacity(rows * nw);
        for r in 0..rows {
            let start = r * total_words + word;
            data.extend_from_slice(&fp_all[start..start + nw]);
        }
        fingerprints.push(FingerprintBlock {
            descriptor: d,
            name: d.fingerprint_name().unwrap_or_default(),
            n_bits: d.feature_len(),
            words_per_row: nw,
            data,
        });
        word += nw;
    }
    drop(fp_all);

    let counts = layout
        .count_fingerprints
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let nnz: usize = count_rows.iter().map(|r| r.get(i).map_or(0, Vec::len)).sum();
            let mut block = CountBlock {
                descriptor: d,
                name: d.count_fingerprint_name().unwrap_or_default(),
                n_keys: d.feature_len(),
                indptr: Vec::with_capacity(rows + 1),
                indices: Vec::with_capacity(nnz),
                counts: Vec::with_capacity(nnz),
            };
            block.indptr.push(0);
            for r in &count_rows {
                if let Some(pairs) = r.get(i) {
                    block.indices.extend(pairs.iter().map(|p| p.0));
                    block.counts.extend(pairs.iter().map(|p| p.1));
                }
                block.indptr.push(block.indices.len() as u32);
            }
            block
        })
        .collect();

    Ok(DescriptorTable {
        rows,
        scalar_names: layout.scalar_names.clone(),
        scalars: Vec::new(),
        fingerprints,
        counts,
        valid,
    })
}

/// All values of one molecule: scalars into `srow` (NaN-filled, `n_scalars`
/// long), the fingerprints side by side into `frow` (zeroed; `fp_words[i]`
/// words for fingerprint `i`) and the non-zero pairs of every sparse count
/// fingerprint appended to `crow`. Runs on the calling thread with its
/// calculation context.
fn compute_row(
    layout: &TableLayout,
    fp_words: &[usize],
    mol: &Molecule,
    srow: &mut [f64],
    frow: &mut [u64],
    crow: &mut Vec<Vec<(u16, u32)>>,
) {
    super::calculator::with_thread_context(|ctx| {
        let mut offset = 0;
        for &d in &layout.scalars {
            d.write_into(mol, srow, &mut offset, ctx);
        }
        debug_assert_eq!(offset, srow.len().min(layout.n_scalars));
        if !layout.count_fingerprints.is_empty() {
            crow.extend(layout.count_fingerprints.iter().map(|&d| sparse_counts(d.calculate(mol, ctx))));
        }
        let mut word = 0;
        for (&d, &nw) in layout.fingerprints.iter().zip(fp_words) {
            pack_bits(d.calculate(mol, ctx), &mut frow[word..word + nw]);
            word += nw;
        }
    });
}

/// Computes and formats CSV rows one record at a time, without a
/// [`DescriptorTable`]: the bytes are the same as [`write_csv_rows`] gives for
/// a table of the same records, but the memory needed is one row of values,
/// however many rows are formatted. Use one per thread (e.g. per task of a
/// streaming pipeline); the molecule is calculated on the calling thread.
///
/// ```
/// use ferrum_core::desc::batch::BatchDescriptor;
/// use ferrum_core::desc::calculator::DescriptorCalculator;
/// use ferrum_core::desc::table::{CsvRowFormatter, FingerprintFormat};
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let calc = DescriptorCalculator::new(&[BatchDescriptor::HeavyAtomCount, BatchDescriptor::MaccsKeys]);
/// let mut rows = CsvRowFormatter::new(calc.layout(), FingerprintFormat::OnBits);
/// let mut out = Vec::new();
/// rows.write_row(&mut out, "CCO", parse_smiles("CCO").ok().as_ref());
/// rows.write_row(&mut out, "bad", None);
/// let text = String::from_utf8(out).unwrap();
/// assert!(text.starts_with("CCO,3,"));
/// assert!(text.ends_with("\nbad,,\n"));
/// ```
pub struct CsvRowFormatter<'a> {
    layout: &'a TableLayout,
    fp_format: FingerprintFormat,
    fp_words: Vec<usize>,
    /// `(first word in `words`, words, bits)` of every fingerprint
    fp_shapes: Vec<(usize, usize, usize)>,
    scalars: Vec<f64>,
    words: Vec<u64>,
    counts: Vec<Vec<(u16, u32)>>,
}

impl<'a> CsvRowFormatter<'a> {
    /// Formatter for the columns of `layout` (see [`write_csv_header`]).
    pub fn new(layout: &'a TableLayout, fp_format: FingerprintFormat) -> Self {
        let fp_words = layout.fingerprint_words();
        let mut word = 0;
        let fp_shapes = layout
            .fingerprints
            .iter()
            .zip(&fp_words)
            .map(|(d, &nw)| {
                word += nw;
                (word - nw, nw, d.feature_len())
            })
            .collect();
        Self {
            layout,
            fp_format,
            fp_shapes,
            scalars: vec![f64::NAN; layout.n_scalars],
            words: vec![0; word],
            counts: Vec::with_capacity(layout.count_fingerprints.len()),
            fp_words,
        }
    }

    /// Calculates `molecule` (`None`: a record that could not be parsed, which
    /// gets empty cells) and appends its CSV row with the trailing newline;
    /// `id` is the first column.
    pub fn write_row(&mut self, out: &mut Vec<u8>, id: &str, molecule: Option<&Molecule>) {
        self.scalars.fill(f64::NAN);
        self.words.fill(0);
        self.counts.clear();
        if let Some(mol) = molecule {
            compute_row(self.layout, &self.fp_words, mol, &mut self.scalars, &mut self.words, &mut self.counts);
        }
        let (words, counts) = (&self.words, &self.counts);
        format_row(
            out,
            id,
            molecule.is_some(),
            &self.scalars,
            self.fp_shapes.iter().map(|&(start, nw, bits)| (&words[start..start + nw], bits)),
            // one cell per count fingerprint, also for rows without a molecule
            (0..self.layout.count_fingerprints.len()).map(|i| counts.get(i).map_or(&[][..], Vec::as_slice).iter().copied()),
            self.fp_format,
        );
    }
}

/// The non-zero `(key index, count)` pairs of a count fingerprint value.
fn sparse_counts(value: DescriptorValue) -> Vec<(u16, u32)> {
    match value {
        DescriptorValue::IntVec(v) => {
            v.into_iter().enumerate().filter(|&(_, c)| c > 0).map(|(i, c)| (i as u16, c)).collect()
        }
        DescriptorValue::IntU16Vec(v) => {
            v.into_iter().enumerate().filter(|&(_, c)| c > 0).map(|(i, c)| (i as u16, c as u32)).collect()
        }
        other => {
            debug_assert!(false, "count fingerprint returned {other:?}");
            Vec::new()
        }
    }
}

/// Packs a fingerprint value into `out` (words already zeroed).
fn pack_bits(value: DescriptorValue, out: &mut [u64]) {
    match value {
        DescriptorValue::Binary(bits) => {
            let n = out.len() * 64;
            for i in bits.iter_ones().take_while(|&i| i < n) {
                out[i / 64] |= 1 << (i % 64);
            }
        }
        other => {
            debug_assert!(false, "fingerprint descriptor returned {other:?}");
        }
    }
}

/// How fingerprints are written to CSV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FingerprintFormat {
    /// One column per fingerprint: hex of the little-endian byte view (bit `i`
    /// is bit `i % 8` of byte `i / 8`).
    #[default]
    Hex,
    /// One column per fingerprint: space-separated indices of the set bits.
    OnBits,
    /// One `0`/`1` column per bit (wide; compatible with older outputs).
    Bits,
    /// No fingerprint columns.
    None,
}

/// Writes the CSV header for `layout`.
pub fn write_csv_header<W: Write>(
    w: &mut W,
    id_column: &str,
    layout: &TableLayout,
    fp_format: FingerprintFormat,
) -> io::Result<()> {
    w.write_all(id_column.as_bytes())?;
    for name in &layout.scalar_names {
        write!(w, ",{name}")?;
    }
    for &d in &layout.fingerprints {
        match fp_format {
            FingerprintFormat::Hex | FingerprintFormat::OnBits => {
                write!(w, ",{}", d.fingerprint_name().unwrap_or_default())?
            }
            FingerprintFormat::Bits => {
                for name in d.feature_names() {
                    write!(w, ",{name}")?;
                }
            }
            FingerprintFormat::None => {}
        }
    }
    for &d in &layout.count_fingerprints {
        write!(w, ",{}", d.count_fingerprint_name().unwrap_or_default())?;
    }
    writeln!(w)
}

/// Writes the rows of `table`; `ids[i]` is the first column of row `i`.
/// Values round-trip exactly (shortest `f64` representation, integers without
/// fraction); invalid rows have empty cells.
pub fn write_csv_rows<W: Write>(
    w: &mut W,
    ids: &[String],
    table: &DescriptorTable,
    fp_format: FingerprintFormat,
) -> io::Result<()> {
    let mut line = Vec::with_capacity(4096);
    for r in 0..table.rows {
        line.clear();
        format_csv_row(&mut line, ids, table, r, fp_format);
        w.write_all(&line)?;
    }
    Ok(())
}

/// The CSV rows of `table` as bytes (same output as [`write_csv_rows`]),
/// formatted in parallel.
pub fn format_csv_rows_par(ids: &[String], table: &DescriptorTable, fp_format: FingerprintFormat) -> Vec<u8> {
    const BLOCK: usize = 64;
    let blocks: Vec<Vec<u8>> = (0..table.rows.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let mut out = Vec::with_capacity(BLOCK * 8192);
            for r in b * BLOCK..((b + 1) * BLOCK).min(table.rows) {
                format_csv_row(&mut out, ids, table, r, fp_format);
            }
            out
        })
        .collect();
    blocks.concat()
}

/// Appends row `r` (with trailing newline) to `line`.
fn format_csv_row(line: &mut Vec<u8>, ids: &[String], table: &DescriptorTable, r: usize, fp_format: FingerprintFormat) {
    format_row(
        line,
        ids.get(r).map(String::as_str).unwrap_or_default(),
        table.valid[r],
        table.scalar_row(r),
        table.fingerprints.iter().map(|fp| (fp.row(r), fp.n_bits)),
        table.counts.iter().map(|block| {
            let (keys, counts) = block.row(r);
            keys.iter().copied().zip(counts.iter().copied())
        }),
        fp_format,
    );
}

/// Appends one CSV row (with trailing newline) to `line`: the identifier, the
/// scalar values, every fingerprint as `(packed words, bits)` and the non-zero
/// `(key, count)` pairs of every sparse count fingerprint. Rows that are not
/// `ok` get empty cells.
fn format_row<'r, P: Iterator<Item = (u16, u32)>>(
    line: &mut Vec<u8>,
    id: &str,
    ok: bool,
    scalars: &[f64],
    fingerprints: impl Iterator<Item = (&'r [u64], usize)>,
    counts: impl Iterator<Item = P>,
    fp_format: FingerprintFormat,
) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    write_csv_field(line, id);
    for &v in scalars {
        line.push(b',');
        if ok {
            write_f64(line, v);
        }
    }
    for (words, n_bits) in fingerprints {
        match fp_format {
            FingerprintFormat::Hex => {
                line.push(b',');
                if ok {
                    for byte in words.iter().flat_map(|w| w.to_le_bytes()).take(n_bits.div_ceil(8)) {
                        line.push(HEX[(byte >> 4) as usize]);
                        line.push(HEX[(byte & 15) as usize]);
                    }
                }
            }
            FingerprintFormat::OnBits => {
                line.push(b',');
                if ok {
                    let mut first = true;
                    for (w, &word) in words.iter().enumerate() {
                        let mut rest = word;
                        while rest != 0 {
                            let bit = w * 64 + rest.trailing_zeros() as usize;
                            rest &= rest - 1;
                            if !first {
                                line.push(b' ');
                            }
                            first = false;
                            line.extend_from_slice(itoa::Buffer::new().format(bit).as_bytes());
                        }
                    }
                }
            }
            FingerprintFormat::Bits => {
                for bit in 0..n_bits {
                    line.push(b',');
                    if ok {
                        line.push(if words[bit / 64] >> (bit % 64) & 1 == 1 { b'1' } else { b'0' });
                    }
                }
            }
            FingerprintFormat::None => {}
        }
    }
    for pairs in counts {
        line.push(b',');
        if ok {
            for (k, (key, count)) in pairs.enumerate() {
                if k > 0 {
                    line.push(b' ');
                }
                line.extend_from_slice(itoa::Buffer::new().format(key).as_bytes());
                line.push(b':');
                line.extend_from_slice(itoa::Buffer::new().format(count).as_bytes());
            }
        }
    }
    line.push(b'\n');
}

/// Shortest round-trip representation; integral values without fraction.
fn write_f64(out: &mut Vec<u8>, v: f64) {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        out.extend_from_slice(itoa::Buffer::new().format(v as i64).as_bytes());
    } else if v.is_finite() {
        // shortest round-trip representation (exponent form for large/small magnitudes)
        out.extend_from_slice(ryu::Buffer::new().format_finite(v).as_bytes());
    } else if v.is_nan() {
        out.extend_from_slice(b"NaN");
    } else {
        out.extend_from_slice(if v > 0.0 { b"inf" } else { b"-inf" });
    }
}

fn write_csv_field(out: &mut Vec<u8>, s: &str) {
    if s.contains([',', '"', '\n', '\r']) {
        out.push(b'"');
        for b in s.bytes() {
            if b == b'"' {
                out.push(b'"');
            }
            out.push(b);
        }
        out.push(b'"');
    } else {
        out.extend_from_slice(s.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::context::CalculationContext;
    use crate::io::smiles::parse_smiles;

    fn table_for(smiles: &[&str], descriptors: &[BatchDescriptor]) -> DescriptorTable {
        let mols: Vec<Option<Molecule>> = smiles.iter().map(|s| parse_smiles(s).ok()).collect();
        crate::desc::calculator::DescriptorCalculator::new(descriptors).calculate(&mols)
    }

    #[test]
    fn scalars_are_f64_and_invalid_rows_are_nan() {
        let t = table_for(
            &["CCO", "not a smiles", "c1ccccc1"],
            &[BatchDescriptor::MolecularWeight, BatchDescriptor::HeavyAtomCount],
        );
        assert_eq!(t.rows, 3);
        assert_eq!(t.scalar_names, vec!["MolWt", "HeavyAtomCount"]);
        assert_eq!(t.valid, vec![true, false, true]);
        assert!((t.scalar_row(0)[0] - 46.069).abs() < 1e-9 && t.scalar_row(0)[1] == 3.0);
        assert!(t.scalar_row(1).iter().all(|v| v.is_nan()));
        assert!(t.fingerprints.is_empty());
    }

    #[test]
    fn fingerprints_are_bit_packed() {
        let descriptors = [BatchDescriptor::MolecularWeight, BatchDescriptor::MaccsKeys, BatchDescriptor::Ecfp4];
        let t = table_for(&["c1ccccc1O", "xx"], &descriptors);
        assert_eq!(t.n_scalars(), 1);
        let maccs = t.fingerprint("MACCS").unwrap();
        assert_eq!((maccs.n_bits, maccs.words_per_row, maccs.bytes_per_row()), (166, 3, 21));
        let ecfp = t.fingerprint("ECFP4").unwrap();
        assert_eq!(ecfp.words_per_row, 32);

        // same bits as the descriptor itself
        let mol = parse_smiles("c1ccccc1O").unwrap();
        let mut ctx = CalculationContext::new();
        let expected: Vec<usize> = BatchDescriptor::MaccsKeys.calculate(&mol, &mut ctx).as_binary().iter_ones().collect();
        assert!(!expected.is_empty());
        assert_eq!(maccs.on_bits(0).collect::<Vec<_>>(), expected);
        assert_eq!(maccs.count_ones(0) as usize, expected.len());
        assert_eq!(maccs.count_ones(1), 0, "invalid row has no bits");

        // byte view: bit i -> byte i/8, bit i%8
        let bytes = maccs.to_bytes();
        assert_eq!(bytes.len(), 2 * 21);
        for &i in &expected {
            assert_eq!(bytes[i / 8] >> (i % 8) & 1, 1);
        }
    }

    #[test]
    fn cancellation_stops_calculation() {
        let mols = vec![parse_smiles("C").ok(); 64];
        let layout = TableLayout::new(&[BatchDescriptor::MolecularWeight]);
        assert_eq!(compute_table(&mols, &layout, &|| true), Err(Cancelled));
    }

    #[test]
    fn csv_formats() {
        let descriptors = [BatchDescriptor::HeavyAtomCount, BatchDescriptor::LogP, BatchDescriptor::MaccsKeys];
        let layout = TableLayout::new(&descriptors);
        let mols: Vec<Option<Molecule>> = vec![parse_smiles("CC(=O)O").ok(), None];
        let t = crate::desc::calculator::DescriptorCalculator::new(&descriptors).calculate(&mols);
        let ids = vec!["CC(=O)O".to_string(), "bad,id".to_string()];
        let render = |f| {
            let mut out = Vec::new();
            write_csv_header(&mut out, "SMILES", &layout, f).unwrap();
            write_csv_rows(&mut out, &ids, &t, f).unwrap();
            String::from_utf8(out).unwrap()
        };

        let hex = render(FingerprintFormat::Hex);
        let lines: Vec<&str> = hex.lines().collect();
        assert_eq!(lines[0], "SMILES,HeavyAtomCount,MolLogP,MACCS");
        let cells: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(cells[1], "4");
        assert_eq!(cells[2].parse::<f64>().unwrap(), t.scalar_row(0)[1], "exact round trip");
        assert_eq!(cells[3].len(), 42);
        assert_eq!(lines[2], "\"bad,id\",,,");

        let on = render(FingerprintFormat::OnBits);
        let maccs = t.fingerprint("MACCS").unwrap();
        let expected: Vec<String> = maccs.on_bits(0).map(|b| b.to_string()).collect();
        assert!(on.lines().nth(1).unwrap().ends_with(&expected.join(" ")));

        let bits = render(FingerprintFormat::Bits);
        assert_eq!(bits.lines().next().unwrap().split(',').count(), 3 + 166);
        let none = render(FingerprintFormat::None);
        assert_eq!(none.lines().next().unwrap(), "SMILES,HeavyAtomCount,MolLogP");
    }

    #[test]
    fn count_fingerprints_as_columns_or_sparse() {
        let descriptors =
            [BatchDescriptor::HeavyAtomCount, BatchDescriptor::CdkSubstructureCount, BatchDescriptor::KlekotaRoth];
        let smiles = ["CC(=O)Oc1ccccc1C(=O)O", "bad", "CCO"];
        let mols: Vec<Option<Molecule>> = smiles.iter().map(|s| parse_smiles(s).ok()).collect();
        let calc = crate::desc::calculator::DescriptorCalculator::new(&descriptors);
        let cols = calc.calculate(&mols);
        assert_eq!(cols.n_scalars(), 1 + 307);
        assert!(cols.counts.is_empty());
        assert_eq!(cols.scalar_names[1], "SubFPC1");

        let calc = calc.with_count_format(CountFingerprintFormat::Sparse);
        let layout = calc.layout();
        assert_eq!(layout.count_fingerprint_shapes(), vec![("CDKSubstructureCount", 307)]);
        let sparse = calc.calculate(&mols);
        assert_eq!(sparse.scalar_names, vec!["HeavyAtomCount"]);
        let block = sparse.count_fingerprint("CDKSubstructureCount").unwrap();
        assert_eq!((block.rows(), block.n_keys), (3, 307));
        let dense = block.to_dense();
        for r in [0, 2] {
            assert!(!block.row(r).0.is_empty());
            for k in 0..307 {
                assert_eq!(dense[r * 307 + k] as f64, cols.scalar_row(r)[1 + k]);
                assert_eq!(block.get(r, k), dense[r * 307 + k]);
            }
        }
        assert!(block.row(1).0.is_empty(), "invalid row has no counts");
        assert_eq!(sparse.fingerprints, cols.fingerprints);

        let ids: Vec<String> = smiles.iter().map(|s| s.to_string()).collect();
        let mut out = Vec::new();
        write_csv_header(&mut out, "SMILES", layout, FingerprintFormat::OnBits).unwrap();
        write_csv_rows(&mut out, &ids, &sparse, FingerprintFormat::OnBits).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "SMILES,HeavyAtomCount,KlekotaRoth,CDKSubstructureCount");
        let (keys, counts) = block.row(2);
        let pairs: Vec<String> = keys.iter().zip(counts).map(|(k, c)| format!("{k}:{c}")).collect();
        assert!(lines[3].ends_with(&format!(",{}", pairs.join(" "))));
        assert_eq!(lines[2], "bad,,,");
        assert_eq!(format_csv_rows_par(&ids, &sparse, FingerprintFormat::OnBits), text.split_once('\n').unwrap().1.as_bytes());
    }

    #[test]
    fn f64_formatting_round_trips() {
        for v in [0.0, 3.0, -2.0, 0.1, 1.0 / 3.0, 3.7e21, 1e-300, -5e-7] {
            let mut out = Vec::new();
            write_f64(&mut out, v);
            assert_eq!(String::from_utf8(out).unwrap().parse::<f64>().unwrap(), v);
        }
        let mut out = Vec::new();
        write_f64(&mut out, f64::NAN);
        assert_eq!(out, b"NaN");
    }

    #[test]
    fn full_bundle_lengths_match_on_torture_set() {
        let torture = [
            "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "Nc1ncnc2[nH]cnc12", "C1=CC=CC=CC=C1", "[cH-]1cccc1",
            "CC[NH3+]", "C[N+](C)(C)C", "O=[N+](O)c1ccccc1", "C12C3C4C1C5C4C3C25", "[13CH3]C([2H])([2H])[2H]",
            "C[C@H](O)C(=O)O", "F/C=C/F", "[NH4+].[Cl-]", "[O-][N+](=O)c1ccccc1", "C", "[Na+]", "[H][H]",
        ];
        let mut ctx = CalculationContext::new();
        for s in torture {
            let mol = parse_smiles(s).unwrap();
            for &d in BatchDescriptor::all() {
                assert_eq!(d.calculate(&mol, &mut ctx).len(), d.feature_len(), "{} on {s}", d.name());
            }
        }
        let t = table_for(&torture, BatchDescriptor::all());
        assert!(t.valid.iter().all(|&v| v));
        assert_eq!(t.scalars.len(), torture.len() * t.n_scalars());
    }
}
