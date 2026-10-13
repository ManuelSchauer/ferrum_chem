//! Batch output layout: `f64` scalars plus bit-packed fingerprints.
//!
//! A [`DescriptorTable`] holds one row per input record:
//! - every non-fingerprint descriptor value as `f64` in a row-major matrix
//!   (`rows x n_scalars`); counts are exact, rows of records that could not be
//!   parsed are NaN and flagged in `valid`;
//! - one [`FingerprintBlock`] per fingerprint, bit-packed into `u64` words:
//!   bit `i` of a row is bit `i % 64` of word `i / 64`. Read as little-endian
//!   bytes, bit `i` is bit `i % 8` (LSB first) of byte `i / 8`, which is the
//!   layout of `numpy.packbits(bits, bitorder="little")`.
//!
//! [`calculate_table`] computes a table in parallel for a chunk of molecules;
//! callers stream large inputs chunk by chunk (see `io::stream`) so memory stays
//! bounded by the chunk size.

use std::io::{self, Write};

use rayon::prelude::*;

use super::batch::BatchDescriptor;
use super::context::CalculationContext;
use super::typed::DescriptorValue;
use crate::mol::molecule::Molecule;

/// Column layout for a descriptor selection: which descriptors are scalars and
/// which are fingerprints, plus the scalar column names.
#[derive(Debug, Clone)]
pub struct TableLayout {
    scalars: Vec<BatchDescriptor>,
    fingerprints: Vec<BatchDescriptor>,
    scalar_names: Vec<String>,
    n_scalars: usize,
}

impl TableLayout {
    /// Splits `descriptors` into scalar and fingerprint descriptors (order is kept
    /// within each group).
    pub fn new(descriptors: &[BatchDescriptor]) -> Self {
        let (fingerprints, scalars): (Vec<_>, Vec<_>) =
            descriptors.iter().copied().partition(|d| d.is_fingerprint());
        let scalar_names = super::batch::feature_names(&scalars);
        let n_scalars = scalar_names.len();
        Self { scalars, fingerprints, scalar_names, n_scalars }
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
        let rows = if self.words_per_row == 0 { 0 } else { self.data.len() / self.words_per_row };
        let mut out = Vec::with_capacity(rows * bpr);
        for r in 0..rows {
            out.extend(self.row(r).iter().flat_map(|w| w.to_le_bytes()).take(bpr));
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
}

/// Returned by [`calculate_table_cancellable`] when cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

/// Computes all descriptors of `layout` for a chunk of records in parallel.
/// `None` entries (unparseable input) give an invalid row.
pub fn calculate_table(molecules: &mut [Option<Molecule>], layout: &TableLayout) -> DescriptorTable {
    match calculate_table_cancellable(molecules, layout, &|| false) {
        Ok(table) => table,
        Err(Cancelled) => unreachable!("never cancelled"),
    }
}

/// [`calculate_table`] with cooperative cancellation between molecules.
pub fn calculate_table_cancellable<F>(
    molecules: &mut [Option<Molecule>],
    layout: &TableLayout,
    should_cancel: &F,
) -> Result<DescriptorTable, Cancelled>
where
    F: Fn() -> bool + Sync,
{
    let rows = molecules.len();
    let ns = layout.n_scalars;
    let fp_words = layout.fingerprint_words();
    let total_words: usize = fp_words.iter().sum();

    let mut scalars = vec![f64::NAN; rows * ns];
    // all fingerprints of a row side by side; split into blocks afterwards
    let mut fp_all = vec![0u64; rows * total_words];
    let mut valid = vec![false; rows];

    // `par_chunks_mut(0)` panics, so empty segments use a dummy chunk size
    let scalar_chunks = scalars.par_chunks_mut(ns.max(1));
    let fp_chunks = fp_all.par_chunks_mut(total_words.max(1));
    let work = molecules.par_iter_mut().zip(valid.par_iter_mut());
    let run = |ctx: &mut CalculationContext,
               (((mol, ok), srow), frow): (((&mut Option<Molecule>, &mut bool), &mut [f64]), &mut [u64])| {
        if should_cancel() {
            return Err(Cancelled);
        }
        let Some(mol) = mol else { return Ok(()) };
        *ok = true;
        ctx.reset_for_new_molecule();
        let mut offset = 0;
        for &d in &layout.scalars {
            d.write_into(mol, srow, &mut offset, ctx);
        }
        debug_assert_eq!(offset, srow.len().min(ns));
        let mut word = 0;
        for (&d, &nw) in layout.fingerprints.iter().zip(&fp_words) {
            pack_bits(d.calculate(mol, ctx), &mut frow[word..word + nw]);
            word += nw;
        }
        Ok(())
    };

    if ns > 0 && total_words > 0 {
        work.zip(scalar_chunks).zip(fp_chunks).try_for_each_init(CalculationContext::new, run)?;
    } else if ns > 0 {
        work.zip(scalar_chunks)
            .try_for_each_init(CalculationContext::new, |ctx, (w, s)| run(ctx, ((w, s), &mut [])))?;
    } else if total_words > 0 {
        work.zip(fp_chunks)
            .try_for_each_init(CalculationContext::new, |ctx, (w, f)| run(ctx, ((w, &mut []), f)))?;
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

    Ok(DescriptorTable {
        rows,
        scalar_names: layout.scalar_names.clone(),
        scalars,
        fingerprints,
        valid,
    })
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
    const HEX: &[u8; 16] = b"0123456789abcdef";
    {
        write_csv_field(line, ids.get(r).map(String::as_str).unwrap_or_default());
        let ok = table.valid[r];
        for &v in table.scalar_row(r) {
            line.push(b',');
            if ok {
                write_f64(line, v);
            }
        }
        for fp in &table.fingerprints {
            match fp_format {
                FingerprintFormat::Hex => {
                    line.push(b',');
                    if ok {
                        for byte in fp.row(r).iter().flat_map(|w| w.to_le_bytes()).take(fp.bytes_per_row()) {
                            line.push(HEX[(byte >> 4) as usize]);
                            line.push(HEX[(byte & 15) as usize]);
                        }
                    }
                }
                FingerprintFormat::OnBits => {
                    line.push(b',');
                    if ok {
                        for (k, bit) in fp.on_bits(r).enumerate() {
                            if k > 0 {
                                line.push(b' ');
                            }
                            line.extend_from_slice(itoa::Buffer::new().format(bit).as_bytes());
                        }
                    }
                }
                FingerprintFormat::Bits => {
                    for bit in 0..fp.n_bits {
                        line.push(b',');
                        if ok {
                            line.push(if fp.get(r, bit) { b'1' } else { b'0' });
                        }
                    }
                }
                FingerprintFormat::None => {}
            }
        }
        line.push(b'\n');
    }
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
    use crate::io::smiles::parse_smiles;

    fn table_for(smiles: &[&str], descriptors: &[BatchDescriptor]) -> DescriptorTable {
        let mut mols: Vec<Option<Molecule>> = smiles.iter().map(|s| parse_smiles(s).ok()).collect();
        calculate_table(&mut mols, &TableLayout::new(descriptors))
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
        let mut mol = parse_smiles("c1ccccc1O").unwrap();
        let mut ctx = CalculationContext::new();
        let expected: Vec<usize> = BatchDescriptor::MaccsKeys.calculate(&mut mol, &mut ctx).as_binary().iter_ones().collect();
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
        let mut mols = vec![parse_smiles("C").ok(); 64];
        let layout = TableLayout::new(&[BatchDescriptor::MolecularWeight]);
        assert_eq!(calculate_table_cancellable(&mut mols, &layout, &|| true), Err(Cancelled));
    }

    #[test]
    fn csv_formats() {
        let descriptors = [BatchDescriptor::HeavyAtomCount, BatchDescriptor::LogP, BatchDescriptor::MaccsKeys];
        let layout = TableLayout::new(&descriptors);
        let mut mols: Vec<Option<Molecule>> = vec![parse_smiles("CC(=O)O").ok(), None];
        let t = calculate_table(&mut mols, &layout);
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
            let mut mol = parse_smiles(s).unwrap();
            for &d in BatchDescriptor::all() {
                assert_eq!(d.calculate(&mut mol, &mut ctx).len(), d.feature_len(), "{} on {s}", d.name());
            }
        }
        let t = table_for(&torture, BatchDescriptor::all());
        assert!(t.valid.iter().all(|&v| v));
        assert_eq!(t.scalars.len(), torture.len() * t.n_scalars());
    }
}
