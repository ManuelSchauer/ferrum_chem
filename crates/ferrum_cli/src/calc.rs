use ferrum_core::desc::calculator::DescriptorCalculator;
use ferrum_core::desc::table::{write_csv_header, CountFingerprintFormat, FingerprintFormat};
use ferrum_core::io::stream::{MoleculeReader, RawRecord};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

/// Default number of records in flight (read but not yet written).
pub const DEFAULT_IN_FLIGHT: usize = 1024;

/// Rows written and records that could not be parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamStats {
    pub rows: usize,
    pub failed: usize,
}

/// Why [`stream_csv`] stopped.
#[derive(Debug)]
pub enum StreamError {
    /// Reading the input failed (rows before the failing record are written).
    Read(String),
    /// Writing the output failed.
    Write(std::io::Error),
}

/// Records per task: about eight tasks per thread fit into `in_flight`, so
/// the other threads keep working while one is busy with a slow molecule
/// (its block holds back the rows after it to keep the input order); 8-9
/// records per task on 14 threads (4 and 36 were slower, 64 used more memory).
fn block_size(in_flight: usize, threads: usize) -> usize {
    (in_flight / (8 * threads.max(1))).clamp(1, 32)
}

/// Streams every record of `reader` through `calc` as CSV rows into `writer`
/// (header not included), in input order.
///
/// Records are read here and handed to the rayon pool in small blocks; each
/// task parses its records, calculates one molecule at a time and formats
/// its rows directly (no descriptor table is materialised), and the rows are
/// written here as soon as all earlier ones are. At most about `in_flight`
/// records are read but not yet written, which bounds memory (the formatted
/// rows of a block are its only buffer) independently of the input size.
/// `progress` gets the number of rows written so far; when `should_cancel`
/// returns true no further records are read and `Ok(None)` is returned once
/// the running tasks have finished.
pub fn stream_csv<W: Write>(
    reader: &mut MoleculeReader,
    calc: &Arc<DescriptorCalculator>,
    writer: &mut W,
    fp_format: FingerprintFormat,
    in_flight: usize,
    progress: &mut dyn FnMut(usize),
    should_cancel: &dyn Fn() -> bool,
) -> Result<Option<StreamStats>, StreamError> {
    let in_flight = in_flight.max(1);
    let block = block_size(in_flight, rayon::current_num_threads());
    // (block index, formatted rows, rows, unparseable rows)
    let (tx, rx) = mpsc::channel::<(usize, Vec<u8>, usize, usize)>();
    let mut pending = BTreeMap::new();
    let (mut spawned, mut written, mut open_rows) = (0usize, 0usize, 0usize);
    let mut stats = StreamStats { rows: 0, failed: 0 };
    let (mut done, mut cancelled) = (false, false);
    let (mut read_error, mut write_error) = (None, None);
    loop {
        while !done && open_rows < in_flight {
            if should_cancel() {
                (done, cancelled) = (true, true);
                break;
            }
            let mut records: Vec<RawRecord> = Vec::with_capacity(block);
            while records.len() < block {
                match reader.next_record() {
                    Ok(Some(r)) => records.push(r),
                    Ok(None) => {
                        done = true;
                        break;
                    }
                    Err(e) => {
                        read_error = Some(e.to_string());
                        done = true;
                        break;
                    }
                }
            }
            if records.is_empty() {
                break;
            }
            open_rows += records.len();
            let (calc, tx, index) = (Arc::clone(calc), tx.clone(), spawned);
            rayon::spawn(move || {
                let n = records.len();
                let mut rows = calc.csv_rows(fp_format);
                let mut out = Vec::new();
                let mut failed = 0;
                for record in records {
                    let record = record.parse();
                    failed += usize::from(record.molecule.is_err());
                    rows.write_row(&mut out, &record.id, record.molecule.as_ref().ok());
                }
                let _ = tx.send((index, out, n, failed));
            });
            spawned += 1;
        }
        if written == spawned {
            break;
        }
        // `tx` is held here, so this waits for the next task (a panicking task aborts)
        let (index, bytes, n, failed) = rx.recv().expect("sender held");
        pending.insert(index, (bytes, n, failed));
        while let Some((bytes, n, failed)) = pending.remove(&written) {
            // after a write error keep collecting the running tasks, but write nothing
            if write_error.is_none()
                && !cancelled
                && let Err(e) = writer.write_all(&bytes)
            {
                write_error = Some(e);
                done = true;
            }
            written += 1;
            open_rows -= n;
            stats.rows += n;
            stats.failed += failed;
            progress(stats.rows);
        }
    }
    match (write_error, read_error) {
        (Some(e), _) => Err(StreamError::Write(e)),
        (None, Some(e)) => Err(StreamError::Read(e)),
        (None, None) if cancelled => Ok(None),
        (None, None) => Ok(Some(stats)),
    }
}

/// Runs batch descriptor calculation from CLI arguments.
///
/// The input is streamed with at most about `in_flight` records read but not
/// yet written (calculated in parallel, written in input order), so memory
/// stays bounded for any input size. Every input record gives one output row;
/// records that cannot be parsed get empty cells.
pub fn run_calc(input: &Path, output: &Path, fp_format: FingerprintFormat, in_flight: usize) -> Result<(), String> {
    run_calc_with_counts(input, output, fp_format, CountFingerprintFormat::Columns, in_flight)
}

/// [`run_calc`] with the output format of the count fingerprints (scalar
/// columns `KRFPC1..`, or one sparse `index:count` column per count fingerprint).
pub fn run_calc_with_counts(
    input: &Path,
    output: &Path,
    fp_format: FingerprintFormat,
    count_format: CountFingerprintFormat,
    in_flight: usize,
) -> Result<(), String> {
    println!("=== FerrumChem Descriptor Calculation ===");
    println!("Input file:  {}", input.display());
    println!("Output file: {}", output.display());

    let mut reader =
        MoleculeReader::open(input).map_err(|e| format!("Failed to open '{}': {}", input.display(), e))?;

    // one calculator, shared by every worker for every molecule
    let calc = Arc::new(DescriptorCalculator::all().with_count_format(count_format));
    let layout = calc.layout();
    println!(
        "Calculating {} scalar descriptors and {} fingerprints ({})...",
        layout.n_scalars(),
        layout.fingerprint_descriptors().len(),
        layout
            .fingerprint_shapes()
            .iter()
            .map(|(name, bits)| format!("{name} {bits} bit"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (name, keys) in layout.count_fingerprint_shapes() {
        println!("Count fingerprint {name} ({keys} keys) as sparse index:count column");
    }

    let out_file = File::create(output)
        .map_err(|e| format!("Failed to create output file '{}': {}", output.display(), e))?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);
    let write_err = |e: std::io::Error| format!("Failed to write '{}': {}", output.display(), e);
    write_csv_header(&mut writer, reader.id_column(), layout, fp_format).map_err(write_err)?;

    let start = Instant::now();
    let stats = match stream_csv(&mut reader, &calc, &mut writer, fp_format, in_flight, &mut |_| {}, &|| false) {
        Ok(Some(stats)) => stats,
        Ok(None) => unreachable!("never cancelled"),
        Err(StreamError::Read(e)) => return Err(format!("Failed to read '{}': {}", input.display(), e)),
        Err(StreamError::Write(e)) => return Err(write_err(e)),
    };
    writer.flush().map_err(write_err)?;

    if stats.rows == 0 {
        return Err(format!("No records found in '{}'.", input.display()));
    }
    let elapsed = start.elapsed();
    println!(
        "Wrote {} rows ({} could not be parsed) in {:.2?} ({:.0} molecules/sec)",
        stats.rows,
        stats.failed,
        elapsed,
        stats.rows as f64 / elapsed.as_secs_f64().max(1e-6)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::desc::batch::BatchDescriptor;
    use ferrum_core::desc::table::write_csv_rows;

    #[test]
    fn streamed_rows_equal_table_rows_in_input_order() {
        let smiles = ["CCO", "not_a_smiles", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "[Na+].[Cl-]", "C1CC1N"];
        let path = std::env::temp_dir().join(format!("ferrum_stream_test_{}.smi", std::process::id()));
        std::fs::write(&path, smiles.join("\n")).unwrap();
        let descriptors = [BatchDescriptor::MolecularWeight, BatchDescriptor::MaccsKeys, BatchDescriptor::CdkSubstructureCount];
        for count_format in [CountFingerprintFormat::Columns, CountFingerprintFormat::Sparse] {
            let calc = Arc::new(DescriptorCalculator::new(&descriptors).with_count_format(count_format));
            let table = calc.calculate_smiles(&smiles);
            let ids: Vec<String> = smiles.iter().map(|s| s.to_string()).collect();
            let mut expected = Vec::new();
            write_csv_rows(&mut expected, &ids, &table, FingerprintFormat::OnBits).unwrap();
            for in_flight in [1, 2, 1024] {
                let mut reader = MoleculeReader::open(&path).unwrap();
                let (mut out, mut seen) = (Vec::new(), Vec::new());
                let stats = stream_csv(&mut reader, &calc, &mut out, FingerprintFormat::OnBits, in_flight, &mut |r| seen.push(r), &|| false)
                    .unwrap()
                    .unwrap();
                assert_eq!(stats, StreamStats { rows: 6, failed: 1 });
                assert_eq!(out, expected);
                assert_eq!(seen.last(), Some(&6));
                let mut reader = MoleculeReader::open(&path).unwrap();
                let cancelled = stream_csv(&mut reader, &calc, &mut Vec::new(), FingerprintFormat::Hex, in_flight, &mut |_| {}, &|| true);
                assert!(matches!(cancelled, Ok(None)));
            }
        }
        let _ = std::fs::remove_file(path);
    }
}
