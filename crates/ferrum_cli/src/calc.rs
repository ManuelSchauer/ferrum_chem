use ferrum_core::desc::calculator::DescriptorCalculator;
use ferrum_core::desc::table::{format_csv_rows_par, write_csv_header, FingerprintFormat};
use ferrum_core::io::stream::MoleculeReader;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// Records calculated concurrently (summed over all chunks in flight). A
/// chunk waits for its slowest molecule; keeping several chunks in flight lets
/// the other cores continue with later chunks meanwhile. Memory scales with it.
const IN_FLIGHT_RECORDS: usize = 16_384;

/// Runs batch descriptor calculation from CLI arguments.
///
/// The input is streamed in chunks of `chunk_size` records (parsed and
/// calculated in parallel, written in input order), so memory stays bounded
/// for any input size. Every input record gives one output row; records that
/// cannot be parsed get empty cells.
pub fn run_calc(
    input: &Path,
    output: &Path,
    fp_format: FingerprintFormat,
    chunk_size: usize,
) -> Result<(), String> {
    println!("=== FerrumChem Descriptor Calculation ===");
    println!("Input file:  {}", input.display());
    println!("Output file: {}", output.display());

    let mut reader = MoleculeReader::open(input)
        .map_err(|e| format!("Failed to open '{}': {}", input.display(), e))?
        .with_chunk_size(chunk_size);

    // one calculator, shared by every worker for every molecule
    let calc = Arc::new(DescriptorCalculator::all());
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

    let out_file = File::create(output)
        .map_err(|e| format!("Failed to create output file '{}': {}", output.display(), e))?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);
    write_csv_header(&mut writer, reader.id_column(), layout, fp_format)
        .map_err(|e| format!("Failed to write '{}': {}", output.display(), e))?;

    let start = Instant::now();
    let in_flight = (IN_FLIGHT_RECORDS / chunk_size.max(1)).max(2);
    let slots = Arc::new((Mutex::new(in_flight), Condvar::new()));
    // (chunk index, formatted CSV rows, rows, unparseable rows)
    let (tx, rx) = mpsc::channel::<(usize, Vec<u8>, usize, usize)>();

    // Writer: receives chunks in completion order, writes them in input order.
    let writer_slots = Arc::clone(&slots);
    let writer_thread = std::thread::spawn(move || -> std::io::Result<(usize, usize)> {
        let (mut rows, mut failed, mut next) = (0usize, 0usize, 0usize);
        let mut pending = BTreeMap::new();
        // after a write error keep draining (and freeing slots) so the reader never blocks
        let mut error = None;
        for (index, bytes, n_rows, n_failed) in rx {
            pending.insert(index, (bytes, n_rows, n_failed));
            while let Some((bytes, n_rows, n_failed)) = pending.remove(&next) {
                if error.is_none() {
                    error = writer.write_all(&bytes).err();
                }
                rows += n_rows;
                failed += n_failed;
                next += 1;
                let (count, cv) = &*writer_slots;
                *count.lock().unwrap() += 1;
                cv.notify_one();
            }
        }
        match error {
            Some(e) => Err(e),
            None => writer.flush().map(|()| (rows, failed)),
        }
    });

    let mut read_error = None;
    let mut index = 0usize;
    loop {
        {
            let (count, cv) = &*slots;
            let mut free = cv.wait_while(count.lock().unwrap(), |n| *n == 0).unwrap();
            *free -= 1;
        }
        let chunk = match reader.next_chunk() {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(e) => {
                read_error = Some(format!("Failed to read '{}': {}", input.display(), e));
                break;
            }
        };
        let (calc, tx) = (Arc::clone(&calc), tx.clone());
        rayon::spawn(move || {
            let (ids, molecules): (Vec<_>, Vec<Option<_>>) =
                chunk.into_iter().map(|r| (r.id, r.molecule.ok())).unzip();
            let table = calc.calculate(&molecules);
            drop(molecules);
            // format in the worker pool, so the writer thread only copies bytes
            let bytes = format_csv_rows_par(&ids, &table, fp_format);
            let failed = table.valid.iter().filter(|&&ok| !ok).count();
            let _ = tx.send((index, bytes, table.rows, failed));
        });
        index += 1;
    }
    drop(tx);
    let (rows, failed) = writer_thread
        .join()
        .map_err(|_| "writer thread panicked".to_string())?
        .map_err(|e| format!("Failed to write '{}': {}", output.display(), e))?;
    if let Some(e) = read_error {
        return Err(e);
    }

    if rows == 0 {
        return Err(format!("No records found in '{}'.", input.display()));
    }
    let elapsed = start.elapsed();
    println!(
        "Wrote {} rows ({} could not be parsed) in {:.2?} ({:.0} molecules/sec)",
        rows,
        failed,
        elapsed,
        rows as f64 / elapsed.as_secs_f64().max(1e-6)
    );
    Ok(())
}
