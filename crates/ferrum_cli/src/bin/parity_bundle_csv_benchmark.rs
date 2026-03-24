#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use csv::ReaderBuilder;
use ferrum_core::desc::batch::BatchDescriptor;
use ferrum_core::io::csv_reader::CsvReader;
use ferrum_core::io::smiles::parse_smiles;
use std::env;
use std::hint::black_box;
use std::time::Instant;
use std::{ffi::c_int, mem::MaybeUninit};

const DEFAULT_LIMIT: usize = 10_000;
const DEFAULT_REPEATS: usize = 1;
const DEFAULT_WARMUP_ROWS: usize = 100;
const DEFAULT_CSV_PATH: &str = "coconut_csv_lite-02-2026.csv";
const DEFAULT_CSV_COLUMN: &str = "canonical_smiles";

#[repr(C)]
struct TimeVal {
    tv_sec: i64,
    tv_usec: i64,
}

#[repr(C)]
struct RUsage {
    ru_utime: TimeVal,
    ru_stime: TimeVal,
    ru_maxrss: i64,
    ru_ixrss: i64,
    ru_idrss: i64,
    ru_isrss: i64,
    ru_minflt: i64,
    ru_majflt: i64,
    ru_nswap: i64,
    ru_inblock: i64,
    ru_oublock: i64,
    ru_msgsnd: i64,
    ru_msgrcv: i64,
    ru_nsignals: i64,
    ru_nvcsw: i64,
    ru_nivcsw: i64,
}

unsafe extern "C" {
    fn getrusage(who: c_int, usage: *mut RUsage) -> c_int;
}

const RUSAGE_SELF: c_int = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CsvBenchMode {
    Buffered,
    Stream,
}

fn bundle_descriptors() -> Vec<BatchDescriptor> {
    vec![
        BatchDescriptor::MolecularWeight,
        BatchDescriptor::LogP,
        BatchDescriptor::MolarRefractivity,
        BatchDescriptor::Tpsa,
        BatchDescriptor::HBondDonor,
        BatchDescriptor::HBondAcceptor,
        BatchDescriptor::RotatableBondCount,
        BatchDescriptor::HeavyAtomCount,
        BatchDescriptor::AtomCount,
        BatchDescriptor::AromaticAtomCount,
        BatchDescriptor::HeteroatomCount,
        BatchDescriptor::AromaticRingCount,
        BatchDescriptor::FormalCharge,
        BatchDescriptor::FractionCSP3,
        BatchDescriptor::BertzCT,
        BatchDescriptor::LabuteAsa,
        BatchDescriptor::Qed,
        BatchDescriptor::NumAmideBonds,
        BatchDescriptor::HallKierAlpha,
        BatchDescriptor::BalabanIndex,
        BatchDescriptor::BondCount,
        BatchDescriptor::SlogpVsa,
        BatchDescriptor::SmrVsa,
        BatchDescriptor::PeoeVsa,
    ]
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn env_bench_mode() -> CsvBenchMode {
    match env::var("FERRUM_BENCH_CSV_MODE")
        .ok()
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("stream") => CsvBenchMode::Stream,
        _ => CsvBenchMode::Buffered,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    values[values.len() / 2]
}

fn peak_rss_bytes() -> Option<u64> {
    let mut usage = MaybeUninit::<RUsage>::uninit();
    // SAFETY: getrusage writes a full RUsage struct into the provided out-pointer.
    let rc = unsafe { getrusage(RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: rc == 0 guarantees `usage` has been initialized by getrusage.
    let usage = unsafe { usage.assume_init() };
    let maxrss = usage.ru_maxrss;
    if maxrss <= 0 {
        return Some(0);
    }
    // Darwin reports ru_maxrss in bytes.
    Some(maxrss as u64)
}

fn bundle_output_len(descriptors: &[BatchDescriptor]) -> usize {
    descriptors.iter().map(|d| d.feature_len()).sum()
}

fn run_bundle_on_molecule(
    mol: &mut ferrum_core::mol::molecule::Molecule,
    descriptors: &[BatchDescriptor],
) {
    let mut ctx = ferrum_core::desc::context::CalculationContext::new();
    for &desc in descriptors {
        black_box(desc.calculate(mol, &mut ctx));
    }
}

fn run_bundle(smiles: &str, descriptors: &[BatchDescriptor]) -> bool {
    match parse_smiles(smiles) {
        Ok(mut mol) => {
            run_bundle_on_molecule(&mut mol, descriptors);
            true
        }
        Err(_) => false,
    }
}

fn run_bundle_stream(
    csv_path: &str,
    csv_column: &str,
    limit: usize,
    descriptors: &[BatchDescriptor],
    chunk_size: usize,
) -> (usize, usize) {
    let reader = CsvReader::from_path(csv_path)
        .expect("open csv benchmark file")
        .with_smiles_column(csv_column)
        .with_properties(false)
        .with_row_limit(limit)
        .with_chunk_size(chunk_size);

    let mut processed = 0usize;
    let mut failed = 0usize;
    let _summary = reader
        .parse_stream(|record| {
            let mut mol = match record.molecule {
                Some(mol) => mol,
                None => {
                    failed += 1;
                    return;
                }
            };
            run_bundle_on_molecule(&mut mol, descriptors);
            processed += 1;
        })
        .expect("stream csv benchmark parse");
    (processed, failed)
}

fn load_smiles(path: &str, column: &str, limit: usize) -> Result<Vec<String>, String> {
    let mut reader = ReaderBuilder::new()
        .from_path(path)
        .map_err(|e| format!("failed to open CSV '{path}': {e}"))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("failed to read CSV headers: {e}"))?
        .clone();
    let smiles_idx = headers
        .iter()
        .position(|header| header == column)
        .ok_or_else(|| format!("SMILES column '{column}' not found in {path}"))?;

    let mut smiles = Vec::with_capacity(limit);
    for record in reader.records().take(limit) {
        let record = record.map_err(|e| format!("failed to read CSV record: {e}"))?;
        smiles.push(record.get(smiles_idx).unwrap_or("").trim().to_string());
    }
    Ok(smiles)
}

fn main() {
    let csv_path = env::var("FERRUM_BENCH_CSV").unwrap_or_else(|_| DEFAULT_CSV_PATH.to_string());
    let csv_column =
        env::var("FERRUM_BENCH_CSV_COLUMN").unwrap_or_else(|_| DEFAULT_CSV_COLUMN.to_string());
    let limit = env_usize("FERRUM_BENCH_CSV_LIMIT", DEFAULT_LIMIT);
    let repeats = env_usize("FERRUM_BENCH_REPEATS", DEFAULT_REPEATS);
    let warmup_rows = env_usize("FERRUM_BENCH_WARMUP", DEFAULT_WARMUP_ROWS);
    let chunk_size = env_usize("FERRUM_BENCH_CSV_CHUNK_SIZE", 10_000);
    let mode = env_bench_mode();

    let descriptors = bundle_descriptors();
    let outputs = bundle_output_len(&descriptors);

    let mut processed_rows = 0usize;
    let mut parse_failures = 0usize;
    let mut timings = Vec::with_capacity(repeats);
    match mode {
        CsvBenchMode::Buffered => {
            let smiles = load_smiles(&csv_path, &csv_column, limit).expect("load benchmark smiles");
            let warmup_rows = warmup_rows.min(smiles.len());
            for smiles in smiles.iter().take(warmup_rows) {
                let _ = run_bundle(smiles, &descriptors);
            }

            for _ in 0..repeats {
                let start = Instant::now();
                let mut processed = 0usize;
                let mut failed = 0usize;
                for smiles in &smiles {
                    if run_bundle(smiles, &descriptors) {
                        processed += 1;
                    } else {
                        failed += 1;
                    }
                }
                timings.push(start.elapsed().as_secs_f64());
                processed_rows = processed;
                parse_failures = failed;
            }
        }
        CsvBenchMode::Stream => {
            let warmup_limit = warmup_rows.min(limit);
            if warmup_limit > 0 {
                let _ = run_bundle_stream(
                    &csv_path,
                    &csv_column,
                    warmup_limit,
                    &descriptors,
                    chunk_size,
                );
            }

            for _ in 0..repeats {
                let start = Instant::now();
                let (processed, failed) =
                    run_bundle_stream(&csv_path, &csv_column, limit, &descriptors, chunk_size);
                timings.push(start.elapsed().as_secs_f64());
                processed_rows = processed;
                parse_failures = failed;
            }
        }
    }

    let median_seconds = median(&mut timings);
    let ms_per_molecule = median_seconds * 1000.0 / processed_rows.max(1) as f64;
    let molecules_per_second = processed_rows as f64 / median_seconds;
    let peak_rss_bytes = peak_rss_bytes().unwrap_or(0);
    let peak_rss_mb = peak_rss_bytes as f64 / (1024.0 * 1024.0);

    println!(
        "tool\tdataset\tcsv_path\trow_limit\tprocessed_rows\tparse_failures\toutputs\tmedian_seconds\tms_per_molecule\tmolecules_per_second\tpeak_rss_bytes\tpeak_rss_mb"
    );
    println!(
        "ferrum\tCOCONUT\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.2}\t{}\t{:.2}",
        csv_path,
        processed_rows + parse_failures,
        processed_rows,
        parse_failures,
        outputs,
        median_seconds,
        ms_per_molecule,
        molecules_per_second,
        peak_rss_bytes,
        peak_rss_mb
    );
}
