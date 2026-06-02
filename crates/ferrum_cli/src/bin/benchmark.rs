use clap::{Parser, Subcommand};
use ferrum_core::desc::batch::{BatchDescriptor, calculate_batch_flat_static};
use ferrum_core::desc::{Descriptor, build_ml_matrix_parallel};
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::ml::random_forest::RandomForestRegressor;
use ferrum_core::mol::molecule::Molecule;
use ndarray::{Array1, Array2};
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "ferrum_benchmark")]
#[command(about = "FerrumChem Performance Benchmark Suite", long_about = None)]
/// Implements Cli.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// System-layer note: component execution and pipeline orchestration.
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
/// Implements Commands.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// System-layer note: component execution and pipeline orchestration.
enum Commands {
    /// Scenario A (CPU-Heavy): 10,000 molecules + ML Descriptors
    ScenarioA {
        #[arg(short, long, default_value_t = 10000)]
        molecules: usize,
    },
    /// Scenario B (Memory-Heavy): 100,000 molecules + ECFP4 Fingerprints
    ScenarioB {
        #[arg(short, long, default_value_t = 100000)]
        molecules: usize,
    },
    /// Scenario C (ML-Heavy): Random Forest training
    ScenarioC {
        #[arg(short, long, default_value_t = 5000)]
        samples: usize,
        #[arg(short, long, default_value_t = 500)]
        trees: usize,
    },
    /// Scenario D (IO-Heavy): Parse large volume of SMILES
    ScenarioD {
        #[arg(short, long, default_value_t = 1000000)]
        count: usize,
    },
    /// Run all scenarios sequentially
    All,
}

/// Spawns a background thread that samples RSS memory usage (macOS / Linux via ps)
fn spawn_memory_monitor() -> (Arc<AtomicUsize>, std::sync::mpsc::Sender<()>) {
    let peak_rss = Arc::new(AtomicUsize::new(0));
    let peak_rss_clone = Arc::clone(&peak_rss);
    let (tx, rx) = std::sync::mpsc::channel();
    
    let pid = std::process::id();
    
    std::thread::spawn(move || {
        loop {
            if rx.try_recv().is_ok() {
                break;
            }
            
            // Call `ps -o rss= -p <pid>`
            if let Ok(output) = std::process::Command::new("ps")
                .args(&["-o", "rss=", "-p", &pid.to_string()])
                .output() 
            {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(kb) = mem_str.trim().parse::<usize>() {
                        let bytes = kb * 1024;
                        let mut current_max = peak_rss_clone.load(Ordering::Relaxed);
                        while bytes > current_max {
                            match peak_rss_clone.compare_exchange_weak(current_max, bytes, Ordering::Relaxed, Ordering::Relaxed) {
                                Ok(_) => break,
                                Err(x) => current_max = x,
                            }
                        }
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });
    
    (peak_rss, tx)
}

/// Prepares a repeated array of molecules for benchmarking
fn prep_molecules(count: usize) -> Vec<Molecule> {
    let seed_smiles = vec![
        "C", "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ];
    let seed_mols: Vec<Molecule> = seed_smiles.iter().map(|s| parse_smiles(s).unwrap()).collect();
    
    (0..count).into_par_iter().map(|i| {
        seed_mols[i % seed_mols.len()].clone()
    }).collect()
}

/// Implements run scenario a.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn run_scenario_a(molecules: usize) {
    println!("\n=== Scenario A (CPU-Heavy) ===");
    println!("Calculating Fast ML Descriptor preset for {} molecules...", molecules);
    
    let mut mols = prep_molecules(molecules);
    let descriptors = BatchDescriptor::fast_ml_preset();
    let num_features = descriptors.iter().map(|d| d.feature_len()).sum::<usize>();
    
    let (mem_tracker, stop_tx) = spawn_memory_monitor();
    let start = Instant::now();
    
    let matrix = calculate_batch_flat_static(&mut mols, descriptors);
    
    let duration = start.elapsed();
    let _ = stop_tx.send(());
    
    let peak_mb = mem_tracker.load(Ordering::Relaxed) as f64 / 1_048_576.0;
    
    assert_eq!(matrix.rows, molecules);
    assert_eq!(matrix.cols, num_features);
    
    println!("Result     : Matrix {} x {}", matrix.rows, matrix.cols);
    println!("Time       : {:.2?}", duration);
    println!("Throughput : {:.0} MPS", molecules as f64 / duration.as_secs_f64());
    println!("Peak RAM   : {:.2} MB", peak_mb);
}

/// Implements run scenario b.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn run_scenario_b(molecules: usize) {
    println!("\n=== Scenario B (Memory-Heavy) ===");
    println!("Calculating ECFP4 (2048-bit) Fingerprints for {} molecules...", molecules);
    
    let mut mols = prep_molecules(molecules);
    // ECFP4 is a single descriptor that outputs 2048 dimensions
    let descriptors = vec![BatchDescriptor::Ecfp4];
    
    let (mem_tracker, stop_tx) = spawn_memory_monitor();
    let start = Instant::now();
    
    let matrix = calculate_batch_flat_static(&mut mols, &descriptors);
    
    let duration = start.elapsed();
    let _ = stop_tx.send(());
    
    let peak_mb = mem_tracker.load(Ordering::Relaxed) as f64 / 1_048_576.0;
    
    assert_eq!(matrix.rows, molecules);
    assert_eq!(matrix.cols, 2048);
    
    println!("Result     : Matrix {} x {}", matrix.rows, matrix.cols);
    println!("Time       : {:.2?}", duration);
    println!("Throughput : {:.0} MPS", molecules as f64 / duration.as_secs_f64());
    println!("Peak RAM   : {:.2} MB", peak_mb);
}

/// Implements run scenario c.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn run_scenario_c(samples: usize, trees: usize) {
    println!("\n=== Scenario C (ML-Heavy) ===");
    println!("Training Random Forest ({} trees) on {} samples x 200 features...", trees, samples);
    
    let features = 200;
    let mut rng = rand::thread_rng();
    
    // Generate random mock dataset
    let mut x_data = Vec::with_capacity(samples * features);
    let mut y_data = Vec::with_capacity(samples);
    for _ in 0..samples {
        y_data.push(rng.gen_range(0.0..1.0));
        for _ in 0..features {
            x_data.push(rng.gen_range(0.0..1.0));
        }
    }
    
    let x_arr = Array2::from_shape_vec((samples, features), x_data).unwrap();
    let y_arr = Array1::from_shape_vec((samples,), y_data).unwrap();
    
    let (mem_tracker, stop_tx) = spawn_memory_monitor();
    let start = Instant::now();
    
    let rf = RandomForestRegressor::fit(
        x_arr.view(),
        y_arr.view(),
        trees,
        15, // max_depth
        2,  // min_samples_split
        0.3, // max_features_fraction
        42, // seed
    );
    
    let duration = start.elapsed();
    let _ = stop_tx.send(());
    
    let peak_mb = mem_tracker.load(Ordering::Relaxed) as f64 / 1_048_576.0;
    
    println!("Result     : RF Regressor with {} trained trees", trees);
    println!("Time       : {:.2?}", duration);
    println!("Peak RAM   : {:.2} MB", peak_mb);
}

/// Implements run scenario d.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn run_scenario_d(count: usize) {
    println!("\n=== Scenario D (IO-Heavy) ===");
    println!("Parsing {} SMILES strings...", count);
    
    let seed_smiles = vec![
        "C", "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ];
    
    // Create an in-memory vector of strings to simulate file IO reading overhead without disk IO bottlenecks
    let mut lines = Vec::with_capacity(count);
    for i in 0..count {
        lines.push(seed_smiles[i % seed_smiles.len()]);
    }
    
    let (mem_tracker, stop_tx) = spawn_memory_monitor();
    let start = Instant::now();
    
    // We use sequential parsing to simulate realistic reading since typical CSV reading is sequential
    let mut success = 0;
    for s in lines {
        if parse_smiles(s).is_ok() {
            success += 1;
        }
    }
    
    let duration = start.elapsed();
    let _ = stop_tx.send(());
    
    let peak_mb = mem_tracker.load(Ordering::Relaxed) as f64 / 1_048_576.0;
    
    println!("Result     : {} successfully parsed", success);
    println!("Time       : {:.2?}", duration);
    println!("Throughput : {:.0} Parses/Sec", count as f64 / duration.as_secs_f64());
    println!("Peak RAM   : {:.2} MB", peak_mb);
}

/// Implements main.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// System-layer note: component execution and pipeline orchestration.
fn main() {
    let cli = Cli::parse();
    
    // Ensure Rayon threadpool is initialized
    let threads = rayon::current_num_threads();
    println!("FerrumChem Benchmark Suite (Using {} threads)\n", threads);

    match cli.command {
        Commands::ScenarioA { molecules } => run_scenario_a(molecules),
        Commands::ScenarioB { molecules } => run_scenario_b(molecules),
        Commands::ScenarioC { samples, trees } => run_scenario_c(samples, trees),
        Commands::ScenarioD { count } => run_scenario_d(count),
        Commands::All => {
            run_scenario_a(10_000);
            run_scenario_b(100_000);
            run_scenario_c(5_000, 500);
            run_scenario_d(1_000_000);
        }
    }
}
