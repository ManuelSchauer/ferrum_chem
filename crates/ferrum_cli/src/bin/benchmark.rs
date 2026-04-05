//! Benchmark binary.
//!
//! This executable is a task-oriented demo or benchmark used to explore
//! FerrumChem behavior from the command line. It is intentionally small and
//! readable so learners can trace data flow from CSV/SMILES input to outputs.
//!
//! Learning focus:
//! - Environment-driven configuration patterns in Rust CLIs.
//! - How FerrumChem APIs are wired into practical workflows.

//! ## File guide: Benchmark
//!
//! This file belongs to FerrumChem's **CLI/demo/benchmark executable layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn main`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Benchmark` is a task-focused executable; CLI environment variables are part of the stable workflow contract.
//! - `Benchmark` should keep parse/descriptor/model behavior aligned with `ferrum_core` defaults to avoid demo drift.
//! - Benchmark/demo paths in `Benchmark` prefer deterministic setup to keep measurements comparable over time.
//! - Specific scope anchor: `crates/ferrum_cli/src/bin/benchmark.rs` defines benchmark behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust CLI patterns (argument/env handling, structured errors, stdout diagnostics) reflected by `Benchmark`.
//! - FerrumChem benchmark/parity methodology documents referenced by `Benchmark` scripts.
//! - Terminal UX libraries (where applicable) used by `Benchmark` for interactive workflows.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_cli/src/bin/benchmark.rs`.

use ferrum_core::desc::{Descriptor, molecular_weight};
use ferrum_core::io::smiles::parse_smiles;
use std::time::Instant;

/// Runs this executable entrypoint and orchestrates its top-level workflow.
///
/// Function context: `crates/ferrum_cli/src/bin/benchmark.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn main() {
    println!("=== FerrumChem Performance Benchmarks ===\n");

    // Test molecules of varying complexity
    let test_smiles = vec![
        ("Methane", "C"),
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        (
            "Complex",
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O.CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        ),
    ];

    println!("--- SMILES Parsing Performance (with automatic implicit H) ---");
    for (name, smiles) in &test_smiles {
        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = parse_smiles(smiles).unwrap();
        }

        let duration = start.elapsed();
        let per_parse = duration.as_nanos() / iterations;
        let per_second = 1_000_000_000.0 / per_parse as f64;

        println!(
            "{:12} ({:2} chars): {:6} ns/parse  ({:8.0} molecules/sec)",
            name,
            smiles.len(),
            per_parse,
            per_second
        );
    }

    println!("\n--- Molecular Weight Calculation ---");
    let descriptor = molecular_weight::MolecularWeight;

    for (name, smiles) in &test_smiles {
        let mut mol = parse_smiles(smiles).unwrap();

        let iterations = 100000;

        let start = Instant::now();
        let mut mw_val = 0.0f32;
        for _ in 0..iterations {
            mw_val = descriptor
                .calculate(
                    &mut mol,
                    &mut ferrum_core::desc::context::CalculationContext::new(),
                )
                .as_float();
        }

        let duration = start.elapsed();
        let per_calc = duration.as_nanos() / iterations;
        let per_second = 1_000_000_000.0 / per_calc as f64;

        println!(
            "{:12}: MW = {:7.2}, {:5} ns/calc  ({:9.0} calcs/sec)",
            name, mw_val, per_calc, per_second
        );
    }

    println!("\n--- Combined Pipeline (Parse + Calculate MW) ---");
    for (name, smiles) in &test_smiles {
        let iterations = 10000;

        let start = Instant::now();
        let mut last_mw = 0.0f32;
        for _ in 0..iterations {
            let mut mol = parse_smiles(smiles).unwrap();
            last_mw = descriptor
                .calculate(
                    &mut mol,
                    &mut ferrum_core::desc::context::CalculationContext::new(),
                )
                .as_float();
        }
        let _ = last_mw;

        let duration = start.elapsed();
        let per_molecule = duration.as_nanos() / iterations;
        let per_second = 1_000_000_000.0 / per_molecule as f64;

        println!(
            "{:12}: {:6} ns/molecule  ({:8.0} molecules/sec)",
            name, per_molecule, per_second
        );
    }

    println!("\n--- Batch Processing Simulation ---");
    // Simulate processing a large dataset
    let dataset_sizes = vec![1000, 10000, 100000];
    let sample_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"; // Ibuprofen

    for size in dataset_sizes {
        let start = Instant::now();

        for _ in 0..size {
            let mut mol = parse_smiles(sample_smiles).unwrap();
            descriptor.calculate(
                &mut mol,
                &mut ferrum_core::desc::context::CalculationContext::new(),
            );
        }

        let duration = start.elapsed();
        let per_molecule = duration.as_micros() as f64 / size as f64;
        let throughput = size as f64 / duration.as_secs_f64();

        println!(
            "{:6} molecules: {:7.2} ms total, {:6.2} μs/mol, {:9.0} mol/sec",
            size,
            duration.as_secs_f64() * 1000.0,
            per_molecule,
            throughput
        );
    }

    println!("\n--- Memory Usage Estimate ---");
    for (name, smiles) in &test_smiles {
        let mol = parse_smiles(smiles).unwrap();
        let atoms = mol.num_atoms();
        let bonds = mol.num_bonds();

        // Rough estimate:
        // - Atom struct: 8 bytes
        // - Bond struct: 6 bytes
        // - Adjacency list: ~8 bytes/atom (SmallVec overhead)
        let estimated_bytes = atoms * 16 + bonds * 6;

        println!(
            "{:12}: {} atoms, {} bonds → ~{} bytes",
            name, atoms, bonds, estimated_bytes
        );
    }
}
