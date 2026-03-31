//! Smarts Demo binary.
//!
//! This executable is a task-oriented demo or benchmark used to explore
//! FerrumChem behavior from the command line. It is intentionally small and
//! readable so learners can trace data flow from CSV/SMILES input to outputs.
//!
//! Learning focus:
//! - Environment-driven configuration patterns in Rust CLIs.
//! - How FerrumChem APIs are wired into practical workflows.


//! ## File guide: Smarts Demo
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
//! - `Smarts Demo` is a task-focused executable; CLI environment variables are part of the stable workflow contract.
//! - `Smarts Demo` should keep parse/descriptor/model behavior aligned with `ferrum_core` defaults to avoid demo drift.
//! - Benchmark/demo paths in `Smarts Demo` prefer deterministic setup to keep measurements comparable over time.
//! - Specific scope anchor: `crates/ferrum_cli/src/bin/smarts_demo.rs` defines smarts demo behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust CLI patterns (argument/env handling, structured errors, stdout diagnostics) reflected by `Smarts Demo`.
//! - FerrumChem benchmark/parity methodology documents referenced by `Smarts Demo` scripts.
//! - Terminal UX libraries (where applicable) used by `Smarts Demo` for interactive workflows.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_cli/src/bin/smarts_demo.rs`.

use ferrum_core::algo::aromaticity;
use ferrum_core::algo::iso::{count_matches, has_substruct_match};
use ferrum_core::io::smarts::parse_smarts;
use ferrum_core::io::smiles::parse_smiles;

/// Runs this executable entrypoint and orchestrates its top-level workflow.
///
/// Function context: `crates/ferrum_cli/src/bin/smarts_demo.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn main() {
    println!("=== SMARTS Pattern Matching Demo ===\n");

    // Test molecules
    let test_mols = vec![
        ("CCO", "Ethanol"),
        ("CC(C)O", "Isopropanol"),
        ("c1ccccc1", "Benzene"),
        ("c1ccccc1O", "Phenol"),
        ("C=O", "Formaldehyde"),
        ("CC(=O)C", "Acetone"),
        ("CC(=O)O", "Acetic acid"),
        ("CCN", "Ethylamine"),
    ];

    // Test patterns
    let patterns = vec![
        ("[C]", "Any carbon"),
        ("[O]", "Any oxygen"),
        ("CO", "Carbon-oxygen"),
        ("C=O", "Carbonyl"),
        ("[c]", "Aromatic carbon"),
        ("[OH]", "Hydroxyl (explicit H)"),
        ("c1ccccc1", "Benzene ring"),
    ];

    println!(
        "Parsing {} molecules and {} patterns...\n",
        test_mols.len(),
        patterns.len()
    );

    // Parse all molecules
    let mut molecules: Vec<_> = test_mols
        .iter()
        .filter_map(|(smiles, name)| {
            match parse_smiles(smiles) {
                Ok(mut mol) => {
                    // Perceive aromaticity for aromatic patterns
                    aromaticity::perceive_aromaticity(&mut mol);
                    Some((mol, *name))
                }
                Err(e) => {
                    eprintln!("Failed to parse {}: {}", smiles, e);
                    None
                }
            }
        })
        .collect();

    // Parse all patterns
    let query_patterns: Vec<_> = patterns
        .iter()
        .filter_map(|(smarts, desc)| match parse_smarts(smarts) {
            Ok(pattern) => Some((pattern, *desc)),
            Err(e) => {
                eprintln!("Failed to parse SMARTS {}: {}", smarts, e);
                None
            }
        })
        .collect();

    println!(
        "Successfully parsed {} molecules and {} patterns\n",
        molecules.len(),
        query_patterns.len()
    );

    // Create match matrix
    println!(
        "{:<20} {}",
        "Pattern",
        test_mols
            .iter()
            .map(|(_, name)| format!("{:>12}", name))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("{}", "-".repeat(120));

    for (pattern, desc) in &query_patterns {
        print!("{:<20}", desc);

        for (mol, _) in &molecules {
            let count = count_matches(mol, pattern);
            if count > 0 {
                print!(" {:>12}", count);
            } else {
                print!(" {:>12}", "-");
            }
        }
        println!();
    }

    println!("\n=== Detailed Analysis ===\n");

    // Show detailed matches for specific patterns
    let hydroxyl = parse_smarts("CO").unwrap();
    println!("Molecules containing C-O:");
    for (mol, name) in &molecules {
        if has_substruct_match(mol, &hydroxyl) {
            let count = count_matches(mol, &hydroxyl);
            println!("  {} - {} matches", name, count);
        }
    }

    println!("\nMolecules containing carbonyl (C=O):");
    let carbonyl = parse_smarts("C=O").unwrap();
    for (mol, name) in &molecules {
        if has_substruct_match(mol, &carbonyl) {
            println!("  {}", name);
        }
    }

    println!("\nAromatic molecules:");
    let aromatic_carbon = parse_smarts("[c]").unwrap();
    for (mol, name) in &molecules {
        if has_substruct_match(mol, &aromatic_carbon) {
            let count = count_matches(mol, &aromatic_carbon);
            println!("  {} - {} aromatic carbons", name, count);
        }
    }

    println!("\n=== Performance Test ===\n");

    // Simple performance benchmark
    let test_mol = parse_smiles("c1ccccc1").unwrap();
    let simple_pattern = parse_smarts("[C]").unwrap();

    let iterations = 100_000;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let _ = has_substruct_match(&test_mol, &simple_pattern);
    }

    let elapsed = start.elapsed();
    let matches_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!(
        "Simple pattern matching: {:.0} matches/sec",
        matches_per_sec
    );
    println!(
        "Time per match: {:.2} µs",
        elapsed.as_micros() as f64 / iterations as f64
    );

    println!("\nDemo complete!");
}
