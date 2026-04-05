//! Example binary.
//!
//! This executable is a task-oriented demo or benchmark used to explore
//! FerrumChem behavior from the command line. It is intentionally small and
//! readable so learners can trace data flow from CSV/SMILES input to outputs.
//!
//! Learning focus:
//! - Environment-driven configuration patterns in Rust CLIs.
//! - How FerrumChem APIs are wired into practical workflows.

//! ## File guide: Example
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
//! - `Example` is a task-focused executable; CLI environment variables are part of the stable workflow contract.
//! - `Example` should keep parse/descriptor/model behavior aligned with `ferrum_core` defaults to avoid demo drift.
//! - Benchmark/demo paths in `Example` prefer deterministic setup to keep measurements comparable over time.
//! - Specific scope anchor: `crates/ferrum_cli/src/bin/example.rs` defines example behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust CLI patterns (argument/env handling, structured errors, stdout diagnostics) reflected by `Example`.
//! - FerrumChem benchmark/parity methodology documents referenced by `Example` scripts.
//! - Terminal UX libraries (where applicable) used by `Example` for interactive workflows.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_cli/src/bin/example.rs`.

use ferrum_core::algo::aromaticity::perceive_aromaticity;
use ferrum_core::algo::rings::find_sssr;
use ferrum_core::desc::Descriptor;
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::io::smiles::parse_smiles;

/// Runs this executable entrypoint and orchestrates its top-level workflow.
///
/// Function context: `crates/ferrum_cli/src/bin/example.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn main() {
    println!("=== FerrumChem Example Usage ===\n");

    // Example 1: Parse simple molecules
    println!("Example 1: Parsing SMILES");
    println!("{}", "-".repeat(40));

    let molecules = vec![
        ("Water", "O"),
        ("Methane", "C"),
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ];

    for (name, smiles) in &molecules {
        match parse_smiles(smiles) {
            Ok(mol) => {
                println!(
                    "{:12} '{}' → {} atoms, {} bonds",
                    name,
                    smiles,
                    mol.num_atoms(),
                    mol.num_bonds()
                );
            }
            Err(e) => println!("{:12} ERROR: {}", name, e),
        }
    }

    println!("\n");

    // Example 2: Calculate molecular weights
    println!("Example 2: Molecular Weight Calculation");
    println!("{}", "-".repeat(40));

    let mw_descriptor = MolecularWeight;

    for (name, smiles) in &molecules {
        if let Ok(mut mol) = parse_smiles(smiles) {
            let value = mw_descriptor.calculate(
                &mut mol,
                &mut ferrum_core::desc::context::CalculationContext::new(),
            );
            println!("{:12} MW = {:7.2} Da", name, value.as_float());
        }
    }

    println!("\n");

    // Example 3: Advanced SMILES features
    println!("Example 3: Advanced SMILES Features");
    println!("{}", "-".repeat(40));

    // Charged molecules
    let sodium_ion = parse_smiles("[Na+]").unwrap();
    println!("Sodium ion [Na+]:");
    println!(
        "  Atoms: {}, Charge: {}",
        sodium_ion.num_atoms(),
        sodium_ion.atoms[0].formal_charge
    );

    // Isotopes
    let mut heavy_water = parse_smiles("[2H]O[2H]").unwrap();
    println!("\nHeavy water [2H]O[2H]:");
    println!("  Atoms: {}", heavy_water.num_atoms());
    println!(
        "  Isotopes: {} and {}",
        heavy_water.atoms[0].isotope, heavy_water.atoms[2].isotope
    );

    let mw_val = mw_descriptor.calculate(
        &mut heavy_water,
        &mut ferrum_core::desc::context::CalculationContext::new(),
    );
    println!(
        "  MW: {:.2} Da (vs 18.01 for normal water)",
        mw_val.as_float()
    );

    // Explicit hydrogens
    let mut methane = parse_smiles("[CH4]").unwrap();
    println!("\nMethane with explicit H [CH4]:");
    println!(
        "  Atoms: {}, Explicit H: {}",
        methane.num_atoms(),
        methane.atoms[0].num_explicit_h
    );

    let mw_val = mw_descriptor.calculate(
        &mut methane,
        &mut ferrum_core::desc::context::CalculationContext::new(),
    );
    println!("  MW: {:.2} Da", mw_val.as_float());

    // Aromatic systems
    let mut benzene = parse_smiles("c1ccccc1").unwrap();
    println!("\nBenzene c1ccccc1:");
    println!(
        "  Atoms: {}, Bonds: {}",
        benzene.num_atoms(),
        benzene.num_bonds()
    );
    let aromatic_count = benzene.atoms.iter().filter(|a| a.is_aromatic()).count();
    println!("  Aromatic atoms: {}", aromatic_count);
    let implicit_h: u8 = benzene.atoms.iter().map(|a| a.num_implicit_h).sum();
    println!("  Implicit H: {} (automatically calculated)", implicit_h);
    let mw_val = mw_descriptor.calculate(
        &mut benzene,
        &mut ferrum_core::desc::context::CalculationContext::new(),
    );
    println!("  MW: {:.2} Da", mw_val.as_float());

    println!("\n");

    // Example 4: Branches and complex structures
    println!("Example 4: Complex Structures");
    println!("{}", "-".repeat(40));

    let complex_molecules = vec![
        ("Isobutane", "CC(C)C"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Penicillin G", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
    ];

    for (name, smiles) in &complex_molecules {
        if let Ok(mut mol) = parse_smiles(smiles) {
            let mw_val = mw_descriptor.calculate(
                &mut mol,
                &mut ferrum_core::desc::context::CalculationContext::new(),
            );
            println!(
                "{:15} → {} atoms, MW = {:.2} Da",
                name,
                mol.num_atoms(),
                mw_val.as_float()
            );
        }
    }

    println!("\n");

    // Example 5: Ring Detection
    println!("Example 5: Ring Detection");
    println!("{}", "-".repeat(40));

    let ring_molecules = vec![
        ("Ethane (no ring)", "CC"),
        ("Benzene (1 ring)", "c1ccccc1"),
        ("Cyclohexane (1 ring)", "C1CCCCC1"),
        ("Toluene (1 ring)", "Cc1ccccc1"),
        ("Naphthalene (2 rings)", "c1ccc2ccccc2c1"),
    ];

    for (name, smiles) in &ring_molecules {
        if let Ok(mut mol) = parse_smiles(smiles) {
            find_sssr(&mut mol);

            let ring_atoms = mol.atoms.iter().filter(|a| a.is_in_ring()).count();
            let ring_bonds = mol.bonds.iter().filter(|b| b.is_in_ring()).count();

            println!(
                "{:20} → {} atoms in rings, {} bonds in rings",
                name, ring_atoms, ring_bonds
            );
        }
    }

    println!("\n");

    // Example 6: Aromaticity Perception
    println!("Example 6: Aromaticity Perception");
    println!("{}", "-".repeat(40));

    let aromatic_molecules = vec![
        ("Benzene", "c1ccccc1"),
        ("Pyridine", "c1ccncc1"),
        ("Toluene", "Cc1ccccc1"),
        ("Cyclohexane (not aromatic)", "C1CCCCC1"),
        ("Naphthalene", "c1ccc2ccccc2c1"),
    ];

    for (name, smiles) in &aromatic_molecules {
        if let Ok(mut mol) = parse_smiles(smiles) {
            perceive_aromaticity(&mut mol);

            let aromatic_atoms = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
            let total_atoms = mol.num_atoms();

            println!(
                "{:30} → {}/{} atoms aromatic",
                name, aromatic_atoms, total_atoms
            );
        }
    }

    println!("\n");

    // Example 7: Error handling
    println!("Example 7: Error Handling");
    println!("{}", "-".repeat(40));

    let invalid_smiles = vec![
        "C(((C", // Unmatched parenthesis
        "C1CC",  // Unclosed ring
        "C-",    // Dangling bond
    ];

    for smiles in &invalid_smiles {
        match parse_smiles(smiles) {
            Ok(_) => println!("'{}' → Unexpectedly parsed OK", smiles),
            Err(e) => println!("'{}' → Error: {}", smiles, e),
        }
    }

    println!("\n=== Example Complete ===");
}
