//! Descriptor Demo binary.
//!
//! This executable is a task-oriented demo or benchmark used to explore
//! FerrumChem behavior from the command line. It is intentionally small and
//! readable so learners can trace data flow from CSV/SMILES input to outputs.
//!
//! Learning focus:
//! - Environment-driven configuration patterns in Rust CLIs.
//! - How FerrumChem APIs are wired into practical workflows.

//! ## File guide: Descriptor Demo
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
//! - `Descriptor Demo` is a task-focused executable; CLI environment variables are part of the stable workflow contract.
//! - `Descriptor Demo` should keep parse/descriptor/model behavior aligned with `ferrum_core` defaults to avoid demo drift.
//! - Benchmark/demo paths in `Descriptor Demo` prefer deterministic setup to keep measurements comparable over time.
//! - Specific scope anchor: `crates/ferrum_cli/src/bin/descriptor_demo.rs` defines descriptor demo behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust CLI patterns (argument/env handling, structured errors, stdout diagnostics) reflected by `Descriptor Demo`.
//! - FerrumChem benchmark/parity methodology documents referenced by `Descriptor Demo` scripts.
//! - Terminal UX libraries (where applicable) used by `Descriptor Demo` for interactive workflows.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_cli/src/bin/descriptor_demo.rs`.

use ferrum_core::algo::aromaticity::perceive_aromaticity;
use ferrum_core::algo::valency::assign_implicit_hydrogens;
use ferrum_core::desc::Descriptor;
use ferrum_core::desc::DescriptorValue;
use ferrum_core::desc::aromatic_ring_count::AromaticRingCount;
use ferrum_core::desc::atom_count::{
    AromaticAtomCount, AtomCount, HeavyAtomCount, SingleAtomCount,
};
use ferrum_core::desc::bond_count::{BondCount, BondCountDetailed, RotatableBondCount};
use ferrum_core::desc::formal_charge::FormalCharge;
use ferrum_core::desc::hbond::{HBondAcceptor, HBondDonor};
use ferrum_core::desc::lipinski::LipinskiViolations;
use ferrum_core::desc::logp::LogP;
use ferrum_core::desc::maccs::MaccsKeys;
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::io::smiles::parse_smiles;

/// Runs this executable entrypoint and orchestrates its top-level workflow.
///
/// Function context: `crates/ferrum_cli/src/bin/descriptor_demo.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
fn main() {
    println!("🧪 FerrumChem - Basic Descriptors Demo\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Test molecules
    let test_cases = vec![
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("c1ccncc1", "Pyridine"),
        ("c1ccc2ccccc2c1", "Naphthalene"),
        ("CCCCCC", "Hexane"),
        ("C=CC=C", "1,3-Butadiene"),
        ("C#N", "Hydrogen Cyanide"),
        ("[NH4+]", "Ammonium Ion"),
        ("[OH-]", "Hydroxide Ion"),
        ("Cc1ccccc1", "Toluene"),
    ];

    for (smiles, name) in test_cases {
        println!("📊 {} ({})", name, smiles);
        println!("───────────────────────────────────────────────────────────────");

        match parse_smiles(smiles) {
            Ok(mut mol) => {
                // Process molecule
                assign_implicit_hydrogens(&mut mol).unwrap();
                perceive_aromaticity(&mut mol);

                // Calculate and display individual descriptors
                let descriptors: Vec<Box<dyn Descriptor>> = vec![
                    Box::new(MolecularWeight),
                    Box::new(AtomCount),
                    Box::new(HeavyAtomCount),
                    Box::new(SingleAtomCount::new("C")),
                    Box::new(SingleAtomCount::new("N")),
                    Box::new(SingleAtomCount::new("O")),
                    Box::new(AromaticAtomCount),
                    Box::new(HBondDonor),
                    Box::new(HBondAcceptor),
                    Box::new(LogP),
                    Box::new(LipinskiViolations),
                    Box::new(MaccsKeys),
                    Box::new(BondCount),
                    Box::new(BondCountDetailed),
                    Box::new(RotatableBondCount),
                    Box::new(AromaticRingCount),
                    Box::new(FormalCharge),
                ];

                for desc in &descriptors {
                    let value = desc.calculate(
                        &mut mol,
                        &mut ferrum_core::desc::context::CalculationContext::new(),
                    );

                    print!("  {:20} ", desc.name());
                    match &value {
                        DescriptorValue::Float(v) => println!("{:.2}", v),
                        DescriptorValue::Float64(v) => println!("{:.2}", v),
                        DescriptorValue::Int(v) => println!("{}", v),
                        DescriptorValue::IntVec(v) => {
                            if v.len() == 4 {
                                println!(
                                    "[Total: {}, Single: {}, Double: {}, Triple: {}]",
                                    v[0], v[1], v[2], v[3]
                                );
                            } else {
                                println!("{:?}", v);
                            }
                        }
                        DescriptorValue::Binary(_bits) => {
                            let mut floats = Vec::new();
                            value.to_floats(&mut floats);
                            let set_count = floats.iter().filter(|&&x| x == 1.0).count();
                            println!("{} / {} keys set", set_count, floats.len());
                        }
                        _ => {
                            let mut floats = Vec::new();
                            value.to_floats(&mut floats);
                            println!("{:?}", floats);
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Error parsing SMILES: {}", e);
            }
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("\n✅ Demo complete! All {} descriptors tested.", 15);
    println!("\nDescriptor Summary:");
    println!("  • Molecular Weight - Accurate mass calculation");
    println!("  • Atom Counts - Total, heavy, specific elements");
    println!("  • Aromatic Atoms - Counts atoms in aromatic systems");
    println!("  • H-Bond Donors - N-H and O-H groups (Lipinski)");
    println!("  • H-Bond Acceptors - N and O atoms (Lipinski)");
    println!("  • LogP - Lipophilicity estimation (RDKit patterns)");
    println!("  • Lipinski Rule of Five - Drug-likeness assessment");
    println!("  • MACCS Keys - 166 structural fingerprint keys");
    println!("  • Bond Counts - Total and detailed (single/double/triple)");
    println!("  • Rotatable Bonds - Conformational flexibility metric");
    println!("  • Aromatic Rings - Aromatic ring detection");
    println!("  • Formal Charge - Net molecular charge");
    println!("\n🚀 Phase 8 Complete: MACCS Keys implementation (all 166 keys)!");
}
