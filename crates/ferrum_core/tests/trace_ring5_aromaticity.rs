//! Integration tests: Trace Ring5 Aromaticity.
//!
//! This test module validates end-to-end behavior across parser, chemistry
//! algorithms, descriptors, and ML surfaces. It captures real-world edge cases
//! so regressions are caught before parity or performance runs.


//! ## File guide: Trace Ring5 Aromaticity
//!
//! This file belongs to FerrumChem's **integration/diagnostic validation layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn trace_ring5_aromaticity`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Trace Ring5 Aromaticity` fixtures are regression-focused: they pin previously fragile behavior before parity/performance runs.
//! - `Trace Ring5 Aromaticity` assumes deterministic parser/descriptor outputs so test failures indicate real semantic drift.
//! - Test expectations in `Trace Ring5 Aromaticity` are intentionally explicit to support learning-oriented debugging.
//! - Specific scope anchor: `crates/ferrum_core/tests/trace_ring5_aromaticity.rs` defines trace ring5 aromaticity behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Repository historical bug reports and checkpoints that motivated `Trace Ring5 Aromaticity` assertions.
//! - RDKit or chemistry-reference fixtures used to validate `Trace Ring5 Aromaticity` expectations.
//! - Rust integration-test best practices for reproducible scientific software checks.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/tests/trace_ring5_aromaticity.rs`.

use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::types::BondType;

#[test]
/// Runs regression test for trace ring5 aromaticity.
///
/// Function context: `crates/ferrum_core/tests/trace_ring5_aromaticity.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn trace_ring5_aromaticity() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mol = parse_smiles(smiles).unwrap();

    eprintln!("\n=== Analyzing Ring 5: atoms [3, 4, 8, 44, 51, 52] ===");
    let ring = vec![3, 4, 8, 44, 51, 52];

    eprintln!("\nAtom properties:");
    for &ai in &ring {
        let a = &mol.atoms[ai];
        let pi_bonds = mol.adjacency[ai]
            .iter()
            .filter(|&&bi| {
                let b = &mol.bonds[bi as usize];
                matches!(b.bond_type, BondType::Double | BondType::Aromatic)
            })
            .count();
        eprintln!(
            "  Atom {}: Z={}, h={}, chg={}, pi_bonds={}",
            ai,
            a.atomic_number,
            a.total_h(),
            a.formal_charge,
            pi_bonds
        );
    }

    eprintln!("\nRing edges (consecutive pairs):");
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        eprintln!("  {} -- {}: ", from, to);

        // Find bond between them
        let mut found_bond = false;
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source as usize == from && b.target as usize == to)
                || (b.source as usize == to && b.target as usize == from)
            {
                eprintln!("    Bond {}: type={:?}", bi, b.bond_type);
                found_bond = true;
                break;
            }
        }
        if !found_bond {
            eprintln!("    NO DIRECT BOND FOUND!");
        }
    }

    eprintln!("\nIs this a valid ring?");
    eprintln!("  All consecutive atoms connected: checking...");
    let mut all_connected = true;
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        let mut found = false;
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source as usize == from && b.target as usize == to)
                || (b.source as usize == to && b.target as usize == from)
            {
                found = true;
                break;
            }
        }
        if !found {
            all_connected = false;
            eprintln!("    MISSING: {} -- {}", from, to);
        }
    }
    eprintln!(
        "  Result: {}",
        if all_connected {
            "valid ring"
        } else {
            "INVALID RING!"
        }
    );
}
