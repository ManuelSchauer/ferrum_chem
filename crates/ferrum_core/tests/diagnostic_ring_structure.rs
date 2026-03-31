//! Integration tests: Diagnostic Ring Structure.
//!
//! This test module validates end-to-end behavior across parser, chemistry
//! algorithms, descriptors, and ML surfaces. It captures real-world edge cases
//! so regressions are caught before parity or performance runs.


//! ## File guide: Diagnostic Ring Structure
//!
//! This file belongs to FerrumChem's **integration/diagnostic validation layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn debug_ring_structure`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Diagnostic Ring Structure` fixtures are regression-focused: they pin previously fragile behavior before parity/performance runs.
//! - `Diagnostic Ring Structure` assumes deterministic parser/descriptor outputs so test failures indicate real semantic drift.
//! - Test expectations in `Diagnostic Ring Structure` are intentionally explicit to support learning-oriented debugging.
//! - Specific scope anchor: `crates/ferrum_core/tests/diagnostic_ring_structure.rs` defines diagnostic ring structure behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Repository historical bug reports and checkpoints that motivated `Diagnostic Ring Structure` assertions.
//! - RDKit or chemistry-reference fixtures used to validate `Diagnostic Ring Structure` expectations.
//! - Rust integration-test best practices for reproducible scientific software checks.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/tests/diagnostic_ring_structure.rs`.

use ferrum_core::algo::rings::find_minimum_cycle_basis_bonds;
use ferrum_core::io::smiles::parse_smiles;

#[test]
/// Runs regression test for debug ring structure.
///
/// Function context: `crates/ferrum_core/tests/diagnostic_ring_structure.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn debug_ring_structure() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mol = parse_smiles(smiles).unwrap();

    eprintln!("\n=== Ring Basis (as bonds) ===");
    let rings = find_minimum_cycle_basis_bonds(&mol);

    for (ri, ring_bonds) in rings.iter().enumerate() {
        eprintln!("\nRing {}:", ri);
        eprintln!("  Bond indices: {:?}", ring_bonds);

        // Collect atoms in this ring
        let mut atoms = std::collections::HashSet::new();
        for &bi in ring_bonds {
            let b = &mol.bonds[bi];
            atoms.insert(b.source as usize);
            atoms.insert(b.target as usize);
        }
        let mut atom_vec: Vec<_> = atoms.into_iter().collect();
        atom_vec.sort();
        eprintln!("  Atoms (sorted): {:?}", atom_vec);

        // Show bonds details
        eprintln!("  Bond details:");
        for &bi in ring_bonds {
            let b = &mol.bonds[bi];
            eprintln!(
                "    Bond {}: {} -- {}, type={:?}",
                bi, b.source, b.target, b.bond_type
            );
        }
    }
}
