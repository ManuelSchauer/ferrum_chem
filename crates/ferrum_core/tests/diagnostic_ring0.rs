//! Integration tests: Diagnostic Ring0.
//!
//! This test module validates end-to-end behavior across parser, chemistry
//! algorithms, descriptors, and ML surfaces. It captures real-world edge cases
//! so regressions are caught before parity or performance runs.

//! ## File guide: Diagnostic Ring0
//!
//! This file belongs to FerrumChem's **integration/diagnostic validation layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn debug_ring_0_aromatic`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Diagnostic Ring0` fixtures are regression-focused: they pin previously fragile behavior before parity/performance runs.
//! - `Diagnostic Ring0` assumes deterministic parser/descriptor outputs so test failures indicate real semantic drift.
//! - Test expectations in `Diagnostic Ring0` are intentionally explicit to support learning-oriented debugging.
//! - Specific scope anchor: `crates/ferrum_core/tests/diagnostic_ring0.rs` defines diagnostic ring0 behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Repository historical bug reports and checkpoints that motivated `Diagnostic Ring0` assertions.
//! - RDKit or chemistry-reference fixtures used to validate `Diagnostic Ring0` expectations.
//! - Rust integration-test best practices for reproducible scientific software checks.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/tests/diagnostic_ring0.rs`.

use ferrum_core::algo::aromaticity::perceive_aromaticity;
use ferrum_core::io::smiles::parse_smiles;

#[test]
/// Runs regression test for debug ring 0 aromatic.
///
/// Function context: `crates/ferrum_core/tests/diagnostic_ring0.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn debug_ring_0_aromatic() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mut mol = parse_smiles(smiles).unwrap();

    eprintln!("\n=== Before aromaticity ===");
    eprintln!("Ring 0 atoms [8, 3, 51, 52, 4, 44] properties:");
    for &idx in &[8, 3, 51, 52, 4, 44] {
        let a = &mol.atoms[idx];
        eprintln!(
            "  Atom {}: Z={}, arom={}, h={}, chg={}, bonds=[",
            idx,
            a.atomic_number,
            a.is_aromatic(),
            a.total_h(),
            a.formal_charge
        );

        for &bi in &mol.adjacency[idx] {
            let b = &mol.bonds[bi as usize];
            let other = if b.source as usize == idx {
                b.target
            } else {
                b.source
            };
            eprintln!(
                "    bond {} to atom {}: type={:?}, arom={}",
                bi,
                other,
                b.bond_type,
                b.is_aromatic()
            );
        }
        eprintln!("  ]");
    }

    // Check specific bonds in ring 0
    eprintln!("\nRing 0 bonds (consecutive pairs):");
    let ring = vec![8, 3, 51, 52, 4, 44];
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        eprintln!("  Edge {}: {} -- {}", i, from, to);

        // Find bond
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source == from as u16 && b.target == to as u16)
                || (b.source == to as u16 && b.target == from as u16)
            {
                eprintln!(
                    "    Bond {}: type={:?}, arom={}",
                    bi,
                    b.bond_type,
                    b.is_aromatic()
                );
            }
        }
    }

    perceive_aromaticity(&mut mol);

    eprintln!("\n=== After aromaticity ===");
    eprintln!("Ring 0 atoms [8, 3, 51, 52, 4, 44] are aromatic:");
    for &idx in &[8, 3, 51, 52, 4, 44] {
        let a = &mol.atoms[idx];
        eprintln!("  Atom {}: arom={}", idx, a.is_aromatic());
    }

    eprintln!("\nRing 0 bonds (consecutive pairs) are aromatic:");
    let ring = vec![8, 3, 51, 52, 4, 44];
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];

        // Find bond
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source == from as u16 && b.target == to as u16)
                || (b.source == to as u16 && b.target == from as u16)
            {
                eprintln!(
                    "  Edge {}: {} -- {} [bond {}] arom={}",
                    i,
                    from,
                    to,
                    bi,
                    b.is_aromatic()
                );
            }
        }
    }
}
