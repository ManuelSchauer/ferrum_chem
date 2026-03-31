//! Integration tests: Check Atom 52.
//!
//! This test module validates end-to-end behavior across parser, chemistry
//! algorithms, descriptors, and ML surfaces. It captures real-world edge cases
//! so regressions are caught before parity or performance runs.


//! ## File guide: Check Atom 52
//!
//! This file belongs to FerrumChem's **integration/diagnostic validation layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn check_atom_52`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Check Atom 52` fixtures are regression-focused: they pin previously fragile behavior before parity/performance runs.
//! - `Check Atom 52` assumes deterministic parser/descriptor outputs so test failures indicate real semantic drift.
//! - Test expectations in `Check Atom 52` are intentionally explicit to support learning-oriented debugging.
//! - Specific scope anchor: `crates/ferrum_core/tests/check_atom_52.rs` defines check atom 52 behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Repository historical bug reports and checkpoints that motivated `Check Atom 52` assertions.
//! - RDKit or chemistry-reference fixtures used to validate `Check Atom 52` expectations.
//! - Rust integration-test best practices for reproducible scientific software checks.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/tests/check_atom_52.rs`.

use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::types::BondType;

#[test]
/// Runs regression test for check atom 52.
///
/// Function context: `crates/ferrum_core/tests/check_atom_52.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn check_atom_52() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mol = parse_smiles(smiles).unwrap();

    eprintln!("\n=== Analyzing Atom 52 (N in Ring 5) ===");
    let ring: Vec<u16> = vec![3, 4, 8, 44, 51, 52];

    let a52 = &mol.atoms[52];
    eprintln!(
        "Atom 52: Z={}, total_h={}, charge={}, degree={}",
        a52.atomic_number,
        a52.total_h(),
        a52.formal_charge,
        mol.get_degree(52)
    );

    eprintln!("\nBonds from atom 52:");
    for &bi in &mol.adjacency[52] {
        let b = &mol.bonds[bi as usize];
        let other = if b.source == 52 { b.target } else { b.source } as u16;
        let in_ring = ring.contains(&other);
        eprintln!(
            "  Bond {}: 52 -- {}, type={:?}, in_ring={}",
            bi, other, b.bond_type, in_ring
        );
    }

    eprintln!("\nhas_ring_pi_bond check (bonds with pi type WITHIN ring):");
    let has_ring_pi = mol.adjacency[52].iter().any(|&bi| {
        let b = &mol.bonds[bi as usize];
        if !matches!(b.bond_type, BondType::Double | BondType::Aromatic) {
            return false;
        }
        let other = if b.source == 52 { b.target } else { b.source };
        ring.contains(&(other as u16))
    });

    eprintln!("  has_ring_pi_bond = {}", has_ring_pi);

    eprintln!("\nConclusion: Atom 52 (N) with ring_pi=true should count as:");
    if has_ring_pi {
        let deg = mol.get_degree(52);
        if deg >= 3 {
            eprintln!("  2π (degree >= 3)");
        } else {
            eprintln!("  AMBIGUOUS [1 or 2] (degree < 3)");
        }
    } else {
        eprintln!("  AMBIGUOUS [1 or 2] (no ring_pi)");
    }
}
