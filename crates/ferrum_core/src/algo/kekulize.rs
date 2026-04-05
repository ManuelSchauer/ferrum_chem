//! Kekulization - assign explicit single/double bonds to aromatic rings.
//!
//! Aromatic rings are typically represented with delocalized electrons,
//! but many algorithms require explicit bond orders. Kekulization finds
//! a valid alternating single/double bond pattern for aromatic systems.
//!
//! ## Algorithm
//! 1. Find all aromatic rings
//! 2. For each ring, assign alternating Single/Double bonds
//! 3. Keep the aromatic flag set on bonds (RDKit style)
//!
//! ## Thread Safety
//! This function modifies the molecule in-place and is not thread-safe.
//! Call it once per molecule before parallel descriptor calculations.

//! ## File guide: Kekulize
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn kekulize`
//! - `fn is_aromatic_ring`
//! - `fn kekulize_ring`
//! - `fn find_bond_between`
//! - `mod tests`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Kekulize` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Kekulize` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Kekulize` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/kekulize.rs` defines kekulize behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Kekulize` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Kekulize` decisions.
//! - FerrumChem regression tests that pin `Kekulize` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/kekulize.rs`.

use crate::algo::rings::find_sssr;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};

/// Kekulize aromatic rings by assigning explicit single/double bond orders.
///
/// After kekulization:
/// - Aromatic bonds will have bond_type Single or Double
/// - The aromatic FLAG remains set (for TPSA and other descriptors)
/// - The molecule representation is canonical and consistent
///
/// # Example
/// ```ignore
/// let mut mol = parse_smiles("c1ccccc1").unwrap(); // benzene
/// perceive_aromaticity(&mut mol);
/// kekulize(&mut mol);
/// // Now bonds have explicit Single/Double types, but aromatic flags are set
/// ```
pub fn kekulize(mol: &mut Molecule) {
    mol.invalidate_cached_properties();
    // Find all aromatic rings
    let rings = find_sssr(mol);

    for ring in rings {
        // Check if this ring is aromatic
        if !is_aromatic_ring(mol, &ring) {
            continue;
        }

        // Assign alternating single/double bonds
        kekulize_ring(mol, &ring);
    }
}

/// Check if all atoms in a ring are aromatic
fn is_aromatic_ring(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    ring.iter()
        .all(|&idx| mol.atoms[idx as usize].is_aromatic())
}

/// Assign alternating single/double bonds to an aromatic ring.
/// Keeps the aromatic flag set on bonds.
fn kekulize_ring(mol: &mut Molecule, ring: &[AtomIdx]) {
    let n = ring.len();

    // For standard aromatic rings (5, 6, 7 members), use simple alternating pattern
    // Start with double bond on first edge
    for i in 0..n {
        let from = ring[i];
        let to = ring[(i + 1) % n];

        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            let bond = &mut mol.bonds[bond_idx];

            // Alternate: double on even indices, single on odd
            bond.bond_type = if i % 2 == 0 {
                BondType::Double
            } else {
                BondType::Single
            };

            // IMPORTANT: Keep aromatic flag set (RDKit style)
            // This allows TPSA and other algorithms to detect aromatic bonds
            // even after kekulization
        }
    }
}

/// Find bond index between two atoms.
fn find_bond_between(mol: &Molecule, from: AtomIdx, to: AtomIdx) -> Option<usize> {
    for (idx, bond) in mol.bonds.iter().enumerate() {
        if (bond.source == from && bond.target == to) || (bond.source == to && bond.target == from)
        {
            return Some(idx);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for kekulize benzene.
    ///
    /// Function context: `crates/ferrum_core/src/algo/kekulize.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_kekulize_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);

        // Before kekulization: all bonds are Aromatic type
        for bond in &mol.bonds {
            assert_eq!(bond.bond_type, BondType::Aromatic);
        }

        kekulize(&mut mol);

        // After kekulization: bonds are Single/Double but aromatic flag is set
        let mut single_count = 0;
        let mut double_count = 0;

        for bond in &mol.bonds {
            assert!(bond.is_aromatic(), "Aromatic flag should remain set");
            match bond.bond_type {
                BondType::Single => single_count += 1,
                BondType::Double => double_count += 1,
                _ => panic!("Unexpected bond type after kekulization"),
            }
        }

        assert_eq!(single_count, 3, "Benzene should have 3 single bonds");
        assert_eq!(double_count, 3, "Benzene should have 3 double bonds");
    }
}
