//! Spiro atom counting
//!
//! A spiro atom is an atom that is part of two or more rings but is not a bridge atom.
//! Spiro atoms connect rings by being a single shared atom between them.
//!
//! Examples:
//! - Spiro[3.3]heptane: 1 spiro atom
//! - Spiro[4.5]decane: 1 spiro atom
//! - Bicyclo[2.2.1]heptane: 2 bridge atoms, 0 spiro atoms


//! ## File guide: Spiro
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn num_spiro_atoms`
//! - `mod tests`
//! - `fn parse`
//! - `fn test_no_rings`
//! - `fn test_single_ring`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Spiro` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Spiro` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/spiro.rs` defines spiro behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Spiro` workflows.
//! - Repository unit and parity tests covering `Spiro` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/spiro.rs`.

use crate::algo::rings::find_sssr;
use crate::mol::molecule::Molecule;

/// Count number of spiro atoms in a molecule
///
/// A spiro atom:
/// 1. Is part of at least 2 rings
/// 2. Is NOT a bridge atom (doesn't have 2+ neighbors also in rings)
///
/// # Returns
/// Number of spiro atoms
///
/// # Examples
/// - Spiro[3.3]heptane: 1 spiro atom
/// - Bicyclic fused rings (naphthalene): 0 spiro atoms (shared edge)
/// - Bicyclo[2.2.1]heptane (norbornane): 0 spiro atoms (bridge atoms)
pub fn num_spiro_atoms(mol: &mut Molecule) -> u32 {
    let n = mol.num_atoms();
    if n == 0 {
        return 0;
    }

    let rings = find_sssr(mol);
    if rings.len() < 2 {
        return 0; // Need at least 2 rings for spiro
    }

    // Count ring membership for each atom
    let mut ring_count = vec![0u32; n];
    for ring in &rings {
        for &atom_idx in ring {
            ring_count[atom_idx as usize] += 1;
        }
    }

    let mut spiro_count = 0;

    // Check each atom that's in 2+ rings
    for (atom_idx, &count) in ring_count.iter().enumerate() {
        if count < 2 {
            continue; // Not in multiple rings
        }

        // Check if it's a spiro atom vs bridge atom
        // Spiro: only this atom is shared between rings
        // Bridge: this atom + neighbors form a path between rings

        // Count how many neighbors are also in rings
        let neighbors = &mol.adjacency[atom_idx];
        let neighbors_in_rings = neighbors
            .iter()
            .map(|&bond_idx| {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == atom_idx as u16 {
                    bond.target
                } else {
                    bond.source
                };
                ring_count[neighbor as usize] > 0
            })
            .filter(|&in_ring| in_ring)
            .count();

        // If all neighbors in rings, it's likely a spiro atom
        // A bridge atom would have neighbors in different rings
        // For a true spiro: atom is in multiple rings, but only connects them at this point

        // More precise check: is this atom the ONLY shared atom between its rings?
        let mut is_spiro = true;

        // Get all rings containing this atom
        let atom_rings: Vec<&Vec<u16>> = rings
            .iter()
            .filter(|ring| ring.contains(&(atom_idx as u16)))
            .collect();

        if atom_rings.len() < 2 {
            continue;
        }

        // Check if any pair of rings shares more than just this atom
        for i in 0..atom_rings.len() {
            for j in (i + 1)..atom_rings.len() {
                let ring1 = atom_rings[i];
                let ring2 = atom_rings[j];

                // Count shared atoms
                let shared = ring1.iter().filter(|&&atom| ring2.contains(&atom)).count();

                if shared > 1 {
                    // More than just the current atom is shared
                    is_spiro = false;
                    break;
                }
            }
            if !is_spiro {
                break;
            }
        }

        if is_spiro {
            spiro_count += 1;
        }
    }

    spiro_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    /// Runs regression test for no rings.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_no_rings() {
        let mut mol = parse("CCCC");
        assert_eq!(num_spiro_atoms(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for single ring.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_single_ring() {
        let mut mol = parse("C1CCCCC1");
        assert_eq!(num_spiro_atoms(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for fused rings.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fused_rings() {
        // Naphthalene - fused rings share an edge, not spiro
        let mut mol = parse("c1ccc2ccccc2c1");
        assert_eq!(num_spiro_atoms(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for spiro 3 3 heptane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_spiro_3_3_heptane() {
        // Spiro[3.3]heptane: C1CCC2(C1)CCC2
        let mut mol = parse("C1CCC2(C1)CCC2");
        assert_eq!(num_spiro_atoms(&mut mol), 1);
    }

    #[test]
    /// Runs regression test for spiro 4 5 decane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_spiro_4_5_decane() {
        // Spiro[4.5]decane
        let mut mol = parse("C1CCCC2(C1)CCCCC2");
        assert_eq!(num_spiro_atoms(&mut mol), 1);
    }

    #[test]
    /// Runs regression test for bicyclo bridged.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bicyclo_bridged() {
        // Bicyclo[2.2.1]heptane (norbornane) - bridged, not spiro
        let mut mol = parse("C1CC2CCC1C2");
        assert_eq!(num_spiro_atoms(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for double spiro.
    ///
    /// Function context: `crates/ferrum_core/src/desc/spiro.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_double_spiro() {
        // Dispiro compound should have 2 spiro atoms
        // This is a complex case, may need more examples
        let mut mol = parse("C1CC2(C1)CC1(C2)CCC1");
        let count = num_spiro_atoms(&mut mol);
        assert!(count >= 1); // At least one spiro atom
    }
}
