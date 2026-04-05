//! Valency algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.

//! ## File guide: Valency
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn assign_implicit_hydrogens`
//! - `fn assign_dative_bonds`
//! - `fn is_donor_atom`
//! - `fn calculate_connectivity`
//! - `mod tests`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Valency` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Valency` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Valency` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/valency.rs` defines valency behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Valency` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Valency` decisions.
//! - FerrumChem regression tests that pin `Valency` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/valency.rs`.

use crate::error::{FerrumError, Result};
use crate::mol::elements::{get_aromatic_valences, get_valences};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// Calculates and assigns implicit hydrogens based on valence rules.
///
/// # Thread Safety
/// This function is thread-safe - it only mutates the molecule passed to it.
/// Each molecule can be processed independently in parallel.
///
/// # Algorithm
/// For non-aromatic atoms:
/// 1. Calculate bond order sum (connectivity)
/// 2. Look up possible valences for the element
/// 3. Find the smallest valence >= connectivity
/// 4. Implicit H = valence - connectivity
///
/// For aromatic atoms (OpenSMILES approach):
/// 1. Count degree (number of neighbors) + non-aromatic bond order extras
/// 2. Look up aromatic valences for the element
/// 3. Find the smallest aromatic valence >= degree
/// 4. Implicit H = aromatic_valence - degree
pub fn assign_implicit_hydrogens(mol: &mut Molecule) -> Result<()> {
    mol.invalidate_cached_properties();
    for atom_idx in 0..mol.atoms.len() {
        let atom = &mol.atoms[atom_idx];

        // Skip if explicit hydrogens are already specified
        // or if atom was in bracket notation (explicit H specification, even if 0)
        if atom.num_explicit_h > 0 || atom.no_implicit_h() {
            mol.atoms[atom_idx].num_implicit_h = 0;
            continue;
        }

        if atom.is_aromatic() {
            // Aromatic atoms: use degree-based approach with aromatic valences
            let aromatic_valences = get_aromatic_valences(atom.atomic_number);
            if aromatic_valences.is_empty() {
                mol.atoms[atom_idx].num_implicit_h = 0;
                continue;
            }

            // Degree = number of neighbors (each bond counts as 1 regardless of order)
            let degree = mol.get_degree(atom_idx as u16) as i16;

            // Adjust for formal charge
            let charge_adj = atom.formal_charge as i16;

            let mut best: Option<i16> = None;
            for &v in aromatic_valences {
                let adjusted = (v as i16) - charge_adj;
                if adjusted >= degree {
                    if best.is_none() || adjusted < best.unwrap() {
                        best = Some(adjusted);
                    }
                }
            }

            if let Some(valence) = best {
                let h = valence - degree;
                mol.atoms[atom_idx].num_implicit_h = if h > 0 { h as u8 } else { 0 };
            } else {
                mol.atoms[atom_idx].num_implicit_h = 0;
            }
        } else {
            // Non-aromatic atoms: use bond-order-based approach
            let valences = get_valences(atom.atomic_number);
            if valences.is_empty() {
                mol.atoms[atom_idx].num_implicit_h = 0;
                continue;
            }

            let connectivity = calculate_connectivity(mol, atom_idx as u16);
            let charge_adj = atom.formal_charge as i16;

            let mut best: Option<i16> = None;
            for &v in valences {
                let adjusted = (v as i16) - charge_adj;
                if adjusted >= connectivity as i16 {
                    if best.is_none() || adjusted < best.unwrap() {
                        best = Some(adjusted);
                    }
                }
            }

            if let Some(valence) = best {
                let h = valence - (connectivity as i16);
                mol.atoms[atom_idx].num_implicit_h = if h > 0 { h as u8 } else { 0 };
            } else {
                mol.atoms[atom_idx].num_implicit_h = 0;
            }
        }
    }

    Ok(())
}

/// Detect and assign dative (coordinate) bonds for transition metal complexes.
///
/// RDKit's sanitization converts bonds between donor atoms (N, O, S, P with lone
/// pairs and no formal charge) and transition metals (no default valence) to dative
/// bonds. Dative bonds don't contribute to the donor atom's valence, allowing
/// correct implicit H calculation (e.g., NH3 in Cisplatin keeps 3H).
///
/// Must be called AFTER `assign_implicit_hydrogens` so that donor atoms already
/// have their implicit H computed relative to dative bonds not counting.
pub fn assign_dative_bonds(mol: &mut Molecule) {
    use crate::mol::elements::is_transition_metal;

    mol.invalidate_cached_properties();
    for bond_idx in 0..mol.bonds.len() {
        let bond = &mol.bonds[bond_idx];
        if bond.bond_type != BondType::Single {
            continue;
        }
        let src = bond.source as usize;
        let tgt = bond.target as usize;
        let z_src = mol.atoms[src].atomic_number;
        let z_tgt = mol.atoms[tgt].atomic_number;

        // Check if one end is a transition metal and the other is a donor atom
        let is_dative = (is_transition_metal(z_src) && is_donor_atom(z_tgt))
            || (is_transition_metal(z_tgt) && is_donor_atom(z_src));

        if is_dative {
            mol.bonds[bond_idx].bond_type = BondType::Dative;
            // Ensure direction: source = donor, target = metal (acceptor)
            // Dative contributes 0 to source valence, 1 to target valence
            if is_transition_metal(z_src) && is_donor_atom(z_tgt) {
                // Metal is source, donor is target → swap
                let old_src = mol.bonds[bond_idx].source;
                mol.bonds[bond_idx].source = mol.bonds[bond_idx].target;
                mol.bonds[bond_idx].target = old_src;
            }
        }
    }
}

/// Donor atoms: N, O, S, P — can donate lone pairs to metals.
fn is_donor_atom(z: u8) -> bool {
    matches!(z, 7 | 8 | 15 | 16)
}

/// Calculate the sum of bond orders for an atom.
///
/// # Thread Safety
/// Read-only function, safe to call concurrently on different molecules.
///
/// # Aromatic Bonds
/// Aromatic bonds are special - they contribute 1.5 to connectivity in reality,
/// but we use integer math. For organic aromatic atoms (C, N, O in rings),
/// we count 2 aromatic bonds as contributing 3 to connectivity.
///
/// Note: In SMILES, aromatic atoms are marked with lowercase letters (c, n, o, etc.)
/// but bonds between them might be marked as Single. We check both the bond type
/// and whether both atoms are aromatic.
fn calculate_connectivity(mol: &Molecule, atom_idx: u16) -> u8 {
    let mut connectivity: u8 = 0;
    let mut aromatic_bonds: u8 = 0;

    let atom = &mol.atoms[atom_idx as usize];
    let is_atom_aromatic = atom.is_aromatic();

    // Sum bond orders from adjacency list
    for &bond_idx in &mol.adjacency[atom_idx as usize] {
        let bond = &mol.bonds[bond_idx as usize];

        // Determine the other atom in this bond
        let other_idx = if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        };
        let other_atom = &mol.atoms[other_idx as usize];

        // Check if this is an aromatic bond
        // Either explicitly marked as Aromatic, or both atoms are aromatic
        let is_aromatic_bond =
            bond.bond_type == BondType::Aromatic || (is_atom_aromatic && other_atom.is_aromatic());

        let bond_order = match bond.bond_type {
            BondType::Single if is_aromatic_bond => {
                // Single bond between two aromatic atoms is actually aromatic
                aromatic_bonds += 1;
                0
            }
            BondType::Single => 1,
            BondType::Double => 2,
            BondType::Triple => 3,
            BondType::Aromatic => {
                // Count aromatic bonds separately
                aromatic_bonds += 1;
                0 // Don't add to connectivity yet
            }
            BondType::Dative => 0,  // Coordinate bond, doesn't count
            BondType::Unknown => 1, // Assume single
        };

        connectivity += bond_order;
    }

    // For aromatic systems, compute the aromatic bond contribution.
    // Most atoms: each aromatic bond contributes ~1.5 (Kekulé alternation).
    // Chalcogens (O, S, Se, Te): each aromatic bond contributes 1, because
    // they donate a lone pair to the π system, and both bonds are effectively single.
    if aromatic_bonds > 0 && atom.is_aromatic() && matches!(atom.atomic_number, 8 | 16 | 34 | 52) {
        connectivity += aromatic_bonds;
    } else {
        connectivity += aromatic_bonds + (aromatic_bonds / 2);
    }

    connectivity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for methane implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_methane_implicit_h() {
        // Methane: C (should get 4 implicit H)
        let mut mol = parse_smiles("C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].atomic_number, 6); // Carbon
        assert_eq!(mol.atoms[0].num_implicit_h, 4); // CH4
    }

    #[test]
    /// Runs regression test for ethane implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_ethane_implicit_h() {
        // Ethane: CC
        let mut mol = parse_smiles("CC").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each carbon has 1 bond to the other carbon, so 3 H each
        assert_eq!(mol.atoms[0].num_implicit_h, 3); // CH3
        assert_eq!(mol.atoms[1].num_implicit_h, 3); // CH3
    }

    #[test]
    /// Runs regression test for water implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_water_implicit_h() {
        // Water: O
        let mut mol = parse_smiles("O").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].atomic_number, 8); // Oxygen
        assert_eq!(mol.atoms[0].num_implicit_h, 2); // H2O
    }

    #[test]
    /// Runs regression test for ammonia implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_ammonia_implicit_h() {
        // Ammonia: N
        let mut mol = parse_smiles("N").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].atomic_number, 7); // Nitrogen
        assert_eq!(mol.atoms[0].num_implicit_h, 3); // NH3
    }

    #[test]
    /// Runs regression test for ethene implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_ethene_implicit_h() {
        // Ethene: C=C
        let mut mol = parse_smiles("C=C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each carbon has 1 double bond (order 2), so 2 H each
        assert_eq!(mol.atoms[0].num_implicit_h, 2); // CH2
        assert_eq!(mol.atoms[1].num_implicit_h, 2); // CH2
    }

    #[test]
    /// Runs regression test for acetylene implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_acetylene_implicit_h() {
        // Acetylene: C#C
        let mut mol = parse_smiles("C#C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each carbon has 1 triple bond (order 3), so 1 H each
        assert_eq!(mol.atoms[0].num_implicit_h, 1); // CH
        assert_eq!(mol.atoms[1].num_implicit_h, 1); // CH
    }

    #[test]
    /// Runs regression test for charged ammonium.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_charged_ammonium() {
        // Ammonium: [NH4+]
        let mut mol = parse_smiles("[NH4+]").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Already has explicit H, so should have 0 implicit
        assert_eq!(mol.atoms[0].num_implicit_h, 0);
        assert_eq!(mol.atoms[0].num_explicit_h, 4);
    }

    #[test]
    /// Runs regression test for benzene implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_benzene_implicit_h() {
        // Benzene: c1ccccc1
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each aromatic carbon has 2 aromatic bonds, so 1 H each
        for atom in &mol.atoms {
            assert_eq!(atom.num_implicit_h, 1);
        }
    }

    #[test]
    /// Runs regression test for ethanol implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/algo/valency.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_ethanol_implicit_h() {
        // Ethanol: CCO
        let mut mol = parse_smiles("CCO").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].num_implicit_h, 3); // CH3
        assert_eq!(mol.atoms[1].num_implicit_h, 2); // CH2
        assert_eq!(mol.atoms[2].num_implicit_h, 1); // OH
    }
}
