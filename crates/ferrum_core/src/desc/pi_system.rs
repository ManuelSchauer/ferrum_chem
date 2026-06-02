//! Largest Pi System descriptor for conjugation analysis.
//!
//! Finds the largest conjugated π-system in a molecule.
//! A π-system consists of atoms connected by double bonds or aromatic bonds,
//! allowing electron delocalization.
//!
//! Important for:
//! - UV-Vis absorption prediction
//! - Aromatic stability
//! - Reactivity prediction

//! ## File guide: Pi System
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn is_pi_atom`
//! - `fn dfs_pi_system`
//! - `struct PiSystemCounts`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Pi System` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Pi System` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/pi_system.rs` defines pi system behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Pi System` workflows.
//! - Repository unit and parity tests covering `Pi System` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/pi_system.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use std::collections::HashSet;

/// Check if an atom can participate in a π-system
fn is_pi_atom(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    let atom = &mol.atoms[atom_idx as usize];

    if atom.is_aromatic() {
        return true;
    }

    for bond in &mol.bonds {
        if bond.source == atom_idx || bond.target == atom_idx {
            match bond.bond_type {
                BondType::Double | BondType::Triple | BondType::Aromatic => return true,
                _ => {}
            }
        }
    }

    false
}

/// DFS to find all atoms in a connected π-system
fn dfs_pi_system(
    mol: &Molecule,
    current: AtomIdx,
    visited: &mut HashSet<AtomIdx>,
    system: &mut HashSet<AtomIdx>,
) {
    if visited.contains(&current) {
        return;
    }

    visited.insert(current);
    system.insert(current);

    for bond in &mol.bonds {
        let neighbor_idx = if bond.source == current {
            Some(bond.target)
        } else if bond.target == current {
            Some(bond.source)
        } else {
            None
        };

        if let Some(neighbor_idx) = neighbor_idx {
            if visited.contains(&neighbor_idx) {
                continue;
            }

            if is_pi_atom(mol, neighbor_idx) {
                dfs_pi_system(mol, neighbor_idx, visited, system);
            }
        }
    }
}

/// Pi system counts descriptor.
///
/// Returns IntU16Vec with 2 values:
/// [largest_pi_system_size, num_pi_systems]
pub struct PiSystemCounts;

impl Descriptor for PiSystemCounts {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "PiSystemCounts"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut visited = HashSet::new();
        let mut largest_size = 0u16;
        let mut num_systems = 0u16;

        for atom_idx in 0..mol.atoms.len() as AtomIdx {
            if visited.contains(&atom_idx) {
                continue;
            }

            if !is_pi_atom(mol, atom_idx) {
                continue;
            }

            let mut system = HashSet::new();
            dfs_pi_system(mol, atom_idx, &mut visited, &mut system);

            if !system.is_empty() {
                num_systems += 1;
                largest_size = largest_size.max(system.len() as u16);
            }
        }

        DescriptorValue::IntU16Vec(vec![largest_size, num_systems])
    }
}

/// Largest pi system size descriptor.
///
/// Returns Int(u32) with the number of atoms in the largest π-system.
pub struct LargestPiSystem;

impl Descriptor for LargestPiSystem {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "LargestPiSystem"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut visited = HashSet::new();
        let mut largest_system_size = 0;

        for i in 0..mol.atoms.len() {
            let atom_idx = i as AtomIdx;

            if visited.contains(&atom_idx) {
                continue;
            }

            if is_pi_atom(mol, atom_idx) {
                let mut system = HashSet::new();
                dfs_pi_system(mol, atom_idx, &mut visited, &mut system);
                largest_system_size = largest_system_size.max(system.len());
            }
        }

        DescriptorValue::Int(largest_system_size as u32)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<PiSystemCounts>;
    let _ = assert_send_sync::<LargestPiSystem>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::algo::valency::assign_implicit_hydrogens;
    use crate::io::smiles::parse_smiles;

    /// Parses and prepare.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse_and_prepare(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        perceive_aromaticity(&mut mol);
        mol
    }

    #[test]
    /// Runs regression test for pi system benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_benzene() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    /// Runs regression test for pi system ethene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_ethene() {
        let mut mol = parse_and_prepare("C=C");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(2)
        );
    }

    #[test]
    /// Runs regression test for pi system butadiene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_butadiene() {
        let mut mol = parse_and_prepare("C=CC=C");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(4)
        );
    }

    #[test]
    /// Runs regression test for pi system ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_ethane() {
        let mut mol = parse_and_prepare("CC");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(0)
        );
    }

    #[test]
    /// Runs regression test for pi system naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_naphthalene() {
        let mut mol = parse_and_prepare("c1ccc2ccccc2c1");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(10)
        );
    }

    #[test]
    /// Runs regression test for pi system styrene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_styrene() {
        let mut mol = parse_and_prepare("C=Cc1ccccc1");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(8)
        );
    }

    #[test]
    /// Runs regression test for pi system separated double bonds.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_separated_double_bonds() {
        let mut mol = parse_and_prepare("C=CCC=C");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(2)
        );
    }

    #[test]
    /// Runs regression test for pi system benzyl alkane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_benzyl_alkane() {
        let mut mol = parse_and_prepare("c1ccccc1CCC");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    /// Runs regression test for pi system triple bond.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_triple_bond() {
        let mut mol = parse_and_prepare("C#C");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(2)
        );
    }

    #[test]
    /// Runs regression test for pi system extended conjugation.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_extended_conjugation() {
        let mut mol = parse_and_prepare("C=CC=CC=C");
        assert_eq!(
            LargestPiSystem.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    /// Runs regression test for pi system counts benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pi_system_counts_benzene() {
        let mut mol = parse_and_prepare("c1ccccc1");
        let value = PiSystemCounts.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        match value {
            DescriptorValue::IntU16Vec(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], 6, "benzene pi system size = 6");
            }
            other => panic!("expected IntU16Vec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for descriptor names.
    ///
    /// Function context: `crates/ferrum_core/src/desc/pi_system.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_descriptor_names() {
        assert_eq!(LargestPiSystem.name(), "LargestPiSystem");
        assert_eq!(PiSystemCounts.name(), "PiSystemCounts");
    }
}
