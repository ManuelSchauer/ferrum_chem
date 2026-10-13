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

/// Largest pi system (CDK 2.13 `LargestPiSystemDescriptor`, `nAtomP`).
///
/// CDK seeds a breadth-first search at every unvisited "pi" atom: an atom with
/// a double/triple bond (Kekulé form), a formal charge, an aromatic flag (CDK
/// Daylight model) or an N/O element. Only atoms reached from a neighbour are
/// counted, so the seed itself is counted only when one of its neighbours in the
/// system has more than one connection. Returns Int(u32).
pub struct LargestPiSystem;

impl Descriptor for LargestPiSystem {
    fn name(&self) -> &str {
        "LargestPiSystem"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(cdk_largest_pi_system(mol))
    }
}

/// CDK `LargestPiSystemDescriptor` value (see [`LargestPiSystem`]).
pub(crate) fn cdk_largest_pi_system(mol: &Molecule) -> u32 {
    use crate::desc::cdk_daylight::{CdkMol, daylight_model_contrib, on_huckel_cycle};
    let n = mol.num_atoms();
    if n < 2 {
        return 0;
    }
    let m = CdkMol::new(mol);
    // atoms whose qualification depends on the aromatic flag only
    let mut contrib: Option<Vec<i32>> = None;
    let qualifies: Vec<bool> = (0..n)
        .map(|a| {
            let z = m.z[a];
            if z == 7 || z == 8 || m.charge[a] != 0 {
                return true;
            }
            if m.adj[a].iter().any(|&(_, bi)| m.order[bi as usize] != 1) {
                return true;
            }
            if m.adj[a].is_empty() {
                return true; // no bonds: max bond order is not SINGLE
            }
            if !m.ring_atom[a] {
                return false;
            }
            let c = contrib.get_or_insert_with(|| daylight_model_contrib(&m));
            on_huckel_cycle(&m, c, a)
        })
        .collect();

    let mut comp = vec![false; n];
    let mut stack = Vec::new();
    let mut best = 0u32;
    for s in 0..n {
        if !qualifies[s] || comp[s] {
            continue;
        }
        comp[s] = true;
        stack.push(s);
        let mut size = 0u32;
        while let Some(a) = stack.pop() {
            size += 1;
            for &(w, _) in &m.adj[a] {
                let w = w as usize;
                if qualifies[w] && !comp[w] {
                    comp[w] = true;
                    stack.push(w);
                }
            }
        }
        // the seed is only re-reached through a neighbour with degree > 1
        let seed_counted = m.adj[s].iter().any(|&(w, _)| qualifies[w as usize] && m.degree(w as usize) > 1);
        let counted = if seed_counted { size } else { size - 1 };
        best = best.max(counted);
    }
    best
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
    use crate::io::smiles::parse_smiles;

    fn parse_and_prepare(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    fn n_atom_p(smiles: &str) -> u32 {
        cdk_largest_pi_system(&parse_and_prepare(smiles))
    }

    #[test]
    /// `nAtomP` values verified against CDK 2.13.
    fn test_largest_pi_system_cdk() {
        assert_eq!(n_atom_p("c1ccccc1"), 6);
        assert_eq!(n_atom_p("c1ccc2ccccc2c1"), 10);
        assert_eq!(n_atom_p("C=CC=C"), 4);
        assert_eq!(n_atom_p("C=CC=CC=C"), 6);
        assert_eq!(n_atom_p("C=Cc1ccccc1"), 8);
        assert_eq!(n_atom_p("c1ccccc1CCC"), 6);
        assert_eq!(n_atom_p("CC"), 0);
        // CDK quirk: the seed atom is only counted when re-reached through a
        // neighbour with more than one connection
        assert_eq!(n_atom_p("C=C"), 1);
        assert_eq!(n_atom_p("C#C"), 1);
        assert_eq!(n_atom_p("C=CCC=C"), 2);
        assert_eq!(n_atom_p("C=C=C"), 3);
        assert_eq!(n_atom_p("CS(C)=O"), 1);
        // N and O always qualify
        assert_eq!(n_atom_p("OP(O)(O)=O"), 5);
        assert_eq!(n_atom_p("CO"), 0);
        // aromatic S/As (CDK Daylight model) qualify without a double bond
        assert_eq!(n_atom_p("c1ccsc1"), 5);
        assert_eq!(n_atom_p("[As]1C=CC=C1"), 5);
    }

    #[test]
    fn test_largest_pi_system_descriptor() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(
            LargestPiSystem.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
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
