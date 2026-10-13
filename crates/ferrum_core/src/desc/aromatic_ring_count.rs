//! Descriptor module: Aromatic Ring Count.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::rings::{find_minimum_cycle_basis_bonds, minimum_cycle_basis_bonds_cached};
use crate::mol::molecule::Molecule;

// ---------------------------------------------------------------------------
// Public computation function
// ---------------------------------------------------------------------------

/// Compute aromatic ring count.
///
/// RDKit's Lipinski aromatic ring descriptor counts aromatic SSSR rings.
/// A ring is aromatic only when every bond around that ring is aromatic;
/// aromatic atoms alone are not sufficient in fused/Kekulized systems.
pub fn compute_aromatic_ring_count(mol: &Molecule) -> u32 {
    find_minimum_cycle_basis_bonds(mol)
        .iter()
        .filter(|ring| ring_bonds_all_aromatic(mol, ring))
        .count() as u32
}

pub fn compute_aromatic_ring_count_cached(mol: &Molecule) -> u32 {
    let cycles = minimum_cycle_basis_bonds_cached(mol);
    let aromatic_bond_mask: Vec<bool> = mol.bonds.iter().map(|bond| bond.is_aromatic()).collect();
    count_aromatic_cycles(cycles, &aromatic_bond_mask)
}

fn ring_bonds_all_aromatic(mol: &Molecule, ring: &[u16]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx as usize].is_aromatic())
}

/// Counts aromatic cycles using a precomputed aromatic-bond mask.
fn count_aromatic_cycles(
    cycles: &[smallvec::SmallVec<[u16; 8]>],
    aromatic_bond_mask: &[bool],
) -> u32 {
    cycles
        .iter()
        .filter(|ring| ring.iter().all(|&bond_idx| aromatic_bond_mask[bond_idx as usize]))
        .count() as u32
}

// ---------------------------------------------------------------------------
// Typed Descriptor
// ---------------------------------------------------------------------------

/// Counts aromatic rings in the molecule.
///
/// Returns Int(u32) with number of aromatic rings.
/// Requires aromaticity perception to be run first.
pub struct AromaticRingCount;

impl Descriptor for AromaticRingCount {
    fn name(&self) -> &str {
        "AromaticRingCount"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_aromatic_ring_count_cached(mol))
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<AromaticRingCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_aromatic_ring_count_benzene() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    fn test_aromatic_ring_count_naphthalene() {
        let mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            2
        );
    }

    #[test]
    fn test_aromatic_ring_count_cyclohexane() {
        let mol = parse_smiles("C1CCCCC1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    fn test_aromatic_ring_count_pyridine() {
        let mol = parse_smiles("c1ccncc1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    fn test_aromatic_ring_count_pyrrole() {
        let mol = parse_smiles("c1cc[nH]c1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    fn test_aromatic_ring_count_no_rings() {
        let mol = parse_smiles("CCCC").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    fn test_compute_aromatic_ring_count() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(compute_aromatic_ring_count(&mol), 1);
    }

    #[test]
    fn test_aromatic_ring_count_requires_aromatic_ring_bonds() {
        let mol = parse_smiles("CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2").unwrap();
        assert_eq!(compute_aromatic_ring_count(&mol), 3);
    }
}
