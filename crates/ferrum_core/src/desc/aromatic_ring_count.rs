use super::typed::{Descriptor, DescriptorValue};
use crate::algo::rings::find_minimum_cycle_basis_bonds;
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

fn ring_bonds_all_aromatic(mol: &Molecule, ring: &[usize]) -> bool {
    ring.iter().all(|&bond_idx| mol.bonds[bond_idx].is_aromatic())
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

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(compute_aromatic_ring_count(mol))
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<AromaticRingCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_aromatic_ring_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticRingCount.calculate(&mut mol).as_int(), 1);
    }

    #[test]
    fn test_aromatic_ring_count_naphthalene() {
        let mut mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticRingCount.calculate(&mut mol).as_int(), 2);
    }

    #[test]
    fn test_aromatic_ring_count_cyclohexane() {
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticRingCount.calculate(&mut mol).as_int(), 0);
    }

    #[test]
    fn test_aromatic_ring_count_pyridine() {
        let mut mol = parse_smiles("c1ccncc1").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticRingCount.calculate(&mut mol).as_int(), 1);
    }

    #[test]
    fn test_aromatic_ring_count_pyrrole() {
        let mut mol = parse_smiles("c1cc[nH]c1").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticRingCount.calculate(&mut mol).as_int(), 1);
    }

    #[test]
    fn test_aromatic_ring_count_no_rings() {
        let mut mol = parse_smiles("CCCC").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticRingCount.calculate(&mut mol).as_int(), 0);
    }

    #[test]
    fn test_compute_aromatic_ring_count() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(compute_aromatic_ring_count(&mol), 1);
    }

    #[test]
    fn test_aromatic_ring_count_requires_aromatic_ring_bonds() {
        let mut mol = parse_smiles("CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(compute_aromatic_ring_count(&mol), 3);
    }

}
