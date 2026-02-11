use super::typed::{DescriptorValue, Descriptor};
use crate::mol::molecule::Molecule;
use crate::algo::rings::find_sssr;

// ---------------------------------------------------------------------------
// Public computation function
// ---------------------------------------------------------------------------

/// Compute aromatic ring count.
///
/// A ring is considered aromatic if all atoms in the ring
/// have their aromatic flag set.
pub fn compute_aromatic_ring_count(mol: &Molecule) -> u32 {
    let mut mol_copy = Molecule {
        atoms: mol.atoms.clone(),
        bonds: mol.bonds.clone(),
        adjacency: mol.adjacency.clone(),
    };
    
    let rings = find_sssr(&mut mol_copy);
    
    let mut aromatic_count = 0u32;
    
    for ring in rings {
        let all_aromatic = ring.iter()
            .all(|&atom_idx| mol.atoms[atom_idx as usize].is_aromatic());
        
        if all_aromatic {
            aromatic_count += 1;
        }
    }
    
    aromatic_count
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
    use crate::io::smiles::parse_smiles;
    use crate::algo::aromaticity::perceive_aromaticity;

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
}
