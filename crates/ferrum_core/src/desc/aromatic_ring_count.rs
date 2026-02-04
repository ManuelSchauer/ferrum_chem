use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;
use crate::algo::rings::find_sssr;

/// Counts aromatic rings in the molecule.
/// 
/// A ring is considered aromatic if all atoms in the ring
/// have their aromatic flag set. Requires aromaticity perception
/// to be run first for accurate results.
/// 
/// # Example
/// ```
/// use ferrum_core::desc::{Descriptor, aromatic_ring_count::AromaticRingCount};
/// use ferrum_core::io::smiles::parse_smiles;
/// use ferrum_core::algo::aromaticity::perceive_aromaticity;
/// 
/// let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene
/// perceive_aromaticity(&mut mol);
/// 
/// let desc = AromaticRingCount;
/// let mut result = vec![0.0; 1];
/// desc.calculate(&mol, &mut result);
/// assert_eq!(result[0], 1.0); // 1 aromatic ring
/// ```
pub struct AromaticRingCount;

impl Descriptor for AromaticRingCount {
    fn name(&self) -> String { "AromaticRingCount".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Clone molecule to avoid modifying original during ring detection
        let mut mol_copy = Molecule {
            atoms: mol.atoms.clone(),
            bonds: mol.bonds.clone(),
            adjacency: mol.adjacency.clone(),
        };
        
        let rings = find_sssr(&mut mol_copy);
        
        let mut aromatic_count = 0;
        
        for ring in rings {
            // Check if all atoms in ring are aromatic
            let all_aromatic = ring.iter()
                .all(|&atom_idx| mol.atoms[atom_idx as usize].is_aromatic());
            
            if all_aromatic {
                aromatic_count += 1;
            }
        }
        
        output[0] = aromatic_count as Real;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::aromaticity::perceive_aromaticity;

    #[test]
    fn test_aromatic_ring_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticRingCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_aromatic_ring_count_naphthalene() {
        let mut mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticRingCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 2.0); // 2 fused aromatic rings
    }

    #[test]
    fn test_aromatic_ring_count_cyclohexane() {
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticRingCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // Not aromatic
    }

    #[test]
    fn test_aromatic_ring_count_pyridine() {
        let mut mol = parse_smiles("c1ccncc1").unwrap();
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticRingCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_aromatic_ring_count_pyrrole() {
        let mut mol = parse_smiles("c1cc[nH]c1").unwrap();
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticRingCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_aromatic_ring_count_no_rings() {
        let mut mol = parse_smiles("CCCC").unwrap();
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticRingCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0);
    }
}
