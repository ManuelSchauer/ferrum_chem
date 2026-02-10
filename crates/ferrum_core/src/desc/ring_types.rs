//! Ring type counting descriptors
//!
//! Classifies rings by saturation and aromaticity:
//! - Saturated rings: no double/triple bonds
//! - Aliphatic rings: non-aromatic
//! - Aromatic rings: marked as aromatic

use crate::mol::molecule::Molecule;
use crate::algo::rings::find_sssr;
use crate::types::BondType;

/// Count number of saturated rings (no double/triple bonds)
///
/// A saturated ring contains only single bonds.
///
/// # Examples
/// - Cyclohexane: 1 saturated ring
/// - Benzene: 0 saturated rings (aromatic double bonds)
/// - Cyclohexene: 0 saturated rings (one double bond)
pub fn num_saturated_rings(mol: &mut Molecule) -> u32 {
    let rings = find_sssr(mol);
    
    rings.iter().filter(|ring| {
        // Check if all bonds in the ring are single bonds
        for i in 0..ring.len() {
            let atom1 = ring[i];
            let atom2 = ring[(i + 1) % ring.len()];
            
            // Find bond between atom1 and atom2
            if let Some(bond_idx) = mol.adjacency[atom1 as usize].iter()
                .find(|&&bond_idx| {
                    let bond = &mol.bonds[bond_idx as usize];
                    (bond.source == atom1 && bond.target == atom2) ||
                    (bond.source == atom2 && bond.target == atom1)
                })
            {
                let bond = &mol.bonds[*bond_idx as usize];
                if bond.bond_type != BondType::Single {
                    return false; // Found a double/triple bond
                }
            }
        }
        true // All bonds are single
    }).count() as u32
}

/// Count number of aliphatic rings (non-aromatic)
///
/// Aliphatic rings are not marked as aromatic by the aromaticity perception.
/// This includes both saturated rings and rings with double bonds that don't
/// meet aromaticity criteria.
///
/// # Examples
/// - Cyclohexane: 1 aliphatic ring
/// - Cyclohexene: 1 aliphatic ring
/// - Benzene: 0 aliphatic rings (aromatic)
/// - Naphthalene: 0 aliphatic rings (both rings aromatic)
pub fn num_aliphatic_rings(mol: &mut Molecule) -> u32 {
    let rings = find_sssr(mol);
    
    rings.iter().filter(|ring| {
        // Check if any atom in the ring is aromatic
        !ring.iter().any(|&atom_idx| {
            mol.atoms[atom_idx as usize].is_aromatic()
        })
    }).count() as u32
}

/// Count number of aromatic rings
///
/// Rings where all atoms are marked as aromatic.
///
/// # Examples
/// - Benzene: 1 aromatic ring
/// - Naphthalene: 2 aromatic rings
/// - Pyridine: 1 aromatic ring
/// - Cyclohexane: 0 aromatic rings
pub fn num_aromatic_rings(mol: &mut Molecule) -> u32 {
    let rings = find_sssr(mol);
    
    rings.iter().filter(|ring| {
        // Check if all atoms in the ring are aromatic
        ring.iter().all(|&atom_idx| {
            mol.atoms[atom_idx as usize].is_aromatic()
        })
    }).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }
    
    #[test]
    fn test_saturated_cyclohexane() {
        let mut mol = parse("C1CCCCC1");
        assert_eq!(num_saturated_rings(&mut mol), 1);
        assert_eq!(num_aliphatic_rings(&mut mol), 1);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }
    
    #[test]
    fn test_aromatic_benzene() {
        let mut mol = parse("c1ccccc1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 1);
    }
    
    #[test]
    fn test_cyclohexene() {
        let mut mol = parse("C1=CCCCC1");
        assert_eq!(num_saturated_rings(&mut mol), 0); // Has double bond
        assert_eq!(num_aliphatic_rings(&mut mol), 1); // Not aromatic
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }
    
    #[test]
    fn test_naphthalene() {
        let mut mol = parse("c1ccc2ccccc2c1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 2); // Two fused aromatic rings
    }
    
    #[test]
    fn test_pyridine() {
        let mut mol = parse("c1ccncc1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 1);
    }
    
    #[test]
    fn test_decalin() {
        // Two fused saturated rings
        let mut mol = parse("C1CCC2CCCCC2C1");
        assert_eq!(num_saturated_rings(&mut mol), 2);
        assert_eq!(num_aliphatic_rings(&mut mol), 2);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }
    
    #[test]
    fn test_acyclic() {
        let mut mol = parse("CCCCCC"); // Hexane
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }
}
