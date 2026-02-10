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

use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::{Real, AtomIdx, BondType};
use std::collections::HashSet;

/// Largest Pi System descriptor.
///
/// Finds the largest conjugated π-system (connected double/aromatic bonds).
/// Returns the number of atoms involved in the largest π-system.
///
/// Examples:
/// - Benzene: 6 (all atoms in aromatic ring)
/// - 1,3-butadiene: 4 (C=C-C=C conjugated system)
/// - Ethene: 2 (single double bond)
/// - Ethane: 0 (no π-bonds)
pub struct LargestPiSystem;

impl LargestPiSystem {
    /// Check if an atom can participate in a π-system
    fn is_pi_atom(mol: &Molecule, atom_idx: AtomIdx) -> bool {
        let atom = &mol.atoms[atom_idx as usize];
        
        // Aromatic atoms are part of π-systems
        if atom.is_aromatic() {
            return true;
        }
        
        // Check if atom has any double or triple bonds
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
        
        // Traverse to neighbors through any bond if both atoms are π-atoms
        // This handles conjugation: C=C-C=C where middle single bond connects two π-atoms
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
                
                // Include neighbor if it's a π-atom
                // This allows traversing conjugated systems through single bonds
                if Self::is_pi_atom(mol, neighbor_idx) {
                    Self::dfs_pi_system(mol, neighbor_idx, visited, system);
                }
            }
        }
    }
}

impl Descriptor for LargestPiSystem {
    fn name(&self) -> String {
        "LargestPiSystem".to_string()
    }
    
    fn size(&self) -> usize {
        1
    }
    
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut visited = HashSet::new();
        let mut largest_system_size = 0;
        
        // Try starting from each π-atom
        for i in 0..mol.atoms.len() {
            let atom_idx = i as AtomIdx;
            
            if visited.contains(&atom_idx) {
                continue;
            }
            
            if Self::is_pi_atom(mol, atom_idx) {
                let mut system = HashSet::new();
                Self::dfs_pi_system(mol, atom_idx, &mut visited, &mut system);
                largest_system_size = largest_system_size.max(system.len());
            }
        }
        
        output[0] = largest_system_size as Real;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::valency::assign_implicit_hydrogens;
    use crate::algo::aromaticity::perceive_aromaticity;

    fn parse_and_prepare(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        perceive_aromaticity(&mut mol);
        mol
    }

    #[test]
    fn test_pi_system_benzene() {
        let mol = parse_and_prepare("c1ccccc1"); // Benzene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Benzene largest π-system: {}", output[0]);
        assert_eq!(output[0], 6.0, "Benzene has 6-atom aromatic π-system");
    }

    #[test]
    fn test_pi_system_ethene() {
        let mol = parse_and_prepare("C=C"); // Ethene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Ethene largest π-system: {}", output[0]);
        assert_eq!(output[0], 2.0, "Ethene has 2-atom π-system (double bond)");
    }

    #[test]
    fn test_pi_system_butadiene() {
        let mol = parse_and_prepare("C=CC=C"); // 1,3-butadiene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("1,3-butadiene largest π-system: {}", output[0]);
        assert_eq!(output[0], 4.0, "Butadiene has 4-atom conjugated π-system");
    }

    #[test]
    fn test_pi_system_ethane() {
        let mol = parse_and_prepare("CC"); // Ethane (no π-bonds)
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Ethane largest π-system: {}", output[0]);
        assert_eq!(output[0], 0.0, "Ethane has no π-system");
    }

    #[test]
    fn test_pi_system_naphthalene() {
        let mol = parse_and_prepare("c1ccc2ccccc2c1"); // Naphthalene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Naphthalene largest π-system: {}", output[0]);
        assert_eq!(output[0], 10.0, "Naphthalene has 10-atom fused aromatic π-system");
    }

    #[test]
    fn test_pi_system_styrene() {
        let mol = parse_and_prepare("C=Cc1ccccc1"); // Styrene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Styrene largest π-system: {}", output[0]);
        assert_eq!(output[0], 8.0, "Styrene has 8-atom conjugated system (vinyl + benzene)");
    }

    #[test]
    fn test_pi_system_separated_double_bonds() {
        let mol = parse_and_prepare("C=CCC=C"); // 1,4-pentadiene (non-conjugated)
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("1,4-pentadiene largest π-system: {}", output[0]);
        assert_eq!(output[0], 2.0, "Largest system is 2 (isolated double bonds)");
    }

    #[test]
    fn test_pi_system_benzyl_alkane() {
        let mol = parse_and_prepare("c1ccccc1CCC"); // Propylbenzene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Propylbenzene largest π-system: {}", output[0]);
        assert_eq!(output[0], 6.0, "Largest system is benzene ring (6 atoms)");
    }

    #[test]
    fn test_pi_system_triple_bond() {
        let mol = parse_and_prepare("C#C"); // Acetylene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Acetylene largest π-system: {}", output[0]);
        assert_eq!(output[0], 2.0, "Acetylene has 2-atom π-system (triple bond)");
    }

    #[test]
    fn test_pi_system_extended_conjugation() {
        let mol = parse_and_prepare("C=CC=CC=C"); // Hexatriene
        let desc = LargestPiSystem;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Hexatriene largest π-system: {}", output[0]);
        assert_eq!(output[0], 6.0, "Hexatriene has 6-atom conjugated system");
    }

    #[test]
    fn test_descriptor_name() {
        assert_eq!(LargestPiSystem.name(), "LargestPiSystem");
        assert_eq!(LargestPiSystem.size(), 1);
    }
}
