//! Structural chain descriptors for molecular complexity analysis.
//!
//! These descriptors measure different aspects of molecular chain structure:
//! - Longest aliphatic chain: Maximum sp3 carbon chain length
//! - Largest chain: Maximum chain through any atoms
//!
//! Useful for predicting:
//! - Flexibility
//! - Lipophilicity
//! - Molecular complexity

use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::{Real, AtomIdx};
use std::collections::HashSet;

/// Longest aliphatic (sp3 carbon) chain descriptor.
///
/// Finds the longest continuous chain of sp3-hybridized carbons (non-aromatic, single bonds).
/// This is important for predicting lipophilicity and molecular flexibility.
///
/// Returns 1 value: the length of the longest aliphatic carbon chain.
pub struct LongestAliphaticChain;

impl LongestAliphaticChain {
    /// DFS to find longest path through aliphatic carbons
    fn dfs_aliphatic(
        mol: &Molecule,
        current: AtomIdx,
        visited: &mut HashSet<AtomIdx>,
        depth: usize,
    ) -> usize {
        visited.insert(current);
        
        let mut max_length = depth;
        
        // Try extending through aliphatic carbon neighbors
        for &neighbor_idx in &mol.adjacency[current as usize] {
            if visited.contains(&neighbor_idx) {
                continue;
            }
            
            let neighbor = &mol.atoms[neighbor_idx as usize];
            
            // Check if neighbor is aliphatic carbon
            if neighbor.atomic_number == 6 && !neighbor.is_aromatic() {
                let length = Self::dfs_aliphatic(mol, neighbor_idx, visited, depth + 1);
                max_length = max_length.max(length);
            }
        }
        
        visited.remove(&current);
        max_length
    }
}

impl Descriptor for LongestAliphaticChain {
    fn name(&self) -> String {
        "LongestAliphaticChain".to_string()
    }
    
    fn size(&self) -> usize {
        1
    }
    
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut max_chain = 0;
        
        // Try starting from each aliphatic carbon
        for i in 0..mol.atoms.len() {
            let atom = &mol.atoms[i];
            
            if atom.atomic_number == 6 && !atom.is_aromatic() {
                let mut visited = HashSet::new();
                let length = Self::dfs_aliphatic(mol, i as AtomIdx, &mut visited, 1);
                max_chain = max_chain.max(length);
            }
        }
        
        output[0] = max_chain as Real;
    }
}

/// Largest chain through any atoms descriptor.
///
/// Finds the longest continuous chain through any atoms (not just carbons).
/// This measures overall molecular backbone length and complexity.
///
/// Returns 1 value: the length of the longest atom chain.
pub struct LargestChain;

impl LargestChain {
    /// DFS to find longest path through any atoms
    fn dfs_any(
        mol: &Molecule,
        current: AtomIdx,
        visited: &mut HashSet<AtomIdx>,
        depth: usize,
    ) -> usize {
        visited.insert(current);
        
        let mut max_length = depth;
        
        // Try extending through all neighbors
        for &neighbor_idx in &mol.adjacency[current as usize] {
            if !visited.contains(&neighbor_idx) {
                let length = Self::dfs_any(mol, neighbor_idx, visited, depth + 1);
                max_length = max_length.max(length);
            }
        }
        
        visited.remove(&current);
        max_length
    }
}

impl Descriptor for LargestChain {
    fn name(&self) -> String {
        "LargestChain".to_string()
    }
    
    fn size(&self) -> usize {
        1
    }
    
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut max_chain = 0;
        
        // Try starting from each atom
        for i in 0..mol.atoms.len() {
            let mut visited = HashSet::new();
            let length = Self::dfs_any(mol, i as AtomIdx, &mut visited, 1);
            max_chain = max_chain.max(length);
        }
        
        output[0] = max_chain as Real;
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
    fn test_longest_aliphatic_chain_linear() {
        let mol = parse_and_prepare("CCCCCC"); // Hexane
        let desc = LongestAliphaticChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Hexane longest aliphatic chain: {}", output[0]);
        assert_eq!(output[0], 6.0, "Hexane should have chain of 6");
    }

    #[test]
    fn test_longest_aliphatic_chain_branched() {
        let mol = parse_and_prepare("CC(C)CCC"); // 2-methylpentane
        let desc = LongestAliphaticChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("2-methylpentane longest aliphatic chain: {}", output[0]);
        // The longest PATH is C-C-C-C-C-C (going through the branch point)
        assert_eq!(output[0], 6.0, "Longest path through carbons is 6");
    }

    #[test]
    fn test_longest_aliphatic_chain_aromatic() {
        let mol = parse_and_prepare("CCCc1ccccc1"); // Propylbenzene
        let desc = LongestAliphaticChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Propylbenzene longest aliphatic chain: {}", output[0]);
        assert_eq!(output[0], 3.0, "Should only count aliphatic propyl chain");
    }

    #[test]
    fn test_longest_aliphatic_chain_aromatic_only() {
        let mol = parse_and_prepare("c1ccccc1"); // Benzene
        let desc = LongestAliphaticChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Benzene longest aliphatic chain: {}", output[0]);
        assert_eq!(output[0], 0.0, "Benzene has no aliphatic carbons");
    }

    #[test]
    fn test_longest_aliphatic_chain_complex() {
        let mol = parse_and_prepare("CCCCC(CC)CCCC"); // 3-ethyloctane
        let desc = LongestAliphaticChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("3-ethyloctane longest aliphatic chain: {}", output[0]);
        // Longest path goes through all carbons: 11 total
        assert_eq!(output[0], 11.0, "Longest path through all carbons");
    }

    #[test]
    fn test_largest_chain_linear() {
        let mol = parse_and_prepare("CCCCCC"); // Hexane
        let desc = LargestChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Hexane largest chain: {}", output[0]);
        assert_eq!(output[0], 6.0, "Hexane chain is 6 carbons");
    }

    #[test]
    fn test_largest_chain_aromatic() {
        let mol = parse_and_prepare("c1ccccc1"); // Benzene
        let desc = LargestChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Benzene largest chain: {}", output[0]);
        assert_eq!(output[0], 6.0, "Benzene ring has 6 atoms");
    }

    #[test]
    fn test_largest_chain_with_heteroatom() {
        let mol = parse_and_prepare("CCOCCOC"); // Diethyl ether with O in chain
        let desc = LargestChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Diethyl ether chain variant largest chain: {}", output[0]);
        // Chain includes heteroatoms: C-C-O-C-C-O-C = 7
        assert_eq!(output[0], 7.0, "Should count through oxygen atoms");
    }

    #[test]
    fn test_largest_chain_branched() {
        let mol = parse_and_prepare("CC(C)(C)C"); // Neopentane
        let desc = LargestChain;
        let mut output = vec![0.0; 1];
        desc.calculate(&mol, &mut output);
        
        println!("Neopentane largest chain: {}", output[0]);
        // Can go from one branch through center to another: 5 carbons total
        assert_eq!(output[0], 5.0, "Longest path is 5 (counting all carbons in path)");
    }

    #[test]
    fn test_descriptor_names() {
        assert_eq!(LongestAliphaticChain.name(), "LongestAliphaticChain");
        assert_eq!(LargestChain.name(), "LargestChain");
    }
}
