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

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::HashSet;

/// Longest aliphatic (sp3 carbon) chain descriptor.
///
/// Finds the longest continuous chain of sp3-hybridized carbons (non-aromatic, single bonds).
/// Returns Int(u32) with the length of the longest aliphatic carbon chain.
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

        for &neighbor_idx in &mol.adjacency[current as usize] {
            if visited.contains(&neighbor_idx) {
                continue;
            }

            let neighbor = &mol.atoms[neighbor_idx as usize];

            if neighbor.atomic_number == 6 && !neighbor.is_aromatic() {
                let length = Self::dfs_aliphatic(mol, neighbor_idx, visited, depth + 1);
                max_length = max_length.max(length);
            }
        }

        visited.remove(&current);
        max_length
    }

    fn compute(mol: &Molecule) -> u32 {
        let mut max_chain = 0;

        for i in 0..mol.atoms.len() {
            let atom = &mol.atoms[i];

            if atom.atomic_number == 6 && !atom.is_aromatic() {
                let mut visited = HashSet::new();
                let length = Self::dfs_aliphatic(mol, i as AtomIdx, &mut visited, 1);
                max_chain = max_chain.max(length);
            }
        }

        max_chain as u32
    }
}

impl Descriptor for LongestAliphaticChain {
    fn name(&self) -> &str {
        "LongestAliphaticChain"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(Self::compute(mol))
    }
}

/// Largest chain through any atoms descriptor.
///
/// Finds the longest continuous chain through any atoms (not just carbons).
/// Returns Int(u32) with the length of the longest atom chain.
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

        for &neighbor_idx in &mol.adjacency[current as usize] {
            if !visited.contains(&neighbor_idx) {
                let length = Self::dfs_any(mol, neighbor_idx, visited, depth + 1);
                max_length = max_length.max(length);
            }
        }

        visited.remove(&current);
        max_length
    }

    fn compute(mol: &Molecule) -> u32 {
        let mut max_chain = 0;

        for i in 0..mol.atoms.len() {
            let mut visited = HashSet::new();
            let length = Self::dfs_any(mol, i as AtomIdx, &mut visited, 1);
            max_chain = max_chain.max(length);
        }

        max_chain as u32
    }
}

impl Descriptor for LargestChain {
    fn name(&self) -> &str {
        "LargestChain"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(Self::compute(mol))
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<LongestAliphaticChain>;
    let _ = assert_send_sync::<LargestChain>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::algo::valency::assign_implicit_hydrogens;
    use crate::io::smiles::parse_smiles;

    fn parse_and_prepare(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        perceive_aromaticity(&mut mol);
        mol
    }

    #[test]
    fn test_longest_aliphatic_chain_linear() {
        let mut mol = parse_and_prepare("CCCCCC");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_branched() {
        let mut mol = parse_and_prepare("CC(C)CCC");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_aromatic() {
        let mut mol = parse_and_prepare("CCCc1ccccc1");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol),
            DescriptorValue::Int(3)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_aromatic_only() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol),
            DescriptorValue::Int(0)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_complex() {
        let mut mol = parse_and_prepare("CCCCC(CC)CCCC");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol),
            DescriptorValue::Int(11)
        );
    }

    #[test]
    fn test_largest_chain_linear() {
        let mut mol = parse_and_prepare("CCCCCC");
        assert_eq!(LargestChain.calculate(&mut mol), DescriptorValue::Int(6));
    }

    #[test]
    fn test_largest_chain_aromatic() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(LargestChain.calculate(&mut mol), DescriptorValue::Int(6));
    }

    #[test]
    fn test_largest_chain_with_heteroatom() {
        let mut mol = parse_and_prepare("CCOCCOC");
        assert_eq!(LargestChain.calculate(&mut mol), DescriptorValue::Int(7));
    }

    #[test]
    fn test_largest_chain_branched() {
        let mut mol = parse_and_prepare("CC(C)(C)C");
        assert_eq!(LargestChain.calculate(&mut mol), DescriptorValue::Int(5));
    }

    #[test]
    fn test_descriptor_names() {
        assert_eq!(LongestAliphaticChain.name(), "LongestAliphaticChain");
        assert_eq!(LargestChain.name(), "LargestChain");
    }
}
