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
use std::collections::VecDeque;

/// Longest aliphatic (sp3 carbon) chain descriptor.
///
/// Finds the longest continuous chain of sp3-hybridized carbons (non-aromatic, single bonds).
/// Returns Int(u32) with the length of the longest aliphatic carbon chain.
pub struct LongestAliphaticChain;

impl LongestAliphaticChain {
    fn neighbor_atom_idx(mol: &Molecule, atom_idx: AtomIdx, bond_idx: u16) -> AtomIdx {
        let bond = &mol.bonds[bond_idx as usize];
        if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        }
    }

    fn farthest_aliphatic(
        mol: &Molecule,
        start: AtomIdx,
        allowed: &[bool],
        dist: &mut [i32],
        queue: &mut VecDeque<AtomIdx>,
    ) -> (AtomIdx, i32) {
        dist.fill(-1);
        queue.clear();
        dist[start as usize] = 0;
        queue.push_back(start);

        let mut far = start;
        let mut far_dist = 0i32;
        while let Some(node) = queue.pop_front() {
            let d = dist[node as usize];
            if d > far_dist {
                far_dist = d;
                far = node;
            }
            for &bond_idx in &mol.adjacency[node as usize] {
                let nei = Self::neighbor_atom_idx(mol, node, bond_idx);
                let ni = nei as usize;
                if !allowed[ni] || dist[ni] != -1 {
                    continue;
                }
                dist[ni] = d + 1;
                queue.push_back(nei);
            }
        }
        (far, far_dist)
    }

    fn component_vertices(
        mol: &Molecule,
        start: AtomIdx,
        allowed: &[bool],
        comp_mark: &mut [bool],
        queue: &mut VecDeque<AtomIdx>,
    ) -> Vec<AtomIdx> {
        queue.clear();
        queue.push_back(start);
        comp_mark[start as usize] = true;

        let mut out = Vec::new();
        while let Some(node) = queue.pop_front() {
            out.push(node);
            for &bond_idx in &mol.adjacency[node as usize] {
                let nei = Self::neighbor_atom_idx(mol, node, bond_idx);
                let ni = nei as usize;
                if !allowed[ni] || comp_mark[ni] {
                    continue;
                }
                comp_mark[ni] = true;
                queue.push_back(nei);
            }
        }
        out
    }

    fn tree_diameter_or_cycle_bound(
        mol: &Molecule,
        vertices: &[AtomIdx],
        allowed: &[bool],
        dist: &mut [i32],
        queue: &mut VecDeque<AtomIdx>,
    ) -> i32 {
        if vertices.is_empty() {
            return 0;
        }

        let mut edge_twice = 0usize;
        for &v in vertices {
            edge_twice += mol.adjacency[v as usize]
                .iter()
                .filter(|&&bond_idx| {
                    let nei = Self::neighbor_atom_idx(mol, v, bond_idx);
                    allowed[nei as usize]
                })
                .count();
        }
        let edges = edge_twice / 2;

        // Exact tree diameter if component is acyclic; otherwise use a tight linear-time upper bound.
        if edges + 1 == vertices.len() {
            let start = vertices[0];
            let (u, _) = Self::farthest_aliphatic(mol, start, allowed, dist, queue);
            let (_, d) = Self::farthest_aliphatic(mol, u, allowed, dist, queue);
            d + 1
        } else {
            vertices.len() as i32
        }
    }

    fn compute(mol: &Molecule) -> u32 {
        let n = mol.atoms.len();
        if n == 0 {
            return 0;
        }

        let mut allowed = vec![false; n];
        for (i, atom) in mol.atoms.iter().enumerate() {
            allowed[i] = atom.atomic_number == 6 && !atom.is_aromatic();
        }

        let mut comp_mark = vec![false; n];
        let mut dist = vec![-1; n];
        let mut queue = VecDeque::new();
        let mut best = 0i32;

        for i in 0..n {
            if !allowed[i] || comp_mark[i] {
                continue;
            }
            let vertices = Self::component_vertices(
                mol,
                i as AtomIdx,
                &allowed,
                &mut comp_mark,
                &mut queue,
            );
            let len =
                Self::tree_diameter_or_cycle_bound(mol, &vertices, &allowed, &mut dist, &mut queue);
            best = best.max(len);
        }

        best.max(0) as u32
    }
}

impl Descriptor for LongestAliphaticChain {
    fn name(&self) -> &str {
        "LongestAliphaticChain"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Int(Self::compute(mol))
    }
}

/// Largest chain through any atoms descriptor.
///
/// Finds the longest continuous chain through any atoms (not just carbons).
/// Returns Int(u32) with the length of the longest atom chain.
pub struct LargestChain;

impl LargestChain {
    fn farthest_any(
        mol: &Molecule,
        start: AtomIdx,
        allowed: &[bool],
        dist: &mut [i32],
        queue: &mut VecDeque<AtomIdx>,
    ) -> (AtomIdx, i32) {
        dist.fill(-1);
        queue.clear();
        dist[start as usize] = 0;
        queue.push_back(start);

        let mut far = start;
        let mut far_dist = 0i32;
        while let Some(node) = queue.pop_front() {
            let d = dist[node as usize];
            if d > far_dist {
                far_dist = d;
                far = node;
            }
            for &bond_idx in &mol.adjacency[node as usize] {
                let nei = Self::neighbor_atom_idx(mol, node, bond_idx);
                let ni = nei as usize;
                if !allowed[ni] || dist[ni] != -1 {
                    continue;
                }
                dist[ni] = d + 1;
                queue.push_back(nei);
            }
        }
        (far, far_dist)
    }

    fn component_vertices(
        mol: &Molecule,
        start: AtomIdx,
        allowed: &[bool],
        comp_mark: &mut [bool],
        queue: &mut VecDeque<AtomIdx>,
    ) -> Vec<AtomIdx> {
        queue.clear();
        queue.push_back(start);
        comp_mark[start as usize] = true;

        let mut out = Vec::new();
        while let Some(node) = queue.pop_front() {
            out.push(node);
            for &bond_idx in &mol.adjacency[node as usize] {
                let nei = Self::neighbor_atom_idx(mol, node, bond_idx);
                let ni = nei as usize;
                if !allowed[ni] || comp_mark[ni] {
                    continue;
                }
                comp_mark[ni] = true;
                queue.push_back(nei);
            }
        }
        out
    }

    fn tree_diameter_or_cycle_bound(
        mol: &Molecule,
        vertices: &[AtomIdx],
        allowed: &[bool],
        dist: &mut [i32],
        queue: &mut VecDeque<AtomIdx>,
    ) -> i32 {
        if vertices.is_empty() {
            return 0;
        }

        let mut edge_twice = 0usize;
        for &v in vertices {
            edge_twice += mol.adjacency[v as usize]
                .iter()
                .filter(|&&bond_idx| {
                    let nei = Self::neighbor_atom_idx(mol, v, bond_idx);
                    allowed[nei as usize]
                })
                .count();
        }
        let edges = edge_twice / 2;

        if edges + 1 == vertices.len() {
            let start = vertices[0];
            let (u, _) = Self::farthest_any(mol, start, allowed, dist, queue);
            let (_, d) = Self::farthest_any(mol, u, allowed, dist, queue);
            d + 1
        } else {
            vertices.len() as i32
        }
    }

    fn neighbor_atom_idx(mol: &Molecule, atom_idx: AtomIdx, bond_idx: u16) -> AtomIdx {
        let bond = &mol.bonds[bond_idx as usize];
        if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        }
    }

    fn compute(mol: &Molecule) -> u32 {
        let n = mol.atoms.len();
        if n == 0 {
            return 0;
        }

        let allowed = vec![true; n];
        let mut comp_mark = vec![false; n];
        let mut dist = vec![-1; n];
        let mut queue = VecDeque::new();
        let mut best = 0i32;

        for i in 0..n {
            if comp_mark[i] {
                continue;
            }
            let vertices = Self::component_vertices(
                mol,
                i as AtomIdx,
                &allowed,
                &mut comp_mark,
                &mut queue,
            );
            let len =
                Self::tree_diameter_or_cycle_bound(mol, &vertices, &allowed, &mut dist, &mut queue);
            best = best.max(len);
        }

        best.max(0) as u32
    }
}

impl Descriptor for LargestChain {
    fn name(&self) -> &str {
        "LargestChain"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
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
            LongestAliphaticChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_branched() {
        let mut mol = parse_and_prepare("CC(C)CCC");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Int(5)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_aromatic() {
        let mut mol = parse_and_prepare("CCCc1ccccc1");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Int(3)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_aromatic_only() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Int(0)
        );
    }

    #[test]
    fn test_longest_aliphatic_chain_complex() {
        let mut mol = parse_and_prepare("CCCCC(CC)CCCC");
        assert_eq!(
            LongestAliphaticChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Int(9)
        );
    }

    #[test]
    fn test_largest_chain_linear() {
        let mut mol = parse_and_prepare("CCCCCC");
        assert_eq!(LargestChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()), DescriptorValue::Int(6));
    }

    #[test]
    fn test_largest_chain_aromatic() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(LargestChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()), DescriptorValue::Int(6));
    }

    #[test]
    fn test_largest_chain_with_heteroatom() {
        let mut mol = parse_and_prepare("CCOCCOC");
        assert_eq!(LargestChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()), DescriptorValue::Int(7));
    }

    #[test]
    fn test_largest_chain_branched() {
        let mut mol = parse_and_prepare("CC(C)(C)C");
        assert_eq!(LargestChain.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()), DescriptorValue::Int(3));
    }

    #[test]
    fn test_descriptor_names() {
        assert_eq!(LongestAliphaticChain.name(), "LongestAliphaticChain");
        assert_eq!(LargestChain.name(), "LargestChain");
    }
}
