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

//! ## File guide: Chains
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct LongestAliphaticChain`
//! - `fn neighbor_atom_idx`
//! - `fn farthest_aliphatic`
//! - `fn component_vertices`
//! - `fn tree_diameter_or_cycle_bound`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Chains` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Chains` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/chains.rs` defines chains behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Chains` workflows.
//! - Repository unit and parity tests covering `Chains` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/chains.rs`.

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
    /// Implements atom idx.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn neighbor_atom_idx(mol: &Molecule, atom_idx: AtomIdx, bond_idx: u16) -> AtomIdx {
        let bond = &mol.bonds[bond_idx as usize];
        if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        }
    }

    /// Implements aliphatic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Implements vertices.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Implements diameter or cycle bound.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Computes the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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
            let vertices =
                Self::component_vertices(mol, i as AtomIdx, &allowed, &mut comp_mark, &mut queue);
            let len =
                Self::tree_diameter_or_cycle_bound(mol, &vertices, &allowed, &mut dist, &mut queue);
            best = best.max(len);
        }

        best.max(0) as u32
    }
}

impl Descriptor for LongestAliphaticChain {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "LongestAliphaticChain"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(Self::compute(mol))
    }
}

/// Largest chain through any atoms descriptor.
///
/// Finds the longest continuous chain through any atoms (not just carbons).
/// Returns Int(u32) with the length of the longest atom chain.
pub struct LargestChain;

impl LargestChain {
    /// Implements any.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Implements vertices.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Implements diameter or cycle bound.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Implements atom idx.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn neighbor_atom_idx(mol: &Molecule, atom_idx: AtomIdx, bond_idx: u16) -> AtomIdx {
        let bond = &mol.bonds[bond_idx as usize];
        if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        }
    }

    /// Computes the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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
            let vertices =
                Self::component_vertices(mol, i as AtomIdx, &allowed, &mut comp_mark, &mut queue);
            let len =
                Self::tree_diameter_or_cycle_bound(mol, &vertices, &allowed, &mut dist, &mut queue);
            best = best.max(len);
        }

        best.max(0) as u32
    }
}

impl Descriptor for LargestChain {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "LargestChain"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(Self::compute(mol))
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Parses and prepare.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse_and_prepare(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        perceive_aromaticity(&mut mol);
        mol
    }

    #[test]
    /// Runs regression test for longest aliphatic chain linear.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_longest_aliphatic_chain_linear() {
        let mut mol = parse_and_prepare("CCCCCC");
        assert_eq!(
            LongestAliphaticChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    /// Runs regression test for longest aliphatic chain branched.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_longest_aliphatic_chain_branched() {
        let mut mol = parse_and_prepare("CC(C)CCC");
        assert_eq!(
            LongestAliphaticChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(5)
        );
    }

    #[test]
    /// Runs regression test for longest aliphatic chain aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_longest_aliphatic_chain_aromatic() {
        let mut mol = parse_and_prepare("CCCc1ccccc1");
        assert_eq!(
            LongestAliphaticChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(3)
        );
    }

    #[test]
    /// Runs regression test for longest aliphatic chain aromatic only.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_longest_aliphatic_chain_aromatic_only() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(
            LongestAliphaticChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(0)
        );
    }

    #[test]
    /// Runs regression test for longest aliphatic chain complex.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_longest_aliphatic_chain_complex() {
        let mut mol = parse_and_prepare("CCCCC(CC)CCCC");
        assert_eq!(
            LongestAliphaticChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(9)
        );
    }

    #[test]
    /// Runs regression test for largest chain linear.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_largest_chain_linear() {
        let mut mol = parse_and_prepare("CCCCCC");
        assert_eq!(
            LargestChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    /// Runs regression test for largest chain aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_largest_chain_aromatic() {
        let mut mol = parse_and_prepare("c1ccccc1");
        assert_eq!(
            LargestChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(6)
        );
    }

    #[test]
    /// Runs regression test for largest chain with heteroatom.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_largest_chain_with_heteroatom() {
        let mut mol = parse_and_prepare("CCOCCOC");
        assert_eq!(
            LargestChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(7)
        );
    }

    #[test]
    /// Runs regression test for largest chain branched.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_largest_chain_branched() {
        let mut mol = parse_and_prepare("CC(C)(C)C");
        assert_eq!(
            LargestChain.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(3)
        );
    }

    #[test]
    /// Runs regression test for descriptor names.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chains.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_descriptor_names() {
        assert_eq!(LongestAliphaticChain.name(), "LongestAliphaticChain");
        assert_eq!(LargestChain.name(), "LargestChain");
    }
}
