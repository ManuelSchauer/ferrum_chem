//! Structural chain descriptors (CDK 2.13 `LongestAliphaticChainDescriptor`
//! and `LargestChainDescriptor`, default parameters).
//!
//! - `nAtomLAC`: number of atoms in the longest chain of acyclic carbons
//!   (any hybridisation). Isolated acyclic carbons (no acyclic carbon
//!   neighbour) do not form a chain, so e.g. toluene gives 0.
//! - `nAtomLC`: number of atoms in the longest shortest-path between two
//!   acyclic non-hydrogen atoms (any element). A lone acyclic atom gives 0.
//!
//! Both subsets are forests (acyclic atoms cannot form a cycle), so the value is
//! the largest tree diameter (in atoms) over components with at least two atoms.
//!
//! Key symbols: `struct LongestAliphaticChain`, `struct LargestChain`,
//! `fn longest_chain_in_forest`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Number of atoms on the longest path in the forest induced by `include`,
/// counting only components with at least two atoms (CDK: 0 otherwise).
fn longest_chain_in_forest(mol: &Molecule, include: impl Fn(usize) -> bool) -> u32 {
    let n = mol.num_atoms();
    let inc: Vec<bool> = (0..n).map(&include).collect();
    // depth from BFS; u32::MAX = unvisited
    let mut dist = vec![u32::MAX; n];
    let mut comp_seen = vec![false; n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    let mut best = 0u32;

    // BFS over the included subgraph from `src`; returns (farthest atom, its depth)
    let bfs = |src: usize, dist: &mut Vec<u32>, queue: &mut Vec<usize>, mark: &mut Vec<bool>| -> (usize, u32) {
        queue.clear();
        queue.push(src);
        dist[src] = 0;
        let mut far = (src, 0u32);
        let mut head = 0;
        while head < queue.len() {
            let a = queue[head];
            head += 1;
            mark[a] = true;
            let d = dist[a];
            if d > far.1 {
                far = (a, d);
            }
            for &bi in &mol.adjacency[a] {
                let b = &mol.bonds[bi as usize];
                let w = if b.source as usize == a { b.target } else { b.source } as usize;
                if inc[w] && dist[w] == u32::MAX {
                    dist[w] = d + 1;
                    queue.push(w);
                }
            }
        }
        for &a in queue.iter() {
            dist[a] = u32::MAX;
        }
        far
    };

    let mut scratch = vec![false; n];
    for s in 0..n {
        if !inc[s] || comp_seen[s] {
            continue;
        }
        let (far, _) = bfs(s, &mut dist, &mut queue, &mut comp_seen);
        let (_, diam) = bfs(far, &mut dist, &mut queue, &mut scratch);
        if diam > 0 {
            best = best.max(diam + 1);
        }
    }
    best
}

/// Longest aliphatic chain (CDK `LongestAliphaticChainDescriptor`, `nAtomLAC`).
///
/// Longest chain of acyclic carbon atoms; returns Int(u32).
pub struct LongestAliphaticChain;

impl LongestAliphaticChain {
    /// Computes `nAtomLAC`.
    pub fn compute(mol: &Molecule) -> u32 {
        longest_chain_in_forest(mol, |a| {
            let at = &mol.atoms[a];
            at.atomic_number == 6 && !at.is_in_ring()
        })
    }
}

impl Descriptor for LongestAliphaticChain {
    fn name(&self) -> &str {
        "LongestAliphaticChain"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(Self::compute(mol))
    }
}

/// Largest chain (CDK `LargestChainDescriptor`, `nAtomLC`).
///
/// Longest shortest-path (in atoms) between acyclic heavy atoms; returns Int(u32).
pub struct LargestChain;

impl LargestChain {
    /// Computes `nAtomLC`.
    pub fn compute(mol: &Molecule) -> u32 {
        longest_chain_in_forest(mol, |a| {
            let at = &mol.atoms[a];
            at.atomic_number != 1 && !at.is_in_ring()
        })
    }
}

impl Descriptor for LargestChain {
    fn name(&self) -> &str {
        "LargestChain"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
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
    use crate::io::smiles::parse_smiles;

    fn lac(smiles: &str) -> u32 {
        LongestAliphaticChain::compute(&parse_smiles(smiles).unwrap())
    }

    fn lc(smiles: &str) -> u32 {
        LargestChain::compute(&parse_smiles(smiles).unwrap())
    }

    // Expected values verified against CDK 2.13.

    #[test]
    fn test_longest_aliphatic_chain() {
        assert_eq!(lac("CCCCCC"), 6);
        assert_eq!(lac("CC(C)CCC"), 5);
        assert_eq!(lac("CCCc1ccccc1"), 3);
        assert_eq!(lac("c1ccccc1"), 0);
        assert_eq!(lac("CCCCC(CC)CCCC"), 9);
        // isolated acyclic carbons are not chains
        assert_eq!(lac("Cc1ccccc1-c1ccccc1"), 0);
        assert_eq!(lac("CS(C)(=O)=O"), 0);
        // unsaturated acyclic carbons count
        assert_eq!(lac("C=C=O"), 2);
        assert_eq!(lac("COCCC#N"), 3);
    }

    #[test]
    fn test_largest_chain() {
        assert_eq!(lc("CCCCCC"), 6);
        assert_eq!(lc("c1ccccc1"), 0);
        assert_eq!(lc("CCOCCOC"), 7);
        assert_eq!(lc("CC(C)(C)C"), 3);
        assert_eq!(lc("CS(C)=O"), 3);
        assert_eq!(lc("C[NH3+]"), 2);
        assert_eq!(lc("[NH4+]"), 0);
    }

    #[test]
    fn test_descriptor_names() {
        assert_eq!(LongestAliphaticChain.name(), "LongestAliphaticChain");
        assert_eq!(LargestChain.name(), "LargestChain");
    }
}
