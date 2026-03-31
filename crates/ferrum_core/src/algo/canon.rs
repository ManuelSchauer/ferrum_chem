//! Canon algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.


//! ## File guide: Canon
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn rank_atoms`
//! - `fn reorder_molecule`
//! - `mod tests`
//! - `fn test_rank_atoms_methane`
//! - `fn test_rank_atoms_ethane`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Canon` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Canon` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Canon` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/canon.rs` defines canon behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Canon` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Canon` decisions.
//! - FerrumChem regression tests that pin `Canon` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/canon.rs`.

use crate::mol::molecule::Molecule;

/// Morgan algorithm for atom ranking.
/// Iteratively refines atom priorities based on neighbor connectivity
/// until the ranking stabilizes.
pub fn rank_atoms(mol: &Molecule) -> Vec<usize> {
    let n = mol.atoms.len();
    if n == 0 {
        return vec![];
    }
    // Initial invariants: atomic number
    let mut rank: Vec<u64> = mol.atoms.iter().map(|a| a.atomic_number as u64).collect();

    // Iterate until stable
    for _ in 0..n {
        let mut new_rank = vec![0u64; n];
        for i in 0..n {
            // Collect sorted neighbor ranks for an order-independent but collision-resistant hash
            let mut nbr_ranks: Vec<u64> = Vec::new();
            for &bi in &mol.adjacency[i] {
                let b = &mol.bonds[bi as usize];
                let nbr = if b.source as usize == i {
                    b.target as usize
                } else {
                    b.source as usize
                };
                nbr_ranks.push(rank[nbr]);
            }
            nbr_ranks.sort();
            // Combine using a polynomial hash to reduce collisions
            let mut h = rank[i].wrapping_mul(1000003);
            for &nr in &nbr_ranks {
                h = h.wrapping_mul(1000003).wrapping_add(nr).wrapping_add(1);
            }
            new_rank[i] = h;
        }
        // Compress to sequential ranks
        let mut sorted: Vec<(u64, usize)> = new_rank
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        sorted.sort();
        let mut compressed = vec![0usize; n];
        let mut cur_rank = 0;
        for (k, &(val, idx)) in sorted.iter().enumerate() {
            if k > 0 && val != sorted[k - 1].0 {
                cur_rank += 1;
            }
            compressed[idx] = cur_rank;
        }
        let n_classes = cur_rank + 1;
        let old_classes = {
            let mut s: Vec<u64> = rank.clone();
            s.sort();
            s.dedup();
            s.len()
        };
        rank = compressed.iter().map(|&r| r as u64).collect();
        if n_classes == old_classes {
            break;
        }
    }
    rank.iter().map(|&r| r as usize).collect()
}

/// Reorder molecule atoms according to the given new ordering.
/// `new_order[i]` is the rank/priority of atom i; atoms are sorted by ascending rank.
pub fn reorder_molecule(mol: Molecule, new_order: &[usize]) -> Molecule {
    let n = mol.atoms.len();
    // Build permutation: sorted by rank
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by_key(|&i| new_order[i]);

    // Inverse map: old_idx -> new_idx
    let mut inv = vec![0usize; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        inv[old_i] = new_i;
    }

    // Reorder atoms
    let mut new_atoms = Vec::with_capacity(n);
    for &old_i in &perm {
        new_atoms.push(mol.atoms[old_i].clone());
    }

    // Remap bonds
    let mut new_bonds = Vec::with_capacity(mol.bonds.len());
    for b in &mol.bonds {
        let mut nb = b.clone();
        nb.source = inv[b.source as usize] as u16;
        nb.target = inv[b.target as usize] as u16;
        new_bonds.push(nb);
    }

    // Rebuild adjacency
    let mut adj = vec![smallvec::smallvec![]; n];
    for (bi, b) in new_bonds.iter().enumerate() {
        adj[b.source as usize].push(bi as u16);
        adj[b.target as usize].push(bi as u16);
    }

    Molecule::from_parts(new_atoms, new_bonds, adj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for rank atoms methane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/canon.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rank_atoms_methane() {
        let mol = parse_smiles("C").unwrap();
        let ranks = rank_atoms(&mol);
        assert_eq!(ranks.len(), 1);
    }

    #[test]
    /// Runs regression test for rank atoms ethane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/canon.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rank_atoms_ethane() {
        let mol = parse_smiles("CC").unwrap();
        let ranks = rank_atoms(&mol);
        assert_eq!(ranks[0], ranks[1]); // symmetric
    }

    #[test]
    /// Runs regression test for rank atoms propane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/canon.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rank_atoms_propane() {
        let mol = parse_smiles("CCC").unwrap();
        let ranks = rank_atoms(&mol);
        // Terminal carbons should have same rank, central different
        assert_eq!(ranks[0], ranks[2]);
        assert_ne!(ranks[0], ranks[1]);
    }

    #[test]
    /// Runs regression test for rank atoms ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/algo/canon.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rank_atoms_ethanol() {
        let mol = parse_smiles("CCO").unwrap();
        let ranks = rank_atoms(&mol);
        // All three atoms should have different ranks
        assert_ne!(ranks[0], ranks[1]);
        assert_ne!(ranks[1], ranks[2]);
        assert_ne!(ranks[0], ranks[2]);
    }

    #[test]
    /// Runs regression test for reorder preserves structure.
    ///
    /// Function context: `crates/ferrum_core/src/algo/canon.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_reorder_preserves_structure() {
        let mol = parse_smiles("CCO").unwrap();
        let ranks = rank_atoms(&mol);
        let reordered = reorder_molecule(mol, &ranks);
        assert_eq!(reordered.atoms.len(), 3);
        assert_eq!(reordered.bonds.len(), 2);
        // Each atom should still have correct degree
        for i in 0..3 {
            assert!(!reordered.adjacency[i].is_empty());
        }
    }

    #[test]
    /// Runs regression test for reorder benzene.
    ///
    /// Function context: `crates/ferrum_core/src/algo/canon.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_reorder_benzene() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        let ranks = rank_atoms(&mol);
        let reordered = reorder_molecule(mol, &ranks);
        assert_eq!(reordered.atoms.len(), 6);
        assert_eq!(reordered.bonds.len(), 6);
    }
}
