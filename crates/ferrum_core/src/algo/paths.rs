//! Paths algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.

//! ## File guide: Paths
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn find_all_paths_of_length`
//! - `fn dfs_paths_of_length_mask`
//! - `fn dfs_paths_of_length_vec`
//! - `mod tests`
//! - `fn test_triangle_counts_closed_three_bond_path_once`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Paths` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Paths` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Paths` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/paths.rs` defines paths behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Paths` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Paths` decisions.
//! - FerrumChem regression tests that pin `Paths` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/paths.rs`.

use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use rustc_hash::FxHashSet;

/// Finds all paths of length.
///
/// Function context: `crates/ferrum_core/src/algo/paths.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub fn find_all_paths_of_length<F>(mol: &Molecule, bond_length: usize, mut callback: F)
where
    F: FnMut(&[AtomIdx]),
{
    if mol.num_atoms() == 0 {
        return;
    }
    if bond_length == 0 {
        for idx in 0..mol.num_atoms() {
            callback(&[idx as AtomIdx]);
        }
        return;
    }

    // For common small/medium molecular graphs we can represent used bonds in a
    // compact 128-bit mask, eliminating Vec contains/clone/sort work in the DFS
    // inner loop while preserving current canonical path de-dup semantics.
    if mol.num_bonds() <= 128 {
        let mut paths_seen = FxHashSet::default();
        let mut atom_path = Vec::with_capacity(bond_length + 1);
        for start in 0..mol.num_atoms() {
            atom_path.clear();
            atom_path.push(start as AtomIdx);
            dfs_paths_of_length_mask(
                mol,
                &mut atom_path,
                0,
                0,
                bond_length,
                &mut paths_seen,
                &mut callback,
            );
        }
        return;
    }

    // Fallback for very large bond counts retains the previous generic keying
    // strategy based on sorted used-bond vectors.
    let mut paths_seen = FxHashSet::default();
    let mut atom_path = Vec::with_capacity(bond_length + 1);
    let mut used_bonds = Vec::with_capacity(bond_length);

    for start in 0..mol.num_atoms() {
        atom_path.clear();
        used_bonds.clear();
        atom_path.push(start as AtomIdx);
        dfs_paths_of_length_vec(
            mol,
            &mut atom_path,
            &mut used_bonds,
            bond_length,
            &mut paths_seen,
            &mut callback,
        );
    }
}

/// Implements paths of length mask.
///
/// Function context: `crates/ferrum_core/src/algo/paths.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn dfs_paths_of_length_mask<F>(
    mol: &Molecule,
    atom_path: &mut Vec<AtomIdx>,
    used_mask_lo: u128,
    depth: usize,
    bond_length: usize,
    seen: &mut FxHashSet<u128>,
    callback: &mut F,
) where
    F: FnMut(&[AtomIdx]),
{
    if depth == bond_length {
        if seen.insert(used_mask_lo) {
            callback(atom_path);
        }
        return;
    }

    let current = *atom_path.last().expect("atom_path is non-empty");
    for &bond_idx in &mol.adjacency[current as usize] {
        let bidx = bond_idx as usize;
        let bit = 1u128 << bidx;
        if (used_mask_lo & bit) != 0 {
            continue;
        }

        let bond = &mol.bonds[bidx];
        let neighbor = if bond.source == current {
            bond.target
        } else {
            bond.source
        };

        atom_path.push(neighbor);
        dfs_paths_of_length_mask(
            mol,
            atom_path,
            used_mask_lo | bit,
            depth + 1,
            bond_length,
            seen,
            callback,
        );
        atom_path.pop();
    }
}

/// Implements paths of length vec.
///
/// Function context: `crates/ferrum_core/src/algo/paths.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn dfs_paths_of_length_vec<F>(
    mol: &Molecule,
    atom_path: &mut Vec<AtomIdx>,
    used_bonds: &mut Vec<usize>,
    bond_length: usize,
    seen: &mut FxHashSet<Vec<usize>>,
    callback: &mut F,
) where
    F: FnMut(&[AtomIdx]),
{
    if used_bonds.len() == bond_length {
        let mut key = used_bonds.clone();
        key.sort_unstable();
        if seen.insert(key) {
            callback(atom_path);
        }
        return;
    }

    let current = *atom_path.last().expect("atom_path is non-empty");
    for &bond_idx in &mol.adjacency[current as usize] {
        let bidx = bond_idx as usize;
        if used_bonds.contains(&bidx) {
            continue;
        }

        let bond = &mol.bonds[bidx];
        let neighbor = if bond.source == current {
            bond.target
        } else {
            bond.source
        };

        used_bonds.push(bidx);
        atom_path.push(neighbor);
        dfs_paths_of_length_vec(mol, atom_path, used_bonds, bond_length, seen, callback);
        atom_path.pop();
        used_bonds.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::find_all_paths_of_length;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for triangle counts closed three bond path once.
    ///
    /// Function context: `crates/ferrum_core/src/algo/paths.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_triangle_counts_closed_three_bond_path_once() {
        let mol = parse_smiles("C1CC1").unwrap();
        let mut paths = 0;
        let mut last_len = 0;
        let mut same_ends = false;
        find_all_paths_of_length(&mol, 3, |path| {
            paths += 1;
            last_len = path.len();
            same_ends = path.first() == path.last();
        });
        assert_eq!(paths, 1);
        assert_eq!(last_len, 4);
        assert!(same_ends);
    }

    #[test]
    /// Runs regression test for cyclobutane counts open and closed paths like rdkit.
    ///
    /// Function context: `crates/ferrum_core/src/algo/paths.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cyclobutane_counts_open_and_closed_paths_like_rdkit() {
        let mol = parse_smiles("C1CCC1").unwrap();
        let mut paths_3 = 0;
        find_all_paths_of_length(&mol, 3, |_| paths_3 += 1);
        assert_eq!(paths_3, 4);

        let mut paths_4 = 0;
        find_all_paths_of_length(&mol, 4, |_| paths_4 += 1);
        assert_eq!(paths_4, 1);
    }
}

use crate::desc::context::CalculationContext;

/// Compute all-pairs unweighted shortest paths using BFS, utilizing the CalculationContext
/// to avoid repeated allocations, and caching the result on the Molecule.

/// Implements compute unweighted distance matrix cached.
///
/// Function context: `crates/ferrum_core/src/algo/paths.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub fn compute_unweighted_distance_matrix_cached<'a>(
    mol: &crate::mol::molecule::Molecule,
    ctx: &'a mut CalculationContext,
) -> &'a [u8] {
    let cache_key = crate::desc::context::molecule_cache_key(mol);
    if ctx.has_dist_u8 && ctx.active_molecule_key == Some(cache_key) {
        return &ctx.dist_u8;
    }
    if let Some(cached) = ctx.dist_u8_global_cache.get(&cache_key) {
        ctx.dist_u8.clear();
        ctx.dist_u8.extend_from_slice(cached);
        ctx.has_dist_u8 = true;
        ctx.active_molecule_key = Some(cache_key);
        return &ctx.dist_u8;
    }

    let n = mol.num_atoms();
    ctx.dist_u8.clear();
    ctx.dist_u8.resize(n * n, 255);

    if n > 0 {
        ctx.bfs_visited_u64.resize(n, 0);
        ctx.bfs_frontier_u64.resize(n, 0);
        ctx.bfs_next_frontier_u64.resize(n, 0);

        let mut chunk_start = 0;
        while chunk_start < n {
            let chunk_size = (n - chunk_start).min(64);
            
            ctx.bfs_visited_u64.fill(0);
            ctx.bfs_frontier_u64.fill(0);
            
            for i in 0..chunk_size {
                ctx.bfs_visited_u64[chunk_start + i] = 1 << i;
                ctx.bfs_frontier_u64[chunk_start + i] = 1 << i;
                ctx.dist_u8[(chunk_start + i) * n + (chunk_start + i)] = 0;
            }
            
            let mut d = 1u8;
            let mut any_active = true;
            
            while any_active {
                any_active = false;
                ctx.bfs_next_frontier_u64.fill(0);
                
                for u in 0..n {
                    let mut new_mask = 0u64;
                    for &bond_idx in &mol.adjacency[u] {
                        let bond = &mol.bonds[bond_idx as usize];
                        let v = if bond.source as usize == u {
                            bond.target as usize
                        } else {
                            bond.source as usize
                        };
                        new_mask |= ctx.bfs_frontier_u64[v];
                    }
                    new_mask &= !ctx.bfs_visited_u64[u];
                    
                    if new_mask != 0 {
                        ctx.bfs_next_frontier_u64[u] = new_mask;
                        ctx.bfs_visited_u64[u] |= new_mask;
                        any_active = true;
                        
                        let mut mask = new_mask;
                        while mask != 0 {
                            let tz = mask.trailing_zeros();
                            mask &= mask - 1; // clear lowest bit
                            let s = chunk_start + tz as usize;
                            ctx.dist_u8[s * n + u] = d;
                        }
                    }
                }
                
                std::mem::swap(&mut ctx.bfs_frontier_u64, &mut ctx.bfs_next_frontier_u64);
                if d < 254 {
                    d += 1;
                }
            }
            chunk_start += 64;
        }
    }

    ctx.has_dist_u8 = true;
    ctx.active_molecule_key = Some(cache_key);
    ctx.dist_u8_global_cache.insert(cache_key, ctx.dist_u8.clone());
    &ctx.dist_u8
}
