//! Paths algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.

use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use rustc_hash::FxHashSet;

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

/// Longest bond path kept in the per-molecule path table
/// ([`cached_paths_of_length`]).
pub const MAX_CACHED_PATH_BONDS: usize = 4;

/// [`find_all_paths_of_length`] for every length `1..=max_bonds` in one
/// depth-first search: `out[len]` receives the atom paths (`len + 1` atoms
/// each, flattened) in exactly the order, orientation and number in which
/// `find_all_paths_of_length(mol, len, ..)` reports them.
///
/// A trail without repeated atoms is a simple path, found once from each
/// end; it is reported when walked from its lower end atom, which is how the
/// bond-set de-duplication sees it first. Only trails that revisit an atom
/// (rings of at most `max_bonds` bonds) need the bond-set hash.
pub fn enumerate_paths_up_to(mol: &Molecule, max_bonds: usize, out: &mut Vec<Vec<AtomIdx>>) {
    out.resize_with(max_bonds + 1, Vec::new);
    for v in out.iter_mut() {
        v.clear();
    }
    let n = mol.num_atoms();
    if n == 0 || max_bonds == 0 {
        return;
    }
    struct Walk<'a> {
        mol: &'a Molecule,
        max: usize,
        atoms: Vec<AtomIdx>,
        bonds: Vec<u32>,
        on_path: Vec<u16>,
        seen: Vec<rustc_hash::FxHashSet<[u32; MAX_CACHED_PATH_BONDS]>>,
    }
    fn dfs(w: &mut Walk, out: &mut [Vec<AtomIdx>], simple: bool) {
        let current = *w.atoms.last().expect("non-empty path") as usize;
        let mol = w.mol;
        for &bond_idx in &mol.adjacency[current] {
            let b = bond_idx as u32;
            if w.bonds.contains(&b) {
                continue;
            }
            let bond = &mol.bonds[b as usize];
            let nb = if bond.source as usize == current { bond.target } else { bond.source };
            let still_simple = simple && w.on_path[nb as usize] == 0;
            w.atoms.push(nb);
            w.bonds.push(b);
            w.on_path[nb as usize] += 1;
            let len = w.bonds.len();
            let report = if still_simple {
                w.atoms[0] < nb
            } else {
                let mut key = [u32::MAX; MAX_CACHED_PATH_BONDS];
                key[..len].copy_from_slice(&w.bonds);
                key[..len].sort_unstable();
                w.seen[len].insert(key)
            };
            if report {
                out[len].extend_from_slice(&w.atoms);
            }
            if len < w.max {
                dfs(w, out, still_simple);
            }
            w.on_path[nb as usize] -= 1;
            w.bonds.pop();
            w.atoms.pop();
        }
    }
    let max = max_bonds.min(MAX_CACHED_PATH_BONDS);
    let mut w = Walk {
        mol,
        max,
        atoms: Vec::with_capacity(max + 1),
        bonds: Vec::with_capacity(max),
        on_path: vec![0; n],
        seen: vec![rustc_hash::FxHashSet::default(); max + 1],
    };
    for start in 0..n {
        w.atoms.push(start as AtomIdx);
        w.on_path[start] = 1;
        dfs(&mut w, out, true);
        w.on_path[start] = 0;
        w.atoms.pop();
    }
}

/// Atom paths of `bond_length` bonds (flattened, `bond_length + 1` atoms each)
/// as [`find_all_paths_of_length`] reports them, enumerated once per molecule
/// for all lengths up to [`MAX_CACHED_PATH_BONDS`] and shared through the
/// calculation context (ChiIndices, ChiNIndices, KappaIndices).
pub fn cached_paths_of_length<'a>(mol: &Molecule, ctx: &'a mut CalculationContext, bond_length: usize) -> &'a [AtomIdx] {
    assert!((1..=MAX_CACHED_PATH_BONDS).contains(&bond_length), "path length {bond_length} not cached");
    let key = crate::desc::context::molecule_cache_key(mol);
    if ctx.paths_key != Some(key) {
        enumerate_paths_up_to(mol, MAX_CACHED_PATH_BONDS, &mut ctx.paths_by_len);
        ctx.paths_key = Some(key);
    }
    &ctx.paths_by_len[bond_length]
}

#[cfg(test)]
mod tests {
    use super::find_all_paths_of_length;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn enumerate_paths_up_to_matches_find_all_paths_of_length() {
        for smi in [
            "C", "CC", "CCO", "C1CC1", "C1CCC1", "C1CC1C", "CC1(C)CC1", "c1ccccc1", "C12C3C4C1C5C2C3C45",
            "CC1(C)C2CC=CC1C2", "CS(=O)(=O)c1ccc(Cl)cc1", "[H]C([H])([H])C1CC1.O", "C1C2CC12",
            "OC1C2CC3CC1CC(C2)C3", "C1CC2(C1)CC2",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let mut table = Vec::new();
            super::enumerate_paths_up_to(&mol, 4, &mut table);
            for len in 1..=4 {
                let mut expected: Vec<crate::types::AtomIdx> = Vec::new();
                find_all_paths_of_length(&mol, len, |p| expected.extend_from_slice(p));
                assert_eq!(table[len], expected, "{smi} length {len}");
            }
        }
    }

    #[test]
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

/// Sentinel for "no path" (disconnected atoms) in the topological distance matrix.
///
/// Distances are stored as `u16`; since atom indices are `u16`, any real
/// shortest path is strictly smaller than this value.
pub const DIST_INF: u16 = u16::MAX;

/// Compute all-pairs unweighted shortest paths, reusing the workspace in
/// `CalculationContext`.
///
/// Pendant trees are peeled off first (repeatedly removing degree-1 atoms):
/// a peeled atom's row is the row of the atom it hangs on plus one. The
/// remaining core is searched with a 64-source bitset BFS over a compact
/// core-only adjacency.
///
/// The matrix is cached in the context for the active molecule, so all
/// descriptors of one molecule share a single computation.
pub fn compute_unweighted_distance_matrix_cached<'a>(
    mol: &crate::mol::molecule::Molecule,
    ctx: &'a mut CalculationContext,
) -> &'a [u16] {
    let cache_key = crate::desc::context::molecule_cache_key(mol);
    if ctx.has_dist && ctx.active_molecule_key == Some(cache_key) {
        return &ctx.dist;
    }

    let n = mol.num_atoms();
    ctx.dist.clear();
    ctx.dist.resize(n * n, DIST_INF);
    if n > 0 {
        unweighted_distances(mol, ctx);
    }

    ctx.has_dist = true;
    ctx.active_molecule_key = Some(cache_key);
    &ctx.dist
}

fn unweighted_distances(mol: &crate::mol::molecule::Molecule, ctx: &mut CalculationContext) {
    let n = mol.num_atoms();
    let dist = &mut ctx.dist;
    let other = |a: usize, bi: u16| {
        let b = &mol.bonds[bi as usize];
        if b.source as usize == a { b.target as usize } else { b.source as usize }
    };

    // peel degree-1 atoms; `peeled` holds (atom, parent) in peel order
    let mut deg: Vec<u32> = (0..n).map(|a| mol.adjacency[a].len() as u32).collect();
    let mut in_core = vec![true; n];
    let mut peeled: Vec<(u32, u32)> = Vec::new();
    let mut stack: Vec<u32> = (0..n as u32).filter(|&a| deg[a as usize] == 1).collect();
    while let Some(x) = stack.pop() {
        let x = x as usize;
        if !in_core[x] || deg[x] != 1 {
            continue;
        }
        let p = mol.adjacency[x]
            .iter()
            .map(|&bi| other(x, bi))
            .find(|&p| in_core[p])
            .expect("degree-1 atom has a core neighbour");
        in_core[x] = false;
        peeled.push((x as u32, p as u32));
        deg[p] -= 1;
        if deg[p] == 1 {
            stack.push(p as u32);
        }
    }

    // compact core adjacency
    let core: Vec<u32> = (0..n as u32).filter(|&a| in_core[a as usize]).collect();
    let c = core.len();
    let mut local = vec![u32::MAX; n];
    for (i, &a) in core.iter().enumerate() {
        local[a as usize] = i as u32;
    }
    let mut start = Vec::with_capacity(c + 1);
    let mut adj: Vec<u32> = Vec::with_capacity(2 * mol.num_bonds());
    for &a in &core {
        start.push(adj.len());
        for &bi in &mol.adjacency[a as usize] {
            let l = local[other(a as usize, bi)];
            if l != u32::MAX {
                adj.push(l);
            }
        }
    }
    start.push(adj.len());

    // 64-source bitset BFS over the core
    let visited = &mut ctx.bfs_visited_u64;
    let frontier = &mut ctx.bfs_frontier_u64;
    let next_frontier = &mut ctx.bfs_next_frontier_u64;
    visited.clear();
    visited.resize(c, 0);
    frontier.clear();
    frontier.resize(c, 0);
    next_frontier.clear();
    next_frontier.resize(c, 0);
    let mut chunk_start = 0;
    while chunk_start < c {
        let chunk_size = (c - chunk_start).min(64);
        visited.fill(0);
        frontier.fill(0);
        for i in 0..chunk_size {
            visited[chunk_start + i] = 1 << i;
            frontier[chunk_start + i] = 1 << i;
            let a = core[chunk_start + i] as usize;
            dist[a * n + a] = 0;
        }
        let mut d: u16 = 1;
        let mut any_active = true;
        while any_active {
            any_active = false;
            for u in 0..c {
                let mut new_mask = 0u64;
                for &v in &adj[start[u]..start[u + 1]] {
                    new_mask |= frontier[v as usize];
                }
                new_mask &= !visited[u];
                next_frontier[u] = new_mask;
                if new_mask != 0 {
                    visited[u] |= new_mask;
                    any_active = true;
                    let col = core[u] as usize;
                    let mut mask = new_mask;
                    while mask != 0 {
                        let tz = mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        dist[core[chunk_start + tz] as usize * n + col] = d;
                    }
                }
            }
            std::mem::swap(frontier, next_frontier);
            d += 1;
        }
        chunk_start += 64;
    }

    // re-attach the pendant atoms, innermost first
    let mut known = core;
    for &(x, p) in peeled.iter().rev() {
        let (x, p) = (x as usize, p as usize);
        dist[x * n + x] = 0;
        for &j in &known {
            let j = j as usize;
            let dp = dist[p * n + j];
            let d = if dp == DIST_INF { DIST_INF } else { dp + 1 };
            dist[x * n + j] = d;
            dist[j * n + x] = d;
        }
        known.push(x as u32);
    }
}

#[cfg(test)]
mod distance_tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn unweighted_distances_match_bfs() {
        for smi in [
            "C", "CC", "CCO", "C1CC1", "c1ccccc1", "CC(C)(C)CC(=O)O", "C1CC1.C1CCC1", "[Na+].[Cl-]", "CC.O",
            "C12C3C4C1C5C2C3C45", "CS(=O)(=O)c1ccc(Cl)cc1", "OC1C2CC3CC1CC(C2)C3",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let n = mol.num_atoms();
            let mut ctx = CalculationContext::new();
            let fast = compute_unweighted_distance_matrix_cached(&mol, &mut ctx).to_vec();
            for s in 0..n {
                let mut d = vec![DIST_INF; n];
                d[s] = 0;
                let mut queue = std::collections::VecDeque::from([s]);
                while let Some(u) = queue.pop_front() {
                    for &bi in &mol.adjacency[u] {
                        let b = &mol.bonds[bi as usize];
                        let v = if b.source as usize == u { b.target as usize } else { b.source as usize };
                        if d[v] == DIST_INF {
                            d[v] = d[u] + 1;
                            queue.push_back(v);
                        }
                    }
                }
                assert_eq!(&fast[s * n..(s + 1) * n], &d[..], "{smi} row {s}");
            }
        }
    }
}
