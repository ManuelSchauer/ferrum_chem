use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;

pub fn find_all_paths_of_length(mol: &Molecule, bond_length: usize) -> Vec<Vec<AtomIdx>> {
    if mol.num_atoms() == 0 {
        return Vec::new();
    }
    if bond_length == 0 {
        return (0..mol.num_atoms())
            .map(|idx| vec![idx as AtomIdx])
            .collect();
    }

    // For common small/medium molecular graphs we can represent used bonds in a
    // compact 128-bit mask, eliminating Vec contains/clone/sort work in the DFS
    // inner loop while preserving current canonical path de-dup semantics.
    if mol.num_bonds() <= 128 {
        let mut paths: Vec<(u128, Vec<AtomIdx>)> = Vec::new();
        let mut atom_path = Vec::with_capacity(bond_length + 1);
        for start in 0..mol.num_atoms() {
            atom_path.clear();
            atom_path.push(start as AtomIdx);
            dfs_paths_of_length_mask(mol, &mut atom_path, 0, 0, bond_length, &mut paths);
        }
        paths.sort_unstable_by_key(|(mask, _)| *mask);
        paths.dedup_by(|(left, _), (right, _)| left == right);
        return paths.into_iter().map(|(_, path)| path).collect();
    }

    // Fallback for very large bond counts retains the previous generic keying
    // strategy based on sorted used-bond vectors.
    let mut paths: Vec<(Vec<usize>, Vec<AtomIdx>)> = Vec::new();
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
            &mut paths,
        );
    }

    paths.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    paths.dedup_by(|(left, _), (right, _)| left == right);
    paths.into_iter().map(|(_, path)| path).collect()
}

fn dfs_paths_of_length_mask(
    mol: &Molecule,
    atom_path: &mut Vec<AtomIdx>,
    used_mask_lo: u128,
    depth: usize,
    bond_length: usize,
    out: &mut Vec<(u128, Vec<AtomIdx>)>,
) {
    if depth == bond_length {
        out.push((used_mask_lo, atom_path.clone()));
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
            out,
        );
        atom_path.pop();
    }
}

fn dfs_paths_of_length_vec(
    mol: &Molecule,
    atom_path: &mut Vec<AtomIdx>,
    used_bonds: &mut Vec<usize>,
    bond_length: usize,
    out: &mut Vec<(Vec<usize>, Vec<AtomIdx>)>,
) {
    if used_bonds.len() == bond_length {
        let mut key = used_bonds.clone();
        key.sort_unstable();
        out.push((key, atom_path.clone()));
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
        dfs_paths_of_length_vec(mol, atom_path, used_bonds, bond_length, out);
        atom_path.pop();
        used_bonds.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::find_all_paths_of_length;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_triangle_counts_closed_three_bond_path_once() {
        let mol = parse_smiles("C1CC1").unwrap();
        let paths = find_all_paths_of_length(&mol, 3);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 4);
        assert_eq!(paths[0].first(), paths[0].last());
    }

    #[test]
    fn test_cyclobutane_counts_open_and_closed_paths_like_rdkit() {
        let mol = parse_smiles("C1CCC1").unwrap();
        assert_eq!(find_all_paths_of_length(&mol, 3).len(), 4);
        assert_eq!(find_all_paths_of_length(&mol, 4).len(), 1);
    }
}

use crate::desc::context::CalculationContext;

/// Compute all-pairs unweighted shortest paths using BFS, utilizing the CalculationContext
/// to avoid repeated allocations, and caching the result on the Molecule.

pub fn compute_unweighted_distance_matrix_cached<'a>(mol: &crate::mol::molecule::Molecule, ctx: &'a mut CalculationContext) -> &'a [u32] {
    if ctx.has_dist_u32 {
        return &ctx.dist_u32;
    }

    let n = mol.num_atoms();
    ctx.dist_u32.clear();
    ctx.dist_u32.resize(n * n, u32::MAX);
    
    if n > 0 {
        ctx.bfs_queue.reserve(n);
        ctx.bfs_dist.resize(n, u32::MAX as usize);

        for start in 0..n {
            ctx.bfs_queue.clear();
            ctx.bfs_dist.fill(u32::MAX as usize);
            
            ctx.bfs_dist[start] = 0;
            ctx.bfs_queue.push(start);
            
            let mut head = 0;
            while head < ctx.bfs_queue.len() {
                let current = ctx.bfs_queue[head];
                head += 1;
                
                let current_dist = ctx.bfs_dist[current];
                let next_dist = current_dist + 1;
                
                for &bond_idx in &mol.adjacency[current] {
                    let bond = &mol.bonds[bond_idx as usize];
                    let neighbor = if bond.source as usize == current {
                        bond.target as usize
                    } else {
                        bond.source as usize
                    };
                    
                    if ctx.bfs_dist[neighbor] == u32::MAX as usize {
                        ctx.bfs_dist[neighbor] = next_dist;
                        ctx.bfs_queue.push(neighbor);
                    }
                }
            }
            
            let start_offset = start * n;
            for i in 0..n {
                ctx.dist_u32[start_offset + i] = ctx.bfs_dist[i] as u32;
            }
        }
    }

    ctx.has_dist_u32 = true;
    &ctx.dist_u32
}
