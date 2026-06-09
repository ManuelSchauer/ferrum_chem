use crate::mol::molecule::Molecule;
use smallvec::SmallVec;
use std::collections::VecDeque;

/// Compute the distance matrix using optimized kernels.
/// With use_bo=true: weight = 1/bond_order (aromatic=1.5, double=2, triple=3).
/// With use_bo=false: weight = 1.0 for all bonds.
pub fn compute_distance_matrix(mol: &Molecule, use_bo: bool) -> Vec<f64> {
    if !use_bo {
        compute_distance_matrix_unweighted(mol)
    } else {
        compute_distance_matrix_weighted(mol)
    }
}

fn compute_distance_matrix_unweighted(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut dist = vec![f64::INFINITY; n * n];
    if n == 0 {
        return dist;
    }

    // Build flat adjacency list (CSR format) for cache-friendly access
    let mut flat_adj = Vec::with_capacity(mol.num_bonds() * 2);
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0);

    for u in 0..n {
        for &bond_idx in &mol.adjacency[u] {
            let bond = &mol.bonds[bond_idx as usize];
            let neighbor = if bond.source as usize == u {
                bond.target as usize
            } else {
                bond.source as usize
            };
            flat_adj.push(neighbor);
        }
        offsets.push(flat_adj.len());
    }

    // Flat arrays for level-synchronous BFS queue
    let mut q = vec![0; n];
    let mut q_next = vec![0; n];

    for start in 0..n {
        let start_offset = start * n;
        dist[start_offset + start] = 0.0;
        
        let mut q_len = 1;
        q[0] = start;
        let mut d = 1.0;

        while q_len > 0 {
            let mut next_len = 0;
            // Iterate over all nodes in the current BFS level
            for i in 0..q_len {
                let u = q[i];
                let start_idx = offsets[u];
                let end_idx = offsets[u + 1];

                // Vectorizable inner loop over neighbors
                for &neighbor in &flat_adj[start_idx..end_idx] {
                    if dist[start_offset + neighbor] == f64::INFINITY {
                        dist[start_offset + neighbor] = d;
                        q_next[next_len] = neighbor;
                        next_len += 1;
                    }
                }
            }
            std::mem::swap(&mut q, &mut q_next);
            q_len = next_len;
            d += 1.0;
        }
    }
    dist
}

#[inline]
fn bond_weight(bond: &crate::mol::bond::Bond) -> f64 {
    let bond_order = if bond.is_aromatic() || bond.bond_type == crate::types::BondType::Aromatic {
        1.5
    } else {
        match bond.bond_type {
            crate::types::BondType::Double => 2.0,
            crate::types::BondType::Triple => 3.0,
            _ => 1.0,
        }
    };
    1.0 / bond_order
}

/// Weighted Dijkstra / Floyd-Warshall kernel for weighted distance matrices.
fn compute_distance_matrix_weighted(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    if n == 0 {
        return Vec::new();
    }

    if n <= 64 {
        // Use Floyd-Warshall for small dense graphs (tight loop, vectorizable)
        let mut dist = vec![f64::INFINITY; n * n];
        for i in 0..n {
            dist[i * n + i] = 0.0;
        }
        for bond in &mol.bonds {
            let src = bond.source as usize;
            let dst = bond.target as usize;
            let w = bond_weight(bond);
            dist[src * n + dst] = w;
            dist[dst * n + src] = w;
        }

        // FW loop
        for k in 0..n {
            let k_n = k * n;
            for i in 0..n {
                let i_n = i * n;
                let dik = dist[i_n + k];
                if dik == f64::INFINITY { continue; }
                for j in 0..n {
                    let dkj = dist[k_n + j];
                    if dkj == f64::INFINITY { continue; }
                    let new_dist = dik + dkj;
                    if new_dist < dist[i_n + j] {
                        dist[i_n + j] = new_dist;
                    }
                }
            }
        }
        dist
    } else {
        // Dense Dijkstra for larger graphs
        let mut neighbors = vec![SmallVec::<[(usize, f64); 4]>::new(); n];
        for bond in &mol.bonds {
            let src = bond.source as usize;
            let dst = bond.target as usize;
            let w = bond_weight(bond);
            neighbors[src].push((dst, w));
            neighbors[dst].push((src, w));
        }

        let mut dist = vec![f64::INFINITY; n * n];
        let mut row = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];

        for start in 0..n {
            row.fill(f64::INFINITY);
            visited.fill(false);
            row[start] = 0.0;

            for _ in 0..n {
                let mut min_dist = f64::INFINITY;
                let mut u = None;
                for v in 0..n {
                    if !visited[v] && row[v] < min_dist {
                        min_dist = row[v];
                        u = Some(v);
                    }
                }
                let Some(u) = u else {
                    break;
                };
                visited[u] = true;

                let base = row[u];
                for &(neighbor, weight) in &neighbors[u] {
                    let new_dist = base + weight;
                    if new_dist < row[neighbor] {
                        row[neighbor] = new_dist;
                    }
                }
            }

            let start_offset = start * n;
            dist[start_offset..start_offset + n].copy_from_slice(&row);
        }
        dist
    }
}
