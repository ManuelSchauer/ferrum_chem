use crate::mol::molecule::Molecule;

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

/// Bond-order weighted distance matrix (RDKit `getDistanceMat(useBO=true)`),
/// `INFINITY` for unreachable pairs.
///
/// Computed exactly in integer sixths (see [`bond_order_distances_sixths`]);
/// a floating-point Floyd-Warshall (RDKit) differs from `D / 6` only by the
/// rounding of its partial sums (a few ulp).
fn compute_distance_matrix_weighted(mol: &Molecule) -> Vec<f64> {
    let mut sixths = Vec::new();
    bond_order_distances_sixths(mol, &mut sixths);
    sixths
        .iter()
        .map(|&d| if d == BO_DIST_INF { f64::INFINITY } else { d as f64 / 6.0 })
        .collect()
}

/// Unreachable marker of [`bond_order_distances_sixths`].
pub const BO_DIST_INF: u32 = u32::MAX;

/// Bond weight `1 / bond order` in sixths: single (and dative/unknown) 6,
/// aromatic 4, double 3, triple 2.
#[inline]
fn bond_weight_sixths(bond: &crate::mol::bond::Bond) -> u32 {
    if bond.is_aromatic() || bond.bond_type == crate::types::BondType::Aromatic {
        4
    } else {
        match bond.bond_type {
            crate::types::BondType::Double => 3,
            crate::types::BondType::Triple => 2,
            _ => 6,
        }
    }
}

/// All-pairs bond-order weighted distances in units of 1/6 (row-major,
/// [`BO_DIST_INF`] = unreachable): every bond weight `1 / order` is a
/// multiple of 1/6, so the distances are exact integers.
///
/// Pendant trees are peeled off first (repeatedly removing degree-1 atoms):
/// every path from a peeled atom leaves through the atom it hung on, so its
/// row is that atom's row plus the bond weight. The remaining core gets one
/// Dial (bucket queue) shortest-path search per atom, O(core (m + max distance)).
pub fn bond_order_distances_sixths(mol: &Molecule, out: &mut Vec<u32>) {
    let n = mol.num_atoms();
    out.clear();
    out.resize(n * n, BO_DIST_INF);
    if n == 0 {
        return;
    }
    let mut start = Vec::with_capacity(n + 1);
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(2 * mol.num_bonds());
    for u in 0..n {
        start.push(edges.len());
        for &bi in &mol.adjacency[u] {
            let b = &mol.bonds[bi as usize];
            let v = if b.source as usize == u { b.target } else { b.source };
            edges.push((v as u32, bond_weight_sixths(b)));
        }
    }
    start.push(edges.len());

    // peel degree-1 atoms; `peeled` holds (atom, parent, weight) in peel order
    let mut deg: Vec<u32> = (0..n).map(|u| (start[u + 1] - start[u]) as u32).collect();
    let mut in_core = vec![true; n];
    let mut peeled: Vec<(u32, u32, u32)> = Vec::new();
    let mut stack: Vec<u32> = (0..n as u32).filter(|&u| deg[u as usize] == 1).collect();
    while let Some(x) = stack.pop() {
        let x = x as usize;
        if !in_core[x] || deg[x] != 1 {
            continue;
        }
        let &(p, w) = edges[start[x]..start[x + 1]]
            .iter()
            .find(|&&(p, _)| in_core[p as usize])
            .expect("degree-1 atom has a core neighbour");
        in_core[x] = false;
        peeled.push((x as u32, p, w));
        let p = p as usize;
        deg[p] -= 1;
        if deg[p] == 1 {
            stack.push(p as u32);
        }
    }

    let core: Vec<u32> = (0..n as u32).filter(|&u| in_core[u as usize]).collect();
    if core.len() <= CORE_FW_LIMIT {
        core_floyd_warshall(&core, &start, &edges, n, out);
    } else {
        core_dial(&in_core, &deg, &start, &edges, n, out);
    }

    // re-attach the pendant atoms, innermost first
    let mut known = core;
    for &(x, p, w) in peeled.iter().rev() {
        let (x, p) = (x as usize, p as usize);
        out[x * n + x] = 0;
        for &j in &known {
            let j = j as usize;
            let dp = out[p * n + j];
            let d = if dp == BO_DIST_INF { BO_DIST_INF } else { dp + w };
            out[x * n + j] = d;
            out[j * n + x] = d;
        }
        known.push(x as u32);
    }
}

/// Cores up to this size use the (vectorised) integer Floyd-Warshall, which
/// beats per-source Dial searches well beyond typical molecule sizes.
const CORE_FW_LIMIT: usize = 256;

/// Integer Floyd-Warshall over the (compacted) core atoms.
fn core_floyd_warshall(core: &[u32], start: &[usize], edges: &[(u32, u32)], n: usize, out: &mut [u32]) {
    const INF: u32 = 1 << 30;
    let c = core.len();
    let mut local = vec![u32::MAX; n];
    for (k, &a) in core.iter().enumerate() {
        local[a as usize] = k as u32;
    }
    let mut m = vec![INF; c * c];
    for (i, &a) in core.iter().enumerate() {
        m[i * c + i] = 0;
        for &(v, w) in &edges[start[a as usize]..start[a as usize + 1]] {
            let j = local[v as usize];
            if j != u32::MAX {
                let e = &mut m[i * c + j as usize];
                *e = (*e).min(w);
            }
        }
    }
    for k in 0..c {
        let (before, rest) = m.split_at_mut(k * c);
        let (row_k, after) = rest.split_at_mut(c);
        for row in before.chunks_exact_mut(c).chain(after.chunks_exact_mut(c)) {
            let dik = row[k];
            if dik >= INF {
                continue;
            }
            for (x, &y) in row.iter_mut().zip(row_k.iter()) {
                *x = (*x).min(dik + y);
            }
        }
    }
    for (i, &a) in core.iter().enumerate() {
        let orow = &mut out[a as usize * n..(a as usize + 1) * n];
        for (j, &b) in core.iter().enumerate() {
            let d = m[i * c + j];
            orow[b as usize] = if d >= INF { BO_DIST_INF } else { d };
        }
    }
}

/// One Dial (bucket queue) shortest-path search per core atom. Weights are
/// 2..=6, so a circular queue of 7 buckets suffices and a relaxation never
/// lands in the bucket being drained.
fn core_dial(in_core: &[bool], deg: &[u32], start: &[usize], edges: &[(u32, u32)], n: usize, out: &mut [u32]) {
    let mut buckets: [Vec<u32>; 7] = Default::default();
    let mut current: Vec<u32> = Vec::new();
    for s in (0..n).filter(|&s| in_core[s]) {
        let row = &mut out[s * n..(s + 1) * n];
        row[s] = 0;
        if deg[s] == 0 {
            continue;
        }
        buckets[0].push(s as u32);
        let mut pending = 1usize;
        let mut d = 0u32;
        while pending > 0 {
            let b = (d % 7) as usize;
            std::mem::swap(&mut current, &mut buckets[b]);
            pending -= current.len();
            for &u in &current {
                let u = u as usize;
                if row[u] != d {
                    continue; // stale entry
                }
                for &(v, w) in &edges[start[u]..start[u + 1]] {
                    let nd = d + w;
                    if nd < row[v as usize] && in_core[v as usize] {
                        row[v as usize] = nd;
                        buckets[(nd % 7) as usize].push(v);
                        pending += 1;
                    }
                }
            }
            current.clear();
            d += 1;
        }
    }
}

/// [`bond_order_distances_sixths`] computed once per molecule and shared
/// through the calculation context (BalabanJ, BertzCT).
pub fn bond_order_distances_sixths_cached<'a>(
    mol: &Molecule,
    ctx: &'a mut crate::desc::context::CalculationContext,
) -> &'a [u32] {
    let key = crate::desc::context::molecule_cache_key(mol);
    if ctx.bo_dist6_key != Some(key) {
        bond_order_distances_sixths(mol, &mut ctx.bo_dist6);
        ctx.bo_dist6_key = Some(key);
    }
    &ctx.bo_dist6
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Floating-point Floyd-Warshall over `1 / order` weights (RDKit semantics).
    fn floyd_warshall(mol: &Molecule) -> Vec<f64> {
        let n = mol.num_atoms();
        let mut dist = vec![f64::INFINITY; n * n];
        for i in 0..n {
            dist[i * n + i] = 0.0;
        }
        for b in &mol.bonds {
            let w = 1.0 / (6.0 / bond_weight_sixths(b) as f64);
            dist[b.source as usize * n + b.target as usize] = w;
            dist[b.target as usize * n + b.source as usize] = w;
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let v = dist[i * n + k] + dist[k * n + j];
                    if v < dist[i * n + j] {
                        dist[i * n + j] = v;
                    }
                }
            }
        }
        dist
    }

    #[test]
    fn bond_order_distances_match_floyd_warshall() {
        for smi in [
            "C", "CC", "C=C", "C#CC=CC", "c1ccccc1", "c1ccc2ccccc2c1C#N", "CC.O", "[Na+].[Cl-]",
            "C1=CC=CC=C1C(=O)OC#C", "O=C1C=CC(=O)C=C1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "C12C3C4C1C5C2C3C45", "OC(=O)C=CC=CC=CC#CC#N", "CC(C)(C)C(C)(C)C=O.N", "C1CC1CC(C=O)c1ccccc1.C",
            "OCC1OC(OC2C(O)C(O)C(O)OC2CO)C(O)C(O)C1O", "[Cl-].C[N+](C)(C)CC#C",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let fw = floyd_warshall(&mol);
            let fast = compute_distance_matrix(&mol, true);
            for (a, b) in fw.iter().zip(&fast) {
                assert!(a == b || (a - b).abs() <= 1e-12 * a.abs(), "{smi}: {a} vs {b}");
            }
        }
    }
}
