//! Mordred (mordred-community 2.0.7) walk counts and topological charge
//! indices on the hydrogen-suppressed graph (`Chem.RemoveHs`; hydrogens RDKit
//! keeps stay graph atoms).
//!
//! - `WalkCount` (21 values): `MWC01..MWC10`, `TMWC10`, `SRW02..SRW10`,
//!   `TSRW10`. With the adjacency matrix `A` and `A` atoms:
//!   `MWC01 = ½ Σ A` (number of bonds), `MWCk = ln(1 + Σ Aᵏ)`,
//!   `SRWk = ln(1 + tr Aᵏ)`; the totals add the atom count and the lower
//!   orders: `TMWC10 = A + Σ_{k=1..10} MWCk`, `TSRW10 = A + Σ_{k=1..10} SRWk`
//!   (`SRW01 = ln 1 = 0`).
//! - `TopologicalCharge` (21 values): `GGI1..GGI10`, `JGI1..JGI10`, `JGT10`.
//!   The charge term matrix is `CT = M - Mᵀ` with `M = A·D⁻²` (`D⁻²` the
//!   element-wise inverse squared distance matrix, zero diagonal);
//!   `GGIk = Σ_{i<j, d(i,j)=k} |CT_ij|`, `JGIk = GGIk / #pairs at distance k`
//!   (0 when there is none) and `JGT10 = Σ_{k=1..10} JGIk`.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::paths::{DIST_INF, compute_unweighted_distance_matrix_cached};
use crate::desc::context::CalculationContext;
use crate::mol::molecule::Molecule;

/// Longest walk of Mordred's WalkCount preset.
const MAX_WALK: usize = 10;

/// Number of WalkCount columns.
pub const MORDRED_WALK_COUNT_LEN: usize = 21;

/// WalkCount column names in Mordred's preset order.
pub fn mordred_walk_count_names() -> Vec<String> {
    let mut v: Vec<String> = (1..=MAX_WALK).map(|k| format!("MWC{k:02}")).collect();
    v.push(format!("TMWC{MAX_WALK:02}"));
    v.extend((2..=MAX_WALK).map(|k| format!("SRW{k:02}")));
    v.push(format!("TSRW{MAX_WALK:02}"));
    v
}

/// Neighbour lists (CSR) of the molecule graph.
fn csr(mol: &Molecule) -> (Vec<u32>, Vec<u32>) {
    let n = mol.num_atoms();
    let mut start = Vec::with_capacity(n + 1);
    let mut adj = Vec::with_capacity(2 * mol.num_bonds());
    for a in 0..n {
        start.push(adj.len() as u32);
        for &bi in &mol.adjacency[a] {
            let b = &mol.bonds[bi as usize];
            adj.push(if b.source as usize == a { b.target } else { b.source } as u32);
        }
    }
    start.push(adj.len() as u32);
    (start, adj)
}

/// `y = A·x`.
#[inline]
fn mul_adj(start: &[u32], adj: &[u32], x: &[f64], y: &mut [f64]) {
    for (i, yi) in y.iter_mut().enumerate() {
        let mut s = 0.0;
        for &j in &adj[start[i] as usize..start[i + 1] as usize] {
            s += x[j as usize];
        }
        *yi = s;
    }
}

/// Mordred WalkCount values in [`mordred_walk_count_names`] order.
pub fn mordred_walk_count(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    let (start, adj) = csr(mol);
    // Walk counts are integers; f64 holds them exactly far beyond any molecule.
    // Σ Aᵏ = 1ᵀ Aᵏ 1
    let mut total = [0.0f64; MAX_WALK + 1];
    let mut x = vec![1.0f64; n];
    let mut y = vec![0.0f64; n];
    for t in total.iter_mut().skip(1) {
        mul_adj(&start, &adj, &x, &mut y);
        std::mem::swap(&mut x, &mut y);
        *t = x.iter().sum();
    }
    // tr Aᵏ = Σ_i (Aʲ e_i)·(A^(k-j) e_i): powers up to 5 per atom
    const HALF: usize = MAX_WALK / 2;
    let mut trace = [0.0f64; MAX_WALK + 1];
    let mut pw: Vec<Vec<f64>> = vec![vec![0.0; n]; HALF + 1];
    for i in 0..n {
        pw[0].fill(0.0);
        pw[0][i] = 1.0;
        for p in 1..=HALF {
            let (lo, hi) = pw.split_at_mut(p);
            mul_adj(&start, &adj, &lo[p - 1], &mut hi[0]);
        }
        for (k, tr) in trace.iter_mut().enumerate().skip(2) {
            let (a, b) = (k / 2, k - k / 2);
            *tr += pw[a].iter().zip(&pw[b]).map(|(u, v)| u * v).sum::<f64>();
        }
    }

    let mut out = Vec::with_capacity(MORDRED_WALK_COUNT_LEN);
    let mwc1 = 0.5 * total[1];
    out.push(mwc1);
    out.extend((2..=MAX_WALK).map(|k| (total[k] + 1.0).ln()));
    let mut t = n as f64 + mwc1;
    for &v in &out[1..MAX_WALK] {
        t += v;
    }
    out.push(t);
    let srw: Vec<f64> = (2..=MAX_WALK).map(|k| (trace[k] + 1.0).ln()).collect();
    // SRW01 = ln(tr A + 1) = 0
    let mut t = n as f64;
    for &v in &srw {
        t += v;
    }
    out.extend(srw);
    out.push(t);
    out
}

/// Mordred WalkCount descriptor family (21 values, see the module docs).
pub struct MordredWalkCount;

impl Descriptor for MordredWalkCount {
    fn name(&self) -> &str {
        "MordredWalkCount"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_walk_count(mol))
    }
}

/// Largest distance of Mordred's TopologicalCharge preset.
const MAX_DIST: usize = 10;

/// Number of TopologicalCharge columns.
pub const MORDRED_TOPOLOGICAL_CHARGE_LEN: usize = 21;

/// TopologicalCharge column names in Mordred's preset order.
pub fn mordred_topological_charge_names() -> Vec<String> {
    let mut v: Vec<String> = (1..=MAX_DIST).map(|k| format!("GGI{k}")).collect();
    v.extend((1..=MAX_DIST).map(|k| format!("JGI{k}")));
    v.push(format!("JGT{MAX_DIST}"));
    v
}

/// Mordred TopologicalCharge values in [`mordred_topological_charge_names`] order.
pub fn mordred_topological_charge(mol: &Molecule, ctx: &mut CalculationContext) -> Vec<f64> {
    let n = mol.num_atoms();
    let (start, adj) = csr(mol);
    let dist = compute_unweighted_distance_matrix_cached(mol, ctx);
    // D⁻² row by row; unreachable pairs never meet a pair at distance <= 10
    // (RDKit's 1e8 gives 1e-16 there, which only enters unselected entries).
    let mut inv2 = vec![0.0f64; n * n];
    for (o, &d) in inv2.iter_mut().zip(dist.iter()) {
        if d != 0 && d != DIST_INF {
            *o = 1.0 / (d as f64 * d as f64);
        }
    }
    // M = A·D⁻²: row i is the sum of the rows of i's neighbours
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        let (row, rest) = (i * n, &adj[start[i] as usize..start[i + 1] as usize]);
        for &l in rest {
            let src = &inv2[l as usize * n..(l as usize + 1) * n];
            for (o, &v) in m[row..row + n].iter_mut().zip(src) {
                *o += v;
            }
        }
    }
    let mut ggi = [0.0f64; MAX_DIST + 1];
    let mut count = [0u64; MAX_DIST + 1];
    for i in 0..n {
        for j in 0..i {
            let d = dist[i * n + j] as usize;
            if d == 0 || d > MAX_DIST {
                continue;
            }
            ggi[d] += (m[i * n + j] - m[j * n + i]).abs();
            count[d] += 1;
        }
    }
    let mut out = Vec::with_capacity(MORDRED_TOPOLOGICAL_CHARGE_LEN);
    out.extend_from_slice(&ggi[1..]);
    let mut jgt = 0.0;
    for k in 1..=MAX_DIST {
        let jgi = if count[k] == 0 { 0.0 } else { ggi[k] / count[k] as f64 };
        jgt += jgi;
        out.push(jgi);
    }
    out.push(jgt);
    out
}

/// Mordred TopologicalCharge descriptor family (21 values, see the module docs).
pub struct MordredTopologicalCharge;

impl Descriptor for MordredTopologicalCharge {
    fn name(&self) -> &str {
        "MordredTopologicalCharge"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_topological_charge(mol, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn walk_counts_of_propane() {
        // path P3: Σ A² = 6 (degrees²: 1 + 4 + 1), tr A² = 4
        let v = mordred_walk_count(&parse_smiles("CCC").unwrap());
        assert_eq!(v[0], 2.0);
        assert!((v[1] - 7f64.ln()).abs() < 1e-12);
        assert!((v[11] - 5f64.ln()).abs() < 1e-12);
        // odd closed walks do not exist in a tree
        assert_eq!(v[12], 0.0);
    }

    #[test]
    fn topological_charge_of_propane() {
        // M_ij = Σ_{l ~ i} D⁻²(l, j)
        let mut ctx = CalculationContext::new();
        let v = mordred_topological_charge(&parse_smiles("CCC").unwrap(), &mut ctx);
        // M_10 = D⁻²(0,0) + D⁻²(2,0) = 1/4, M_01 = D⁻²(1,1) = 0 -> |CT| = 1/4 (twice)
        assert!((v[0] - 0.5).abs() < 1e-12);
        // M_20 = D⁻²(1,0) = 1, M_02 = D⁻²(1,2) = 1 -> 0
        assert_eq!(v[1], 0.0);
        assert!((v[10] - 0.25).abs() < 1e-12); // JGI1 = 0.5 / 2 pairs
    }
}
