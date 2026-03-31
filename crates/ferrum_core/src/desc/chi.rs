//! Kier-Hall Chi (χ) molecular connectivity indices.
//!
//! Chi indices encode molecular branching and shape using the valence delta
//! (δᵥ = valence electrons - H count) as atom weights.
//!
//! Five descriptor classes matching CDK/RDKit:
//! - **ChiPath** (χ⁰..χ⁷): connectivity over linear paths of length 0-7
//! - **ChiValencePath** (χᵥ⁰..χᵥ⁷): uses valence delta instead of simple degree
//! - **ChiCluster** (χ³c, χ⁴c): star/cluster subgraphs
//! - **ChiChain** (χ³ch..χ⁷ch): chain subgraphs (paths minus clusters)
//! - **ChiPathCluster** (χ⁴pc, χ⁵pc, χ⁶pc): Y-shaped subgraphs
//!
//! Total: 26 values.
//!
//! Reference: Kier, L.B. & Hall, L.H. (1986) "Molecular Connectivity in
//!            Structure-Activity Analysis", Research Studies Press.


//! ## File guide: Chi
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn simple_deltas`
//! - `fn valence_deltas`
//! - `fn chi0`
//! - `fn chi1`
//! - `fn chi_path`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - Visited-bitset DFS enumerates simple paths/clusters without revisiting atoms, preserving Kier-Hall semantics.
//! - Running-product accumulation assumes numerical stability in f64 for higher-order chi terms.
//! - Path and cluster caches in `calculate` must remain coherent with each order-specific helper.
//! ### Chemistry/ML references
//! - Kier, L. B. & Hall, L. H. Molecular connectivity chi indices.
//! - Hall/Kier valence-modified connectivity formulations used in cheminformatics toolkits.
//! - RDKit Chi descriptor outputs used as practical parity targets.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::paths::find_all_paths_of_length;
use crate::mol::elements::outer_electrons;
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;

/// Compute simple delta (graph degree) for each heavy atom.
/// δ_i = number of heavy-atom neighbors (i.e., graph degree).
fn simple_deltas(mol: &Molecule) -> Vec<f64> {
    (0..mol.num_atoms())
        .map(|i| mol.get_degree(i as AtomIdx) as f64)
        .collect()
}

/// Compute valence delta for each heavy atom.
/// δᵥ = (Zᵥ - Hᵢ) / (Z - Zᵥ - 1)
/// where Zᵥ = valence electrons, Hᵢ = total H count, Z = atomic number.
/// For hydrogen or denominator=0, returns 0.
fn valence_deltas(mol: &Molecule) -> Vec<f64> {
    mol.atoms
        .iter()
        .map(|atom| {
            let z = atom.atomic_number as f64;
            let zv = outer_electrons(atom.atomic_number) as f64;
            let h = atom.total_h() as f64;
            let denom = z - zv - 1.0;
            if denom <= 0.0 || atom.atomic_number <= 1 {
                // For H and 1st-row elements where Z-Zv-1=0, use simple delta
                // δᵥ(C) = (4-H)/(6-4-1) = (4-H)/1 = 4-H
                // For H itself: return 0
                if atom.atomic_number == 1 {
                    return 0.0;
                }
                // 1st row: C(6),N(7),O(8),F(9) → denom=1,1,1,1
                // But O has Z-Zv-1 = 8-6-1=1, N has 7-5-1=1, C has 6-4-1=1
                // Only H (1-1-1=-1) truly fails. For other elements where
                // denom is exactly 0 (shouldn't happen for common elements), fallback:
                zv - h
            } else {
                (zv - h) / denom
            }
        })
        .collect()
}

/// Chi-0 (zero-order): χ⁰ = Σ 1/√δᵢ for each atom with δ > 0.
fn chi0(deltas: &[f64]) -> f64 {
    deltas
        .iter()
        .filter(|&&d| d > 0.0)
        .map(|&d| 1.0 / d.sqrt())
        .sum()
}

/// Chi-1 (first-order): χ¹ = Σ 1/√(δᵢ·δⱼ) over all bonds.
fn chi1(mol: &Molecule, deltas: &[f64]) -> f64 {
    mol.bonds
        .iter()
        .map(|bond| {
            let di = deltas[bond.source as usize];
            let dj = deltas[bond.target as usize];
            let prod = di * dj;
            if prod > 0.0 {
                1.0 / prod.sqrt()
            } else {
                0.0
            }
        })
        .sum()
}

/// Find all paths of exactly `length` bonds and compute chi contribution.
/// χⁿ = Σ 1/√(δ₁·δ₂·...·δₙ₊₁) over all n-bond paths.
fn chi_path(mol: &Molecule, deltas: &[f64], length: usize) -> f64 {
    if length == 0 {
        return chi0(deltas);
    }
    if length == 1 {
        return chi1(mol, deltas);
    }

    let n = mol.num_atoms();
    if n < length + 1 {
        return 0.0;
    }

    // DFS to enumerate all paths of exact length
    let mut sum = 0.0f64;
    let mut visited = vec![false; n];

    for start in 0..n {
        let d0 = deltas[start];
        if d0 <= 0.0 {
            continue;
        }
        visited[start] = true;
        dfs_paths_acc(
            mol,
            deltas,
            start as AtomIdx,
            1,
            length + 1,
            d0,
            &mut visited,
            &mut sum,
        );
        visited[start] = false;
    }

    // Each path is found twice (forward + reverse), so divide by 2
    sum / 2.0
}

/// Implements path rdkit like.
///
/// Function context: `crates/ferrum_core/src/desc/chi.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn chi_path_rdkit_like(mol: &Molecule, deltas: &[f64], length: usize) -> f64 {
    let mut sum = 0.0;
    find_all_paths_of_length(mol, length, |path| {
        let atoms = if path.first() == path.last() {
            &path[..path.len() - 1]
        } else {
            &path[..]
        };

        let mut prod = 1.0f64;
        for &idx in atoms {
            prod *= deltas[idx as usize];
        }
        if prod > 0.0 {
            sum += 1.0 / prod.sqrt();
        }
    });
    sum
}

/// Implements paths acc.
///
/// Function context: `crates/ferrum_core/src/desc/chi.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn dfs_paths_acc(
    mol: &Molecule,
    deltas: &[f64],
    current: AtomIdx,
    depth: usize,
    target_depth: usize,
    prod: f64,
    visited: &mut [bool],
    sum: &mut f64,
) {
    if depth == target_depth {
        *sum += 1.0 / prod.sqrt();
        return;
    }

    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let neighbor = if bond.source == current {
            bond.target
        } else {
            bond.source
        };

        let ni = neighbor as usize;
        if visited[ni] {
            continue;
        }
        let d = deltas[ni];
        if d <= 0.0 {
            continue;
        }
        visited[ni] = true;
        dfs_paths_acc(
            mol,
            deltas,
            neighbor,
            depth + 1,
            target_depth,
            prod * d,
            visited,
            sum,
        );
        visited[ni] = false;
    }
}

// (Legacy descriptors removed — use ChiIndices below)

/// Compute cluster chi for star subgraphs of given order.
/// Order 3: central atom with 3 neighbors chosen → C(deg, 3) combinations.
/// Order 4: central atom with 4 neighbors chosen → C(deg, 4) combinations.
fn chi_cluster(mol: &Molecule, deltas: &[f64], order: usize) -> f64 {
    let n = mol.num_atoms();
    let k = order; // need k neighbors around the center
    let mut sum = 0.0;

    for center in 0..n {
        let deg = mol.get_degree(center as AtomIdx);
        if deg < k {
            continue;
        }

        let dc = deltas[center];
        if dc <= 0.0 {
            continue;
        }

        // Collect neighbor deltas
        let neighbor_deltas: Vec<f64> = mol.adjacency[center]
            .iter()
            .map(|&bi| {
                let bond = &mol.bonds[bi as usize];
                let nb = if bond.source == center as AtomIdx {
                    bond.target
                } else {
                    bond.source
                };
                deltas[nb as usize]
            })
            .collect();

        sum += sum_cluster_combinations(&neighbor_deltas, k, dc);
    }

    sum
}

/// Accumulates cluster combinations.
///
/// Function context: `crates/ferrum_core/src/desc/chi.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn sum_cluster_combinations(neighbor_deltas: &[f64], k: usize, center_delta: f64) -> f64 {
    if k == 0 || neighbor_deltas.len() < k || center_delta <= 0.0 {
        return 0.0;
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn recurse(
        vals: &[f64],
        k: usize,
        start: usize,
        picked: usize,
        prod: f64,
        acc: &mut f64,
    ) {
        if picked == k {
            *acc += 1.0 / prod.sqrt();
            return;
        }
        let need = k - picked;
        if vals.len() < need || start > vals.len() - need {
            return;
        }
        let last_start = vals.len() - need;
        for i in start..=last_start {
            let d = vals[i];
            if d <= 0.0 {
                continue;
            }
            recurse(vals, k, i + 1, picked + 1, prod * d, acc);
        }
    }

    let mut sum = 0.0;
    recurse(neighbor_deltas, k, 0, 0, center_delta, &mut sum);
    sum
}

#[derive(Clone)]
struct ArmPath {
    atoms: Vec<AtomIdx>,
    prod: f64,
}

/// Compute path-cluster chi for arbitrary order.
/// A path-cluster of order n is an n-atom connected subgraph with n-1 bonds
/// where exactly one atom has degree 3 in the subgraph (branch point).
/// This is a "T" or "Y" shaped subgraph.
///
/// For order 4: B has 3 neighbors A, C, D (all at distance 1 from B).
/// For order 5+: B has 2 neighbors + one arm extends further.
///
/// We enumerate by finding all connected subgraphs of size `order` with
/// exactly one node of degree 3 (and no node of degree > 3).
fn chi_pathcluster_order(mol: &Molecule, deltas: &[f64], order: usize) -> f64 {
    if order < 4 {
        return 0.0;
    }

    let n = mol.num_atoms();
    let mut sum = 0.0;

    // For each potential branch point B with degree >= 3
    for b in 0..n {
        let b_idx = b as AtomIdx;
        let b_neighbors: Vec<AtomIdx> = mol.adjacency[b]
            .iter()
            .map(|&bi| {
                let bond = &mol.bonds[bi as usize];
                if bond.source == b_idx {
                    bond.target
                } else {
                    bond.source
                }
            })
            .collect();

        if b_neighbors.len() < 3 {
            continue;
        }

        let db = deltas[b];
        if db <= 0.0 {
            continue;
        }
        let combos = combinations3(&b_neighbors);
        if combos.is_empty() {
            continue;
        }

        // Pick 3 arms from B's neighbors, then extend each arm to total `order` atoms
        // The branch point + 3 arms must total `order` atoms.
        // Arm lengths: a1 + a2 + a3 = order - 1, each arm >= 1.
        // Enumerate all (a1, a2, a3) partitions with a1 >= a2 >= a3 >= 1.
        let remaining = order - 1; // atoms distributed across 3 arms

        for a1 in 1..=remaining {
            for a2 in 1..=(remaining - a1) {
                let a3 = remaining - a1 - a2;
                if a3 < 1 || a2 > a1 || a3 > a2 {
                    continue;
                }

                // Skip pure clusters: all arms of length 1 = star subgraph
                if a1 == 1 && a2 == 1 && a3 == 1 {
                    continue;
                }

                // Generate all distinct permutations of arm lengths
                let perms = distinct_permutations(a1, a2, a3);

                // For each combination of 3 neighbors as arm starts
                for combo in &combos {
                    let arm_starts = [combo[0], combo[1], combo[2]];

                    for perm in &perms {
                        let arm_lens = [perm[0], perm[1], perm[2]];
                        let val = extend_arms_chi(mol, deltas, b_idx, db, &arm_starts, &arm_lens);
                        sum += val;
                    }
                }
            }
        }
    }

    // Each subgraph is found once per branch-point (branch point is unique)
    sum
}

/// Extend 3 arms from a branch point, computing chi contribution.
/// Returns sum of 1/√(product of deltas) for all valid extensions.
fn extend_arms_chi(
    mol: &Molecule,
    deltas: &[f64],
    branch: AtomIdx,
    d_branch: f64,
    starts: &[AtomIdx; 3],
    lengths: &[usize; 3],
) -> f64 {
    if d_branch <= 0.0 {
        return 0.0;
    }

    let n = mol.num_atoms();
    // Build arm paths via DFS, then combine.
    let mut arm_paths: [Vec<ArmPath>; 3] = [vec![], vec![], vec![]];

    for arm in 0..3 {
        let start = starts[arm] as usize;
        let d_start = deltas[start];
        if d_start <= 0.0 {
            return 0.0;
        }

        let mut visited = vec![false; n];
        visited[branch as usize] = true;
        // Exclude other arm starts to prevent overlap.
        for other in 0..3 {
            if other != arm {
                visited[starts[other] as usize] = true;
            }
        }
        visited[start] = true;
        let mut path = vec![starts[arm]];
        find_arm_paths_acc(
            mol,
            deltas,
            lengths[arm],
            d_start,
            &mut path,
            &mut visited,
            &mut arm_paths[arm],
        );
    }

    let mut sum = 0.0;
    let mut used = vec![false; n];

    // Combine all valid arm combinations.
    for p0 in &arm_paths[0] {
        for &a in &p0.atoms {
            used[a as usize] = true;
        }
        for p1 in &arm_paths[1] {
            if p1.atoms.iter().any(|&a| used[a as usize]) {
                continue;
            }
            for &a in &p1.atoms {
                used[a as usize] = true;
            }
            for p2 in &arm_paths[2] {
                if p2.atoms.iter().any(|&a| used[a as usize]) {
                    continue;
                }
                let prod = d_branch * p0.prod * p1.prod * p2.prod;
                if prod > 0.0 {
                    sum += 1.0 / prod.sqrt();
                }
            }
            for &a in &p1.atoms {
                used[a as usize] = false;
            }
        }
        for &a in &p0.atoms {
            used[a as usize] = false;
        }
    }

    sum
}

/// Find all simple arm paths of exactly `target_len` atoms.
fn find_arm_paths_acc(
    mol: &Molecule,
    deltas: &[f64],
    target_len: usize,
    current_prod: f64,
    path: &mut Vec<AtomIdx>,
    visited: &mut [bool],
    results: &mut Vec<ArmPath>,
) {
    if path.len() == target_len {
        results.push(ArmPath {
            atoms: path.clone(),
            prod: current_prod,
        });
        return;
    }

    let current = *path.last().expect("path non-empty");
    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let next = if bond.source == current {
            bond.target
        } else {
            bond.source
        };
        let ni = next as usize;
        if visited[ni] {
            continue;
        }
        let d = deltas[ni];
        if d <= 0.0 {
            continue;
        }
        visited[ni] = true;
        path.push(next);
        find_arm_paths_acc(
            mol,
            deltas,
            target_len,
            current_prod * d,
            path,
            visited,
            results,
        );
        path.pop();
        visited[ni] = false;
    }
}

/// Generate combinations of 3 elements from a slice.
fn combinations3(items: &[AtomIdx]) -> Vec<[AtomIdx; 3]> {
    let mut out = Vec::new();
    if items.len() < 3 {
        return out;
    }
    for i in 0..(items.len() - 2) {
        for j in (i + 1)..(items.len() - 1) {
            for k in (j + 1)..items.len() {
                out.push([items[i], items[j], items[k]]);
            }
        }
    }
    out
}

/// Generate all distinct permutations of three values.
fn distinct_permutations(a: usize, b: usize, c: usize) -> Vec<[usize; 3]> {
    let mut perms = vec![];
    let candidates = [
        [a, b, c],
        [a, c, b],
        [b, a, c],
        [b, c, a],
        [c, a, b],
        [c, b, a],
    ];
    for p in &candidates {
        if !perms.contains(p) {
            perms.push(*p);
        }
    }
    perms
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements chi.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_chi(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match ChiIndices.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for chi0 ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi0_ethane() {
        let c = calc_chi("CC");
        // Each C has degree 1: χ⁰ = 1/√1 + 1/√1 = 2
        assert!((c[0] - 2.0).abs() < 0.01, "χ⁰ = {}", c[0]);
    }

    #[test]
    /// Runs regression test for chi1 ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi1_ethane() {
        let c = calc_chi("CC");
        // One bond, both degree 1: χ¹ = 1/√(1*1) = 1
        assert!((c[1] - 1.0).abs() < 0.01, "χ¹ = {}", c[1]);
    }

    #[test]
    /// Runs regression test for chi0 propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi0_propane() {
        let c = calc_chi("CCC");
        // Degrees: [1, 2, 1] → χ⁰ = 1/1 + 1/√2 + 1/1 = 2 + 0.707 = 2.707
        assert!((c[0] - 2.707).abs() < 0.01, "χ⁰ = {}", c[0]);
    }

    #[test]
    /// Runs regression test for chi1 propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi1_propane() {
        let c = calc_chi("CCC");
        // Bonds: (1,2) and (2,1): χ¹ = 1/√2 + 1/√2 = 1.414
        assert!((c[1] - 1.414).abs() < 0.01, "χ¹ = {}", c[1]);
    }

    #[test]
    /// Runs regression test for chi path butane vs isobutane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi_path_butane_vs_isobutane() {
        let lin = calc_chi("CCCC");
        let bra = calc_chi("CC(C)C");
        assert!(lin[1] > bra[1], "lin χ¹={} > bra χ¹={}", lin[1], bra[1]);
    }

    #[test]
    /// Runs regression test for chi2 butane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi2_butane() {
        let c = calc_chi("CCCC");
        assert!((c[2] - 1.0).abs() < 0.01, "χ² = {}", c[2]);
    }

    #[test]
    /// Runs regression test for chi benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi_benzene() {
        let c = calc_chi("c1ccccc1");
        // χ⁰ = 6/√2 = 4.243
        assert!((c[0] - 4.243).abs() < 0.01, "χ⁰ = {}", c[0]);
        // χ¹ = 6/2 = 3.0
        assert!((c[1] - 3.0).abs() < 0.01, "χ¹ = {}", c[1]);
    }

    #[test]
    /// Runs regression test for chi size.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi_size() {
        let c = calc_chi("c1ccccc1");
        assert_eq!(c.len(), 26, "chi indices should have 26 values");
    }

    #[test]
    /// Runs regression test for chi valence differs.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi_valence_differs() {
        // Valence path indices [8..16] should differ from simple path [0..8] for heteroatom molecules
        let c = calc_chi("CCO");
        assert!(
            (c[1] - c[9]).abs() > 0.001,
            "simple χ¹={} vs valence χᵥ¹={}",
            c[1],
            c[9]
        );
    }

    #[test]
    /// Runs regression test for cluster linear zero.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_cluster_linear_zero() {
        let c = calc_chi("CCCCC");
        // χ³c at index 16 should be 0 for linear chain
        assert!(c[16].abs() < 0.01, "linear χ³c should be 0: {}", c[16]);
    }

    #[test]
    /// Runs regression test for cluster isobutane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_cluster_isobutane() {
        let c = calc_chi("CC(C)C");
        assert!(c[16] > 0.0, "isobutane should have χ³c > 0: {}", c[16]);
    }

    #[test]
    /// Runs regression test for chain pentane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chain_pentane() {
        let c = calc_chi("CCCCC");
        // χ³ch at index 18 should equal χ³path at index 3 for linear chain
        assert!(
            (c[18] - c[3]).abs() < 0.01,
            "pentane χ³ch={} should equal χ³path={}",
            c[18],
            c[3]
        );
    }

    #[test]
    /// Runs regression test for pathcluster linear zero.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pathcluster_linear_zero() {
        let c = calc_chi("CCCCCC");
        // χ⁴pc at index 23 should be 0 for linear chain
        assert!(c[23].abs() < 0.01, "linear χ⁴pc should be 0: {}", c[23]);
    }

    #[test]
    /// Runs regression test for pathcluster 2methylpentane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pathcluster_2methylpentane() {
        let c = calc_chi("CC(C)CCC");
        // χ⁵pc at index 24 should be > 0
        assert!(c[24] > 0.0, "should have χ⁵pc > 0: {}", c[24]);
    }

    #[test]
    /// Runs regression test for chi3v counts three membered ring closure like rdkit.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi3v_counts_three_membered_ring_closure_like_rdkit() {
        let c = calc_chi("C1CC1");
        assert!((c[11] - 0.35355338).abs() < 1e-5, "χ3v = {}", c[11]);
    }

    #[test]
    /// Runs regression test for chi4v counts four membered ring closure like rdkit.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi4v_counts_four_membered_ring_closure_like_rdkit() {
        let c = calc_chi("C1CCC1");
        assert!((c[12] - 0.25).abs() < 1e-5, "χ4v = {}", c[12]);
    }

    #[test]
    /// Runs regression test for chi4v bicyclic regression matches rdkit.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_chi4v_bicyclic_regression_matches_rdkit() {
        let c = calc_chi("CC1(C)C2CC=CC1C2");
        assert!((c[12] - 2.1899135).abs() < 1e-5, "χ4v = {}", c[12]);
    }
}

// ===========================================================================
// Typed API: All Chi Indices Combined
// ===========================================================================

/// All Chi indices.
///
/// Returns FloatVec with 26 values:
/// - [0-7]: ChiPath (χ⁰..χ⁷)
/// - [8-15]: ChiValencePath (χᵥ⁰..χᵥ⁷)
/// - [16-17]: ChiCluster (χ³c, χ⁴c)
/// - [18-22]: ChiChain (χ³ch..χ⁷ch)
/// - [23-25]: ChiPathCluster (χ⁴pc, χ⁵pc, χ⁶pc)
pub struct ChiIndices;

impl Descriptor for ChiIndices {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "ChiIndices"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut values = Vec::with_capacity(26);

        // ChiPath (8 values)
        let simple_delta = simple_deltas(mol);
        let mut simple_path_vals = [0.0f64; 8];
        for order in 0..8 {
            let v = chi_path(mol, &simple_delta, order);
            simple_path_vals[order] = v;
            values.push(v as f32);
        }

        // ChiValencePath (8 values)
        let valence_delta = valence_deltas(mol);
        for order in 0..8 {
            let value = if order >= 3 {
                chi_path_rdkit_like(mol, &valence_delta, order)
            } else {
                chi_path(mol, &valence_delta, order)
            };
            values.push(value as f32);
        }

        // ChiCluster (cache orders 3..7 once)
        let mut cluster_vals = [0.0f64; 8];
        for order in 3..=7 {
            cluster_vals[order] = chi_cluster(mol, &simple_delta, order);
        }
        values.push(cluster_vals[3] as f32);
        values.push(cluster_vals[4] as f32);

        // ChiPathCluster (cache orders 4..7 once)
        let mut pathcluster_vals = [0.0f64; 8];
        for order in 4..=7 {
            pathcluster_vals[order] = chi_pathcluster_order(mol, &simple_delta, order);
        }

        // ChiChain (5 values)
        for i in 0..5 {
            let order = i + 3; // orders 3-7
            let path = simple_path_vals[order];
            let cluster = cluster_vals[order];
            let pc = if order >= 4 {
                pathcluster_vals[order]
            } else {
                0.0
            };
            values.push((path - cluster - pc).max(0.0) as f32);
        }

        // ChiPathCluster (3 values)
        values.push(pathcluster_vals[4] as f32);
        values.push(pathcluster_vals[5] as f32);
        values.push(pathcluster_vals[6] as f32);

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/chi.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ChiIndices>;
};
