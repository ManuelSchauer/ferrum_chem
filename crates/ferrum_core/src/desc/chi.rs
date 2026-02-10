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

use super::Descriptor;
use crate::mol::elements::outer_electrons;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, Real};

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
    mol.atoms.iter().map(|atom| {
        let z = atom.atomic_number as f64;
        let zv = outer_electrons(atom.atomic_number) as f64;
        let h = atom.total_h() as f64;
        let denom = z - zv - 1.0;
        if denom <= 0.0 || atom.atomic_number <= 1 {
            // For H and 1st-row elements where Z-Zv-1=0, use simple delta
            // δᵥ(C) = (4-H)/(6-4-1) = (4-H)/1 = 4-H
            // For H itself: return 0
            if atom.atomic_number == 1 { return 0.0; }
            // 1st row: C(6),N(7),O(8),F(9) → denom=1,1,1,1
            // But O has Z-Zv-1 = 8-6-1=1, N has 7-5-1=1, C has 6-4-1=1
            // Only H (1-1-1=-1) truly fails. For other elements where
            // denom is exactly 0 (shouldn't happen for common elements), fallback:
            zv - h
        } else {
            (zv - h) / denom
        }
    }).collect()
}

/// Chi-0 (zero-order): χ⁰ = Σ 1/√δᵢ for each atom with δ > 0.
fn chi0(deltas: &[f64]) -> f64 {
    deltas.iter()
        .filter(|&&d| d > 0.0)
        .map(|&d| 1.0 / d.sqrt())
        .sum()
}

/// Chi-1 (first-order): χ¹ = Σ 1/√(δᵢ·δⱼ) over all bonds.
fn chi1(mol: &Molecule, deltas: &[f64]) -> f64 {
    mol.bonds.iter()
        .map(|bond| {
            let di = deltas[bond.source as usize];
            let dj = deltas[bond.target as usize];
            let prod = di * dj;
            if prod > 0.0 { 1.0 / prod.sqrt() } else { 0.0 }
        })
        .sum()
}

/// Find all paths of exactly `length` bonds and compute chi contribution.
/// χⁿ = Σ 1/√(δ₁·δ₂·...·δₙ₊₁) over all n-bond paths.
fn chi_path(mol: &Molecule, deltas: &[f64], length: usize) -> f64 {
    if length == 0 { return chi0(deltas); }
    if length == 1 { return chi1(mol, deltas); }

    let n = mol.num_atoms();
    if n < length + 1 { return 0.0; }

    // DFS to enumerate all paths of exact length
    let mut sum = 0.0f64;
    let mut path = Vec::with_capacity(length + 1);

    for start in 0..n {
        path.clear();
        path.push(start as AtomIdx);
        dfs_paths(mol, deltas, &mut path, length, &mut sum);
    }

    // Each path is found twice (forward + reverse), so divide by 2
    sum / 2.0
}

fn dfs_paths(
    mol: &Molecule,
    deltas: &[f64],
    path: &mut Vec<AtomIdx>,
    target_len: usize,
    sum: &mut f64,
) {
    if path.len() == target_len + 1 {
        // Compute product of deltas along the path
        let prod: f64 = path.iter().map(|&idx| deltas[idx as usize]).product();
        if prod > 0.0 {
            *sum += 1.0 / prod.sqrt();
        }
        return;
    }

    let current = *path.last().unwrap();
    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let neighbor = if bond.source == current { bond.target } else { bond.source };

        // Avoid revisiting atoms in the current path (simple paths only)
        if !path.contains(&neighbor) {
            path.push(neighbor);
            dfs_paths(mol, deltas, path, target_len, sum);
            path.pop();
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

/// Chi Path Indices: χ⁰, χ¹, χ², χ³, χ⁴, χ⁵, χ⁶, χ⁷ (simple connectivity).
///
/// Uses simple graph degree as atom weight.
/// Returns 8 values.
pub struct ChiPath;

impl Descriptor for ChiPath {
    fn name(&self) -> String { "ChiPath".to_string() }
    fn size(&self) -> usize { 8 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let deltas = simple_deltas(mol);
        for order in 0..8 {
            output[order] = chi_path(mol, &deltas, order) as Real;
        }
    }
}

/// Chi Valence Path Indices: χᵥ⁰..χᵥ⁷ (valence connectivity).
///
/// Uses valence delta δᵥ = (Zᵥ - H) / (Z - Zᵥ - 1) as atom weight.
/// Encodes electronic information in addition to topology.
/// Returns 8 values.
pub struct ChiValencePath;

impl Descriptor for ChiValencePath {
    fn name(&self) -> String { "ChiValencePath".to_string() }
    fn size(&self) -> usize { 8 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let deltas = valence_deltas(mol);
        for order in 0..8 {
            output[order] = chi_path(mol, &deltas, order) as Real;
        }
    }
}

/// Chi Cluster Indices: χ³c, χ⁴c (3-atom and 4-atom star/cluster subgraphs).
///
/// A "cluster" of order n is a star pattern: one central atom bonded to n-1 others.
/// χ³c = Σ 1/√(δ_center · δ₁ · δ₂ · δ₃) for all 3-star subgraphs.
/// Returns 2 values: [χ³c, χ⁴c].
pub struct ChiCluster;

impl Descriptor for ChiCluster {
    fn name(&self) -> String { "ChiCluster".to_string() }
    fn size(&self) -> usize { 2 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let deltas = simple_deltas(mol);
        output[0] = chi_cluster(mol, &deltas, 3) as Real;
        output[1] = chi_cluster(mol, &deltas, 4) as Real;
    }
}

/// Chi Chain Indices: χ³ch, χ⁴ch, χ⁵ch, χ⁶ch, χ⁷ch.
///
/// "Chain" = path subgraphs where no atom has degree > 2 in the subgraph.
/// Chain(n) = Path(n) - Cluster(n) - PathCluster(n).
/// Returns 5 values: [χ³ch..χ⁷ch].
pub struct ChiChain;

impl Descriptor for ChiChain {
    fn name(&self) -> String { "ChiChain".to_string() }
    fn size(&self) -> usize { 5 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let deltas = simple_deltas(mol);
        for i in 0..5 {
            let order = i + 3; // orders 3-7
            let path = chi_path(mol, &deltas, order);
            let cluster = chi_cluster(mol, &deltas, order);
            let pc = chi_pathcluster_order(mol, &deltas, order);
            output[i] = (path - cluster - pc).max(0.0) as Real;
        }
    }
}

/// Chi Path-Cluster Indices: χ⁴pc, χ⁵pc, χ⁶pc.
///
/// Path-cluster subgraphs have one atom with degree 3 in the subgraph
/// (a "branch point") connected to a path. Y-shaped subgraphs.
/// Returns 3 values: [χ⁴pc, χ⁵pc, χ⁶pc].
pub struct ChiPathCluster;

impl Descriptor for ChiPathCluster {
    fn name(&self) -> String { "ChiPathCluster".to_string() }
    fn size(&self) -> usize { 3 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let deltas = simple_deltas(mol);
        output[0] = chi_pathcluster_order(mol, &deltas, 4) as Real;
        output[1] = chi_pathcluster_order(mol, &deltas, 5) as Real;
        output[2] = chi_pathcluster_order(mol, &deltas, 6) as Real;
    }
}

/// Compute cluster chi for star subgraphs of given order.
/// Order 3: central atom with 3 neighbors chosen → C(deg, 3) combinations.
/// Order 4: central atom with 4 neighbors chosen → C(deg, 4) combinations.
fn chi_cluster(mol: &Molecule, deltas: &[f64], order: usize) -> f64 {
    let n = mol.num_atoms();
    let k = order; // need k neighbors around the center
    let mut sum = 0.0;

    for center in 0..n {
        let deg = mol.get_degree(center as AtomIdx);
        if deg < k { continue; }

        let dc = deltas[center];
        if dc <= 0.0 { continue; }

        // Collect neighbor deltas
        let neighbor_deltas: Vec<f64> = mol.adjacency[center].iter()
            .map(|&bi| {
                let bond = &mol.bonds[bi as usize];
                let nb = if bond.source == center as AtomIdx { bond.target } else { bond.source };
                deltas[nb as usize]
            })
            .collect();

        // Generate all combinations of k neighbors
        let combos = combinations(&neighbor_deltas, k);
        for combo in combos {
            let prod: f64 = dc * combo.iter().product::<f64>();
            if prod > 0.0 {
                sum += 1.0 / prod.sqrt();
            }
        }
    }

    sum
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
    if order < 4 { return 0.0; }

    let n = mol.num_atoms();
    let mut sum = 0.0;
    let mut subgraph: Vec<AtomIdx> = Vec::with_capacity(order);

    // For each potential branch point B with degree >= 3
    for b in 0..n {
        let b_idx = b as AtomIdx;
        let b_neighbors: Vec<AtomIdx> = mol.adjacency[b].iter()
            .map(|&bi| {
                let bond = &mol.bonds[bi as usize];
                if bond.source == b_idx { bond.target } else { bond.source }
            })
            .collect();

        if b_neighbors.len() < 3 { continue; }

        let db = deltas[b];
        if db <= 0.0 { continue; }

        // Pick 3 arms from B's neighbors, then extend each arm to total `order` atoms
        // The branch point + 3 arms must total `order` atoms.
        // Arm lengths: a1 + a2 + a3 = order - 1, each arm >= 1.
        // Enumerate all (a1, a2, a3) partitions with a1 >= a2 >= a3 >= 1.
        let remaining = order - 1; // atoms distributed across 3 arms

        for a1 in 1..=remaining {
            for a2 in 1..=(remaining - a1) {
                let a3 = remaining - a1 - a2;
                if a3 < 1 || a2 > a1 || a3 > a2 { continue; }

                // Skip pure clusters: all arms of length 1 = star subgraph
                if a1 == 1 && a2 == 1 && a3 == 1 { continue; }

                // Generate all distinct permutations of arm lengths
                let perms = distinct_permutations(a1, a2, a3);

                // For each combination of 3 neighbors as arm starts
                let combos = combinations_idx(&b_neighbors, 3);
                for combo in &combos {
                    let arm_starts = [combo[0], combo[1], combo[2]];

                    for perm in &perms {
                        let arm_lens = [perm[0], perm[1], perm[2]];
                        let val = extend_arms_chi(
                            mol, deltas, b_idx, db,
                            &arm_starts, &arm_lens,
                        );
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
    // Build arm paths via DFS, then combine
    let mut arm_paths: [Vec<Vec<AtomIdx>>; 3] = [vec![], vec![], vec![]];

    for arm in 0..3 {
        let mut path = vec![starts[arm]];
        let mut excluded: Vec<AtomIdx> = vec![branch];
        // Exclude other arm starts to prevent overlap
        for other in 0..3 {
            if other != arm { excluded.push(starts[other]); }
        }
        find_arm_paths(mol, &mut path, lengths[arm], &excluded, &mut arm_paths[arm]);
    }

    let mut sum = 0.0;

    // Combine all valid arm combinations
    for p0 in &arm_paths[0] {
        for p1 in &arm_paths[1] {
            // Check no overlap between p0 and p1
            if p0.iter().any(|a| p1.contains(a)) { continue; }
            for p2 in &arm_paths[2] {
                // Check no overlap with p0 or p1
                if p2.iter().any(|a| p0.contains(a) || p1.contains(a)) { continue; }

                let mut prod = d_branch;
                for &idx in p0.iter().chain(p1.iter()).chain(p2.iter()) {
                    prod *= deltas[idx as usize];
                }
                if prod > 0.0 {
                    sum += 1.0 / prod.sqrt();
                }
            }
        }
    }

    sum
}

/// Find all simple paths of exactly `length` atoms starting from `path[0]`,
/// avoiding atoms in `excluded`.
fn find_arm_paths(
    mol: &Molecule,
    path: &mut Vec<AtomIdx>,
    length: usize,
    excluded: &[AtomIdx],
    results: &mut Vec<Vec<AtomIdx>>,
) {
    if path.len() == length {
        results.push(path.clone());
        return;
    }

    let current = *path.last().unwrap();
    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let nb = if bond.source == current { bond.target } else { bond.source };
        if path.contains(&nb) || excluded.contains(&nb) { continue; }
        path.push(nb);
        find_arm_paths(mol, path, length, excluded, results);
        path.pop();
    }
}

/// Generate combinations of k elements from a slice (by value).
fn combinations_idx(items: &[AtomIdx], k: usize) -> Vec<Vec<AtomIdx>> {
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    fn recurse(items: &[AtomIdx], k: usize, start: usize, combo: &mut Vec<AtomIdx>, result: &mut Vec<Vec<AtomIdx>>) {
        if combo.len() == k {
            result.push(combo.clone());
            return;
        }
        for i in start..items.len() {
            combo.push(items[i]);
            recurse(items, k, i + 1, combo, result);
            combo.pop();
        }
    }
    recurse(items, k, 0, &mut combo, &mut result);
    result
}

/// Generate all distinct permutations of three values.
fn distinct_permutations(a: usize, b: usize, c: usize) -> Vec<[usize; 3]> {
    let mut perms = vec![];
    let candidates = [
        [a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a],
    ];
    for p in &candidates {
        if !perms.contains(p) {
            perms.push(*p);
        }
    }
    perms
}

/// Generate all combinations of `k` elements from `items`.
fn combinations(items: &[f64], k: usize) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    fn recurse(items: &[f64], k: usize, start: usize, combo: &mut Vec<f64>, result: &mut Vec<Vec<f64>>) {
        if combo.len() == k {
            result.push(combo.clone());
            return;
        }
        for i in start..items.len() {
            combo.push(items[i]);
            recurse(items, k, i + 1, combo, result);
            combo.pop();
        }
    }
    recurse(items, k, 0, &mut combo, &mut result);
    result
}

// ---------------------------------------------------------------------------
// Compile-time thread safety checks
// ---------------------------------------------------------------------------
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ChiPath>;
    let _ = assert_send_sync::<ChiValencePath>;
    let _ = assert_send_sync::<ChiCluster>;
    let _ = assert_send_sync::<ChiChain>;
    let _ = assert_send_sync::<ChiPathCluster>;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_chi_path(smiles: &str) -> Vec<Real> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = vec![0.0; 8];
        ChiPath.calculate(&mol, &mut output);
        output
    }

    fn calc_chi_vpath(smiles: &str) -> Vec<Real> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = vec![0.0; 8];
        ChiValencePath.calculate(&mol, &mut output);
        output
    }

    fn calc_chi_cluster(smiles: &str) -> [Real; 2] {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = [0.0; 2];
        ChiCluster.calculate(&mol, &mut output);
        output
    }

    fn calc_chi_chain(smiles: &str) -> Vec<Real> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = vec![0.0; 5];
        ChiChain.calculate(&mol, &mut output);
        output
    }

    fn calc_chi_pc(smiles: &str) -> [Real; 3] {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = [0.0; 3];
        ChiPathCluster.calculate(&mol, &mut output);
        output
    }

    // -- Chi Path tests --

    #[test]
    fn test_chi0_ethane() {
        let c = calc_chi_path("CC");
        // Each C has degree 1: χ⁰ = 1/√1 + 1/√1 = 2
        assert!((c[0] - 2.0).abs() < 0.01, "χ⁰ = {}", c[0]);
    }

    #[test]
    fn test_chi1_ethane() {
        let c = calc_chi_path("CC");
        // One bond, both degree 1: χ¹ = 1/√(1*1) = 1
        assert!((c[1] - 1.0).abs() < 0.01, "χ¹ = {}", c[1]);
    }

    #[test]
    fn test_chi0_propane() {
        let c = calc_chi_path("CCC");
        // Degrees: [1, 2, 1] → χ⁰ = 1/1 + 1/√2 + 1/1 = 2 + 0.707 = 2.707
        assert!((c[0] - 2.707).abs() < 0.01, "χ⁰ = {}", c[0]);
    }

    #[test]
    fn test_chi1_propane() {
        let c = calc_chi_path("CCC");
        // Bonds: (1,2) and (2,1): χ¹ = 1/√2 + 1/√2 = 1.414
        assert!((c[1] - 1.414).abs() < 0.01, "χ¹ = {}", c[1]);
    }

    #[test]
    fn test_chi_path_butane_vs_isobutane() {
        let lin = calc_chi_path("CCCC");
        let bra = calc_chi_path("CC(C)C");
        // Same χ⁰ (same degree sum)
        // χ¹ differs: linear has (1,2),(2,2),(2,1) = 1/√2+1/2+1/√2 = 1.914
        //             branched has (1,3),(1,3),(1,3) = 3/√3 = 1.732
        assert!(lin[1] > bra[1], "lin χ¹={} > bra χ¹={}", lin[1], bra[1]);
    }

    #[test]
    fn test_chi2_butane() {
        let c = calc_chi_path("CCCC");
        // 2-bond paths: (1,2,2) and (2,2,1) → χ² = 2/√(1*2*2) = 2/2 = 1.0
        assert!((c[2] - 1.0).abs() < 0.01, "χ² = {}", c[2]);
    }

    #[test]
    fn test_chi3_pentane() {
        let c = calc_chi_path("CCCCC");
        // Should have nonzero χ³
        assert!(c[3] > 0.0, "χ³ = {}", c[3]);
    }

    #[test]
    fn test_chi_benzene() {
        let c = calc_chi_path("c1ccccc1");
        // Benzene: all degrees = 2
        // χ⁰ = 6/√2 = 4.243
        assert!((c[0] - 4.243).abs() < 0.01, "χ⁰ = {}", c[0]);
        // χ¹ = 6 * 1/√(2*2) = 6/2 = 3.0
        assert!((c[1] - 3.0).abs() < 0.01, "χ¹ = {}", c[1]);
    }

    #[test]
    fn test_chi4_exists() {
        let c = calc_chi_path("CCCCCC");
        assert!(c[4] > 0.0, "hexane should have χ⁴");
    }

    #[test]
    fn test_chi567_long_chain() {
        let c = calc_chi_path("CCCCCCCCCC"); // decane
        assert!(c[5] > 0.0, "decane should have χ⁵: {}", c[5]);
        assert!(c[6] > 0.0, "decane should have χ⁶: {}", c[6]);
        assert!(c[7] > 0.0, "decane should have χ⁷: {}", c[7]);
    }

    #[test]
    fn test_chi_path_size() {
        assert_eq!(ChiPath.size(), 8);
        assert_eq!(ChiValencePath.size(), 8);
    }

    // -- Chi Valence Path tests --

    #[test]
    fn test_chiv_ethanol() {
        let c = calc_chi_vpath("CCO");
        // Valence deltas differ from simple deltas (O has different Zv)
        assert!(c[0] > 0.0);
        assert!(c[1] > 0.0);
    }

    #[test]
    fn test_chiv_ethane_vs_ethanol() {
        let ethane = calc_chi_vpath("CC");
        let ethanol = calc_chi_vpath("CCO");
        // Ethanol has more atoms → higher χᵥ⁰
        assert!(ethanol[0] > ethane[0]);
    }

    #[test]
    fn test_chiv_amine_vs_alcohol() {
        // N and O have different valence deltas
        let amine = calc_chi_vpath("CCN");
        let alcohol = calc_chi_vpath("CCO");
        // Should differ due to different electronegativities
        assert!((amine[1] - alcohol[1]).abs() > 0.001,
            "amine χᵥ¹={} vs alcohol χᵥ¹={}", amine[1], alcohol[1]);
    }

    // -- Cluster tests --

    #[test]
    fn test_cluster_linear_zero() {
        let c = calc_chi_cluster("CCCCC");
        // Linear chain has no atom with 3+ neighbors → χ³c = 0
        assert!((c[0]).abs() < 0.01, "linear χ³c should be 0: {}", c[0]);
    }

    #[test]
    fn test_cluster_isobutane() {
        let c = calc_chi_cluster("CC(C)C");
        // Central C has degree 3 → one 3-star
        assert!(c[0] > 0.0, "isobutane should have χ³c > 0: {}", c[0]);
    }

    #[test]
    fn test_cluster_neopentane() {
        let c = calc_chi_cluster("CC(C)(C)C");
        // Central C has degree 4 → C(4,3)=4 three-stars, C(4,4)=1 four-star
        assert!(c[0] > 0.0, "neopentane χ³c = {}", c[0]);
        assert!(c[1] > 0.0, "neopentane χ⁴c = {}", c[1]);
    }

    // -- Chain tests --

    #[test]
    fn test_chain_pentane() {
        let c = calc_chi_chain("CCCCC");
        // Linear → all paths are chains, no clusters
        let p = calc_chi_path("CCCCC");
        assert!((c[0] - p[3]).abs() < 0.01,
            "pentane χ³ch={} should equal χ³path={}", c[0], p[3]);
    }

    #[test]
    fn test_chain_neopentane_zero() {
        let c = calc_chi_chain("CC(C)(C)C");
        // Neopentane: all 3-bond subgraphs are clusters, so chain should be 0
        assert!((c[0]).abs() < 0.01,
            "neopentane χ³ch should be ~0: {}", c[0]);
    }

    #[test]
    fn test_chain_size() {
        assert_eq!(ChiChain.size(), 5);
    }

    #[test]
    fn test_chain_long_molecule() {
        let c = calc_chi_chain("CCCCCCCCCC"); // decane
        // Linear chain: all subgraphs are chains (no branch points)
        assert!(c[0] > 0.0, "χ³ch > 0 for decane");
        assert!(c[1] > 0.0, "χ⁴ch > 0 for decane");
        assert!(c[2] > 0.0, "χ⁵ch > 0 for decane");
    }

    // -- Path-cluster tests --

    #[test]
    fn test_pathcluster_size() {
        assert_eq!(ChiPathCluster.size(), 3);
    }

    #[test]
    fn test_pathcluster_linear_zero() {
        let pc = calc_chi_pc("CCCCCC");
        // No branch points in linear chain
        assert!((pc[0]).abs() < 0.01, "linear χ⁴pc should be 0: {}", pc[0]);
    }

    #[test]
    fn test_pathcluster_isobutane_zero() {
        let pc = calc_chi_pc("CC(C)C");
        // Isobutane: 4 atoms. Path-cluster order 4 needs branch + 3 arms = 4 atoms,
        // but the only partition (1,1,1) is a pure cluster. So χ⁴pc = 0.
        assert!((pc[0]).abs() < 0.01, "isobutane χ⁴pc should be 0: {}", pc[0]);
    }

    #[test]
    fn test_pathcluster_2methylpentane() {
        // CC(C)CCC: 6 atoms, branch at C1 (degree 3)
        let pc = calc_chi_pc("CC(C)CCC");
        println!("2-methylpentane χpc: {:?}", pc);
        // χ⁴pc = 0 (no 4-atom path-clusters exist, partition (1,1,1) excluded)
        // χ⁵pc should be > 0: branch + arms (2,1,1) or (2,1,1) permutations
        assert!(pc[1] > 0.0, "should have χ⁵pc > 0: {}", pc[1]);
    }
}
