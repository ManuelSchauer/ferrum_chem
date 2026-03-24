//! Balaban J Index descriptor
//!
//! The Balaban J Index is a topological index that measures the degree of branching
//! in a molecular graph. It's based on the distance matrix and vertex degrees.
//!
//! Formula: J = m/(μ+1) × Σ[(d_i × d_j)^(-0.5)] over all edges
//! where:
//! - m = number of edges (bonds)
//! - μ = cyclomatic number (m - n + 1 for connected graphs)
//! - d_i = sum of distances from vertex i to all other vertices
//!
//! Reference: Balaban, A.T. (1982) Chemical Applications of Graph Theory

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use smallvec::SmallVec;

// Weighted adjacency representation used by Balaban/Bertz shared BO-distance kernels.
//
// Each atom keeps a compact neighbor list of (neighbor_atom_index, edge_weight).
// SmallVec avoids heap allocation for common organic valence (typically <=4 neighbors).
type WeightedNeighbors = Vec<SmallVec<[(usize, f64); 4]>>;

/// Compute the distance matrix using BFS/Dijkstra with bond order weights.
/// With useBO=true: weight = 1/bond_order (aromatic=1.5, double=2, triple=3).
/// With useBO=false: weight = 1.0 for all bonds.
pub fn compute_distance_matrix(mol: &Molecule, use_bo: bool) -> Vec<f64> {
    let n = mol.num_atoms();
    if n == 0 {
        return Vec::new();
    }

    let neighbors = build_weighted_neighbors(mol, use_bo);
    if n <= 64 {
        compute_distance_matrix_small(&neighbors, n)
    } else {
        compute_distance_matrix_large(&neighbors, n)
    }
}

#[inline]
fn bond_weight(bond: &crate::mol::bond::Bond, use_bo: bool) -> f64 {
    if !use_bo {
        return 1.0;
    }
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

/// Builds weighted adjacency once per distance-matrix call.
///
/// This is intentionally separate from Dijkstra loops so all expensive bond-type branching
/// (`single/double/triple/aromatic`) happens once and the inner relaxation loop is branch-light.
fn build_weighted_neighbors(mol: &Molecule, use_bo: bool) -> WeightedNeighbors {
    let mut neighbors = vec![SmallVec::new(); mol.num_atoms()];
    for bond in &mol.bonds {
        let src = bond.source as usize;
        let dst = bond.target as usize;
        let weight = bond_weight(bond, use_bo);
        neighbors[src].push((dst, weight));
        neighbors[dst].push((src, weight));
    }
    neighbors
}

/// O(n^3) dense Dijkstra variant optimized for small molecules (`n <= 64`).
///
/// Uses stack arrays for `row` and `visited` to reduce allocator traffic in common
/// drug-like cases. This keeps behavior identical to the large kernel.
///
/// Implementation detail:
/// - We use a `u64` bitmask for visited-state tracking to avoid writes to a bool array and
///   improve cache behavior in the tight min-selection loop.
fn compute_distance_matrix_small(neighbors: &[SmallVec<[(usize, f64); 4]>], n: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; n * n];
    let mut row = [f64::INFINITY; 64];
    let mut visited_mask = 0u64;

    for start in 0..n {
        row.fill(f64::INFINITY);
        visited_mask = 0;
        row[start] = 0.0;

        for _ in 0..n {
            let mut min_dist = f64::INFINITY;
            let mut u = usize::MAX;
            for v in 0..n {
                if ((visited_mask >> v) & 1) == 0 && row[v] < min_dist {
                    min_dist = row[v];
                    u = v;
                }
            }
            if u == usize::MAX {
                break;
            }
            visited_mask |= 1u64 << u;

            let base = row[u];
            for &(neighbor, weight) in &neighbors[u] {
                let new_dist = base + weight;
                if new_dist < row[neighbor] {
                    row[neighbor] = new_dist;
                }
            }
        }

        let start_offset = start * n;
        for j in 0..n {
            dist[start_offset + j] = row[j];
        }
    }

    dist
}

/// O(n^3) dense Dijkstra variant for larger molecules (`n > 64`).
///
/// Keeps the algorithm equivalent to the small kernel while using heap-backed scratch buffers.
fn compute_distance_matrix_large(neighbors: &[SmallVec<[(usize, f64); 4]>], n: usize) -> Vec<f64> {
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

pub(crate) fn compute_distance_matrix_cached(mol: &mut Molecule, use_bo: bool) -> Vec<f64> {
    if !use_bo {
        return compute_distance_matrix(mol, false);
    }
    if let Some(cached) = mol.cached_balaban_bo_dist() {
        return cached.clone();
    }
    let dist = compute_distance_matrix(mol, true);
    mol.store_balaban_bo_dist(dist.clone());
    dist
}

pub(crate) fn ensure_bo_distance_matrix_cached(mol: &mut Molecule) {
    if mol.cached_balaban_bo_dist().is_none() {
        let dist = compute_distance_matrix(mol, true);
        mol.store_balaban_bo_dist(dist);
    }
}

pub(crate) fn cached_bo_distance_matrix_ref(mol: &Molecule) -> Option<&[f64]> {
    mol.cached_balaban_bo_dist().map(|dist| dist.as_slice())
}

fn calculate_with_distance_matrix(mol: &Molecule, dist: &[f64]) -> f64 {
    if crate::algo::components::num_fragments(mol) > 1 {
        return 0.0;
    }

    let n = mol.num_atoms();
    let m = mol.num_bonds();

    if n == 0 || m == 0 {
        return 0.0;
    }

    if n == 1 {
        return 0.0;
    }

    // Connected molecules have finite distances for every pair, so this is a tight contiguous
    // row sum with no per-entry branching. This is kept explicit for readability and to make
    // future SIMD/blocked-sum experiments straightforward.
    let mut s = vec![0.0f64; n];
    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        s[i] = row.iter().sum();
    }

    // Cyclomatic number μ = m - n + 1
    let mu = m as i32 - n as i32 + 1;

    // Sum over all edges (using simple adjacency, not bond orders)
    let mut sum = 0.0;
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        let si = s[i];
        let sj = s[j];
        if si <= 0.0 || sj <= 0.0 {
            return f64::NAN;
        }
        sum += 1.0 / (si * sj).sqrt();
    }

    // J = m/(μ+1) × sum
    if mu + 1 != 0 {
        (m as f64 / (mu + 1) as f64) * sum
    } else {
        0.0
    }
}

/// Calculate Balaban J Index
///
/// # Returns
/// The Balaban J index value (typically 1.5-3.0 for most organic molecules)
///
/// # Examples
/// - Benzene: J ≈ 2.00 (highly symmetric)
/// - Adamantane: J ≈ 2.31
/// - Linear alkanes: J decreases with length
///
/// # Panics
/// Returns NaN for molecules with no bonds
pub fn calculate(mol: &Molecule) -> f64 {
    let dist = compute_distance_matrix(mol, true);
    calculate_with_distance_matrix(mol, &dist)
}

pub(crate) fn calculate_cached(mol: &mut Molecule) -> f64 {
    ensure_bo_distance_matrix_cached(mol);
    let dist = cached_bo_distance_matrix_ref(mol).expect("Balaban BO distance cache initialized");
    calculate_with_distance_matrix(mol, dist)
}

/// Balaban J Index descriptor.
///
/// Returns Float with the Balaban J index.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct BalabanIndex;

impl Descriptor for BalabanIndex {
    fn name(&self) -> &str {
        "BalabanJ"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Float(calculate_cached(mol) as f32)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BalabanIndex>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    fn test_balaban_methane() {
        let mol = parse("C");
        let j = calculate(&mol);
        println!("Methane J = {}", j);
        assert!(j.is_nan() || j == 0.0); // Single atom, no bonds
    }

    #[test]
    fn test_balaban_ethane() {
        let mol = parse("CC");
        let j = calculate(&mol);
        // n=2, m=1, μ=0, d_sums=[1,1]
        // sum = 1/sqrt(1*1) = 1.0
        // J = 1/(0+1) * 1.0 = 1.0
        assert!((j - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_balaban_propane() {
        let mol = parse("CCC");
        let j = calculate(&mol);
        // Linear chain should have J around 1.63
        assert!(j > 1.5 && j < 1.8);
    }

    #[test]
    fn test_balaban_isobutane() {
        let mol = parse("CC(C)C");
        let j = calculate(&mol);
        // Branched should have higher J than linear
        // Isobutane J ≈ 2.32
        assert!(j > 2.2 && j < 2.4, "J = {}", j);
    }

    #[test]
    fn test_balaban_benzene() {
        let mol = parse("c1ccccc1");
        let j = calculate(&mol);
        // Benzene: J = 3.0 (with bond-order-weighted distances)
        assert!((j - 3.0).abs() < 0.1, "J = {}", j);
    }

    #[test]
    fn test_balaban_cyclohexane() {
        let mol = parse("C1CCCCC1");
        let j = calculate(&mol);
        // Cyclohexane: all single bonds, J ≈ 2.0
        assert!((j - 2.0).abs() < 0.1, "J = {}", j);
    }

    #[test]
    fn test_balaban_naphthalene() {
        let mol = parse("c1ccc2ccccc2c1");
        let j = calculate(&mol);
        // Naphthalene: J ≈ 2.89 (with BO-weighted distances)
        assert!(j > 2.7 && j < 3.0, "J = {}", j);
    }

    #[test]
    fn test_balaban_disconnected() {
        // Two separate methane molecules
        let mol = parse("C.C");
        let j = calculate(&mol);
        // Should handle disconnected graphs
        assert!(j.is_nan() || j >= 0.0);
    }
}
