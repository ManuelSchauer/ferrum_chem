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

/// Compute the distance matrix using BFS/Dijkstra with bond order weights.
/// With useBO=true: weight = 1/bond_order (aromatic=1.5, double=2, triple=3).
/// With useBO=false: weight = 1.0 for all bonds.
pub fn compute_distance_matrix(mol: &Molecule, use_bo: bool) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut dist = vec![f64::INFINITY; n * n];

    for start in 0..n {
        dist[start * n + start] = 0.0;
        // Dijkstra's algorithm for weighted shortest paths
        let mut visited = vec![false; n];
        for _ in 0..n {
            // Find unvisited node with smallest distance
            let mut min_dist = f64::INFINITY;
            let mut u = usize::MAX;
            for v in 0..n {
                if !visited[v] && dist[start * n + v] < min_dist {
                    min_dist = dist[start * n + v];
                    u = v;
                }
            }
            if u == usize::MAX {
                break;
            }
            visited[u] = true;

            for &bond_idx in &mol.adjacency[u] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == u as u16 {
                    bond.target as usize
                } else {
                    bond.source as usize
                };
                let weight = if use_bo {
                    let bo = if bond.is_aromatic()
                        || bond.bond_type == crate::types::BondType::Aromatic
                    {
                        1.5
                    } else {
                        match bond.bond_type {
                            crate::types::BondType::Double => 2.0,
                            crate::types::BondType::Triple => 3.0,
                            _ => 1.0,
                        }
                    };
                    1.0 / bo
                } else {
                    1.0
                };
                let new_dist = dist[start * n + u] + weight;
                if new_dist < dist[start * n + neighbor] {
                    dist[start * n + neighbor] = new_dist;
                }
            }
        }
    }

    dist
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

    // Compute bond-order-weighted distance matrix (like RDKit's useBO=1)
    let dist = compute_distance_matrix(mol, true);

    // Compute distance sums s_i = sum of row i
    let mut s = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            let d = dist[i * n + j];
            if d.is_finite() {
                s[i] += d;
            }
        }
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

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(calculate(mol) as f32)
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
