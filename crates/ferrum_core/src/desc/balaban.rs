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
use std::collections::VecDeque;

/// Compute the distance matrix using BFS from each atom
fn compute_distance_matrix(mol: &Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    let mut dist = vec![u32::MAX; n * n];
    
    for start in 0..n {
        let mut queue = VecDeque::new();
        queue.push_back(start);
        dist[start * n + start] = 0;
        
        while let Some(current) = queue.pop_front() {
            let current_dist = dist[start * n + current];
            
            // adjacency[i] stores bond indices, not atom indices!
            for &bond_idx in &mol.adjacency[current] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == current as u16 {
                    bond.target as usize
                } else {
                    bond.source as usize
                };
                
                if dist[start * n + neighbor] == u32::MAX {
                    dist[start * n + neighbor] = current_dist + 1;
                    queue.push_back(neighbor);
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
    let n = mol.num_atoms();
    let m = mol.num_bonds();
    
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    
    if n == 1 {
        return 0.0;
    }
    
    // Compute distance matrix
    let dist = compute_distance_matrix(mol);
    
    // Compute distance sums d_i for each atom
    let mut distance_sums = vec![0u32; n];
    for i in 0..n {
        for j in 0..n {
            let d = dist[i * n + j];
            if d != u32::MAX {
                distance_sums[i] += d;
            }
        }
    }
    
    // Cyclomatic number μ = m - n + 1 (for connected graph)
    let mu = m as i32 - n as i32 + 1;
    
    // Sum over all edges: (d_i × d_j)^(-0.5)
    let mut sum = 0.0;
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        let d_i = distance_sums[i] as f64;
        let d_j = distance_sums[j] as f64;
        if d_i == 0.0 || d_j == 0.0 {
            // Isolated atom or calculation error
            return f64::NAN;
        }
        sum += 1.0 / (d_i * d_j).sqrt();
    }
    
    // J = m/(μ+1) × sum
    let j = (m as f64 / (mu + 1) as f64) * sum;
    j
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
        // Benzene is highly symmetric: J ≈ 2.00
        assert!((j - 2.0).abs() < 0.1);
    }
    
    #[test]
    fn test_balaban_cyclohexane() {
        let mol = parse("C1CCCCC1");
        let j = calculate(&mol);
        // Regular hexagon, similar to benzene
        assert!((j - 2.0).abs() < 0.1);
    }
    
    #[test]
    fn test_balaban_naphthalene() {
        let mol = parse("c1ccc2ccccc2c1");
        let j = calculate(&mol);
        // Fused rings, J slightly higher than benzene
        assert!(j > 1.9 && j < 2.2);
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
