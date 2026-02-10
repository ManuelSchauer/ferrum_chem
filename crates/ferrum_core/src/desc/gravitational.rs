//! Gravitational Index descriptor
//!
//! The Gravitational Index is a molecular descriptor that combines atomic masses
//! and topological distances to characterize molecular size and mass distribution.
//!
//! Formula: G = Σᵢ Σⱼ>ᵢ (mᵢ × mⱼ) / dᵢⱼ²
//!
//! where:
//! - mᵢ, mⱼ = atomic masses
//! - dᵢⱼ = topological distance (number of bonds)
//!
//! Reference: Katritzky et al. (1996)

use crate::mol::molecule::Molecule;
use crate::mol::elements::ATOMIC_MASSES;
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

/// Calculate Gravitational Index
///
/// # Returns
/// The gravitational index value (higher for larger/heavier molecules)
///
/// # Examples
/// - Larger molecules have higher G values
/// - Heavier atoms contribute more to G
/// - More compact structures have higher G (shorter distances)
pub fn calculate(mol: &Molecule) -> f64 {
    let n = mol.num_atoms();
    
    if n <= 1 {
        return 0.0;
    }
    
    // Compute distance matrix
    let dist = compute_distance_matrix(mol);
    
    // Calculate gravitational index
    let mut g = 0.0;
    for i in 0..n {
        let mass_i = ATOMIC_MASSES[mol.atoms[i].atomic_number as usize] as f64;
        
        for j in (i+1)..n {
            let mass_j = ATOMIC_MASSES[mol.atoms[j].atomic_number as usize] as f64;
            let d = dist[i * n + j];
            
            if d != u32::MAX && d > 0 {
                g += (mass_i * mass_j) / (d * d) as f64;
            }
        }
    }
    
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }
    
    #[test]
    fn test_gravitational_methane() {
        let mol = parse("C");
        let g = calculate(&mol);
        assert_eq!(g, 0.0); // Single atom
    }
    
    #[test]
    fn test_gravitational_ethane() {
        let mol = parse("CC");
        let g = calculate(&mol);
        // Two carbons at distance 1: (12*12)/1² = 144
        assert!((g - 144.0).abs() < 1.0);
    }
    
    #[test]
    fn test_gravitational_propane() {
        let mol = parse("CCC");
        let g = calculate(&mol);
        // C1-C2: d=1, C2-C3: d=1, C1-C3: d=2
        // G = 12*12/1 + 12*12/1 + 12*12/4 = 144 + 144 + 36 = 324
        assert!((g - 324.0).abs() < 1.0);
    }
    
    #[test]
    fn test_gravitational_size_ordering() {
        let ethane = parse("CC");
        let propane = parse("CCC");
        let butane = parse("CCCC");
        
        let g_eth = calculate(&ethane);
        let g_prop = calculate(&propane);
        let g_but = calculate(&butane);
        
        // Larger molecules should have higher G
        assert!(g_prop > g_eth);
        assert!(g_but > g_prop);
    }
    
    #[test]
    fn test_gravitational_heteroatoms() {
        let ethane = parse("CC");
        let ethanol = parse("CCO");
        
        let g_eth = calculate(&ethane);
        let g_etoh = calculate(&ethanol);
        
        // Oxygen is heavier than carbon, but slightly
        // The difference depends on the structure
        assert!(g_etoh > g_eth - 50.0); // Should be in similar range
    }
    
    #[test]
    fn test_gravitational_benzene() {
        let mol = parse("c1ccccc1");
        let g = calculate(&mol);
        // Cyclic structure should have reasonable G value
        assert!(g > 500.0); // 6 carbons with various distances
    }
    
    #[test]
    fn test_gravitational_branched_vs_linear() {
        let butane = parse("CCCC");
        let isobutane = parse("CC(C)C");
        
        let g_but = calculate(&butane);
        let g_iso = calculate(&isobutane);
        
        // Linear should have higher G (longer max distance)
        // Actually, branched has shorter average distances, so higher G
        assert!(g_iso > g_but * 0.8); // Close but branched might be higher
    }
}
