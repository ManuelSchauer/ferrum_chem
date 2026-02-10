//! Petitjean Indices
//!
//! Graph diameter and shape descriptors based on vertex eccentricity.
//!
//! - Eccentricity ε(v) = maximum distance from vertex v to any other vertex
//! - Graph Diameter D = maximum eccentricity
//! - Graph Radius R = minimum eccentricity
//! - Petitjean Index = (D - R) / D (shape measure, 0 = sphere, 1 = rod)
//!
//! Reference: Petitjean (1992) J. Chem. Inf. Comput. Sci.

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

/// Calculate Petitjean Shape Index and related descriptors
///
/// # Returns
/// (radius, diameter, petitjean_index)
///
/// where:
/// - radius = minimum eccentricity (R)
/// - diameter = maximum eccentricity (D)
/// - petitjean_index = (D - R) / D (shape measure)
///
/// # Examples
/// - Linear molecules: high diameter, Petitjean → 1
/// - Compact molecules: low diameter, Petitjean → 0
/// - Cyclic molecules: diameter depends on size
pub fn calculate(mol: &Molecule) -> (u32, u32, f64) {
    let n = mol.num_atoms();
    
    if n == 0 {
        return (0, 0, 0.0);
    }
    
    if n == 1 {
        return (0, 0, 0.0);
    }
    
    // Compute distance matrix
    let dist = compute_distance_matrix(mol);
    
    // Calculate eccentricity for each atom
    let mut eccentricities = Vec::with_capacity(n);
    for i in 0..n {
        let mut max_dist = 0u32;
        for j in 0..n {
            let d = dist[i * n + j];
            if d != u32::MAX && d > max_dist {
                max_dist = d;
            }
        }
        eccentricities.push(max_dist);
    }
    
    // Radius = min eccentricity, Diameter = max eccentricity
    let radius = *eccentricities.iter().min().unwrap();
    let diameter = *eccentricities.iter().max().unwrap();
    
    // Petitjean Index
    let petitjean = if diameter > 0 {
        (diameter - radius) as f64 / diameter as f64
    } else {
        0.0
    };
    
    (radius, diameter, petitjean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }
    
    #[test]
    fn test_petitjean_methane() {
        let mol = parse("C");
        let (r, d, p) = calculate(&mol);
        assert_eq!(r, 0);
        assert_eq!(d, 0);
        assert_eq!(p, 0.0);
    }
    
    #[test]
    fn test_petitjean_ethane() {
        let mol = parse("CC");
        let (r, d, p) = calculate(&mol);
        assert_eq!(r, 1); // Both atoms have eccentricity 1
        assert_eq!(d, 1);
        assert_eq!(p, 0.0); // Perfect symmetry
    }
    
    #[test]
    fn test_petitjean_propane() {
        let mol = parse("CCC");
        let (r, d, p) = calculate(&mol);
        assert_eq!(r, 1); // Center atom has eccentricity 1
        assert_eq!(d, 2); // End atoms have eccentricity 2
        assert!((p - 0.5).abs() < 0.01); // (2-1)/2 = 0.5
    }
    
    #[test]
    fn test_petitjean_linear_vs_branched() {
        let linear = parse("CCCCC"); // Pentane
        let branched = parse("CC(C)CC"); // Isopentane
        
        let (r_lin, d_lin, p_lin) = calculate(&linear);
        let (r_bra, d_bra, p_bra) = calculate(&branched);
        
        // Linear should have higher diameter
        assert!(d_lin > d_bra);
        
        // Linear should have higher Petitjean (more rod-like)
        assert!(p_lin > p_bra);
    }
    
    #[test]
    fn test_petitjean_benzene() {
        let mol = parse("c1ccccc1");
        let (r, d, p) = calculate(&mol);
        
        // Benzene is symmetric, all atoms have same eccentricity
        assert_eq!(r, d); // Radius = Diameter for symmetric rings
        assert_eq!(p, 0.0); // Perfect symmetry
    }
    
    #[test]
    fn test_petitjean_cyclohexane() {
        let mol = parse("C1CCCCC1");
        let (r, d, p) = calculate(&mol);
        
        // Same as benzene structure-wise
        assert_eq!(r, d);
        assert_eq!(p, 0.0);
    }
    
    #[test]
    fn test_petitjean_naphthalene() {
        let mol = parse("c1ccc2ccccc2c1");
        let (r, d, p) = calculate(&mol);
        
        // Fused rings, somewhat elongated
        assert!(d >= 4); // Distance from one end to the other
        assert!(p > 0.0 && p < 0.5); // Not perfectly symmetric
    }
    
    #[test]
    fn test_petitjean_long_chain() {
        let mol = parse("CCCCCCCCCC"); // Decane
        let (r, d, p) = calculate(&mol);
        
        println!("Decane: radius={}, diameter={}, petitjean={}", r, d, p);
        
        // Long linear chain
        assert_eq!(d, 9); // 10 carbons, max distance 9
        // For a chain of 10: center atoms at positions 4,5 have eccentricity 5
        // End atoms have eccentricity 9
        // So radius should be around 5, giving (9-5)/9 ≈ 0.44
        assert!(p > 0.4 && p < 0.6, "Petitjean = {}", p); // Moderately rod-like
    }
}
