//! Eccentric Connectivity Index
//!
//! The Eccentric Connectivity Index is a topological descriptor that combines
//! vertex eccentricity with vertex degree.
//!
//! Eccentricity ε(i) = maximum distance from vertex i to any other vertex
//! ECI = Σ [degree(i) × ε(i)] for all vertices
//!
//! Higher values indicate more extended structures.
//!
//! Reference: Sharma et al. (1997) Eccentric Connectivity Index

use crate::mol::molecule::Molecule;
use std::collections::VecDeque;

/// Compute the distance matrix using BFS from each atom
#[allow(dead_code)]
fn _compute_distance_matrix_unused(mol: &Molecule) -> Vec<u32> {
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

/// Calculate Eccentric Connectivity Index
///
/// # Returns
/// The eccentric connectivity index as a u32 (sum of degree × eccentricity)
///
/// # Examples
/// - Path graphs: increasing with length
/// - Cyclic: lower than linear of same size
/// - Branched: varies with structure
pub fn calculate(mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> u32 {
    let n = mol.num_atoms();

    if n == 0 {
        return 0;
    }

    if n == 1 {
        return 0; // Single atom has no eccentricity
    }

    // Compute distance matrix
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);

    // Calculate eccentricity for each atom
    let mut eccentricities = vec![0u32; n];
    for i in 0..n {
        let mut max_dist = 0u8;
        for j in 0..n {
            let d = dist[i * n + j];
            if d != 255 && d > max_dist {
                max_dist = d;
            }
        }
        eccentricities[i] = max_dist as u32;
    }

    // Sum degree × eccentricity
    let mut eci = 0u32;
    for i in 0..n {
        let degree = mol.get_degree(i as u16) as u32;
        eci += degree * eccentricities[i];
    }

    eci
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    fn test_eccentric_methane() {
        let mut mol = parse("C");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        assert_eq!(eci, 0); // Single atom
    }

    #[test]
    fn test_eccentric_ethane() {
        let mut mol = parse("CC");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // n=2, both atoms have degree=1, eccentricity=1
        // ECI = 1*1 + 1*1 = 2
        assert_eq!(eci, 2);
    }

    #[test]
    fn test_eccentric_propane() {
        let mut mol = parse("CCC");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // C1: degree=1, ecc=2; C2: degree=2, ecc=1; C3: degree=1, ecc=2
        // ECI = 1*2 + 2*1 + 1*2 = 6
        assert_eq!(eci, 6);
    }

    #[test]
    fn test_eccentric_butane() {
        let mut mol = parse("CCCC");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // Linear C4: end atoms ecc=3, middle atoms ecc=2
        // ECI = 1*3 + 2*2 + 2*2 + 1*3 = 14
        assert_eq!(eci, 14);
    }

    #[test]
    fn test_eccentric_isobutane() {
        let mut mol = parse("CC(C)C");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // Central C: degree=3, ecc=1; others: degree=1, ecc=2
        // ECI = 1*2 + 3*1 + 1*2 + 1*2 = 9
        assert_eq!(eci, 9);
    }

    #[test]
    fn test_eccentric_benzene() {
        let mut mol = parse("c1ccccc1");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // All atoms: degree=2, eccentricity=3 (opposite side of ring)
        // ECI = 6 * (2*3) = 36
        assert_eq!(eci, 36);
    }

    #[test]
    fn test_eccentric_cyclohexane() {
        let mut mol = parse("C1CCCCC1");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // Same as benzene structure-wise
        assert_eq!(eci, 36);
    }

    #[test]
    fn test_eccentric_naphthalene() {
        let mut mol = parse("c1ccc2ccccc2c1");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // 10 atoms, varies by position
        assert!(eci > 40); // More extended than benzene
    }

    #[test]
    fn test_eccentric_path_vs_branched() {
        let mut linear = parse("CCCCC"); // Pentane
        let mut branched = parse("CC(C)CC"); // Isopentane
        let lin_eci = calculate(&mut linear, &mut crate::desc::context::CalculationContext::new());
        let bra_eci = calculate(&mut branched, &mut crate::desc::context::CalculationContext::new());
        // Linear should have higher ECI (more extended)
        assert!(lin_eci > bra_eci);
    }
}
