use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::VecDeque;

/// Wiener Index: Sum of all shortest path distances between atom pairs.
///
/// The Wiener Index (W) is a fundamental topological descriptor that measures
/// the compactness of a molecule. It's defined as the sum of all shortest
/// topological distances (in bonds) between all pairs of non-hydrogen atoms.
///
/// # Formula
/// W = (1/2) × Σᵢ Σⱼ d(i,j)
///
/// where d(i,j) is the shortest path distance between atoms i and j.
///
/// # Properties
/// - Linear molecules have higher W than branched molecules of same size
/// - Correlates with boiling point, molar volume, and surface area
/// - Larger molecules have higher W
///
/// # Algorithm
/// Uses BFS (Breadth-First Search) from each atom to compute shortest paths.
/// Time complexity: O(n² + nm) where n = atoms, m = bonds
///
/// # Examples
/// - Methane (CH₄): 0 (only C-H bonds, no heavy-heavy paths)
/// - Ethane (C₂H₆): 1 (one C-C bond)
/// - Propane (C₃H₈): 4 (C0-C1=1, C1-C2=1, C0-C2=2)
/// - Benzene (C₆H₆): 27
///
/// # Thread Safety
/// This descriptor is stateless and thread-safe (Send + Sync).
pub struct WienerIndex;

impl WienerIndex {
    /// Compute all-pairs shortest path distances using BFS.
    fn compute_distance_matrix(mol: &Molecule) -> Vec<u32> {
        let n = mol.num_atoms();
        let mut distances = vec![u32::MAX; n * n];

        for start in 0..n {
            let start_idx = start as AtomIdx;
            let mut queue = VecDeque::new();
            queue.push_back(start_idx);
            distances[start * n + start] = 0;

            while let Some(current) = queue.pop_front() {
                let current_dist = distances[start * n + current as usize];

                for &bond_idx in &mol.adjacency[current as usize] {
                    let bond = &mol.bonds[bond_idx as usize];
                    let neighbor = if bond.source == current {
                        bond.target
                    } else {
                        bond.source
                    };

                    let neighbor_dist = &mut distances[start * n + neighbor as usize];
                    if *neighbor_dist == u32::MAX {
                        *neighbor_dist = current_dist + 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances
    }

    /// Calculate Wiener Index from distance matrix.
    fn wiener_from_distances(distances: &[u32], n: usize) -> f32 {
        let mut sum: u64 = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = distances[i * n + j];
                if dist != u32::MAX {
                    sum += dist as u64;
                }
            }
        }

        sum as f32
    }
}

impl Descriptor for WienerIndex {
    fn name(&self) -> &str {
        "WienerIndex"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let n = mol.num_atoms();

        if n <= 1 {
            return DescriptorValue::Float(0.0);
        }

        let distances = Self::compute_distance_matrix(mol);
        let wiener = Self::wiener_from_distances(&distances, n);

        DescriptorValue::Float(wiener)
    }
}

// Static assertion: WienerIndex must be Send + Sync for thread safety
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<WienerIndex>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calculate_wiener(smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        WienerIndex.calculate(&mut mol).as_float()
    }

    #[test]
    fn test_wiener_methane() {
        assert_eq!(calculate_wiener("C"), 0.0);
    }

    #[test]
    fn test_wiener_ethane() {
        assert_eq!(calculate_wiener("CC"), 1.0);
    }

    #[test]
    fn test_wiener_propane() {
        assert_eq!(calculate_wiener("CCC"), 4.0);
    }

    #[test]
    fn test_wiener_butane() {
        assert_eq!(calculate_wiener("CCCC"), 10.0);
    }

    #[test]
    fn test_wiener_isobutane() {
        assert_eq!(calculate_wiener("CC(C)C"), 9.0);
    }

    #[test]
    fn test_wiener_benzene() {
        assert_eq!(calculate_wiener("c1ccccc1"), 27.0);
    }

    #[test]
    fn test_wiener_cyclohexane() {
        assert_eq!(calculate_wiener("C1CCCCC1"), 27.0);
    }

    #[test]
    fn test_wiener_naphthalene() {
        let wiener = calculate_wiener("c1ccc2ccccc2c1");
        assert!(wiener > 27.0, "Naphthalene should have higher W than benzene");
    }

    #[test]
    fn test_wiener_linear_vs_branched() {
        let linear = calculate_wiener("CCCC");
        let branched = calculate_wiener("CC(C)C");
        assert!(linear > branched);
    }

    #[test]
    fn test_wiener_comprehensive() {
        let test_cases = vec![
            ("C", 0.0, "methane"),
            ("CC", 1.0, "ethane"),
            ("CCC", 4.0, "propane"),
            ("CCCC", 10.0, "butane"),
            ("CC(C)C", 9.0, "isobutane"),
            ("CCCCC", 20.0, "pentane"),
            ("c1ccccc1", 27.0, "benzene"),
            ("C1CCCCC1", 27.0, "cyclohexane"),
        ];

        for (smiles, expected, name) in test_cases {
            let actual = calculate_wiener(smiles);
            assert_eq!(actual, expected, "{} ({}): expected W={}, got W={}", name, smiles, expected, actual);
        }
    }
}
