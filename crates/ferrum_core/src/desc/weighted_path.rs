//! Weighted Path descriptors.
//!
//! Molecular path descriptors weighted by atomic properties.
//! For each bond-path of length k (k=1..5), compute:
//!   WP(k) = Σ_paths (∏ w_i for atoms in path) / k
//!
//! Property weights: mass, electronegativity.
//!
//! Reference: Randić, M. (1993) "Novel graph theoretical approach to
//!            heteroatoms in QSAR", Chemometrics & Intell. Lab. Systems.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;

fn atomic_mass(an: u8) -> f64 {
    crate::mol::elements::ATOMIC_MASSES[an as usize] as f64
}

fn electronegativity(an: u8) -> f64 {
    match an {
        1 => 2.20,
        5 => 2.04,
        6 => 2.55,
        7 => 3.04,
        8 => 3.44,
        9 => 3.98,
        14 => 1.90,
        15 => 2.19,
        16 => 2.58,
        17 => 3.16,
        35 => 2.96,
        53 => 2.66,
        _ => 2.00,
    }
}

const MAX_PATH_LEN: usize = 5;

/// Compute weighted path sums for given atom weights.
/// Returns [WP1, WP2, WP3, WP4, WP5].
fn weighted_paths(mol: &Molecule, weights: &[f64]) -> [f64; MAX_PATH_LEN] {
    let n = mol.num_atoms();
    let mut wp = [0.0f64; MAX_PATH_LEN];

    if n < 2 {
        return wp;
    }

    let mut path = Vec::with_capacity(MAX_PATH_LEN + 1);

    for start in 0..n {
        path.clear();
        path.push(start as AtomIdx);
        dfs_wp(mol, weights, &mut path, &mut wp);
    }

    // Each path found twice (forward + reverse)
    for v in wp.iter_mut() {
        *v /= 2.0;
    }

    wp
}

fn dfs_wp(mol: &Molecule, weights: &[f64], path: &mut Vec<AtomIdx>, wp: &mut [f64; MAX_PATH_LEN]) {
    let depth = path.len();
    if depth >= 2 {
        let k = depth - 1; // path length in bonds
        if k <= MAX_PATH_LEN {
            let prod: f64 = path.iter().map(|&idx| weights[idx as usize]).product();
            wp[k - 1] += prod / (k as f64);
        }
    }
    if depth > MAX_PATH_LEN {
        return;
    }

    let current = *path.last().unwrap();
    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let nb = if bond.source == current {
            bond.target
        } else {
            bond.source
        };
        if !path.contains(&nb) {
            path.push(nb);
            dfs_wp(mol, weights, path, wp);
            path.pop();
        }
    }
}

/// Weighted Path descriptors.
///
/// Returns FloatVec with 10 values:
/// [WPm1..WPm5, WPe1..WPe5]
/// m = mass-weighted, e = electronegativity-weighted.
pub struct WeightedPath;

impl Descriptor for WeightedPath {
    fn name(&self) -> &str {
        "WeightedPath"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut values = vec![0.0f32; MAX_PATH_LEN * 2];

        let n = mol.num_atoms();
        if n < 2 {
            return DescriptorValue::FloatVec(values);
        }

        let w_mass: Vec<f64> = (0..n)
            .map(|i| atomic_mass(mol.atoms[i].atomic_number))
            .collect();
        let w_en: Vec<f64> = (0..n)
            .map(|i| electronegativity(mol.atoms[i].atomic_number))
            .collect();

        let wp_mass = weighted_paths(mol, &w_mass);
        let wp_en = weighted_paths(mol, &w_en);

        for i in 0..MAX_PATH_LEN {
            values[i] = wp_mass[i] as f32;
            values[MAX_PATH_LEN + i] = wp_en[i] as f32;
        }

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<WeightedPath>;
};

#[cfg(test)]
mod tests {
    use super::super::typed::Descriptor;
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_wp(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match WeightedPath.calculate(&mut mol) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    fn test_wp_ethane() {
        let wp = calc_wp("CC");
        assert!(wp[0] > 100.0, "WPm1 = {}", wp[0]);
        assert!(wp[1].abs() < 0.001, "WPm2 should be 0");
    }

    #[test]
    fn test_wp_propane() {
        let wp = calc_wp("CCC");
        assert!(wp[0] > 0.0, "WPm1 > 0");
        assert!(wp[1] > 0.0, "WPm2 > 0");
    }

    #[test]
    fn test_wp_ethanol_vs_ethane() {
        let ethane = calc_wp("CC");
        let ethanol = calc_wp("CCO");
        assert_ne!(ethane[0], ethanol[0]);
    }

    #[test]
    fn test_wp_symmetry() {
        let a = calc_wp("CCC");
        let b = calc_wp("CCC");
        assert_eq!(a, b);
    }

    #[test]
    fn test_wp_single_atom() {
        let wp = calc_wp("C");
        for v in &wp {
            assert!(v.abs() < 0.001);
        }
    }

    #[test]
    fn test_wp_benzene() {
        let wp = calc_wp("c1ccccc1");
        assert!(wp[0] > 0.0);
        assert!(wp[1] > 0.0);
        assert!(wp[2] > 0.0);
    }
}
