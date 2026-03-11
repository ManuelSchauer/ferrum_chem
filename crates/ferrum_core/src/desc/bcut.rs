//! BCUT (Burden-CAS-UT) descriptors.
//!
//! Modified Burden adjacency matrix with atomic property weights on the
//! diagonal. The highest and lowest eigenvalues capture molecular shape
//! and property extremes.
//!
//! Weights: atomic mass, electronegativity, polarizability, van der Waals volume.
//! Each produces 2 eigenvalues (highest, lowest) → 8 total.
//!
//! Reference: Pearlman, R.S. & Smith, K.M. (1999) "Metric Validation and
//!            the Receptor-Relevant Subspace Concept", JCICS 39, 28-35.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

// ---------------------------------------------------------------------------
// Atomic property functions (shared with autocorrelation)
// ---------------------------------------------------------------------------

fn atomic_mass(atomic_number: u8) -> f64 {
    crate::mol::elements::ATOMIC_MASSES[atomic_number as usize] as f64
}

fn electronegativity(atomic_number: u8) -> f64 {
    match atomic_number {
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

fn polarizability(atomic_number: u8) -> f64 {
    match atomic_number {
        1 => 0.67,
        5 => 3.03,
        6 => 1.76,
        7 => 1.10,
        8 => 0.80,
        9 => 0.56,
        14 => 5.38,
        15 => 3.63,
        16 => 2.90,
        17 => 2.18,
        35 => 3.05,
        53 => 5.35,
        _ => 1.50,
    }
}

fn vdw_volume(atomic_number: u8) -> f64 {
    let r: f64 = match atomic_number {
        1 => 1.20,
        5 => 1.92,
        6 => 1.70,
        7 => 1.55,
        8 => 1.52,
        9 => 1.47,
        14 => 2.10,
        15 => 1.80,
        16 => 1.80,
        17 => 1.75,
        35 => 1.85,
        53 => 1.98,
        _ => 1.70,
    };
    (4.0 / 3.0) * std::f64::consts::PI * r.powi(3)
}

// ---------------------------------------------------------------------------
// Burden matrix + eigenvalues
// ---------------------------------------------------------------------------

/// Build the modified Burden matrix:
/// - Diagonal: atomic property weight
/// - Off-diagonal connected: bond order / √(bond_order)
/// - Off-diagonal disconnected: 0.001 (small perturbation)
fn build_burden_matrix(mol: &Molecule, weights: &[f64]) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut mat = vec![0.001; n * n]; // small off-diagonal default

    // Diagonal: property weights
    for i in 0..n {
        mat[i * n + i] = weights[i];
    }

    // Connected atoms: bond contribution
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        let bo = match bond.bond_type {
            crate::types::BondType::Single => 1.0,
            crate::types::BondType::Double => 2.0,
            crate::types::BondType::Triple => 3.0,
            crate::types::BondType::Aromatic => 1.5,
            _ => 1.0,
        };
        let val = (bo as f64).sqrt() * 0.1;
        mat[i * n + j] = val;
        mat[j * n + i] = val;
    }

    mat
}

/// Compute eigenvalues of a symmetric matrix using Jacobi iteration.
/// Returns sorted eigenvalues (ascending).
fn eigenvalues_symmetric(matrix: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![matrix[0]];
    }

    let mut a = matrix.to_vec();
    let max_iter = 100 * n * n;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i * n + j].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                new_a[i * n + p] = c * aip + s * aiq;
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * aip + c * aiq;
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;

        a = new_a;
    }

    // Extract diagonal (eigenvalues)
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

/// BCUT descriptors.
///
/// For each of 4 atomic properties (mass, electronegativity, polarizability,
/// vdW volume), computes the highest and lowest eigenvalue of the modified
/// Burden matrix.
///
/// Returns FloatVec with 8 values:
/// [mass_low, mass_high, en_low, en_high, pol_low, pol_high, vdw_low, vdw_high]
pub struct BCUT;

impl Descriptor for BCUT {
    fn name(&self) -> &str {
        "BCUT"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut values = vec![0.0f32; 8];

        let n = mol.num_atoms();
        if n == 0 {
            return DescriptorValue::FloatVec(values);
        }

        let weight_fns: [fn(u8) -> f64; 4] =
            [atomic_mass, electronegativity, polarizability, vdw_volume];

        for (wi, wfn) in weight_fns.iter().enumerate() {
            let weights: Vec<f64> = (0..n).map(|i| wfn(mol.atoms[i].atomic_number)).collect();
            let mat = build_burden_matrix(mol, &weights);
            let eigs = eigenvalues_symmetric(&mat, n);

            if !eigs.is_empty() {
                values[wi * 2] = eigs[0] as f32;
                values[wi * 2 + 1] = *eigs.last().unwrap() as f32;
            }
        }

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BCUT>;
};

#[cfg(test)]
mod tests {
    use super::super::typed::Descriptor;
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_bcut(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match BCUT.calculate(&mut mol) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    fn test_bcut_len() {
        let b = calc_bcut("CC");
        assert_eq!(b.len(), 8);
    }

    #[test]
    fn test_bcut_ethane() {
        let b = calc_bcut("CC");
        assert!(b[1] > 0.0, "mass high > 0");
    }

    #[test]
    fn test_bcut_ethanol_different_from_ethane() {
        let ethane = calc_bcut("CC");
        let ethanol = calc_bcut("CCO");
        assert_ne!(ethane, ethanol);
    }

    #[test]
    fn test_bcut_high_ge_low() {
        let b = calc_bcut("c1ccccc1");
        for i in 0..4 {
            assert!(
                b[i * 2 + 1] >= b[i * 2],
                "prop {}: high {} >= low {}",
                i,
                b[i * 2 + 1],
                b[i * 2]
            );
        }
    }

    #[test]
    fn test_bcut_deterministic() {
        let b1 = calc_bcut("CCCC");
        let b2 = calc_bcut("CCCC");
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_bcut_single_atom() {
        let b = calc_bcut("C");
        assert!(b[0] > 0.0);
        assert!((b[0] - b[1]).abs() < 0.01, "single atom: low ≈ high");
    }

    #[test]
    fn test_bcut_aspirin() {
        let b = calc_bcut("CC(=O)Oc1ccccc1C(=O)O");
        assert!(b[1] > b[0], "mass: high > low for mixed atoms");
    }

    #[test]
    fn test_eigenvalues_2x2() {
        let mat = vec![4.0, 1.0, 1.0, 4.0];
        let eigs = eigenvalues_symmetric(&mat, 2);
        assert!((eigs[0] - 3.0).abs() < 0.01, "λ₁ = {}", eigs[0]);
        assert!((eigs[1] - 5.0).abs() < 0.01, "λ₂ = {}", eigs[1]);
    }

    #[test]
    fn test_eigenvalues_identity() {
        let mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let eigs = eigenvalues_symmetric(&mat, 3);
        for e in &eigs {
            assert!((e - 1.0).abs() < 0.01, "identity eigenvalue = {}", e);
        }
    }
}
