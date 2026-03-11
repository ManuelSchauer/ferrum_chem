//! Kier-Hall Kappa shape indices.
//!
//! Three shape descriptors based on path counts:
//! - κ₁: based on 1-bond paths (edges)
//! - κ₂: based on 2-bond paths
//! - κ₃: based on 3-bond paths
//!
//! Reference: Kier, L.B. (1985) "A Shape Index from Molecular Graphs"
//!            Quant. Struct.-Act. Relat. 4, 109-116.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::paths::find_all_paths_of_length;
use crate::mol::molecule::Molecule;

// (Legacy descriptor removed — use KappaIndices below)

// ===========================================================================
// Typed API: Kappa Indices
// ===========================================================================

/// Kier-Hall Kappa Shape Indices (κ₁, κ₂, κ₃).
///
/// Measures molecular shape/branching:
/// - κ₁ = n(n-1)² / p₁²  where p₁ = number of bonds
/// - κ₂ = (n-1)(n-2)² / p₂²  where p₂ = number of 2-bond paths
/// - κ₃ depends on parity of n, using p₃ = number of 3-bond paths
///
/// Returns FloatVec with 3 values: [κ₁, κ₂, κ₃]
pub struct KappaIndices;

impl Descriptor for KappaIndices {
    fn name(&self) -> &str {
        "KappaIndices"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let n = mol.num_atoms();

        if n == 0 {
            return DescriptorValue::FloatVec(vec![0.0, 0.0, 0.0]);
        }

        let a = n as f64;

        // Hall-Kier alpha correction
        let alpha = crate::desc::extra_counts::compute_hall_kier_alpha_f64(mol);
        let a_alpha = a + alpha;

        // P₁ = number of bonds
        let p1 = mol.num_bonds() as f64;

        // P₂ = sum of C(degree_i, 2) for each atom
        let p2: f64 = (0..n)
            .map(|i| {
                let d = mol.get_degree(i as u16) as f64;
                d * (d - 1.0) / 2.0
            })
            .sum();

        // RDKit counts explicit 3-bond paths and includes ring-closing paths.
        let p3 = find_all_paths_of_length(mol, 3).len() as f64;

        // κ₁ = (A + α)(A + α - 1)² / (P1 + α)²
        let denom1 = p1 + alpha;
        let kappa1 = if denom1.abs() > 1e-10 {
            (a_alpha * (a_alpha - 1.0).powi(2) / denom1.powi(2)) as f32
        } else {
            0.0
        };

        // κ₂ = (A + α - 1)(A + α - 2)² / (P2 + α)²
        let denom2 = p2 + alpha;
        let kappa2 = if denom2.abs() > 1e-10 {
            ((a_alpha - 1.0) * (a_alpha - 2.0).powi(2) / denom2.powi(2)) as f32
        } else {
            0.0
        };

        // κ₃ depends on parity of A
        let denom3 = p3 + alpha;
        let kappa3 = if denom3.abs() > 1e-10 {
            if n % 2 == 1 {
                ((a_alpha - 1.0) * (a_alpha - 3.0).powi(2) / denom3.powi(2)) as f32
            } else {
                ((a_alpha - 2.0) * (a_alpha - 3.0).powi(2) / denom3.powi(2)) as f32
            }
        } else {
            0.0
        };

        DescriptorValue::FloatVec(vec![kappa1, kappa2, kappa3])
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<KappaIndices>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_kappa(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match KappaIndices.calculate(&mut mol) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    fn test_kappa_methane() {
        let k = calc_kappa("C");
        assert_eq!(k[0], 0.0);
        assert_eq!(k[1], 0.0);
        assert_eq!(k[2], 0.0);
    }

    #[test]
    fn test_kappa_ethane() {
        let k = calc_kappa("CC");
        assert!((k[0] - 2.0).abs() < 0.01, "κ₁ of ethane = {}", k[0]);
        assert_eq!(k[1], 0.0);
        assert_eq!(k[2], 0.0);
    }

    #[test]
    fn test_kappa_propane() {
        let k = calc_kappa("CCC");
        assert!((k[0] - 3.0).abs() < 0.01, "κ₁ = {}", k[0]);
        assert!((k[1] - 2.0).abs() < 0.01, "κ₂ = {}", k[1]);
    }

    #[test]
    fn test_kappa_butane_vs_isobutane() {
        let k_linear = calc_kappa("CCCC");
        let k_branched = calc_kappa("CC(C)C");
        assert!((k_linear[0] - k_branched[0]).abs() < 0.01);
        assert!(
            k_linear[1] > k_branched[1],
            "linear κ₂={} should > branched κ₂={}",
            k_linear[1],
            k_branched[1]
        );
    }

    #[test]
    fn test_kappa_pentane_linear() {
        let k = calc_kappa("CCCCC");
        assert!((k[0] - 5.0).abs() < 0.01, "κ₁ = {}", k[0]);
    }

    #[test]
    fn test_kappa_benzene() {
        let k = calc_kappa("c1ccccc1");
        assert!(k[0] > 3.0 && k[0] < 6.0, "κ₁ = {}", k[0]);
        assert!(k[1] > 0.0);
        assert!(k[2] > 0.0);
    }

    #[test]
    fn test_kappa3_epoxide_regression_matches_rdkit() {
        let k = calc_kappa("CC1OC1");
        assert!((k[2] - 0.20616508).abs() < 1e-5, "κ₃ = {}", k[2]);
    }

    #[test]
    fn test_kappa_size() {
        let k = calc_kappa("C");
        assert_eq!(k.len(), 3);
    }
}
