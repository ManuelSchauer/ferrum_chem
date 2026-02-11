//! Kier-Hall Kappa shape indices.
//!
//! Three shape descriptors based on path counts:
//! - κ₁: based on 1-bond paths (edges)
//! - κ₂: based on 2-bond paths
//! - κ₃: based on 3-bond paths
//!
//! Reference: Kier, L.B. (1985) "A Shape Index from Molecular Graphs"
//!            Quant. Struct.-Act. Relat. 4, 109-116.

use super::typed::{DescriptorValue, Descriptor};
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

        let nf = n as f64;

        // p₁ = number of bonds (paths of length 1)
        let p1 = mol.num_bonds() as f64;

        // p₂ = sum of C(degree_i, 2) for each atom
        let p2: f64 = (0..n)
            .map(|i| {
                let d = mol.get_degree(i as u16) as f64;
                d * (d - 1.0) / 2.0
            })
            .sum();

        // p₃ = sum over all bonds of (deg_i - 1)(deg_j - 1)
        let p3: f64 = mol.bonds.iter()
            .map(|bond| {
                let di = mol.get_degree(bond.source) as f64;
                let dj = mol.get_degree(bond.target) as f64;
                (di - 1.0) * (dj - 1.0)
            })
            .sum();

        // κ₁ = n(n-1)² / p₁²
        let kappa1 = if p1 > 0.0 {
            (nf * (nf - 1.0).powi(2) / p1.powi(2)) as f32
        } else {
            0.0
        };

        // κ₂ = (n-1)(n-2)² / p₂²
        let kappa2 = if p2 > 0.0 && n >= 3 {
            ((nf - 1.0) * (nf - 2.0).powi(2) / p2.powi(2)) as f32
        } else {
            0.0
        };

        // κ₃: depends on parity of n
        let kappa3 = if p3 > 0.0 && n >= 4 {
            if n % 2 == 1 {
                // odd n: (n-1)(n-3)² / p₃²
                ((nf - 1.0) * (nf - 3.0).powi(2) / p3.powi(2)) as f32
            } else {
                // even n: (n-3)(n-2)² / p₃²
                ((nf - 3.0) * (nf - 2.0).powi(2) / p3.powi(2)) as f32
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
        assert!(k_linear[1] > k_branched[1],
            "linear κ₂={} should > branched κ₂={}", k_linear[1], k_branched[1]);
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
    fn test_kappa_size() {
        let k = calc_kappa("C");
        assert_eq!(k.len(), 3);
    }
}
