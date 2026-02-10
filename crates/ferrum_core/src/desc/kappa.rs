//! Kier-Hall Kappa shape indices.
//!
//! Three shape descriptors based on path counts:
//! - κ₁: based on 1-bond paths (edges)
//! - κ₂: based on 2-bond paths
//! - κ₃: based on 3-bond paths
//!
//! Reference: Kier, L.B. (1985) "A Shape Index from Molecular Graphs"
//!            Quant. Struct.-Act. Relat. 4, 109-116.

use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Kier-Hall Kappa Shape Indices (κ₁, κ₂, κ₃).
///
/// Measures molecular shape/branching:
/// - κ₁ = n(n-1)² / p₁²  where p₁ = number of bonds
/// - κ₂ = (n-1)(n-2)² / p₂²  where p₂ = number of 2-bond paths
/// - κ₃ depends on parity of n, using p₃ = number of 3-bond paths
///
/// Higher values indicate more branched/compact molecules.
/// Returns 3 values: [κ₁, κ₂, κ₃].
pub struct KappaIndices;

impl Descriptor for KappaIndices {
    fn name(&self) -> String {
        "Kappa".to_string()
    }

    fn size(&self) -> usize {
        3
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let n = mol.num_atoms();

        if n == 0 {
            output[0] = 0.0;
            output[1] = 0.0;
            output[2] = 0.0;
            return;
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
        output[0] = if p1 > 0.0 {
            (nf * (nf - 1.0).powi(2) / p1.powi(2)) as Real
        } else {
            0.0
        };

        // κ₂ = (n-1)(n-2)² / p₂²
        output[1] = if p2 > 0.0 && n >= 3 {
            ((nf - 1.0) * (nf - 2.0).powi(2) / p2.powi(2)) as Real
        } else {
            0.0
        };

        // κ₃: depends on parity of n
        output[2] = if p3 > 0.0 && n >= 4 {
            if n % 2 == 1 {
                // odd n: (n-1)(n-3)² / p₃²
                ((nf - 1.0) * (nf - 3.0).powi(2) / p3.powi(2)) as Real
            } else {
                // even n: (n-3)(n-2)² / p₃²
                ((nf - 3.0) * (nf - 2.0).powi(2) / p3.powi(2)) as Real
            }
        } else {
            0.0
        };
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

    fn calc_kappa(smiles: &str) -> [Real; 3] {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = [0.0; 3];
        KappaIndices.calculate(&mol, &mut output);
        output
    }

    #[test]
    fn test_kappa_methane() {
        let k = calc_kappa("C");
        assert_eq!(k[0], 0.0); // no bonds
        assert_eq!(k[1], 0.0);
        assert_eq!(k[2], 0.0);
    }

    #[test]
    fn test_kappa_ethane() {
        let k = calc_kappa("CC");
        // n=2, p1=1: κ₁ = 2*1/1 = 2
        assert!((k[0] - 2.0).abs() < 0.01, "κ₁ of ethane = {}", k[0]);
        assert_eq!(k[1], 0.0); // n<3
        assert_eq!(k[2], 0.0); // n<4
    }

    #[test]
    fn test_kappa_propane() {
        let k = calc_kappa("CCC");
        // n=3, p1=2: κ₁ = 3*4/4 = 3
        assert!((k[0] - 3.0).abs() < 0.01, "κ₁ = {}", k[0]);
        // n=3, p2=1: κ₂ = 2*1/1 = 2
        assert!((k[1] - 2.0).abs() < 0.01, "κ₂ = {}", k[1]);
    }

    #[test]
    fn test_kappa_butane_vs_isobutane() {
        let k_linear = calc_kappa("CCCC");
        let k_branched = calc_kappa("CC(C)C");
        println!("Butane κ: {:?}", k_linear);
        println!("Isobutane κ: {:?}", k_branched);
        // Same atom count and bond count → same κ₁
        assert!((k_linear[0] - k_branched[0]).abs() < 0.01);
        // But κ₂ differs: branched has more 2-paths through central atom
        // Butane p2: degrees [1,2,2,1] → C(1,2)=0 + C(2,2)=1 + C(2,2)=1 + C(1,2)=0 = 2
        // Isobutane p2: degrees [1,3,1,1] → 0 + C(3,2)=3 + 0 + 0 = 3
        // κ₂ = (n-1)(n-2)²/p₂² → butane: 3*4/4=3, isobutane: 3*4/9≈1.33
        assert!(k_linear[1] > k_branched[1],
            "linear κ₂={} should > branched κ₂={}", k_linear[1], k_branched[1]);
    }

    #[test]
    fn test_kappa_pentane_linear() {
        let k = calc_kappa("CCCCC");
        // n=5, p1=4: κ₁ = 5*16/16 = 5
        assert!((k[0] - 5.0).abs() < 0.01, "κ₁ = {}", k[0]);
    }

    #[test]
    fn test_kappa_neopentane() {
        let k = calc_kappa("CC(C)(C)C");
        println!("Neopentane κ: {:?}", k);
        // Most branched C5: κ₂ should be lowest (most branched = smallest p₂ fraction)
        let k_linear = calc_kappa("CCCCC");
        // Same κ₁ (same bond count), but different κ₂
        assert!((k[0] - k_linear[0]).abs() < 0.01);
        assert!(k[1] < k_linear[1],
            "neopentane κ₂={} should < pentane κ₂={}", k[1], k_linear[1]);
    }

    #[test]
    fn test_kappa_benzene() {
        let k = calc_kappa("c1ccccc1");
        println!("Benzene κ: {:?}", k);
        // n=6, p1=6: κ₁ = 6*25/36 ≈ 4.17
        assert!(k[0] > 3.0 && k[0] < 6.0, "κ₁ = {}", k[0]);
        assert!(k[1] > 0.0);
        assert!(k[2] > 0.0);
    }

    #[test]
    fn test_kappa_cyclohexane() {
        let k = calc_kappa("C1CCCCC1");
        println!("Cyclohexane κ: {:?}", k);
        assert!(k[0] > 3.0 && k[0] < 6.0);
    }
}
