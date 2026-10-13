//! Molecular Distance Edge descriptors (CDK `MDEDescriptor`).
//!
//! Atoms of one element are split by the number of non-hydrogen neighbours
//! (degree 1-4 for carbon, 1-2 for oxygen, 1-3 for nitrogen). For two such
//! classes i, j the descriptor is
//!
//! ```text
//! MDE(i, j) = n / (prod d)^(1/n)
//! ```
//!
//! over the n atom pairs of the two classes, i.e. the pair count divided by
//! the geometric mean of their topological distances. Output order (19
//! values): MDEC-11, 12, 13, 14, 22, 23, 24, 33, 34, 44, MDEO-11, 12, 22,
//! MDEN-11, 12, 13, 22, 23, 33.
//!
//! The product is accumulated in `f64` in CDK's loop order (ordered pairs and
//! a square root for equal classes), so CDK's overflow to infinity (value 0)
//! on large classes is reproduced; disconnected pairs use CDK's distance
//! 999 999 999.
//!
//! Reference: Liu, S. et al. (1998) "Molecular Distance Edge (MDE) Vector",
//! JCICS 38, 536-540.

use super::cdk_graph::cdk_distance;
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// (element, class i, class j) for each output, in CDK order.
const MDE_TERMS: [(u8, u8, u8); 19] = [
    (6, 1, 1),
    (6, 1, 2),
    (6, 1, 3),
    (6, 1, 4),
    (6, 2, 2),
    (6, 2, 3),
    (6, 2, 4),
    (6, 3, 3),
    (6, 3, 4),
    (6, 4, 4),
    (8, 1, 1),
    (8, 1, 2),
    (8, 2, 2),
    (7, 1, 1),
    (7, 1, 2),
    (7, 1, 3),
    (7, 2, 2),
    (7, 2, 3),
    (7, 3, 3),
];

/// CDK `MDEDescriptor.evalCValue` for the atom lists `v1`, `v2`.
fn eval_c_value(dist: &[u16], n_atoms: usize, v1: &[usize], v2: &[usize], same: bool) -> f64 {
    let mut lambda = 1.0f64;
    let mut n = 0.0f64;
    for &a in v1 {
        let row = &dist[a * n_atoms..(a + 1) * n_atoms];
        for &b in v2 {
            if a == b {
                continue;
            }
            lambda *= cdk_distance(row[b]) as f64;
            n += 1.0;
        }
    }
    if same {
        lambda = lambda.sqrt();
        n /= 2.0;
    }
    if n == 0.0 {
        0.0
    } else {
        let root = lambda.powf(1.0 / (2.0 * n));
        n / (root * root)
    }
}

/// MDE descriptors (CDK `MDEDescriptor`), 19 values.
pub struct MDE;

impl Descriptor for MDE {
    fn name(&self) -> &str {
        "MDE"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        // CDK computes on `removeHydrogens(container)`
        if let Some(mut suppressed) = crate::algo::sanitize::cdk_remove_hydrogens(mol) {
            return self.calculate(&mut suppressed, ctx);
        }
        let n = mol.num_atoms();
        // classes[element][degree]: atom indices in ascending order
        let mut classes: [[Vec<usize>; 5]; 3] = Default::default();
        let mut any_pair = false;
        for (a, atom) in mol.atoms.iter().enumerate() {
            let e = match atom.atomic_number {
                6 => 0,
                8 => 1,
                7 => 2,
                _ => continue,
            };
            let degree = mol.adjacency[a].len();
            if (1..=4).contains(&degree) {
                classes[e][degree].push(a);
                any_pair = true;
            }
        }
        let mut values = vec![0.0f64; MDE_TERMS.len()];
        if !any_pair || n < 2 {
            return DescriptorValue::FloatVec(values);
        }
        let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
        for (out, &(z, i, j)) in values.iter_mut().zip(MDE_TERMS.iter()) {
            let e = match z {
                6 => 0,
                8 => 1,
                _ => 2,
            };
            let (v1, v2) = (&classes[e][i as usize], &classes[e][j as usize]);
            if v1.is_empty() || v2.is_empty() {
                continue;
            }
            *out = eval_c_value(dist, n, v1, v2, i == j) as f64;
        }
        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MDE>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_mde(smiles: &str) -> Vec<f64> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        MDE.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_float_vec().to_vec()
    }

    fn close(a: f64, b: f64) -> bool {
        (a as f64 - b).abs() < 1e-5
    }

    #[test]
    fn test_mde_len() {
        assert_eq!(calc_mde("CC").len(), 19);
        assert_eq!(calc_mde("C").len(), 19);
    }

    // Expected values from CDK 2.13 MDEDescriptor.
    #[test]
    fn test_mde_benzene() {
        let m = calc_mde("c1ccccc1");
        // 15 pairs of degree-2 carbons: distances 1 (x6), 2 (x6), 3 (x3)
        assert!(close(m[4], 15.0 / (64.0f64 * 27.0).powf(1.0 / 15.0)), "MDEC-22 = {}", m[4]);
        assert!(m.iter().enumerate().all(|(i, &v)| i == 4 || v == 0.0));
    }

    #[test]
    fn test_mde_ethanol() {
        // C1 (degree 1) - C2 (degree 2) - O (degree 1)
        let m = calc_mde("CCO");
        assert!(close(m[1], 1.0), "MDEC-12 = {}", m[1]);
        assert_eq!(m[10], 0.0); // a single degree-1 O: no O-O pair
    }

    #[test]
    fn test_mde_isobutane() {
        // three methyls at mutual distance 2 -> 3 / 2
        let m = calc_mde("CC(C)C");
        assert!(close(m[0], 1.5), "MDEC-11 = {}", m[0]);
        assert!(close(m[2], 3.0), "MDEC-13 = {}", m[2]);
    }

    #[test]
    fn test_mde_nitrogen() {
        // NCCN: two degree-1 N at distance 3
        let m = calc_mde("NCCN");
        assert!(close(m[13], 1.0 / 3.0), "MDEN-11 = {}", m[13]);
    }

    #[test]
    fn test_mde_disconnected_uses_cdk_sentinel() {
        let m = calc_mde("CC.CC");
        // methyl pairs: 2 at distance 1, 4 at 999999999
        let expected = 6.0 / (999_999_999.0f64.powi(4)).powf(1.0 / 6.0);
        assert!(close(m[0], expected), "MDEC-11 = {}", m[0]);
    }

    #[test]
    fn test_mde_deterministic() {
        assert_eq!(calc_mde("CCCC"), calc_mde("CCCC"));
    }
}
