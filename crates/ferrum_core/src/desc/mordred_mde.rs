//! Mordred molecular distance edge descriptors.
//!
//! Mordred (mordred-community 2.0.7) `MolecularDistanceEdge` on the
//! hydrogen-suppressed graph: atoms of one element (C, O, N) are classed by
//! their degree (`Valence`, the column sums of RDKit's adjacency matrix), and
//! for classes `a <= b`
//!
//! ```text
//! MDE(a, b) = n / exp(Σ ln d / (2n))²
//! ```
//!
//! over the n unordered atom pairs with one atom in each class (pairs within
//! the class when `a == b`), i.e. n divided by the geometric mean of their
//! topological distances. Disconnected pairs use RDKit's distance 1e8, and the
//! value is missing (NaN) when there is no pair. Output order (19 values):
//! MDEC-11, 12, 13, 14, 22, 23, 24, 33, 34, 44, MDEO-11, 12, 22, MDEN-11, 12,
//! 13, 22, 23, 33, written as `Mordred_MDEC-11` etc. because the plain names
//! hold the CDK variant (`mde.rs`, value 0 instead of NaN, CDK's own
//! hydrogen suppression and disconnected distance).
//!
//! Reference: Liu, S. et al. (1998) "Molecular Distance Edge (MDE) Vector",
//! JCICS 38, 536-540.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::paths::{DIST_INF, compute_unweighted_distance_matrix_cached};
use crate::mol::molecule::Molecule;
use smallvec::SmallVec;
use std::sync::OnceLock;

/// Mordred names (without the `Mordred_` prefix), in Mordred's order.
pub const MORDRED_MDE_NAMES: [&str; 19] = [
    "MDEC-11", "MDEC-12", "MDEC-13", "MDEC-14", "MDEC-22", "MDEC-23", "MDEC-24", "MDEC-33",
    "MDEC-34", "MDEC-44", "MDEO-11", "MDEO-12", "MDEO-22", "MDEN-11", "MDEN-12", "MDEN-13",
    "MDEN-22", "MDEN-23", "MDEN-33",
];

/// RDKit `GetDistanceMatrix` value for unreachable atom pairs.
const RDKIT_UNREACHABLE: f64 = 1e8;

/// (element class, maximum degree, output offset): C 1..=4, O 1..=2, N 1..=3.
const ELEMENTS: [(u8, usize, usize); 3] = [(6, 4, 0), (8, 2, 10), (7, 3, 13)];

/// `ln(d)` for small topological distances, built once.
fn ln_table() -> &'static [f64; 256] {
    static TABLE: OnceLock<[f64; 256]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|d| (d as f64).ln()))
}

/// Output index of the class pair `a <= b` (1-based degrees) for an element
/// with classes `1..=m` starting at `offset` (row-major upper triangle).
#[inline]
fn term_index(offset: usize, m: usize, a: usize, b: usize) -> usize {
    // rows 1..a-1 hold m, m-1, ... entries
    offset + (a - 1) * m - (a - 1) * (a - 2) / 2 + (b - a)
}

/// Mordred `MolecularDistanceEdge`, 19 values.
pub struct MordredMDE;

impl Descriptor for MordredMDE {
    fn name(&self) -> &str {
        "MordredMDE"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let n = mol.num_atoms();
        // (atom, degree) of the C / O / N atoms whose degree has a class,
        // grouped by element (`groups[e]` is the range of element `e`), each
        // group in atom order like Mordred's `i < j` loop
        let mut members: SmallVec<[(usize, usize); 64]> = SmallVec::new();
        let mut groups = [(0usize, 0usize); 3];
        for (group, &(z, m, _)) in groups.iter_mut().zip(&ELEMENTS) {
            let start = members.len();
            for (a, atom) in mol.atoms.iter().enumerate() {
                let degree = mol.adjacency[a].len();
                if atom.atomic_number == z && (1..=m).contains(&degree) {
                    members.push((a, degree));
                }
            }
            *group = (start, members.len());
        }
        let mut pairs = [0u32; 19];
        let mut log_sums = [0.0f64; 19];
        if groups.iter().any(|&(s, e)| e - s >= 2) {
            let dist = compute_unweighted_distance_matrix_cached(mol, ctx);
            let ln_cache = ln_table();
            for (&(s, e), &(_, m, offset)) in groups.iter().zip(&ELEMENTS) {
                let group = &members[s..e];
                for (p, &(i, di)) in group.iter().enumerate() {
                    let row = &dist[i * n..(i + 1) * n];
                    for &(j, dj) in &group[p + 1..] {
                        let t = term_index(offset, m, di.min(dj), di.max(dj));
                        let d = row[j];
                        let ln_d = if d == DIST_INF {
                            RDKIT_UNREACHABLE.ln()
                        } else if (d as usize) < ln_cache.len() {
                            ln_cache[d as usize]
                        } else {
                            (d as f64).ln()
                        };
                        pairs[t] += 1;
                        log_sums[t] += ln_d;
                    }
                }
            }
        }
        let values = pairs
            .iter()
            .zip(log_sums)
            .map(|(&k, s)| {
                if k == 0 {
                    return f64::NAN; // Mordred: 0 / 0 -> missing
                }
                let k = k as f64;
                let dx = (s / (2.0 * k)).exp();
                k / (dx * dx)
            })
            .collect();
        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MordredMDE>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        MordredMDE
            .calculate(&mol, &mut crate::desc::context::CalculationContext::new())
            .as_float_vec()
            .to_vec()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn test_term_index_order() {
        let mut got = Vec::new();
        for &(_, m, offset) in &ELEMENTS {
            for a in 1..=m {
                for b in a..=m {
                    got.push(term_index(offset, m, a, b));
                }
            }
        }
        assert_eq!(got, (0..19).collect::<Vec<_>>());
    }

    #[test]
    fn test_benzene() {
        let m = calc("c1ccccc1");
        // 15 pairs of degree-2 carbons: distances 1 (x6), 2 (x6), 3 (x3)
        assert!(
            close(m[4], 15.0 / (64.0f64 * 27.0).powf(1.0 / 15.0)),
            "MDEC-22 = {}",
            m[4]
        );
        assert!(m.iter().enumerate().all(|(i, &v)| i == 4 || v.is_nan()));
    }

    #[test]
    fn test_isobutane_and_disconnected() {
        let m = calc("CC(C)C");
        assert!(close(m[0], 1.5), "MDEC-11 = {}", m[0]);
        assert!(close(m[2], 3.0), "MDEC-13 = {}", m[2]);
        // methyl pairs: 2 at distance 1, 4 at RDKit's 1e8
        let m = calc("CC.CC");
        assert!(
            close(m[0], 6.0 / 1e32f64.powf(1.0 / 6.0)),
            "MDEC-11 = {}",
            m[0]
        );
    }
}
