//! CDK 2.13 `AtomPairs2DFingerprinter` (780 bits).
//!
//! One bit per (topological distance 1..=10, element pair) for the elements
//! C, N, O, S, P, F, Cl, Br, I, B, Si, plus "X" (any halogen) pairs: for every
//! pair of such atoms at most ten bonds apart CDK sets `dist_A_B` (element
//! symbols in that order of the type list) and, if one of them is a halogen,
//! also the pair with the halogen(s) written as `X`. The bit index is the
//! position of `dist_A_B` in CDK's key enumeration (distance-major, then the
//! upper triangle of the 12 x 12 type pairs).

use bitvec::prelude::*;

use super::context::CalculationContext;
use super::typed::{Descriptor, DescriptorValue};
use crate::algo::paths::{DIST_INF, compute_unweighted_distance_matrix_cached};
use crate::mol::molecule::Molecule;

/// Number of bits (10 distances x 78 type pairs).
pub const ATOM_PAIRS_2D_LEN: usize = 780;
const MAX_DISTANCE: u16 = 10;
const N_TYPES: usize = 12;
/// Type index of "X" (halogen) in CDK's type list.
const HALOGEN: usize = 11;

/// CDK type index (`C N O S P F Cl Br I B Si`), `None` for atoms CDK skips.
fn atom_type(z: u8) -> Option<usize> {
    Some(match z {
        6 => 0,
        7 => 1,
        8 => 2,
        16 => 3,
        15 => 4,
        9 => 5,
        17 => 6,
        35 => 7,
        53 => 8,
        5 => 9,
        14 => 10,
        _ => return None,
    })
}

fn is_halogen(t: usize) -> bool {
    (5..=8).contains(&t)
}

/// Bit of `dist_A_B` with `a <= b` (CDK `pathToBit`).
fn bit(dist: u16, a: usize, b: usize) -> usize {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    // the pairs (i, j >= i) of the types before `a`, then the offset of `b`
    let row_start = a * N_TYPES - a * a.saturating_sub(1) / 2;
    (dist as usize - 1) * (N_TYPES * (N_TYPES + 1) / 2) + row_start + (b - a)
}

/// CDK `AtomPairs2DFingerprinter`, bit fingerprint (780 bits).
pub struct AtomPairs2D;

impl Descriptor for AtomPairs2D {
    fn name(&self) -> &str {
        "AtomPairs2D"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        let mut bits = bitvec![u8, Lsb0; 0; ATOM_PAIRS_2D_LEN];
        let n = mol.num_atoms();
        let types: Vec<Option<usize>> = mol.atoms.iter().map(|a| atom_type(a.atomic_number)).collect();
        if types.iter().flatten().count() < 2 {
            return DescriptorValue::Binary(bits);
        }
        let dist = compute_unweighted_distance_matrix_cached(mol, ctx);
        for i in 0..n {
            let Some(ti) = types[i] else { continue };
            for j in i + 1..n {
                let Some(tj) = types[j] else { continue };
                let d = dist[i * n + j];
                if d == DIST_INF || d > MAX_DISTANCE {
                    continue;
                }
                bits.set(bit(d, ti, tj), true);
                if is_halogen(ti) || is_halogen(tj) {
                    let hi = if is_halogen(ti) { HALOGEN } else { ti };
                    let hj = if is_halogen(tj) { HALOGEN } else { tj };
                    bits.set(bit(d, hi, hj), true);
                }
            }
        }
        DescriptorValue::Binary(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn bit_layout() {
        assert_eq!(bit(1, 0, 0), 0); // 1_C_C
        assert_eq!(bit(1, 0, 11), 11); // 1_C_X
        assert_eq!(bit(1, 1, 1), 12); // 1_N_N
        assert_eq!(bit(1, 11, 11), 77); // 1_X_X
        assert_eq!(bit(10, 11, 11), 779);
        assert_eq!(bit(2, 2, 0), 78 + 2); // 2_C_O
    }

    #[test]
    fn chloroethanol() {
        let mol = parse_smiles("OCCCl").unwrap();
        let on: Vec<usize> =
            AtomPairs2D.calculate(&mol, &mut CalculationContext::new()).as_binary().iter_ones().collect();
        // 1_C_C, 1_C_O, 1_C_Cl, 1_C_X, 2_C_O, 2_C_Cl, 2_C_X, 3_O_Cl, 3_O_X
        let expect = [
            bit(1, 0, 0),
            bit(1, 0, 2),
            bit(1, 0, 6),
            bit(1, 0, 11),
            bit(2, 0, 2),
            bit(2, 0, 6),
            bit(2, 0, 11),
            bit(3, 2, 6),
            bit(3, 2, 11),
        ];
        let mut expect = expect.to_vec();
        expect.sort_unstable();
        assert_eq!(on, expect);
    }
}
