//! Mordred atom-type E-State descriptors.
//!
//! Mordred (mordred-community 2.0.7) `EState.AtomTypeEState`: for each of the
//! 79 RDKit E-State atom types `X` (`estate::ESTATE_TYPE_NAMES`, same order as
//! Mordred's `es_types`)
//!
//! - `NX`: number of atoms of type `X`,
//! - `SX`: sum of their E-State indices (0 when there is none),
//! - `MAXX` / `MINX`: largest / smallest of their E-State indices (missing,
//!   i.e. NaN, when there is none).
//!
//! Output order follows Mordred: all `N`, then all `S`, `MAX`, `MIN` (316
//! values).
//!
//! Mordred types atoms with RDKit `EState.TypeAtoms` (every centre of a
//! non-unique SMARTS match, `estate::estate_type_atoms`) on the
//! hydrogen-suppressed molecule and takes RDKit `EStateIndices`. This is not
//! the RDKit `FingerprintMol` counting behind the `SEstate_<type>` columns,
//! which counts each *unique* match once (ethane: one `sCH3` match, so
//! `SEstate_sCH3` = 2.0 while Mordred `SsCH3` = 4.0), so the Mordred columns
//! are separate, not aliases.
//!
//! Reference: Hall, L.H. & Kier, L.B. (1995) "Electrotopological State
//! Indices for Atom Types", JCICS 35, 1039-1045.

use super::estate::{NUM_ESTATE_TYPES, cached_estates, estate_type_atoms};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Number of output values: N, S, MAX, MIN per E-State type.
pub const MORDRED_ESTATE_LEN: usize = 4 * NUM_ESTATE_TYPES;

/// Mordred `AtomTypeEState` (N / S / MAX / MIN per E-State type), 316 values.
pub struct MordredEState;

impl Descriptor for MordredEState {
    fn name(&self) -> &str {
        "MordredEState"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        const T: usize = NUM_ESTATE_TYPES;
        let mut values = vec![0.0f64; MORDRED_ESTATE_LEN];
        values[2 * T..].fill(f64::NAN);
        if mol.num_atoms() == 0 {
            return DescriptorValue::FloatVec(values);
        }
        let mut types = Vec::new();
        estate_type_atoms(mol, &mut types);
        let estates = cached_estates(mol, ctx);
        let (counts, rest) = values.split_at_mut(T);
        let (sums, rest) = rest.split_at_mut(T);
        let (maxs, mins) = rest.split_at_mut(T);
        // atoms in index order, as Mordred's `sum` / `max` / `min` iterate
        for (&bits, &e) in types.iter().zip(estates) {
            let mut bits = bits;
            while bits != 0 {
                let t = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if counts[t] == 0.0 {
                    maxs[t] = e;
                    mins[t] = e;
                } else {
                    maxs[t] = maxs[t].max(e);
                    mins[t] = mins[t].min(e);
                }
                counts[t] += 1.0;
                sums[t] += e;
            }
        }
        DescriptorValue::FloatVec(values)
    }
}

/// Column names in output order (`NsLi`, ..., `MINssssPb`).
pub fn mordred_estate_names() -> Vec<String> {
    let mut v = Vec::with_capacity(MORDRED_ESTATE_LEN);
    for prefix in ["N", "S", "MAX", "MIN"] {
        v.extend(
            super::estate::ESTATE_TYPE_NAMES
                .iter()
                .map(|t| format!("{prefix}{t}")),
        );
    }
    v
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MordredEState>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::estate::ESTATE_TYPE_NAMES;
    use crate::io::smiles::parse_smiles;

    fn calc(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        MordredEState
            .calculate(&mol, &mut crate::desc::context::CalculationContext::new())
            .as_float_vec()
            .to_vec()
    }

    fn idx(name: &str) -> usize {
        ESTATE_TYPE_NAMES.iter().position(|&n| n == name).unwrap()
    }

    #[test]
    fn test_len_and_names() {
        assert_eq!(calc("CCO").len(), MORDRED_ESTATE_LEN);
        let names = mordred_estate_names();
        assert_eq!(names.len(), MORDRED_ESTATE_LEN);
        assert_eq!(names[0], "NsLi");
        assert_eq!(names[NUM_ESTATE_TYPES + 6], "SsCH3");
        assert_eq!(names[MORDRED_ESTATE_LEN - 1], "MINssssPb");
    }

    /// Mordred (mordred-community 2.0.7) reference values.
    #[test]
    fn test_ethane_types_both_carbons() {
        let v = calc("CC");
        let t = idx("sCH3");
        assert_eq!(v[t], 2.0);
        assert_eq!(v[NUM_ESTATE_TYPES + t], 4.0);
        assert_eq!(v[2 * NUM_ESTATE_TYPES + t], 2.0);
        assert_eq!(v[3 * NUM_ESTATE_TYPES + t], 2.0);
        // absent type: count / sum 0, max / min missing
        let o = idx("sOH");
        assert_eq!(v[o], 0.0);
        assert_eq!(v[NUM_ESTATE_TYPES + o], 0.0);
        assert!(v[2 * NUM_ESTATE_TYPES + o].is_nan());
        assert!(v[3 * NUM_ESTATE_TYPES + o].is_nan());
    }

    /// `maxMatches = 1000` counts non-unique matches: 2 per `ssCH2` centre,
    /// so only the first 500 CH2 of a long chain are typed.
    #[test]
    fn test_non_unique_match_limit() {
        let v = calc(&"C".repeat(1100));
        assert_eq!(v[idx("ssCH2")], 500.0);
        assert_eq!(v[idx("sCH3")], 2.0);
        // 24 matches per quaternary centre: 41 full centres + one partial
        let v = calc(&format!("C{}C", "C(C)(C)".repeat(60)));
        assert_eq!(v[idx("ssssC")], 42.0);
    }

    /// `estate_type_atoms` reproduces RDKit `TypeAtoms` on the SMARTS engine:
    /// all (non-unique) matches by ascending centre, at most 1000 per type.
    #[test]
    fn test_type_atoms_match_smarts() {
        use crate::algo::iso::visit_matches;
        use crate::desc::estate::{ESTATE_TYPE_SMARTS, estate_type_atoms};
        use crate::io::smarts::parse_smarts;
        let patterns: Vec<_> = ESTATE_TYPE_SMARTS
            .iter()
            .map(|s| parse_smarts(s).expect("valid E-State SMARTS"))
            .collect();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/parity_molecules.smi");
        let text = std::fs::read_to_string(path).expect("parity molecules");
        let quaternary = format!("C{}C", "C(C)(C)".repeat(60));
        let extra = [
            "CC",
            "C1CC1",
            "[Li]C",
            "C[Be]C",
            "C[BH]C",
            "C[B-](C)(C)C",
            "C[Si](C)(C)C",
            "c1cc[se]c1",
            "C[Se](=O)(=O)C",
            "C[As](=O)(C)C",
            "C[As](C)(C)(C)C",
            "C[Ge](C)(C)C",
            "C[Sn](C)(C)C",
            "C[Pb](C)(C)C",
            "C[N+](=O)[O-]",
            "CS(=O)(=O)C",
            "CP(C)(C)(C)C",
            "[H]C([H])([H])O[H]",
            "[2H]OC",
            "c1cc[nH]c1",
            quaternary.as_str(),
        ];
        let mut types = Vec::new();
        for smi in text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .chain(extra)
        {
            let Ok(mol) = parse_smiles(smi) else {
                continue;
            };
            estate_type_atoms(&mol, &mut types);
            for (ti, patt) in patterns.iter().enumerate() {
                let mut centres = Vec::new();
                visit_matches(&mol, patt, |m| {
                    centres.push(m[0]);
                    true
                });
                centres.sort_unstable();
                centres.truncate(1000);
                for (i, &bits) in types.iter().enumerate() {
                    assert_eq!(
                        bits >> ti & 1 == 1,
                        centres.contains(&(i as u16)),
                        "{smi}: atom {i} type {}",
                        ESTATE_TYPE_NAMES[ti]
                    );
                }
            }
        }
    }
}
