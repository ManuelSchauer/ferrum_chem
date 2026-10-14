//! CDK MACCS keys: a port of CDK 2.13 `MACCSFingerprinter` (166 bits).
//!
//! The keys are CDK's own definitions (`org/openscience/cdk/fingerprint/data/maccs.txt`,
//! derived from RDKit's but matched with CDK SMARTS semantics on the molecule
//! as `SmartsPattern.prepare` leaves it, see [`CdkSmartsTarget`]). Bit `i` is
//! CDK's bit `i` (MACCS key `i + 1`). As in CDK:
//!
//! - key 1 (`[!0]`) is set when an atom has a mass number;
//! - keys with a count `k > 0` need at least `k + 1` matches with distinct atom
//!   sets (`Mappings.uniqueAtoms().atLeast(k + 1)`);
//! - the ring-size keys (3- to 8-membered rings) come from all simple cycles of
//!   up to 8 atoms (`AllCycles`), bit 162 is set by the first and bit 144 by a
//!   second 6-cycle, and bit 124 (key 125, `[!*]`) by a second aromatic 6-cycle;
//! - key 166 (`(*).(*)`) is set for disconnected molecules.

use super::cdk_smarts_target::{CdkSmartsTarget, parse_cdk_smarts};
use super::context::CalculationContext;
use super::typed::{BitVector, Descriptor, DescriptorValue};
use crate::algo::iso::{IsoTarget, QueryScreen, has_at_least_matches_target, has_substruct_match_target};
use crate::io::smarts::QueryGraph;
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use bitvec::prelude::*;
use std::sync::OnceLock;

/// Number of CDK MACCS bits.
pub const CDK_MACCS_SIZE: usize = 166;

/// CDK `maccs.txt`: (SMARTS, count) per key, keys 1..=166.
const KEYS: [(&str, u8); CDK_MACCS_SIZE] = [
    ("[!0]", 0),
    ("[#104,#105,#106,#107,#108,#109,#110,#111,#112]", 0),
    ("[Ge,As,as,Se,se,Sn,Sb,Te,Tl,Pb,Bi]", 0),
    ("[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]", 0),
    ("[Sc,Ti,Y,Zr,Hf]", 0),
    ("[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]", 0),
    ("[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]", 0),
    ("[!#6;!#1]1~*~*~*~1", 0),
    ("[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]", 0),
    ("[Be,Mg,Ca,Sr,Ba,Ra]", 0),
    ("[R]1@*@*@*@1", 0),
    ("[Cu,Zn,Ag,Cd,Au,Hg]", 0),
    ("[#8]~[#7](~[#6])~[#6]", 0),
    ("[#16]-[#16]", 0),
    ("[#8]~[#6](~[#8])~[#8]", 0),
    ("[R!#6!#1]1~*~*~1", 0),
    ("[#6]#[#6]", 0),
    ("[#5,Al,Ga,In,Tl]", 0),
    ("[R]1@*@*@*@*@*@*@1", 0),
    ("[Si]", 0),
    ("[#6]=[#6](~[!#6;!#1])~[!#6;!#1]", 0),
    ("[R]1@*@*@1", 0),
    ("[#7]~[#6](~[#8])~[#8]", 0),
    ("[#7]-[#8]", 0),
    ("[#7]~[#6](~[#7])~[#7]", 0),
    ("[#6]=;@[#6](@*)@*", 0),
    ("[I]", 0),
    ("[!#6;!#1]~[CH2]~[!#6;!#1]", 0),
    ("[#15]", 0),
    ("[#6]~[!#6;!#1](~[#6])(~[#6])~*", 0),
    ("[!#6;!#1]~[F,Cl,Br,I]", 0),
    ("[#6]~[#16]~[#7]", 0),
    ("[#7]~[#16]", 0),
    ("[CH2]=*", 0),
    ("[Li,Na,K,Rb,Cs,Fr]", 0),
    ("[#16R]", 0),
    ("[#7]~[#6](~[#8])~[#7]", 0),
    ("[#7]~[#6](~[#6])~[#7]", 0),
    ("[#8]~[#16](~[#8])~[#8]", 0),
    ("[#16]-[#8]", 0),
    ("[#6]#[#7]", 0),
    ("F", 0),
    ("[!C;!c;!#1;!H0]~*~[!C;!c;!#1;!H0]", 0),
    ("[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]", 0),
    ("[#6]=[#6]~[#7]", 0),
    ("Br", 0),
    ("[#16]~*~[#7]", 0),
    ("[#8]~[!#6;!#1](~[#8])(~[#8])", 0),
    ("[!+0]", 0),
    ("[#6]=[#6](~[#6])~[#6]", 0),
    ("[#6]~[#16]~[#8]", 0),
    ("[#7]~[#7]", 0),
    ("[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]", 0),
    ("[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]", 0),
    ("[#8]~[#16]~[#8]", 0),
    ("[#8]~[#7](~[#8])~[#6]", 0),
    ("[#8R]", 0),
    ("[!#6;!#1]~[#16]~[!#6;!#1]", 0),
    ("[#16]!:*:*", 0),
    ("[#16]=[#8]", 0),
    ("[#16](~*)(~*)~*", 0),
    ("[R]!@[R]", 0),
    ("[#7]=[#8]", 0),
    ("[#16]!@*@*", 0),
    ("n:c", 0),
    ("[#6]~[#6](~[#6])(~[#6])~*", 0),
    ("[!#6;!#1]~[#16]", 0),
    ("[!#6;!#1;!H0]~[!#6;!#1;!H0]", 0),
    ("[!#6;!#1]~[!#6;!#1;!H0]", 0),
    ("[!#6;!#1]~[#7]~[!#6;!#1]", 0),
    ("[#7]~[#8]", 0),
    ("[#8]~*~*~[#8]", 0),
    ("[#16]=*", 0),
    ("[CH3]~*~[CH3]", 0),
    ("[#7](!@*)@*", 0),
    ("[#6]=[#6](~*)~*", 0),
    ("[#7]~*~[#7]", 0),
    ("[#6]=[#7]", 0),
    ("[#7]~*~*~[#7]", 0),
    ("[#7]~*~*~*~[#7]", 0),
    ("[#16]~*(~*)~*", 0),
    ("[CH2](~*)~[!#6;!#1;!H0]", 0),
    ("[!#6;!#1]1~*~*~*~*~1", 0),
    ("[NH2]", 0),
    ("[#6]~[#7](~[#6])~[#6]", 0),
    ("[C;H2,H3][!#6;!#1][C;H2,H3]", 0),
    ("[F,Cl,Br,I]!@*@*", 0),
    ("[#16]", 0),
    ("[#8]~*~*~*~[#8]", 0),
    ("[#6H2](~*~*~[!#6!#1!H0])~*", 0),
    ("[#6H2](~*~*~*~[!#6!#1!H0])~*", 0),
    ("[#8]~[#6](~[#7])~[#6]", 0),
    ("[!#6;!#1]~[CH3]", 0),
    ("[!#6;!#1]~[#7]", 0),
    ("[#7]~*~*~[#8]", 0),
    ("[R]1@*@*@*@*@1", 0),
    ("[#8]~*~*~*~[#7]", 0),
    ("[R!#6!#1]1@*@*@*@*@*@1", 0),
    ("[#6]=[#6]", 0),
    ("[#7]~[CH2]~*", 0),
    ("[R]1@*@*@*@*@*@*@*@1", 0),
    ("[!#6;!#1]~[#8]", 0),
    ("Cl", 0),
    ("[!#6;!#1;!H0]~*~[CH2]~*", 0),
    ("[R!x2](@*)(@*)@*", 0),
    ("[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]", 0),
    ("[F,Cl,Br,I]~*(~*)~*", 0),
    ("[CH3]~*~*~*~[CH2]~*", 0),
    ("[#8]~[CH2]~*", 0),
    ("[#8]~[#6]~[#7]", 0),
    ("[#7]~*~[CH2]~*", 0),
    ("*(~*)(~*)(~*)~*", 0),
    ("[#8]!:*:*", 0),
    ("[CH3]~[CH2]~*", 0),
    ("[CH3]~*~[CH2]~*", 0),
    ("[#6H3]~*~*~[#6H2]~*", 0),
    ("[#7]~*~[#8]", 0),
    ("[#6H2]([#6H2]*)*", 1),
    ("[#7]=*", 0),
    ("[!#6;R]", 1),
    ("[#7;R]", 0),
    ("[#7](~*)(~*)~*", 0),
    ("[#8]~[#6]~[#8]", 0),
    ("[!#6;!#1]~[!#6;!#1]", 0),
    ("[!*]", 0),
    ("[#8](!@*)!@*", 0),
    ("[#8]!@[R]@*", 1),
    ("[#6H2](~*~*~*~[#6H2]~*)~*", 0),
    ("[#6H2](~*~*~[#6H2]~*)~*", 0),
    ("[!#6;!#1]~[!#6;!#1]", 1),
    ("[!#6;!#1;!H0]", 1),
    ("[#8]~*~[CH2]~*", 0),
    ("[#7]!@*@*", 0),
    ("[F,Cl,Br,I]", 0),
    ("[#7]!:*:*", 0),
    ("[#8]=*", 1),
    ("[!C;!c;R]", 0),
    ("[!#6;!#1]~[CH2]~*", 1),
    ("[O;!H0]", 0),
    ("[#8]", 3),
    ("[CH3]", 2),
    ("[#7]", 1),
    ("[#8]!@[R]@*", 0),
    ("a(!:*):a!:*", 0),
    ("[R]1@*@*@*@*@*@1", 1),
    ("[#8]", 2),
    ("[#6H2](~[#6H2]~*)~*", 0),
    ("[!#6;!#1](~*)(~*)~*", 0),
    ("[C;H3,H4]", 1),
    ("[R](!@*)@[R]!@*", 0),
    ("[#7;!H0]", 0),
    ("[#8]~[#6](~[#6])~[#6]", 0),
    ("[!#6;!#1]~[CH2]~*", 0),
    ("[#6]=[#8]", 0),
    ("[CH2](!@*)!@*", 0),
    ("[#7]~*(~*)~*", 0),
    ("[#6]-[#8]", 0),
    ("[#6]-[#7]", 0),
    ("[#8]", 1),
    ("[C;H3,H4]", 0),
    ("[#7]", 0),
    ("a", 0),
    ("[R]1@*@*@*@*@*@1", 0),
    ("[#8]", 0),
    ("[R]", 0),
    ("(*).(*)", 0),
];

/// A compiled SMARTS key: pattern, screen and the number of unique matches needed.
type Key = (QueryGraph, QueryScreen, usize);

/// The compiled keys; `None` for keys set from the cycle census, the isotope
/// check or the component count.
fn keys() -> &'static [Option<Key>] {
    static KEYS_COMPILED: OnceLock<Vec<Option<Key>>> = OnceLock::new();
    KEYS_COMPILED.get_or_init(|| {
        KEYS.iter()
            .map(|&(smarts, count)| {
                if smarts.starts_with("[R]1") || matches!(smarts, "[!0]" | "[!*]" | "(*).(*)") {
                    return None;
                }
                let q = parse_cdk_smarts(smarts);
                let screen = QueryScreen::build(&q);
                Some((q, screen, count as usize + 1))
            })
            .collect()
    })
}

/// Sets the ring bits from all simple cycles of up to `min(8, atoms)` atoms
/// (CDK `AllCycles`; `isAromPath` reads the Daylight-model bond flags).
fn cycle_census(t: &Molecule) -> Census {
    let n = t.num_atoms();
    let mut census = Census::default();
    if n <= 2 {
        return census;
    }
    let max_len = n.min(8);
    let mut path: Vec<usize> = Vec::with_capacity(8);
    let mut on_path = vec![false; n];
    for s in 0..n {
        if !t.atoms[s].is_in_ring() {
            continue;
        }
        path.clear();
        path.push(s);
        on_path[s] = true;
        cycles_from(t, s, s, max_len, &mut path, &mut on_path, &mut census);
        on_path[s] = false;
    }
    census
}

/// Sets the ring bits from the cycle census.
fn set_cycle_bits(census: &Census, fp: &mut BitVector) {
    for (len, bit) in [(3, 21), (4, 10), (5, 95), (7, 18), (8, 100)] {
        if census.seen[len] {
            fp.set(bit, true);
        }
    }
    if census.six >= 1 {
        fp.set(162, true);
    }
    if census.six >= 2 {
        fp.set(144, true);
    }
    if census.arom_six >= 2 {
        fp.set(124, true);
    }
}

#[derive(Default)]
struct Census {
    seen: [bool; 9],
    six: u32,
    arom_six: u32,
}

/// Simple cycles through ring bonds whose smallest atom is `s`, each found once
/// (second atom smaller than the last).
fn cycles_from(
    t: &Molecule,
    s: usize,
    a: usize,
    max_len: usize,
    path: &mut Vec<usize>,
    on_path: &mut [bool],
    census: &mut Census,
) {
    for &bi in &t.adjacency[a] {
        let b = &t.bonds[bi as usize];
        if !b.is_in_ring() {
            continue;
        }
        let w = if b.source as usize == a { b.target } else { b.source } as usize;
        if w == s {
            let len = path.len();
            if len >= 3 && path[1] < path[len - 1] {
                census.seen[len] = true;
                if len == 6 {
                    census.six += 1;
                    if census.arom_six < 2 && cycle_aromatic(t, path) {
                        census.arom_six += 1;
                    }
                }
            }
            continue;
        }
        if w < s || on_path[w] || path.len() >= max_len {
            continue;
        }
        on_path[w] = true;
        path.push(w);
        cycles_from(t, s, w, max_len, path, on_path, census);
        path.pop();
        on_path[w] = false;
    }
}

fn cycle_aromatic(t: &Molecule, path: &[usize]) -> bool {
    (0..path.len()).all(|i| {
        let (u, v) = (path[i], path[(i + 1) % path.len()]);
        t.adjacency[u].iter().any(|&bi| {
            let b = &t.bonds[bi as usize];
            (b.source as usize == v || b.target as usize == v) && b.bond_type == BondType::Aromatic
        })
    })
}

/// CDK MACCS keys (CDK 2.13 `MACCSFingerprinter`, 166 bits).
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::cdk_maccs::CdkMaccs;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mol = parse_smiles("c1ccccc1O").unwrap();
/// let fp = CdkMaccs.calculate(&mol, &mut ferrum_core::desc::context::CalculationContext::new());
/// let bits = fp.as_binary();
/// assert_eq!(bits.len(), 166);
/// assert!(bits[162] && bits[164]); // 6-membered ring, ring atom
/// ```
pub struct CdkMaccs;

impl Descriptor for CdkMaccs {
    fn name(&self) -> &str {
        "CdkMaccs"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        let mut fp = bitvec![u8, Lsb0; 0; CDK_MACCS_SIZE];
        CdkSmartsTarget::with_cached(mol, ctx, |target| {
            let t = &target.mol;
            let iso = IsoTarget::new(t);
            let census = cycle_census(t);
            for (i, key) in keys().iter().enumerate() {
                if let Some((q, screen, need)) = key {
                    // ring-closure keys need a cycle of their size
                    let ring = match i {
                        15 => census.seen[3],
                        7 => census.seen[4],
                        82 => census.seen[5],
                        97 => census.six > 0,
                        _ => true,
                    };
                    let hit = ring
                        && screen.passes_target(&iso)
                        && if *need == 1 {
                            has_substruct_match_target(&iso, q)
                        } else {
                            has_at_least_matches_target(&iso, q, *need)
                        };
                    if hit {
                        fp.set(i, true);
                    }
                }
            }
            if t.atoms.iter().any(|a| a.isotope != 0) {
                fp.set(0, true);
            }
            if t.num_atoms() > 1 && crate::algo::components::num_fragments(t) > 1 {
                fp.set(165, true);
            }
            set_cycle_bits(&census, &mut fp);
        });
        DescriptorValue::Binary(fp)
    }
}
