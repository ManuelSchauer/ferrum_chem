//! CDK 2.13 atom types (`CDKAtomTypeMatcher`) of the elements the reduced
//! matcher of `cdk_perception` does not cover: H, the halogens, B, Si, Se,
//! the metals, Te, Sb, Sn and the noble gases. Only what the CDK count
//! descriptors need is reproduced: whether an atom gets a type (otherwise CDK
//! uses `X`) and that type's formal neighbour count, from which
//! `CDKHydrogenAdder` derives the implicit hydrogens.
//!
//! Each element follows its `perceive*` method; a candidate type is taken
//! when `isAcceptable` holds (neighbours plus implicit hydrogens within the
//! formal neighbour count, bond orders within the type's maximum, charge
//! equal, valence within the type's valency). The SMILES parser keeps no
//! radicals, so the radical branches never apply. Calcium never gets a type
//! (the `isAcceptable` branches of `perceiveCalcium` return nothing) and
//! `Zn.2plus` has a formal neighbour count of 2, so CDK adds two hydrogens to
//! `[Zn+2]`; both are reproduced.

use crate::desc::cdk_daylight::CdkMol;
use crate::desc::cdk_perception::CdkView;
use crate::mol::molecule::Molecule;

/// Formal neighbour count of the CDK atom type of atom `a` (`None`: type `X`,
/// no match). C, N, O, P, S and As come from `CdkView`, the other elements are
/// typed here.
pub(crate) fn formal_neighbour_count(mol: &Molecule, m: &CdkMol, view: &CdkView, a: usize) -> Option<usize> {
    let z = m.z[a];
    let q = view.charge[a];
    // CDK counts the hydrogen atoms of the input (`[H][Si](C)(C)C`) as
    // neighbours, not as implicit hydrogens
    let removed = mol.num_removed_hydrogens(a);
    let deg = m.adj[a].len() + removed;
    let h = (mol.atoms[a].total_h() as usize).saturating_sub(removed);
    let orders = || m.adj[a].iter().map(|&(_, bi)| m.order[bi as usize]);
    let max_order = orders().max().unwrap_or(1).max(1);
    let n_double = orders().filter(|&o| o == 2).count();
    // isAcceptable: (formal neighbour count, max bond order (0: none), charge, valency)
    let ok = |fnc: usize, max: u8, charge: i8, val: usize| {
        let valence: usize = orders().map(|o| o as usize).sum::<usize>() + removed + h;
        (deg + h <= fnc && (max == 0 || orders().all(|o| o <= max)) && q == charge && valence <= val)
            .then_some(fnc)
    };
    if matches!(z, 6 | 7 | 8 | 15 | 16 | 33) {
        let t = view.types[a].map(|t| t.nb as usize);
        // An alkyne carbon that RDKit puts in an aromatic ring (a cyclic
        // dehydroannulene written in Kekulé form) is `C.sp` in CDK, whose
        // parser leaves it non-aromatic; `CdkView` types aromatic carbons sp2.
        if t.is_none() && z == 6 && q == 0 && mol.atoms[a].is_aromatic() && max_order == 3 {
            return ok(2, 3, 0, 4);
        }
        return t;
    }
    match z {
        1 => match deg {
            0 => match q {
                0 => ok(1, 1, 0, 1),
                1 => ok(0, 0, 1, 0),
                -1 => ok(0, 0, -1, 0),
                _ => None,
            },
            1 if q == 0 => ok(1, 1, 0, 1),
            _ => None,
        },
        9 => match q {
            -1 => ok(0, 0, -1, 0),
            1 => match max_order {
                2 => ok(1, 2, 1, 2),
                1 => ok(2, 1, 1, 2),
                _ => None,
            },
            0 if deg <= 1 => ok(1, 1, 0, 1),
            _ => None,
        },
        17 => match q {
            0 => {
                if max_order == 2 {
                    match deg {
                        2 => ok(2, 2, 0, 3),
                        3 => ok(3, 2, 0, 5),
                        4 => ok(4, 2, 0, 7),
                        _ => None,
                    }
                } else if deg <= 1 {
                    ok(1, 1, 0, 1)
                } else {
                    None
                }
            }
            -1 => ok(0, 0, -1, 0),
            1 => match max_order {
                2 => ok(1, 2, 1, 2),
                1 => ok(2, 1, 1, 2),
                _ => None,
            },
            3 if deg == 4 => ok(4, 1, 3, 4),
            _ => {
                if deg == 3 && n_double == 2 {
                    ok(3, 2, 0, 5)
                } else if deg == 4 && n_double == 3 {
                    ok(4, 2, 0, 7)
                } else {
                    None
                }
            }
        },
        35 => match q {
            -1 => ok(0, 0, -1, 0),
            1 => {
                // atomContainer.getMaximumBondOrder: UNSET without bonds and hydrogens
                let max = if deg == 0 && h == 0 { 0 } else { max_order };
                match max {
                    2 => ok(1, 2, 1, 2),
                    1 => ok(2, 1, 1, 2),
                    _ => None,
                }
            }
            _ => match deg {
                0 | 1 => ok(1, 1, 0, 1),
                3 => ok(3, 2, 0, 5),
                _ => None,
            },
        },
        53 => match q {
            -1 => {
                if deg == 0 {
                    ok(0, 0, -1, 0)
                } else {
                    ok(2, 1, -1, 2)
                }
            }
            1 => match max_order {
                2 => ok(1, 2, 1, 2),
                1 => ok(2, 1, 1, 2),
                _ => None,
            },
            0 => match deg {
                3 => {
                    if n_double == 2 {
                        ok(3, 2, 0, 5)
                    } else {
                        ok(3, 1, 0, 3)
                    }
                }
                2 if max_order == 2 => ok(2, 2, 0, 3),
                0 | 1 => ok(1, 1, 0, 1),
                _ => None,
            },
            _ => None,
        },
        14 => match q {
            0 => match deg {
                2 => ok(2, 2, 0, 4),
                3 => ok(3, 2, 0, 4),
                4 => ok(4, 1, 0, 4),
                _ => None,
            },
            -2 => ok(6, 1, -2, 6),
            _ => None,
        },
        5 => {
            // atomContainer.getMaximumBondOrder: UNSET without bonds and hydrogens
            let single = (deg > 0 || h > 0) && max_order == 1;
            if q == -1 && single && deg <= 4 {
                ok(4, 1, -1, 4)
            } else if q == 3 && deg == 4 {
                ok(4, 1, 3, 4)
            } else if deg <= 3 {
                ok(3, 1, 0, 3)
            } else {
                None
            }
        }
        34 => match q {
            0 => match deg {
                0 => {
                    if h == 0 {
                        ok(0, 0, 0, 0)
                    } else {
                        ok(2, 1, 0, 2)
                    }
                }
                1 => match n_double {
                    1 => ok(1, 2, 0, 2),
                    0 => ok(2, 1, 0, 2),
                    _ => None,
                },
                2 => match n_double {
                    0 => ok(2, 1, 0, 2),
                    2 => ok(2, 2, 0, 4),
                    _ => None,
                },
                3 => ok(3, 2, 0, 4),
                4 => match n_double {
                    2 => ok(4, 2, 0, 6),
                    0 => ok(4, 1, 0, 4),
                    _ => None,
                },
                5 => ok(5, 2, 0, 6),
                _ => None,
            },
            4 if deg == 0 => ok(0, 0, 4, 0),
            1 if deg == 3 => ok(3, 1, 1, 3),
            -2 if deg == 0 => ok(0, 0, -2, 0),
            _ => None,
        },
        _ => {
            // getMaximumBondOrder(atom) == SINGLE (UNSET without bonds and hydrogens)
            let single = if deg == 0 { h > 0 } else { max_order == 1 };
            metal_candidates(z, q, deg, single).iter().find_map(|&name| {
                let &(_, fnc, max, charge, val) = OTHER_TYPES.iter().find(|t| t.0 == name)?;
                ok(fnc as usize, max, charge, val as usize)
            })
        }
    }
}


/// Atom types of the elements `metal_candidates` types (`cdk-atom-types.owl`):
/// name, formal neighbour count, maximum bond order (0: none), formal charge,
/// valency.
#[rustfmt::skip]
const OTHER_TYPES: [(&str, u8, u8, i8, u8); 132] = [
    ("Ag.1", 1, 1, 0, 1), ("Ag.neutral", 0, 0, 0, 0), ("Ag.plus", 0, 0, 1, 0),
    ("Al", 3, 1, 0, 3), ("Al.3minus", 6, 1, -3, 6), ("Al.3plus", 0, 0, 3, 0),
    ("Ar", 0, 0, 0, 0), ("Au.1", 1, 1, 0, 1), ("Ba.2plus", 0, 0, 2, 0),
    ("Be.2minus", 4, 1, -2, 4), ("Be.neutral", 0, 0, 0, 0), ("Ca.1", 1, 2, 0, 2),
    ("Ca.2", 2, 1, 0, 2), ("Ca.2plus", 0, 0, 2, 0), ("Cd.2", 2, 1, 0, 2),
    ("Cd.2plus", 0, 0, 2, 0), ("Cd.metallic", 0, 0, 0, 0), ("Co.1", 1, 1, 0, 1),
    ("Co.2", 2, 1, 0, 2), ("Co.2plus", 0, 0, 2, 0), ("Co.3plus", 0, 0, 3, 0),
    ("Co.4", 4, 1, 0, 4), ("Co.6", 6, 1, 0, 6), ("Co.metallic", 0, 0, 0, 0),
    ("Co.plus", 0, 0, 1, 0), ("Co.plus.1", 1, 1, 1, 1), ("Co.plus.2", 2, 1, 1, 2),
    ("Co.plus.4", 4, 1, 1, 4), ("Co.plus.5", 5, 1, 1, 5), ("Co.plus.6", 6, 1, 1, 6),
    ("Cr", 6, 1, 0, 6), ("Cr.3plus", 0, 0, 3, 0), ("Cr.4", 4, 2, 0, 6),
    ("Cr.6plus", 0, 0, 6, 0), ("Cr.neutral", 0, 0, 0, 0), ("Cu.1", 1, 1, 0, 1),
    ("Cu.2plus", 0, 0, 2, 0), ("Cu.metallic", 0, 0, 0, 0), ("Cu.plus", 0, 0, 1, 0),
    ("Fe.2", 2, 1, 0, 2), ("Fe.2minus", 6, 1, -2, 6), ("Fe.2plus", 0, 0, 2, 0),
    ("Fe.3", 3, 1, 0, 3), ("Fe.3minus", 6, 1, -3, 6), ("Fe.3plus", 0, 0, 3, 0),
    ("Fe.4", 4, 1, 0, 4), ("Fe.4minus", 6, 1, -4, 6), ("Fe.5", 5, 1, 0, 5),
    ("Fe.6", 6, 1, 0, 6), ("Fe.metallic", 0, 0, 0, 0), ("Fe.plus", 2, 1, 1, 2),
    ("Ga", 3, 1, 0, 3), ("Ga.3plus", 0, 0, 3, 0), ("Gd.3plus", 0, 0, 3, 0),
    ("Ge", 4, 1, 0, 4), ("Ge.3", 3, 2, 0, 4), ("He", 0, 0, 0, 0),
    ("Hg.1", 1, 2, 0, 2), ("Hg.2", 2, 1, 0, 2), ("Hg.2plus", 0, 0, 2, 0),
    ("Hg.metallic", 0, 0, 0, 0), ("Hg.minus", 2, 1, -1, 2), ("Hg.plus", 1, 1, 1, 1),
    ("In", 0, 0, 0, 0), ("In.1", 1, 3, 0, 3), ("In.3", 3, 1, 0, 3),
    ("In.3plus", 0, 0, 3, 0), ("K.metallic", 0, 0, 0, 0), ("K.neutral", 1, 1, 0, 1),
    ("K.plus", 0, 0, 1, 0), ("Kr", 0, 0, 0, 0), ("Li", 1, 1, 0, 1),
    ("Li.neutral", 0, 0, 0, 0), ("Li.plus", 0, 0, 1, 0), ("Mg.2plus", 0, 0, 2, 0),
    ("Mg.neutral", 4, 1, 0, 4), ("Mg.neutral.1", 1, 2, 0, 2), ("Mg.neutral.2", 2, 1, 0, 2),
    ("Mn.2", 2, 1, 0, 2), ("Mn.2plus", 0, 0, 2, 0), ("Mn.3plus", 0, 0, 3, 0),
    ("Mn.metallic", 0, 0, 0, 0), ("Mo.4", 4, 2, 0, 6), ("Mo.metallic", 0, 0, 0, 0),
    ("Na", 1, 1, 0, 1), ("Na.neutral", 0, 0, 0, 0), ("Na.plus", 0, 0, 1, 0),
    ("Ne", 0, 0, 0, 0), ("Ni", 2, 1, 0, 2), ("Ni.2plus", 0, 0, 2, 0),
    ("Ni.metallic", 0, 0, 0, 0), ("Ni.plus", 1, 1, 1, 1), ("Pb.1", 1, 2, 0, 2),
    ("Pb.2plus", 0, 0, 2, 0), ("Pb.neutral", 0, 0, 0, 0), ("Po", 2, 1, 0, 2),
    ("Pt.2", 2, 1, 0, 2), ("Pt.2plus", 0, 0, 2, 0), ("Pt.2plus.4", 4, 1, 2, 4),
    ("Pt.4", 4, 1, 0, 4), ("Pt.6", 6, 1, 0, 6), ("Pu", 0, 0, 0, 0),
    ("Ra.neutral", 0, 0, 0, 0), ("Rb.neutral", 1, 1, 0, 1), ("Rb.plus", 0, 0, 1, 0),
    ("Rn", 0, 0, 0, 0), ("Ru.2minus.6", 6, 1, -2, 6), ("Ru.3minus.6", 6, 1, -3, 6),
    ("Ru.6", 6, 1, 0, 6), ("Sb.3", 3, 1, 0, 3), ("Sb.4", 4, 2, 0, 5),
    ("Sc.3minus", 6, 1, -3, 6), ("Sn.sp3", 4, 1, 0, 4), ("Sr.2plus", 0, 0, 2, 0),
    ("Te.3", 2, 1, 0, 2), ("Te.4plus", 0, 0, 4, 0), ("Th", 0, 0, 0, 0),
    ("Ti.2", 2, 2, 0, 4), ("Ti.3minus", 6, 1, -3, 6), ("Ti.sp3", 4, 1, 0, 4),
    ("Tl", 0, 0, 0, 0), ("Tl.1", 1, 1, 0, 1), ("Tl.plus", 0, 0, 1, 0),
    ("V.3minus", 6, 1, -3, 6), ("V.3minus.4", 4, 2, -3, 5), ("W.metallic", 0, 0, 0, 0),
    ("Xe", 0, 0, 0, 0), ("Xe.3", 4, 1, 0, 4), ("Zn", 2, 1, 0, 2),
    ("Zn.1", 1, 2, 0, 2), ("Zn.2plus", 2, 1, 2, 2), ("Zn.metallic", 0, 0, 0, 0),
];

/// Candidate CDK atom types of an element handled by the element-specific
/// `perceive*` methods of `CDKAtomTypeMatcher` (metals, Te, Sb, Sn, noble
/// gases, ...), in the order CDK tries them; the first acceptable one is the
/// type. `single`: `getMaximumBondOrder(atom) == SINGLE` (UNSET for an atom
/// without bonds and hydrogens).
fn metal_candidates(z: u8, q: i8, nc: usize, single: bool) -> &'static [&'static str] {
    match z {
        3 => match (nc, q) {
            (1, 0) => &["Li"],
            (0, 0) => &["Li.neutral"],
            (0, 1) => &["Li.plus"],
            _ => &[],
        },
        4 => {
            if q == -2 && single && nc <= 4 {
                &["Be.2minus"]
            } else if q == 0 && nc == 0 {
                &["Be.neutral"]
            } else {
                &[]
            }
        }
        24 => match (q, nc) {
            (0, 6) => &["Cr"],
            (0, 4) => &["Cr.4"],
            (6, 0) => &["Cr.6plus"],
            (0, 0) => &["Cr.neutral"],
            (3, 0) => &["Cr.3plus"],
            _ => &[],
        },
        42 => match (q, nc) {
            (0, 4) => &["Mo.4", "Mo.metallic"],
            (0, _) => &["Mo.metallic"],
            _ => &[],
        },
        37 => match q {
            1 => &["Rb.plus"],
            0 => &["Rb.neutral"],
            _ => &[],
        },
        52 => {
            if q == 0 && single && nc <= 2 {
                &["Te.3"]
            } else if q == 4 && nc == 0 {
                &["Te.4plus"]
            } else {
                &[]
            }
        }
        29 => match (q, nc) {
            (2, _) => &["Cu.2plus"],
            (0, 1) => &["Cu.1"],
            (0, _) => &["Cu.metallic"],
            (1, _) => &["Cu.plus"],
            _ => &[],
        },
        56 if q == 2 => &["Ba.2plus"],
        31 => {
            if q == 0 && single && nc <= 3 {
                &["Ga"]
            } else if q == 3 {
                &["Ga.3plus"]
            } else {
                &[]
            }
        }
        44 => match q {
            0 => &["Ru.6"],
            -2 => &["Ru.2minus.6"],
            -3 => &["Ru.3minus.6"],
            _ => &[],
        },
        30 => match (nc, q) {
            (0, 0) => &["Zn.metallic"],
            (0, 2) => &["Zn.2plus"],
            (1, 0) => &["Zn.1"],
            (2, 0) => &["Zn"],
            _ => &[],
        },
        13 => match (q, nc) {
            (3, 0) => &["Al.3plus"],
            (0, 3) => &["Al"],
            (-3, 6) => &["Al.3minus"],
            _ => &[],
        },
        28 => match (q, nc) {
            (2, _) => &["Ni.2plus"],
            (0, 2) => &["Ni"],
            (0, 0) => &["Ni.metallic"],
            (1, 1) => &["Ni.plus"],
            _ => &[],
        },
        64 if q == 3 && nc == 0 => &["Gd.3plus"],
        // two independent tests: Ge (uncharged, single bonds, <= 4), then Ge.3
        32 => match (q == 0 && single && nc <= 4, q == 0 && nc == 3) {
            (true, true) => &["Ge", "Ge.3"],
            (true, false) => &["Ge"],
            (false, true) => &["Ge.3"],
            (false, false) => &[],
        },
        27 => match (q, nc) {
            (2, _) => &["Co.2plus"],
            (3, _) => &["Co.3plus"],
            (0, 2) => &["Co.2"],
            (0, 4) => &["Co.4"],
            (0, 6) => &["Co.6"],
            (0, 1) => &["Co.1"],
            (0, _) => &["Co.metallic"],
            (1, 2) => &["Co.plus.2"],
            (1, 4) => &["Co.plus.4"],
            (1, 1) => &["Co.plus.1"],
            (1, 6) => &["Co.plus.6"],
            (1, 5) => &["Co.plus.5"],
            (1, _) => &["Co.plus"],
            _ => &[],
        },
        23 => match (q, nc) {
            (-3, 6) => &["V.3minus"],
            (-3, 4) => &["V.3minus.4"],
            _ => &[],
        },
        22 => match (q, nc) {
            (-3, 6) => &["Ti.3minus"],
            (0, 4) => &["Ti.sp3"],
            (0, 2) => &["Ti.2"],
            _ => &[],
        },
        38 if q == 2 => &["Sr.2plus"],
        82 => match (q, nc) {
            (0, 0) => &["Pb.neutral"],
            (2, 0) => &["Pb.2plus"],
            (0, 1) => &["Pb.1"],
            _ => &[],
        },
        81 => match (q, nc) {
            (1, 0) => &["Tl.plus"],
            (0, 0) => &["Tl"],
            (0, 1) => &["Tl.1"],
            _ => &[],
        },
        51 => match (q, nc) {
            (0, 3) => &["Sb.3"],
            (0, 4) => &["Sb.4"],
            _ => &[],
        },
        78 => match (q, nc) {
            (2, 4) => &["Pt.2plus.4"],
            (2, _) => &["Pt.2plus"],
            (0, 2) => &["Pt.2"],
            (0, 4) => &["Pt.4"],
            (0, 6) => &["Pt.6"],
            _ => &[],
        },
        80 => match (q, nc) {
            (-1, _) => &["Hg.minus"],
            (2, _) => &["Hg.2plus"],
            (1, 0 | 1) => &["Hg.plus"],
            (0, 2) => &["Hg.2"],
            (0, 1) => &["Hg.1"],
            (0, 0) => &["Hg.metallic"],
            _ => &[],
        },
        26 => match (q, nc) {
            (0, 0) => &["Fe.metallic"],
            (0, 2) => &["Fe.2"],
            (0, 3) => &["Fe.3"],
            (0, 4) => &["Fe.4"],
            (0, 5) => &["Fe.5"],
            (0, 6) => &["Fe.6"],
            (2, 0 | 1) => &["Fe.2plus"],
            (1, 2) => &["Fe.plus"],
            (3, _) => &["Fe.3plus"],
            (-2, _) => &["Fe.2minus"],
            (-3, _) => &["Fe.3minus"],
            (-4, _) => &["Fe.4minus"],
            _ => &[],
        },
        88 if q == 0 => &["Ra.neutral"],
        79 if q == 0 && nc == 1 => &["Au.1"],
        47 => match (q, nc) {
            (0, 1) => &["Ag.1", "Ag.neutral"],
            (0, _) => &["Ag.neutral"],
            (1, _) => &["Ag.plus"],
            _ => &[],
        },
        49 => match (q, nc) {
            (0, 3) => &["In.3"],
            (3, 0) => &["In.3plus"],
            (0, 1) => &["In.1"],
            _ => &["In"],
        },
        94 if q == 0 && nc == 0 => &["Pu"],
        90 if q == 0 && nc == 0 => &["Th"],
        19 => match (q, nc) {
            (1, _) => &["K.plus"],
            (0, 1) => &["K.neutral", "K.metallic"],
            (0, _) => &["K.metallic"],
            _ => &[],
        },
        25 => match (q, nc) {
            (0, 2) => &["Mn.2"],
            (0, 0) => &["Mn.metallic"],
            (2, _) => &["Mn.2plus"],
            (3, _) => &["Mn.3plus"],
            _ => &[],
        },
        12 => match (q, nc) {
            (0, 4) => &["Mg.neutral"],
            (0, 2) => &["Mg.neutral.2"],
            (0, 1) => &["Mg.neutral.1"],
            (0, _) => &["Mg.neutral"],
            (2, _) => &["Mg.2plus"],
            _ => &[],
        },
        11 => match (q, nc) {
            (1, _) => &["Na.plus"],
            (0, 1) => &["Na"],
            (0, 0) => &["Na.neutral"],
            _ => &[],
        },
        48 => match (q, nc) {
            (2, _) => &["Cd.2plus"],
            (0, 0) => &["Cd.metallic"],
            (0, 2) => &["Cd.2"],
            _ => &[],
        },
        // perceiveCalcium never returns a type (its `isAcceptable` branches are empty)
        20 => &[],
        // perceiveCommonSalts / perceiveOrganometallicCenters / perceiveNobelGases
        74 if q == 0 => &["W.metallic"],
        84 if nc == 2 => &["Po"],
        50 if q == 0 && nc <= 4 => &["Sn.sp3"],
        21 if q == -3 && nc == 6 => &["Sc.3minus"],
        2 if q == 0 => &["He"],
        10 if q == 0 => &["Ne"],
        18 if q == 0 => &["Ar"],
        36 if q == 0 => &["Kr"],
        54 if q == 0 => {
            if nc == 0 {
                &["Xe"]
            } else {
                &["Xe.3"]
            }
        }
        86 if q == 0 => &["Rn"],
        _ => &[],
    }
}
