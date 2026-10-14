//! CDK 2.13 `KierHallSmartsDescriptor` (`khs.sLi` ... `khs.ssssPb`).
//!
//! CDK counts the unique matches of the 79 E-State atom-type SMARTS of
//! `EStateFragments` on the hydrogen-suppressed molecule after
//! `SmartsPattern.prepare` (Daylight aromaticity model). These are CDK's own
//! SMARTS, not RDKit's E-State typer, and are evaluated directly here:
//!
//! - every pattern is a centre atom with fixed degree `D` plus all `D`
//!   neighbours, so a match covers the centre's closed neighbourhood and
//!   `countUnique` (unique matched atom sets) counts centres with distinct
//!   closed neighbourhoods (the two carbons of ethane give one `sCH3` match);
//! - `-`, `=`, `#` are non-aromatic bonds of that order, `:` aromatic bonds;
//! - `C`, `N`, `O`, ... (B, Si, P, S, Ge, As, Se) require a non-aromatic atom,
//!   `[C,c;...]` any carbon; Li, Be, F, Cl, Br, I, Sn and Pb match by element;
//! - `H` is the total hydrogen count, `D` the explicit degree.

use crate::desc::cdk_daylight::{CdkMol, smarts_daylight_aromaticity};
use crate::desc::cdk_perception::CdkView;
use crate::desc::context::CalculationContext;
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Number of Kier-Hall E-State SMARTS (`EStateFragments`).
pub const KIER_HALL_LEN: usize = 79;

/// `EStateFragments.getNames()` (output columns are `khs.<name>`).
pub const KIER_HALL_NAMES: [&str; KIER_HALL_LEN] = [
    "sLi", "ssBe", "ssssBe", "ssBH", "sssB", "ssssB", "sCH3", "dCH2", "ssCH2", "tCH", "dsCH", "aaCH", "sssCH",
    "ddC", "tsC", "dssC", "aasC", "aaaC", "ssssC", "sNH3", "sNH2", "ssNH2", "dNH", "ssNH", "aaNH", "tN", "sssNH",
    "dsN", "aaN", "sssN", "ddsN", "aasN", "ssssN", "sOH", "dO", "ssO", "aaO", "sF", "sSiH3", "ssSiH2", "sssSiH",
    "ssssSi", "sPH2", "ssPH", "sssP", "dsssP", "sssssP", "sSH", "dS", "ssS", "aaS", "dssS", "ddssS", "sCl", "sGeH3",
    "ssGeH2", "sssGeH", "ssssGe", "sAsH2", "ssAsH", "sssAs", "sssdAs", "sssssAs", "sSeH", "dSe", "ssSe", "aaSe",
    "dssSe", "ddssSe", "sBr", "sSnH3", "ssSnH2", "sssSnH", "ssssSn", "sI", "sPbH3", "ssPbH2", "sssPbH", "ssssPb",
];

/// Column names (`khs.sLi`, ...).
pub fn kier_hall_names() -> Vec<String> {
    KIER_HALL_NAMES.iter().map(|n| format!("khs.{n}")).collect()
}

/// CDK 2.13 `KierHallSmartsDescriptor`: IntU16Vec of the 79 counts in the order
/// of [`KIER_HALL_NAMES`].
pub struct CdkKierHallSmarts;

impl Descriptor for CdkKierHallSmarts {
    fn name(&self) -> &str {
        "KierHallSmarts"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        let counts = CdkView::with_legacy_cached(mol, ctx, |view, _| kier_hall_counts(mol, view));
        DescriptorValue::IntU16Vec(counts.to_vec())
    }
}

/// Per-atom data the patterns test.
#[derive(Clone, Copy, Default)]
struct Env {
    z: u8,
    arom: bool,
    /// Explicit degree.
    d: u8,
    /// Total hydrogen count.
    h: u8,
    /// Non-aromatic single / double / triple and aromatic bonds.
    s: u8,
    db: u8,
    t: u8,
    ar: u8,
}

impl Env {
    /// Bond counts are exactly (single, double, triple, aromatic).
    #[inline]
    fn bonds(&self, s: u8, db: u8, t: u8, ar: u8) -> bool {
        self.s == s && self.db == db && self.t == t && self.ar == ar
    }
    /// Degree, hydrogens and single bonds only (`[XDdHh](-*)...`).
    #[inline]
    fn singles(&self, d: u8, h: u8) -> bool {
        self.d == d && self.h == h && self.bonds(d, 0, 0, 0)
    }
}

/// Counts of the 79 Kier-Hall SMARTS exactly as CDK 2.13 computes them.
pub(crate) fn kier_hall_counts(mol: &Molecule, view: &CdkView) -> [u16; KIER_HALL_LEN] {
    if mol.num_atoms() == 0 {
        return [0; KIER_HALL_LEN];
    }
    // AtomContainerManipulator.removeHydrogens, then SmartsPattern.prepare
    match crate::algo::sanitize::cdk_remove_hydrogens(mol) {
        Some(suppressed) => {
            let view = CdkView::new(&suppressed);
            counts_on(&crate::desc::cdk_simple::cdk_parser_mol(&suppressed, &view))
        }
        None => crate::desc::cdk_simple::with_parser_mol(mol, view, counts_on),
    }
}

/// [`kier_hall_counts`] on CDK's view of the hydrogen-suppressed molecule.
fn counts_on(m: &CdkMol) -> [u16; KIER_HALL_LEN] {
    let mut counts = [0u16; KIER_HALL_LEN];
    let (arom_atom, arom_bond) = smarts_daylight_aromaticity(m);
    let n = m.n();
    let env: Vec<Env> = (0..n)
        .map(|a| {
            let mut e = Env {
                z: m.z[a],
                arom: arom_atom[a],
                d: m.adj[a].len() as u8,
                h: m.hcount[a],
                ..Env::default()
            };
            for &(w, bi) in &m.adj[a] {
                if m.z[w as usize] == 1 {
                    e.h += 1;
                }
                if arom_bond[bi as usize] {
                    e.ar += 1;
                } else {
                    match m.order[bi as usize] {
                        1 => e.s += 1,
                        2 => e.db += 1,
                        3 => e.t += 1,
                        _ => {}
                    }
                }
            }
            e
        })
        .collect();
    // `[OD1H0]` (aliphatic) neighbour test used by ddsN / ddssS
    let terminal_o = |w: usize| {
        let e = &env[w];
        e.z == 8 && !e.arom && e.d == 1 && e.h == 0
    };
    // patterns matched per centre atom (at most two)
    let mut atom_hits: Vec<[u8; 2]> = vec![[u8::MAX; 2]; n];
    let mut hits: Vec<usize> = Vec::with_capacity(2);
    for a in 0..n {
        let e = env[a];
        hits.clear();
        let aliph = !e.arom;
        match e.z {
            3 if e.d == 1 && e.bonds(1, 0, 0, 0) => hits.push(0),
            4 => {
                if e.d == 2 && e.bonds(2, 0, 0, 0) {
                    hits.push(1);
                }
                if e.d == 4 && e.bonds(4, 0, 0, 0) {
                    hits.push(2);
                }
            }
            5 if aliph => {
                if e.singles(2, 1) {
                    hits.push(3);
                }
                if e.d == 3 && e.bonds(3, 0, 0, 0) {
                    hits.push(4);
                }
                if e.d == 4 && e.bonds(4, 0, 0, 0) {
                    hits.push(5);
                }
            }
            6 => {
                if aliph {
                    let idx = match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                        (1, 3, 1, 0, 0, 0) => Some(6),
                        (1, 2, 0, 1, 0, 0) => Some(7),
                        (2, 2, 2, 0, 0, 0) => Some(8),
                        (1, 1, 0, 0, 1, 0) => Some(9),
                        (2, 1, 1, 1, 0, 0) => Some(10),
                        (3, 1, 3, 0, 0, 0) => Some(12),
                        (2, 0, 0, 2, 0, 0) => Some(13),
                        (2, 0, 1, 0, 1, 0) => Some(14),
                        (3, 0, 2, 1, 0, 0) => Some(15),
                        (4, 0, 4, 0, 0, 0) => Some(18),
                        _ => None,
                    };
                    hits.extend(idx);
                }
                // [C,c;...]: any carbon
                match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                    (2, 1, 0, 0, 0, 2) => hits.push(11),
                    (3, 0, 1, 0, 0, 2) => hits.push(16),
                    (3, 0, 0, 0, 0, 3) => hits.push(17),
                    _ => {}
                }
            }
            7 => {
                if aliph {
                    let idx = match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                        (1, 3, 1, 0, 0, 0) => Some(19),
                        (1, 2, 1, 0, 0, 0) => Some(20),
                        (2, 2, 2, 0, 0, 0) => Some(21),
                        (1, 1, 0, 1, 0, 0) => Some(22),
                        (2, 1, 2, 0, 0, 0) => Some(23),
                        (1, 0, 0, 0, 1, 0) => Some(25),
                        (3, 1, 3, 0, 0, 0) => Some(26),
                        (2, 0, 1, 1, 0, 0) => Some(27),
                        (3, 0, 3, 0, 0, 0) => Some(29),
                        (4, 0, 4, 0, 0, 0) => Some(32),
                        _ => None,
                    };
                    hits.extend(idx);
                    // [ND3H0](~[OD1H0])(~[OD1H0])-,:*
                    if e.d == 3 && e.h == 0 {
                        let nb = &m.adj[a];
                        let found = (0..3).any(|k| {
                            let bk = nb[k].1 as usize;
                            (arom_bond[bk] || m.order[bk] == 1)
                                && (0..3).filter(|&j| j != k).all(|j| terminal_o(nb[j].0 as usize))
                        });
                        if found {
                            hits.push(30);
                        }
                    }
                } else {
                    // [N,nD2H](:*):*, [N,nD2H0](:*):*, [N,nD3H0](:*)(:*)-,:*
                    match (e.d, e.h, e.ar, e.s) {
                        (2, 1, 2, 0) => hits.push(24),
                        (2, 0, 2, 0) => hits.push(28),
                        (3, 0, 3, 0) | (3, 0, 2, 1) => hits.push(31),
                        _ => {}
                    }
                }
            }
            8 => {
                if aliph {
                    match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                        (1, 1, 1, 0, 0, 0) => hits.push(33),
                        (1, 0, 0, 1, 0, 0) => hits.push(34),
                        (2, 0, 2, 0, 0, 0) => hits.push(35),
                        _ => {}
                    }
                } else if e.d == 2 && e.h == 0 && e.bonds(0, 0, 0, 2) {
                    hits.push(36);
                }
            }
            9 if e.d == 1 && e.bonds(1, 0, 0, 0) => hits.push(37),
            14 if aliph => {
                for (k, (d, h)) in [(1, 3), (2, 2), (3, 1), (4, 0)].into_iter().enumerate() {
                    if e.singles(d, h) {
                        hits.push(38 + k);
                    }
                }
            }
            15 if aliph => match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                (1, 2, 1, 0, 0, 0) => hits.push(42),
                (2, 1, 2, 0, 0, 0) => hits.push(43),
                (3, 0, 3, 0, 0, 0) => hits.push(44),
                (4, 0, 3, 1, 0, 0) => hits.push(45),
                (5, 0, 5, 0, 0, 0) => hits.push(46),
                _ => {}
            },
            16 => {
                if aliph {
                    match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                        (1, 1, 1, 0, 0, 0) => hits.push(47),
                        (1, 0, 0, 1, 0, 0) => hits.push(48),
                        (2, 0, 2, 0, 0, 0) => hits.push(49),
                        (3, 0, 2, 1, 0, 0) => hits.push(51),
                        _ => {}
                    }
                    // [SD4H0](~[OD1H0])(~[OD1H0])(-*)-*
                    if e.d == 4 && e.h == 0 {
                        let nb = &m.adj[a];
                        let single = |k: usize| {
                            let bk = nb[k].1 as usize;
                            !arom_bond[bk] && m.order[bk] == 1
                        };
                        let found = (0..4).any(|i| {
                            (i + 1..4).any(|j| {
                                terminal_o(nb[i].0 as usize)
                                    && terminal_o(nb[j].0 as usize)
                                    && (0..4).filter(|&k| k != i && k != j).all(single)
                            })
                        });
                        if found {
                            hits.push(52);
                        }
                    }
                } else if e.d == 2 && e.h == 0 && e.bonds(0, 0, 0, 2) {
                    hits.push(50);
                }
            }
            17 if e.d == 1 && e.bonds(1, 0, 0, 0) => hits.push(53),
            32 if aliph => {
                for (k, (d, h)) in [(1, 3), (2, 2), (3, 1), (4, 0)].into_iter().enumerate() {
                    if e.singles(d, h) {
                        hits.push(54 + k);
                    }
                }
            }
            33 if aliph => match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                (1, 2, 1, 0, 0, 0) => hits.push(58),
                (2, 1, 2, 0, 0, 0) => hits.push(59),
                (3, 0, 3, 0, 0, 0) => hits.push(60),
                (4, 0, 3, 1, 0, 0) => hits.push(61),
                (5, 0, 5, 0, 0, 0) => hits.push(62),
                _ => {}
            },
            // `[SeD2H0](:*):*` (aaSe) needs an aliphatic Se with aromatic bonds: never matches
            34 if aliph => match (e.d, e.h, e.s, e.db, e.t, e.ar) {
                (1, 1, 1, 0, 0, 0) => hits.push(63),
                (1, 0, 0, 1, 0, 0) => hits.push(64),
                (2, 0, 2, 0, 0, 0) => hits.push(65),
                (3, 0, 2, 1, 0, 0) => hits.push(67),
                (4, 0, 2, 2, 0, 0) => hits.push(68),
                _ => {}
            },
            35 if e.d == 1 && e.bonds(1, 0, 0, 0) => hits.push(69),
            50 => {
                for (k, (d, h)) in [(1, 3), (2, 2), (3, 1), (4, 0)].into_iter().enumerate() {
                    if e.singles(d, h) {
                        hits.push(70 + k);
                    }
                }
            }
            53 if e.d == 1 && e.bonds(1, 0, 0, 0) => hits.push(74),
            82 => {
                for (k, (d, h)) in [(1, 3), (2, 2), (3, 1), (4, 0)].into_iter().enumerate() {
                    if e.singles(d, h) {
                        hits.push(75 + k);
                    }
                }
            }
            _ => {}
        }
        debug_assert!(hits.len() <= 2);
        for (k, &p) in hits.iter().enumerate() {
            atom_hits[a][k] = p as u8;
        }
    }
    // countUnique: centres with the same closed neighbourhood (necessarily
    // adjacent atoms) give one match
    for a in 0..n {
        for &p in &atom_hits[a] {
            if p == u8::MAX {
                continue;
            }
            let dup = m.adj[a].iter().any(|&(b, _)| {
                (b as usize) < a && atom_hits[b as usize].contains(&p) && same_closed_neighbourhood(m, a, b as usize)
            });
            if !dup {
                counts[p as usize] += 1;
            }
        }
    }
    counts
}

/// Whether adjacent atoms `a` and `b` have the same closed neighbourhood.
fn same_closed_neighbourhood(m: &CdkMol, a: usize, b: usize) -> bool {
    if m.adj[a].len() != m.adj[b].len() || m.bond(a, b).is_none() {
        return false;
    }
    m.adj[a]
        .iter()
        .all(|&(w, _)| w as usize == b || m.adj[b].iter().any(|&(x, _)| x == w))
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<CdkKierHallSmarts>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn khs(smi: &str, name: &str) -> u16 {
        let i = KIER_HALL_NAMES.iter().position(|&n| n == name).unwrap();
        let mol = parse_smiles(smi).unwrap();
        kier_hall_counts(&mol, &CdkView::new(&mol))[i]
    }

    #[test]
    fn unique_matches_collapse_symmetric_centres() {
        assert_eq!(khs("CC", "sCH3"), 1);
        assert_eq!(khs("CCC", "sCH3"), 2);
        assert_eq!(khs("C1CC1", "ssCH2"), 1);
        assert_eq!(khs("c1ccccc1", "aaCH"), 6);
        assert_eq!(khs("CC(=O)O", "dO"), 1);
        assert_eq!(khs("C[N+](=O)[O-]", "ddsN"), 1);
    }
}
