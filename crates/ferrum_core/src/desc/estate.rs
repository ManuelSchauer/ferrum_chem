#![allow(dead_code)]
//! Electrotopological State (E-State) indices.
//!
//! Kier & Hall E-State encodes both electronic and topological
//! information for each atom:
//!
//!   S_i = I_i + ΔI_i
//!
//! where I_i is the intrinsic state (based on valence electrons and
//! connectivity) and ΔI_i is the perturbation from all other atoms
//! (decaying with squared topological distance).
//!
//! Output:
//! - Summary: sum, min, max of all atom E-States (3 values)
//! - 79 atom-type E-State sums (RDKit `rdkit.Chem.EState.Fingerprinter.FingerprintMol`)
//!
//! Reference: Kier, L.B. & Hall, L.H. (1999) "Molecular Structure
//!            Description: The Electrotopological State", Academic Press.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::elements::outer_electrons;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Core E-State computation
// ---------------------------------------------------------------------------

/// Compute intrinsic state I_i for each heavy atom.
/// I_i = ((2/N)² δᵥ + 1) / δ
/// where N = principal quantum number, δᵥ = valence delta, δ = sigma electrons (degree).
/// The E-State values are computed once per molecule and shared through the
/// calculation context (EStateIndices, EStateAbs, EState_VSA, VSA_EState).
pub(crate) fn compute_estates(
    mol: &Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
) -> Vec<f64> {
    let key = crate::desc::context::molecule_cache_key(mol);
    if ctx.estates_key != Some(key) {
        let estates = compute_estates_uncached(mol, ctx);
        ctx.estates = estates;
        ctx.estates_key = Some(key);
    }
    ctx.estates.clone()
}

fn compute_estates_uncached(
    mol: &Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
) -> Vec<f64> {
    let n = mol.num_atoms();
    if n == 0 {
        return vec![];
    }

    // Step 1: compute intrinsic state for each atom
    let intrinsic: Vec<f64> = (0..n)
        .map(|i| {
            // RDKit EStateIndices: every atom with at least one neighbour
            // (explicit H and dummy atoms included) gets an intrinsic state.
            let atom = &mol.atoms[i];
            let an = atom.atomic_number;

            let pqn = principal_quantum_number(an) as f64;
            let zv = outer_electrons(an) as f64;
            let h = atom.total_h() as f64;
            let delta = mol.get_degree(i as AtomIdx) as f64;

            if delta == 0.0 {
                return 0.0;
            }

            let delta_v = zv - h;

            // I = ((2/N)^2 * δᵥ + 1) / δ
            ((2.0 / pqn).powi(2) * delta_v + 1.0) / delta
        })
        .collect();

    // Step 2: compute BFS distance matrix
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);

    // Step 3: compute perturbation ΔI for each atom
    let mut estates = intrinsic.clone();
    for i in 0..n {
        let mut delta_i = 0.0f64;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = dist[i * n + j];
            if d == 0 || d == crate::algo::paths::DIST_INF {
                continue;
            }
            // ΔI_ij = (I_i - I_j) / (d_ij + 1)^2
            delta_i += (intrinsic[i] - intrinsic[j]) / ((d as f64 + 1.0).powi(2));
        }
        estates[i] = intrinsic[i] + delta_i;
    }

    estates
}

fn principal_quantum_number(atomic_number: u8) -> u8 {
    match atomic_number {
        0..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        _ => 7,
    }
}

#[allow(dead_code)]
fn _compute_distance_matrix_unused(mol: &Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    let mut distances = vec![u32::MAX; n * n];

    for start in 0..n {
        distances[start * n + start] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start as AtomIdx);

        while let Some(current) = queue.pop_front() {
            let current_dist = distances[start * n + current as usize];
            for &bond_idx in &mol.adjacency[current as usize] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                };
                let nd = &mut distances[start * n + neighbor as usize];
                if *nd == u32::MAX {
                    *nd = current_dist + 1;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    distances
}

// ---------------------------------------------------------------------------
// E-State atom types (RDKit `rdkit.Chem.EState.AtomTypes`)
// ---------------------------------------------------------------------------
//
// RDKit types atoms with 79 SMARTS patterns (Hall & Kier, JCICS 35, 1039
// (1995), Table 1, with RDKit's modifications) and
// `rdkit.Chem.EState.Fingerprinter.FingerprintMol` sums the E-State index of
// the first atom of every *unique* match of each pattern
// (`GetSubstructMatches(patt, uniquify=True)`, default `maxMatches=1000`).
//
// Every pattern is a star: a centre atom plus single-atom branches. The
// direct classifier below reproduces RDKit's matching semantics exactly:
// - bracket symbols of the SMARTS organic subset (B C N O P S F Cl Br I) are
//   aliphatic-only, all other element symbols (Li, Be, Si, Ge, As, Se, Sn,
//   Pb) match aliphatic and aromatic atoms alike;
// - `D` is the explicit degree, `H` the total H count including explicit H
//   neighbours (`getTotalNumHs(true)`), `*` matches any atom including H;
// - `-`, `=`, `#`, `:` compare the bond type, `-,:` is single or aromatic,
//   `~` any bond;
// - several matches covering the same atom set count once, attributed to the
//   match with the lowest centre index (e.g. ethane has one `sCH3` match,
//   cyclopropane one `ssCH2` match);
// - an atom matching several patterns is counted for each of them;
// - at most 1000 unique matches per pattern are counted (lowest centres).
// `tests::test_estate_types_match_rdkit_smarts` checks the classifier against
// the SMARTS engine for every pattern.

/// Number of RDKit E-State atom types.
pub const NUM_ESTATE_TYPES: usize = 79;

/// RDKit E-State atom type names (`AtomTypes._rawD`), in RDKit's order.
pub const ESTATE_TYPE_NAMES: [&str; NUM_ESTATE_TYPES] = [
    "sLi", "ssBe", "ssssBe", "ssBH", "sssB", "ssssB", "sCH3", "dCH2", "ssCH2", "tCH", "dsCH",
    "aaCH", "sssCH", "ddC", "tsC", "dssC", "aasC", "aaaC", "ssssC", "sNH3", "sNH2", "ssNH2", "dNH",
    "ssNH", "aaNH", "tN", "sssNH", "dsN", "aaN", "sssN", "ddsN", "aasN", "ssssN", "sOH", "dO",
    "ssO", "aaO", "sF", "sSiH3", "ssSiH2", "sssSiH", "ssssSi", "sPH2", "ssPH", "sssP", "dsssP",
    "sssssP", "sSH", "dS", "ssS", "aaS", "dssS", "ddssS", "sCl", "sGeH3", "ssGeH2", "sssGeH",
    "ssssGe", "sAsH2", "ssAsH", "sssAs", "sssdAs", "sssssAs", "sSeH", "dSe", "ssSe", "aaSe",
    "dssSe", "ddssSe", "sBr", "sSnH3", "ssSnH2", "sssSnH", "ssssSn", "sI", "sPbH3", "ssPbH2",
    "sssPbH", "ssssPb",
];

/// RDKit's SMARTS definitions (`AtomTypes._rawD`), aligned with
/// [`ESTATE_TYPE_NAMES`]. The classifier does not parse them; they document
/// the types and are checked against the classifier in the tests.
pub const ESTATE_TYPE_SMARTS: [&str; NUM_ESTATE_TYPES] = [
    "[LiD1]-*",
    "[BeD2](-*)-*",
    "[BeD4](-*)(-*)(-*)-*",
    "[BD2H](-*)-*",
    "[BD3](-*)(-*)-*",
    "[BD4](-*)(-*)(-*)-*",
    "[CD1H3]-*",
    "[CD1H2]=*",
    "[CD2H2](-*)-*",
    "[CD1H]#*",
    "[CD2H](=*)-*",
    "[C,c;D2H](:*):*",
    "[CD3H](-*)(-*)-*",
    "[CD2H0](=*)=*",
    "[CD2H0](#*)-*",
    "[CD3H0](=*)(-*)-*",
    "[C,c;D3H0](:*)(:*)-*",
    "[C,c;D3H0](:*)(:*):*",
    "[CD4H0](-*)(-*)(-*)-*",
    "[ND1H3]-*",
    "[ND1H2]-*",
    "[ND2H2](-*)-*",
    "[ND1H]=*",
    "[ND2H](-*)-*",
    "[N,nD2H](:*):*",
    "[ND1H0]#*",
    "[ND3H](-*)(-*)-*",
    "[ND2H0](=*)-*",
    "[N,nD2H0](:*):*",
    "[ND3H0](-*)(-*)-*",
    "[ND3H0](~[OD1H0])(~[OD1H0])-,:*",
    "[N,nD3H0](:*)(:*)-,:*",
    "[ND4H0](-*)(-*)(-*)-*",
    "[OD1H]-*",
    "[OD1H0]=*",
    "[OD2H0](-*)-*",
    "[O,oD2H0](:*):*",
    "[FD1]-*",
    "[SiD1H3]-*",
    "[SiD2H2](-*)-*",
    "[SiD3H1](-*)(-*)-*",
    "[SiD4H0](-*)(-*)(-*)-*",
    "[PD1H2]-*",
    "[PD2H1](-*)-*",
    "[PD3H0](-*)(-*)-*",
    "[PD4H0](=*)(-*)(-*)-*",
    "[PD5H0](-*)(-*)(-*)(-*)-*",
    "[SD1H1]-*",
    "[SD1H0]=*",
    "[SD2H0](-*)-*",
    "[S,sD2H0](:*):*",
    "[SD3H0](=*)(-*)-*",
    "[SD4H0](~[OD1H0])(~[OD1H0])(-*)-*",
    "[ClD1]-*",
    "[GeD1H3](-*)",
    "[GeD2H2](-*)-*",
    "[GeD3H1](-*)(-*)-*",
    "[GeD4H0](-*)(-*)(-*)-*",
    "[AsD1H2]-*",
    "[AsD2H1](-*)-*",
    "[AsD3H0](-*)(-*)-*",
    "[AsD4H0](=*)(-*)(-*)-*",
    "[AsD5H0](-*)(-*)(-*)(-*)-*",
    "[SeD1H1]-*",
    "[SeD1H0]=*",
    "[SeD2H0](-*)-*",
    "[SeD2H0](:*):*",
    "[SeD3H0](=*)(-*)-*",
    "[SeD4H0](=*)(=*)(-*)-*",
    "[BrD1]-*",
    "[SnD1H3]-*",
    "[SnD2H2](-*)-*",
    "[SnD3H1](-*)(-*)-*",
    "[SnD4H0](-*)(-*)(-*)-*",
    "[ID1]-*",
    "[PbD1H3]-*",
    "[PbD2H2](-*)-*",
    "[PbD3H1](-*)(-*)-*",
    "[PbD4H0](-*)(-*)(-*)-*",
];

/// RDKit `GetSubstructMatches` default `maxMatches`.
const MAX_MATCHES_PER_TYPE: usize = 1000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arom {
    /// Element symbol outside the SMARTS organic subset: no aromaticity test.
    Any,
    Aliphatic,
    Aromatic,
}

/// One OR-alternative of a pattern's centre atom.
#[derive(Clone, Copy)]
struct CenterAlt {
    z: u8,
    arom: Arom,
    degree: Option<u8>,
    total_h: Option<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotBond {
    Single,
    Double,
    Triple,
    Aromatic,
    SingleOrAromatic,
    Any,
}

/// One branch `(<bond>*)` of a pattern; `terminal_o` restricts the
/// neighbour to `[OD1H0]`.
#[derive(Clone, Copy)]
struct Slot {
    bond: SlotBond,
    terminal_o: bool,
}

struct EStateType {
    alts: &'static [CenterAlt],
    slots: &'static [Slot],
}

const fn alt(z: u8, arom: Arom, degree: Option<u8>, total_h: Option<u8>) -> CenterAlt {
    CenterAlt {
        z,
        arom,
        degree,
        total_h,
    }
}
const fn slot(bond: SlotBond) -> Slot {
    Slot {
        bond,
        terminal_o: false,
    }
}

const S: Slot = slot(SlotBond::Single);
const D: Slot = slot(SlotBond::Double);
const T: Slot = slot(SlotBond::Triple);
const A: Slot = slot(SlotBond::Aromatic);
const SA: Slot = slot(SlotBond::SingleOrAromatic);
/// `~[OD1H0]`
const OX: Slot = Slot {
    bond: SlotBond::Any,
    terminal_o: true,
};

macro_rules! et {
    ([$($alt:expr),+], [$($slot:expr),*]) => {
        EStateType { alts: &[$($alt),+], slots: &[$($slot),*] }
    };
}

use Arom::{Aliphatic as AL, Any as AN, Aromatic as AR};

/// Direct encoding of [`ESTATE_TYPE_SMARTS`].
const ESTATE_TYPES: [EStateType; NUM_ESTATE_TYPES] = [
    et!([alt(3, AN, Some(1), None)], [S]),             // sLi
    et!([alt(4, AN, Some(2), None)], [S, S]),          // ssBe
    et!([alt(4, AN, Some(4), None)], [S, S, S, S]),    // ssssBe
    et!([alt(5, AL, Some(2), Some(1))], [S, S]),       // ssBH
    et!([alt(5, AL, Some(3), None)], [S, S, S]),       // sssB
    et!([alt(5, AL, Some(4), None)], [S, S, S, S]),    // ssssB
    et!([alt(6, AL, Some(1), Some(3))], [S]),          // sCH3
    et!([alt(6, AL, Some(1), Some(2))], [D]),          // dCH2
    et!([alt(6, AL, Some(2), Some(2))], [S, S]),       // ssCH2
    et!([alt(6, AL, Some(1), Some(1))], [T]),          // tCH
    et!([alt(6, AL, Some(2), Some(1))], [D, S]),       // dsCH
    et!([alt(6, AN, Some(2), Some(1))], [A, A]),       // aaCH ([C,c;...])
    et!([alt(6, AL, Some(3), Some(1))], [S, S, S]),    // sssCH
    et!([alt(6, AL, Some(2), Some(0))], [D, D]),       // ddC
    et!([alt(6, AL, Some(2), Some(0))], [T, S]),       // tsC
    et!([alt(6, AL, Some(3), Some(0))], [D, S, S]),    // dssC
    et!([alt(6, AN, Some(3), Some(0))], [A, A, S]),    // aasC ([C,c;...])
    et!([alt(6, AN, Some(3), Some(0))], [A, A, A]),    // aaaC ([C,c;...])
    et!([alt(6, AL, Some(4), Some(0))], [S, S, S, S]), // ssssC
    et!([alt(7, AL, Some(1), Some(3))], [S]),          // sNH3
    et!([alt(7, AL, Some(1), Some(2))], [S]),          // sNH2
    et!([alt(7, AL, Some(2), Some(2))], [S, S]),       // ssNH2
    et!([alt(7, AL, Some(1), Some(1))], [D]),          // dNH
    et!([alt(7, AL, Some(2), Some(1))], [S, S]),       // ssNH
    et!(
        [alt(7, AL, None, None), alt(7, AR, Some(2), Some(1))],
        [A, A]
    ), // aaNH
    et!([alt(7, AL, Some(1), Some(0))], [T]),          // tN
    et!([alt(7, AL, Some(3), Some(1))], [S, S, S]),    // sssNH
    et!([alt(7, AL, Some(2), Some(0))], [D, S]),       // dsN
    et!(
        [alt(7, AL, None, None), alt(7, AR, Some(2), Some(0))],
        [A, A]
    ), // aaN
    et!([alt(7, AL, Some(3), Some(0))], [S, S, S]),    // sssN
    et!([alt(7, AL, Some(3), Some(0))], [OX, OX, SA]), // ddsN
    et!(
        [alt(7, AL, None, None), alt(7, AR, Some(3), Some(0))],
        [A, A, SA]
    ), // aasN
    et!([alt(7, AL, Some(4), Some(0))], [S, S, S, S]), // ssssN
    et!([alt(8, AL, Some(1), Some(1))], [S]),          // sOH
    et!([alt(8, AL, Some(1), Some(0))], [D]),          // dO
    et!([alt(8, AL, Some(2), Some(0))], [S, S]),       // ssO
    et!(
        [alt(8, AL, None, None), alt(8, AR, Some(2), Some(0))],
        [A, A]
    ), // aaO
    et!([alt(9, AL, Some(1), None)], [S]),             // sF
    et!([alt(14, AN, Some(1), Some(3))], [S]),         // sSiH3
    et!([alt(14, AN, Some(2), Some(2))], [S, S]),      // ssSiH2
    et!([alt(14, AN, Some(3), Some(1))], [S, S, S]),   // sssSiH
    et!([alt(14, AN, Some(4), Some(0))], [S, S, S, S]), // ssssSi
    et!([alt(15, AL, Some(1), Some(2))], [S]),         // sPH2
    et!([alt(15, AL, Some(2), Some(1))], [S, S]),      // ssPH
    et!([alt(15, AL, Some(3), Some(0))], [S, S, S]),   // sssP
    et!([alt(15, AL, Some(4), Some(0))], [D, S, S, S]), // dsssP
    et!([alt(15, AL, Some(5), Some(0))], [S, S, S, S, S]), // sssssP
    et!([alt(16, AL, Some(1), Some(1))], [S]),         // sSH
    et!([alt(16, AL, Some(1), Some(0))], [D]),         // dS
    et!([alt(16, AL, Some(2), Some(0))], [S, S]),      // ssS
    et!(
        [alt(16, AL, None, None), alt(16, AR, Some(2), Some(0))],
        [A, A]
    ), // aaS
    et!([alt(16, AL, Some(3), Some(0))], [D, S, S]),   // dssS
    et!([alt(16, AL, Some(4), Some(0))], [OX, OX, S, S]), // ddssS
    et!([alt(17, AL, Some(1), None)], [S]),            // sCl
    et!([alt(32, AN, Some(1), Some(3))], [S]),         // sGeH3
    et!([alt(32, AN, Some(2), Some(2))], [S, S]),      // ssGeH2
    et!([alt(32, AN, Some(3), Some(1))], [S, S, S]),   // sssGeH
    et!([alt(32, AN, Some(4), Some(0))], [S, S, S, S]), // ssssGe
    et!([alt(33, AN, Some(1), Some(2))], [S]),         // sAsH2
    et!([alt(33, AN, Some(2), Some(1))], [S, S]),      // ssAsH
    et!([alt(33, AN, Some(3), Some(0))], [S, S, S]),   // sssAs
    et!([alt(33, AN, Some(4), Some(0))], [D, S, S, S]), // sssdAs
    et!([alt(33, AN, Some(5), Some(0))], [S, S, S, S, S]), // sssssAs
    et!([alt(34, AN, Some(1), Some(1))], [S]),         // sSeH
    et!([alt(34, AN, Some(1), Some(0))], [D]),         // dSe
    et!([alt(34, AN, Some(2), Some(0))], [S, S]),      // ssSe
    et!([alt(34, AN, Some(2), Some(0))], [A, A]),      // aaSe
    et!([alt(34, AN, Some(3), Some(0))], [D, S, S]),   // dssSe
    et!([alt(34, AN, Some(4), Some(0))], [D, D, S, S]), // ddssSe
    et!([alt(35, AL, Some(1), None)], [S]),            // sBr
    et!([alt(50, AN, Some(1), Some(3))], [S]),         // sSnH3
    et!([alt(50, AN, Some(2), Some(2))], [S, S]),      // ssSnH2
    et!([alt(50, AN, Some(3), Some(1))], [S, S, S]),   // sssSnH
    et!([alt(50, AN, Some(4), Some(0))], [S, S, S, S]), // ssssSn
    et!([alt(53, AL, Some(1), None)], [S]),            // sI
    et!([alt(82, AN, Some(1), Some(3))], [S]),         // sPbH3
    et!([alt(82, AN, Some(2), Some(2))], [S, S]),      // ssPbH2
    et!([alt(82, AN, Some(3), Some(1))], [S, S, S]),   // sssPbH
    et!([alt(82, AN, Some(4), Some(0))], [S, S, S, S]), // ssssPb
];

/// Candidate type indices per centre element and explicit degree (degrees
/// above 5 share the last bucket), built once.
fn types_by_element_and_degree() -> &'static [[SmallVec<[u8; 6]>; 7]; 119] {
    static TABLE: OnceLock<[[SmallVec<[u8; 6]>; 7]; 119]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t: [[SmallVec<[u8; 6]>; 7]; 119] =
            std::array::from_fn(|_| std::array::from_fn(|_| SmallVec::new()));
        for (ti, ty) in ESTATE_TYPES.iter().enumerate() {
            for (d, bucket) in t[ty.alts[0].z as usize].iter_mut().enumerate() {
                if ty
                    .alts
                    .iter()
                    .any(|a| a.degree.is_none_or(|x| d < 6 && x as usize == d))
                {
                    bucket.push(ti as u8);
                }
            }
        }
        t
    })
}

fn slot_accepts(slot: &Slot, mol: &Molecule, bond: BondType, nbr: usize) -> bool {
    let bond_ok = match slot.bond {
        SlotBond::Single => bond == BondType::Single,
        SlotBond::Double => bond == BondType::Double,
        SlotBond::Triple => bond == BondType::Triple,
        SlotBond::Aromatic => bond == BondType::Aromatic,
        SlotBond::SingleOrAromatic => bond == BondType::Single || bond == BondType::Aromatic,
        SlotBond::Any => true,
    };
    bond_ok
        && (!slot.terminal_o || {
            let a = &mol.atoms[nbr];
            a.atomic_number == 8
                && !a.is_aromatic()
                && mol.get_degree(nbr as AtomIdx) == 1
                && mol.total_hs_with_neighbors(nbr) == 0
        })
}

/// Collects every distinct neighbour subset (bit mask over `nbrs`) that an
/// injective slot -> neighbour assignment can cover. With `first_only` the
/// search stops at the first complete assignment.
fn assign_slots(
    slots: &[Slot],
    mol: &Molecule,
    nbrs: &[(usize, BondType)],
    used: u64,
    first_only: bool,
    out: &mut SmallVec<[u64; 4]>,
) -> bool {
    let Some((slot, rest)) = slots.split_first() else {
        if !out.contains(&used) {
            out.push(used);
        }
        return first_only;
    };
    for (k, &(nbr, bond)) in nbrs.iter().enumerate() {
        if used & (1 << k) == 0
            && slot_accepts(slot, mol, bond, nbr)
            && assign_slots(rest, mol, nbrs, used | (1 << k), first_only, out)
        {
            return true;
        }
    }
    false
}

type AtomSet = SmallVec<[u16; 8]>;

type Neighbours = SmallVec<[(usize, BondType); 8]>;

fn neighbours(mol: &Molecule, i: usize) -> Neighbours {
    mol.adjacency[i]
        .iter()
        .map(|&b| {
            let bond = &mol.bonds[b as usize];
            let other = if bond.source as usize == i {
                bond.target
            } else {
                bond.source
            };
            (other as usize, bond.bond_type)
        })
        .collect()
}

/// Atom sets (centre plus covered neighbours, sorted) of the distinct matches
/// of `ty` centred on atom `i` (with neighbours `nbrs`); empty when the
/// centre does not match.
fn match_sets(
    mol: &Molecule,
    i: usize,
    nbrs: &[(usize, BondType)],
    ty: &EStateType,
    out: &mut SmallVec<[AtomSet; 2]>,
) {
    out.clear();
    let degree = nbrs.len();
    if ty.slots.len() > degree || degree > 64 {
        return; // patterns have at most 5 branches
    }
    let aromatic = mol.atoms[i].is_aromatic();
    let mut total_h = None;
    let centre_ok = ty.alts.iter().any(|a| {
        (match a.arom {
            Arom::Any => true,
            Arom::Aliphatic => !aromatic,
            Arom::Aromatic => aromatic,
        }) && a.degree.is_none_or(|d| d as usize == degree)
            && a.total_h.is_none_or(|h| {
                h as u32 == *total_h.get_or_insert_with(|| mol.total_hs_with_neighbors(i))
            })
    });
    if !centre_ok {
        return;
    }
    let mut masks: SmallVec<[u64; 4]> = SmallVec::new();
    // when every neighbour must be covered there is only one subset
    assign_slots(ty.slots, mol, nbrs, 0, ty.slots.len() == degree, &mut masks);
    for mask in masks {
        let mut set: AtomSet = SmallVec::new();
        set.push(i as u16);
        set.extend(
            nbrs.iter()
                .enumerate()
                .filter(|(k, _)| mask & (1 << k) != 0)
                .map(|(_, &(n, _))| n as u16),
        );
        set.sort_unstable();
        out.push(set);
    }
}

/// RDKit `FingerprintMol(mol)[1]`: per type, the sum of the E-State indices
/// of the first atom of each unique match.
fn estate_type_sums(mol: &Molecule, estates: &[f64]) -> [f64; NUM_ESTATE_TYPES] {
    let mut sums = [0.0f64; NUM_ESTATE_TYPES];
    let mut counts = [0usize; NUM_ESTATE_TYPES];
    let table = types_by_element_and_degree();
    // bit t: the atom is the centre of at least one match of type t
    let mut centre_of = vec![0u128; mol.num_atoms()];
    let mut sets: SmallVec<[AtomSet; 2]> = SmallVec::new();
    let mut earlier: SmallVec<[AtomSet; 2]> = SmallVec::new();

    for i in 0..mol.num_atoms() {
        let degree = mol.get_degree(i as AtomIdx).min(6);
        let candidates = &table[(mol.atoms[i].atomic_number as usize).min(118)][degree];
        if candidates.is_empty() {
            continue;
        }
        let nbrs = neighbours(mol, i);
        for &ti in candidates {
            let ti = ti as usize;
            if counts[ti] >= MAX_MATCHES_PER_TYPE {
                continue;
            }
            let ty = &ESTATE_TYPES[ti];
            match_sets(mol, i, &nbrs, ty, &mut sets);
            if sets.is_empty() {
                continue;
            }
            let bit = 1u128 << ti;
            centre_of[i] |= bit;
            for set in &sets {
                // A set already matched from a lower centre counts once; that
                // centre is a member of the set, bonded to all other members.
                let duplicate = set.iter().any(|&c| {
                    let c = c as usize;
                    c < i && centre_of[c] & bit != 0 && {
                        let c_nbrs = neighbours(mol, c);
                        set.iter().all(|&m| {
                            m as usize == c || c_nbrs.iter().any(|&(n, _)| n == m as usize)
                        }) && {
                            match_sets(mol, c, &c_nbrs, ty, &mut earlier);
                            earlier.contains(set)
                        }
                    }
                });
                if !duplicate && counts[ti] < MAX_MATCHES_PER_TYPE {
                    counts[ti] += 1;
                    sums[ti] += estates[i];
                }
            }
        }
    }
    sums
}

// ===========================================================================
// Typed API: E-State Indices
// ===========================================================================

/// E-State indices.
///
/// Returns FloatVec with 82 values:
/// - [0-2]: Summary (sum, min, max), RDKit `EStateIndices`
/// - [3-81]: sums per RDKit E-State atom type ([`ESTATE_TYPE_NAMES`]),
///   RDKit `Fingerprinter.FingerprintMol(mol)[1]`
pub struct EStateIndices;

impl Descriptor for EStateIndices {
    fn name(&self) -> &str {
        "EStateIndices"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = Vec::with_capacity(3 + NUM_ESTATE_TYPES);

        let estates = compute_estates(mol, ctx);

        // Summary (3 values)
        if estates.is_empty() {
            values.push(0.0);
            values.push(0.0);
            values.push(0.0);
        } else {
            let sum: f64 = estates.iter().sum();
            let min = estates.iter().cloned().fold(f64::MAX, f64::min);
            let max = estates.iter().cloned().fold(f64::MIN, f64::max);
            values.push(sum);
            values.push(min);
            values.push(max);
        }

        // RDKit E-State atom-type sums (79 values)
        values.extend(estate_type_sums(mol, &estates));

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<EStateIndices>;
};

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

// (Legacy descriptors removed — use EStateIndices below)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_estate(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        match EStateIndices.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    fn type_idx(name: &str) -> usize {
        ESTATE_TYPE_NAMES
            .iter()
            .position(|&n| n == name)
            .expect("known type")
    }

    /// RDKit `FingerprintMol` reference values (RDKit 2026.03.2).
    #[test]
    fn test_estate_type_sums_rdkit_semantics() {
        // matches covering the same atom set count once
        let v = calc_estate("CC");
        assert_eq!(v[3 + type_idx("sCH3")], 2.0);
        let v = calc_estate("C1CC1");
        assert_eq!(v[3 + type_idx("ssCH2")], 1.5);
        let v = calc_estate("OO");
        assert_eq!(v[3 + type_idx("sOH")], 6.0);
        let v = calc_estate("C12C3C1C23");
        assert!((v[3 + type_idx("sssCH")] - 4.0 / 3.0).abs() < 1e-12);
        // at most 1000 unique matches per type (lowest centres)
        let v = calc_estate(&"C".repeat(1100));
        assert!((v[3 + type_idx("ssCH2")] - 1499.6734618864543).abs() < 1e-9);
        assert!((v[3 + type_idx("sCH3")] - 4.644024562590775).abs() < 1e-12);
    }

    /// The direct classifier reproduces RDKit's SMARTS semantics: for every
    /// pattern, unique matches (by atom set, lowest centre kept, at most 1000)
    /// found by the SMARTS engine give the same sums.
    #[test]
    fn test_estate_types_match_rdkit_smarts() {
        use crate::algo::iso::visit_matches;
        use crate::io::smarts::parse_smarts;
        let patterns: Vec<_> = ESTATE_TYPE_SMARTS
            .iter()
            .map(|s| parse_smarts(s).expect("valid E-State SMARTS"))
            .collect();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/parity_molecules.smi");
        let text = std::fs::read_to_string(path).expect("parity molecules");
        let extra = [
            "CC",
            "C1CC1",
            "OO",
            "C12C3C1C23",
            "[Li]C",
            "C[Be]C",
            "C[BH]C",
            "CB(C)C",
            "C[B-](C)(C)C",
            "C[SiH3]",
            "C[SiH2]C",
            "C[SiH](C)C",
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
            "O=[N+]([O-])[O-]",
            "[O-][N+]([O-])C",
            "C[PH2]",
            "CP(C)(C)(C)C",
            "[H]C([H])([H])O[H]",
            "[2H]OC",
            "c1cc2ccc3cccc4ccc(c1)c2c34",
            "C=C=C",
            "C#CC#N",
            "Cn1cccc1",
            "C[n+]1ccccc1",
            "[O-][n+]1ccccc1",
            "c1cc[nH]c1",
            "c1ccoc1",
            "c1ccsc1",
            "C1=CC=CC=C1",
        ];
        let mut checked = 0;
        for smi in text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .chain(extra)
        {
            let Ok(mol) = parse_smiles(smi) else {
                continue;
            };
            let estates = compute_estates(
                &mol,
                &mut crate::desc::context::CalculationContext::new(),
            );
            let got = estate_type_sums(&mol, &estates);
            for (ti, patt) in patterns.iter().enumerate() {
                let mut seen = std::collections::HashSet::new();
                let mut matches: Vec<(u16, Vec<u16>)> = Vec::new();
                visit_matches(&mol, patt, |m| {
                    let mut set = m.to_vec();
                    set.sort_unstable();
                    matches.push((m[0], set));
                    true
                });
                matches.sort();
                let mut sum = 0.0;
                let mut n = 0;
                for (centre, set) in matches {
                    if n < MAX_MATCHES_PER_TYPE && seen.insert(set) {
                        sum += estates[centre as usize];
                        n += 1;
                    }
                }
                assert!(
                    (sum - got[ti]).abs() <= 1e-9 * (1.0 + sum.abs()),
                    "{smi}: {} SMARTS {sum} vs classifier {}",
                    ESTATE_TYPE_NAMES[ti],
                    got[ti]
                );
            }
            checked += 1;
        }
        assert!(checked > 1000);
    }

    fn estates_raw(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        compute_estates(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
    }

    #[test]
    fn test_estate_ethane() {
        let e = estates_raw("CC");
        assert_eq!(e.len(), 2);
        assert!(
            (e[0] - e[1]).abs() < 0.001,
            "symmetric: {} vs {}",
            e[0],
            e[1]
        );
    }

    #[test]
    fn test_estate_summary_ethane() {
        let v = calc_estate("CC");
        // v[0]=sum, v[1]=min, v[2]=max
        assert!(v[0] > 0.0, "sum > 0");
        assert!((v[1] - v[2]).abs() < 0.01, "min ≈ max for symmetric mol");
    }

    #[test]
    fn test_estate_ethanol() {
        let e = estates_raw("CCO");
        assert!(e[2] > e[0], "O E-State={} > C E-State={}", e[2], e[0]);
    }

    #[test]
    fn test_estate_sum_conservation() {
        let mol = parse_smiles("CCCC").expect("valid");
        let estates = compute_estates(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        let n = mol.num_atoms();
        let intrinsic_sum: f64 = (0..n)
            .map(|i| {
                let atom = &mol.atoms[i];
                let pqn = principal_quantum_number(atom.atomic_number) as f64;
                let zv = outer_electrons(atom.atomic_number) as f64;
                let h = atom.total_h() as f64;
                let delta = mol.get_degree(i as AtomIdx) as f64;
                if delta == 0.0 {
                    return 0.0;
                }
                ((2.0 / pqn).powi(2) * (zv - h) + 1.0) / delta
            })
            .sum();
        let estate_sum: f64 = estates.iter().sum();
        assert!(
            (estate_sum - intrinsic_sum).abs() < 0.001,
            "E-State sum {} ≈ intrinsic sum {}",
            estate_sum,
            intrinsic_sum
        );
    }

    #[test]
    fn test_estate_atom_types_ethanol() {
        let v = calc_estate("CCO");
        // v[3 + ESTATE_sCH3] and v[3 + ESTATE_sOH] should be nonzero
        assert!(v[3 + type_idx("sCH3")] != 0.0, "should have sCH3");
        assert!(v[3 + type_idx("sOH")] != 0.0, "should have sOH");
    }

    #[test]
    fn test_estate_atom_types_benzene() {
        let v = calc_estate("c1ccccc1");
        assert!(
            v[3 + type_idx("aaCH")] != 0.0,
            "benzene should have aaCH: {}",
            v[3 + type_idx("aaCH")]
        );
    }

    #[test]
    fn test_estate_size() {
        let v = calc_estate("CCO");
        assert_eq!(v.len(), 82, "E-State has 82 values");
    }

    #[test]
    fn test_estate_benzene_symmetry() {
        let e = estates_raw("c1ccccc1");
        let first = e[0];
        for val in &e[1..] {
            assert!(
                (val - first).abs() < 0.001,
                "benzene atoms should be equal: {} vs {}",
                val,
                first
            );
        }
    }

    #[test]
    fn test_estate_aspirin() {
        let v = calc_estate("CC(=O)Oc1ccccc1C(=O)O");
        assert!(v[0] > 0.0, "aspirin E-State sum = {}", v[0]);
        assert!(v[2] > v[1], "max > min for complex mol");
    }
}
