//! Stereochemistry perception (port of RDKit 2026.03 `Chirality.cpp`,
//! `FindStereo.cpp` and the parts of `new_canon` they rely on).
//!
//! Three layers, each mirroring RDKit exactly for molecules read from SMILES:
//!
//! 1. [`legacy_stereo_perception`]: RDKit's *legacy* `assignStereochemistry`
//!    (`cleanIt = true, force = true, flagPossibleStereoCenters = true`), the
//!    default perception that `MolFromSmiles` runs. It produces the legacy CIP
//!    ranks, R/S codes, `_ChiralityPossible` flags, cleaned chiral tags and
//!    E/Z labels of double bonds written with `/` `\`.
//! 2. [`find_potential_stereo_bonds`]: legacy `FindPotentialStereoBonds`
//!    (`cleanIt = false`), which marks candidate double bonds `STEREOANY`.
//! 3. [`find_potential_tetrahedral_centers`]: the new-style
//!    `FindPotentialStereo(cleanIt = false, flagPossible = true)` as used by
//!    `Chem.FindMolChiralCenters(useLegacyImplementation=False,
//!    includeUnassigned=True)`.
//!
//! The descriptors built on top are `NumAtomStereoCenters`,
//! `NumUnspecifiedAtomStereoCenters` (`Lipinski.cpp`) and `SPS`
//! (`rdkit/Chem/SpacialScore.py`).
//!
//! Chirality conventions: an atom's [`ChiralTag`] is relative to the order of
//! its bonds in `Molecule::adjacency`, with an implicit H (or lone pair) last.
//! Bond directions (`/`, `\`) are stored by the SMILES parser in the bond
//! flags (see [`bond_direction`]) relative to the bond's `source` atom.

use std::collections::VecDeque;

use smallvec::SmallVec;

use crate::mol::atom::ChiralTag;
use crate::mol::bond::Bond;
use crate::mol::elements::ELEMENT_SYMBOLS;
use crate::mol::molecule::{Molecule, RingInfo};
use crate::mol::rdkit_data::ELEMENTS;
use crate::types::{BondType, Hybridization};

// ---------------------------------------------------------------------------
// Bond directions
// ---------------------------------------------------------------------------

/// Bond flag bit holding the direction sense (set: `/`, clear: `\`). Only
/// meaningful when `Bond::has_direction()` is set.
const BOND_DIR_UP_BIT: u8 = 0b1_0000;

/// RDKit single-bond direction used for double-bond stereo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondDir {
    None,
    /// `/` (RDKit `ENDUPRIGHT`), relative to the bond's source atom.
    EndUpRight,
    /// `\` (RDKit `ENDDOWNRIGHT`), relative to the bond's source atom.
    EndDownRight,
}

impl BondDir {
    fn is_set(self) -> bool {
        self != BondDir::None
    }
    fn flipped(self) -> BondDir {
        match self {
            BondDir::EndUpRight => BondDir::EndDownRight,
            BondDir::EndDownRight => BondDir::EndUpRight,
            BondDir::None => BondDir::None,
        }
    }
}

/// Marks `bond` as directional (`up`: `/`, otherwise `\`).
pub(crate) fn set_bond_direction(bond: &mut Bond, up: bool) {
    bond.set_has_direction(true);
    if up {
        bond.flags |= BOND_DIR_UP_BIT;
    } else {
        bond.flags &= !BOND_DIR_UP_BIT;
    }
}

/// Direction of a bond as written in SMILES.
pub fn bond_direction(bond: &Bond) -> BondDir {
    if !bond.has_direction() {
        BondDir::None
    } else if bond.flags & BOND_DIR_UP_BIT != 0 {
        BondDir::EndUpRight
    } else {
        BondDir::EndDownRight
    }
}

/// Parity of the permutation turning `from` into `to` (RDKit
/// `countSwapsToInterconvert` modulo 2). Both slices hold the same elements.
pub(crate) fn permutation_is_odd(from: &[usize], to: &[usize]) -> bool {
    let mut probe: SmallVec<[usize; 8]> = from.iter().copied().collect();
    let mut swaps = 0usize;
    for (i, &want) in to.iter().enumerate() {
        if probe[i] != want
            && let Some(j) = (i + 1..probe.len()).find(|&j| probe[j] == want) {
                probe.swap(i, j);
                swaps += 1;
            }
    }
    swaps % 2 == 1
}

// ---------------------------------------------------------------------------
// Molecule helpers
// ---------------------------------------------------------------------------

static EMPTY_RING_INFO: std::sync::OnceLock<RingInfo> = std::sync::OnceLock::new();

struct Ctx<'a> {
    mol: &'a Molecule,
    ri: &'a RingInfo,
    /// (neighbour, bond) in adjacency order.
    nbrs: Vec<SmallVec<[(usize, usize); 4]>>,
    total_h: Vec<u32>,
    min_bond_ring_size: Vec<usize>,
}

impl<'a> Ctx<'a> {
    fn new(mol: &'a Molecule) -> Self {
        let ri = mol
            .ring_info()
            .unwrap_or_else(|| EMPTY_RING_INFO.get_or_init(RingInfo::default));
        let nbrs = (0..mol.num_atoms())
            .map(|a| {
                mol.adjacency[a]
                    .iter()
                    .map(|&bi| {
                        let b = &mol.bonds[bi as usize];
                        let o = if b.source as usize == a { b.target } else { b.source };
                        (o as usize, bi as usize)
                    })
                    .collect()
            })
            .collect();
        let total_h = mol.atoms.iter().map(|a| a.total_h() as u32).collect();
        let mut min_bond_ring_size = vec![usize::MAX; mol.num_bonds()];
        for ring in &ri.bond_rings {
            for &b in ring {
                let m = &mut min_bond_ring_size[b as usize];
                *m = (*m).min(ring.len());
            }
        }
        Ctx { mol, ri, nbrs, total_h, min_bond_ring_size }
    }

    fn n(&self) -> usize {
        self.mol.num_atoms()
    }
    fn anum(&self, a: usize) -> u8 {
        self.mol.atoms[a].atomic_number
    }
    fn degree(&self, a: usize) -> usize {
        self.nbrs[a].len()
    }
    fn total_degree(&self, a: usize) -> u32 {
        self.degree(a) as u32 + self.total_h[a]
    }
    /// `getTotalNumHs(includeNeighbors = true)`.
    fn total_h_with_nbrs(&self, a: usize) -> u32 {
        self.total_h[a] + self.nbrs[a].iter().filter(|&&(o, _)| self.anum(o) == 1).count() as u32
    }
    /// RDKit bond type (aromatic bonds are `AROMATIC`).
    fn btype(&self, b: usize) -> BondType {
        let bond = &self.mol.bonds[b];
        if bond.is_aromatic() { BondType::Aromatic } else { bond.bond_type }
    }
    fn bond_begin(&self, b: usize) -> usize {
        self.mol.bonds[b].source as usize
    }
    fn bond_end(&self, b: usize) -> usize {
        self.mol.bonds[b].target as usize
    }
    fn bond_between(&self, a: usize, o: usize) -> Option<usize> {
        self.nbrs[a].iter().find(|&&(n, _)| n == o).map(|&(_, b)| b)
    }
    fn num_atom_rings(&self, a: usize) -> usize {
        self.ri.num_atom_rings(a)
    }
    fn num_bond_rings(&self, b: usize) -> usize {
        self.ri.num_bond_rings(b)
    }
    fn is_atom_in_ring_of_size(&self, a: usize, size: usize) -> bool {
        self.ri.atom_rings.iter().any(|r| r.len() == size && r.iter().any(|&x| x as usize == a))
    }
    /// RDKit `RingInfo::atomMembers`: indices of the rings containing `a`.
    fn atom_members(&self, a: usize) -> SmallVec<[usize; 4]> {
        self.ri
            .atom_rings
            .iter()
            .enumerate()
            .filter(|(_, r)| r.iter().any(|&x| x as usize == a))
            .map(|(i, _)| i)
            .collect()
    }
    /// `shouldDetectDoubleBondStereo`.
    fn should_detect_double_bond_stereo(&self, b: usize) -> bool {
        self.num_bond_rings(b) == 0 || self.min_bond_ring_size[b] >= 8
    }
    /// `bondAffectsAtomChirality`.
    fn bond_affects_atom_chirality(&self, b: usize, a: usize) -> bool {
        match self.mol.bonds[b].bond_type {
            BondType::Unknown => false,
            BondType::Dative => self.bond_begin(b) != a,
            _ => true,
        }
    }
    /// `getAtomNonzeroDegree`.
    fn nonzero_degree(&self, a: usize) -> u32 {
        self.nbrs[a].iter().filter(|&&(_, b)| self.bond_affects_atom_chirality(b, a)).count() as u32
    }
    fn has_protium_neighbor(&self, a: usize) -> bool {
        self.nbrs[a].iter().any(|&(o, _)| self.anum(o) == 1 && self.mol.atoms[o].isotope == 0)
    }
    fn has_conjugated_bond(&self, a: usize) -> bool {
        self.nbrs[a].iter().any(|&(_, b)| self.mol.bonds[b].is_conjugated())
    }
    fn explicit_valence(&self, a: usize) -> i32 {
        let atom = &self.mol.atoms[a];
        crate::algo::sanitize::total_valence(self.mol, a)
            - if atom.no_implicit_h() { 0 } else { atom.num_implicit_h as i32 }
    }
    /// `queryIsAtomBridgehead`.
    fn is_bridgehead(&self, a: usize) -> bool {
        if self.degree(a) < 3 {
            return false;
        }
        let ring_bonds: SmallVec<[usize; 6]> =
            self.nbrs[a].iter().map(|&(_, b)| b).filter(|&b| self.num_bond_rings(b) > 0).collect();
        if ring_bonds.len() < 3 {
            return false;
        }
        let brings = &self.ri.bond_rings;
        let mut overlaps = vec![false; brings.len()];
        for i in 0..brings.len() {
            if !brings[i].iter().any(|&b| ring_bonds.contains(&(b as usize))) {
                continue;
            }
            for j in i + 1..brings.len() {
                let in_j = brings[j].iter().any(|&b| ring_bonds.contains(&(b as usize)));
                let overlap = brings[j].iter().filter(|b| brings[i].contains(b)).count();
                if overlap >= 2 && in_j {
                    overlaps[i] = true;
                    overlaps[j] = true;
                }
            }
            if !overlaps[i] {
                return false;
            }
        }
        true
    }
}

fn twice_bond_type(t: BondType) -> usize {
    match t {
        BondType::Single | BondType::Dative => 2,
        BondType::Double => 4,
        BondType::Triple => 6,
        BondType::Aromatic => 3,
        BondType::Unknown => 0,
    }
}

// ---------------------------------------------------------------------------
// Legacy CIP ranks (buildCIPInvariants / iterateCIPRanks / rerankAtoms)
// ---------------------------------------------------------------------------

fn build_cip_invariants(ctx: &Ctx) -> Vec<i64> {
    const N_MASS_BITS: u32 = 10;
    const MAX_MASS: i64 = 1 << N_MASS_BITS;
    ctx.mol
        .atoms
        .iter()
        .map(|atom| {
            let num = (atom.atomic_number % 128) as i64;
            let mut mass: i64 = 0;
            if atom.isotope != 0 {
                let common = ELEMENTS.get(atom.atomic_number as usize).map_or(0, |e| e.common_isotope);
                mass = atom.isotope as i64 - common as i64;
                if mass >= 0 {
                    mass += 1;
                }
            }
            mass += MAX_MASS / 2;
            mass = if mass < 0 { 0 } else { mass % MAX_MASS };
            let mut inv = num;
            inv = (inv << N_MASS_BITS) | mass;
            // atom map numbers are not kept by the parser: mapnum = -1 -> 0
            inv <<= 10;
            inv
        })
        .collect()
}

/// RDKit `iterateCIPRanks`.
///
/// RDKit ranks the atoms by their entries `[invariant or atomic number, the
/// ranks kept so far..., neighbour features...]` (lexicographic, dense ranks).
/// The prefix before the features always orders exactly like the current
/// ranks (each rank refines the values before it), so a round is a partition
/// refinement: only atoms sharing a rank are compared, by their features.
fn iterate_cip_ranks(ctx: &Ctx, invars: &[i64], ranks: &mut Vec<u32>, seed_with_invars: bool) {
    let n = ctx.n();
    ranks.resize(n, 0);
    // atoms in rank order (ties adjacent)
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by_key(|&i| invars[i]);
    let mut num_ranks = 0usize;
    for k in 0..n {
        if k > 0 && invars[idx[k]] != invars[idx[k - 1]] {
            num_ranks += 1;
        }
        ranks[idx[k]] = num_ranks as u32;
    }
    if n > 0 {
        num_ranks += 1;
    }
    let _ = seed_with_invars; // the prefix layout does not change the order
    // per atom: (count, neighbour)
    let mut features: Vec<SmallVec<[(usize, usize); 4]>> = (0..n)
        .map(|a| {
            ctx.nbrs[a]
                .iter()
                .map(|&(o, b)| {
                    let t = ctx.btype(b);
                    let p_special = t == BondType::Double
                        && ctx.anum(o) == 15
                        && matches!(ctx.degree(o), 3 | 4);
                    (if p_special { 1 } else { twice_bond_type(t) }, o)
                })
                .collect()
        })
        .collect();
    let mut flat: Vec<i64> = Vec::new();
    let mut off = vec![0usize; n + 1];
    let mut start = vec![0usize; n];
    let mut old = vec![0u32; n];
    let max_its = n / 2 + 1;
    let mut num_its = 0;
    let mut last_num_ranks: i64 = -1;
    while num_ranks < n && num_its < max_its && (last_num_ranks < 0 || (last_num_ranks as usize) < num_ranks) {
        // features of the atoms that share their rank
        flat.clear();
        for k in 0..n {
            let i = idx[k];
            let tied = (k > 0 && ranks[idx[k - 1]] == ranks[i]) || (k + 1 < n && ranks[idx[k + 1]] == ranks[i]);
            start[i] = flat.len();
            if tied {
                // neighbours by decreasing rank; equal ranks contribute equal values
                features[i].sort_unstable_by(|x, y| ranks[y.1].cmp(&ranks[x.1]));
                for &(count, o) in &features[i] {
                    flat.extend(std::iter::repeat_n(ranks[o] as i64 + 1, count));
                }
                flat.extend(std::iter::repeat_n(0, ctx.total_h[i] as usize));
            }
            off[i] = flat.len();
        }
        let feat = |i: usize| &flat[start[i]..off[i]];
        let mut k = 0;
        while k < n {
            let r = ranks[idx[k]];
            let mut e = k + 1;
            while e < n && ranks[idx[e]] == r {
                e += 1;
            }
            if e - k > 1 {
                idx[k..e].sort_unstable_by(|&a, &b| feat(a).cmp(feat(b)));
            }
            k = e;
        }
        old.copy_from_slice(ranks);
        let mut rank = 0u32;
        for k in 0..n {
            let (i, p) = (idx[k], if k > 0 { idx[k - 1] } else { usize::MAX });
            if k > 0 && (old[i] != old[p] || feat(i) != feat(p)) {
                rank += 1;
            }
            ranks[i] = rank;
        }
        last_num_ranks = num_ranks as i64;
        num_ranks = if n == 0 { 0 } else { rank as usize + 1 };
        num_its += 1;
    }
}

fn assign_atom_cip_ranks(ctx: &Ctx, ranks: &mut Vec<u32>) {
    let invars = build_cip_invariants(ctx);
    iterate_cip_ranks(ctx, &invars, ranks, false);
}

/// Double-bond stereo as assigned by RDKit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondStereo {
    None,
    Any,
    /// E relative to the bond's stereo atoms.
    E,
    /// Z relative to the bond's stereo atoms.
    Z,
}

/// Legacy CIP code assigned to an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipCode {
    R,
    S,
}

/// Result of RDKit's legacy `assignStereochemistry(cleanIt, force, flagPossible)`.
#[derive(Debug, Clone)]
pub struct LegacyStereo {
    /// `_ChiralityPossible` flag per atom.
    pub chirality_possible: Vec<bool>,
    /// `_CIPCode` per atom (legacy, approximate CIP).
    pub cip_codes: Vec<Option<CipCode>>,
    /// Chiral tags after cleanup (tags on non-stereocentres removed).
    pub chiral_tags: Vec<ChiralTag>,
    /// Stereo of each bond (only double bonds can be non-`None`).
    pub bond_stereo: Vec<BondStereo>,
    /// Stereo atoms (begin side, end side) of stereo double bonds.
    pub stereo_atoms: Vec<Option<(usize, usize)>>,
    /// Final legacy CIP ranks (`_CIPRank`).
    pub cip_ranks: Vec<u32>,
}

fn rerank_atoms(ctx: &Ctx, st: &LegacyStereo, ranks: &mut Vec<u32>) {
    let n = ctx.n();
    let mut factor: i64 = 100;
    while factor < n as i64 {
        factor *= 10;
    }
    let invars: Vec<i64> = (0..n)
        .map(|i| {
            let mut v = ranks[i] as i64 * factor;
            match st.cip_codes[i] {
                Some(CipCode::S) => v += 10,
                Some(CipCode::R) => v += 20,
                None => {}
            }
            for &(_, b) in &ctx.nbrs[i] {
                if ctx.btype(b) == BondType::Double {
                    match st.bond_stereo[b] {
                        BondStereo::E => v += 1,
                        BondStereo::Z => v += 2,
                        _ => {}
                    }
                }
            }
            v
        })
        .collect();
    iterate_cip_ranks(ctx, &invars, ranks, true);
}

// ---------------------------------------------------------------------------
// Legacy chiral centres
// ---------------------------------------------------------------------------

/// `isAtomPotentialChiralCenter`: (legal centre, duplicate neighbour ranks,
/// (rank, bond) of the neighbours).
fn is_atom_potential_chiral_center(
    ctx: &Ctx,
    a: usize,
    ranks: &[u32],
) -> (bool, bool, SmallVec<[(u32, usize); 4]>) {
    let mut legal = true;
    let mut dupes = false;
    let mut nbrs = SmallVec::new();
    let nz = ctx.nonzero_degree(a);
    let h = ctx.total_h[a];
    let tnz = nz + h;
    let z = ctx.anum(a);
    if !(3..=4).contains(&tnz) {
        legal = false;
    } else if nz < 3 && z != 15 && z != 33 {
        legal = false;
    } else if nz == 3 {
        if h == 1 {
            if ctx.has_protium_neighbor(a) {
                legal = false;
            }
        } else {
            legal = false;
            if z == 7 {
                if ctx.mol.atoms[a].hybridization == Hybridization::SP3
                    && !ctx.has_conjugated_bond(a)
                    && (ctx.is_atom_in_ring_of_size(a, 3) || ctx.is_bridgehead(a))
                {
                    legal = true;
                }
            } else if z == 15 || z == 33 {
                legal = true;
            } else if z == 16 || z == 34 {
                let ev = ctx.explicit_valence(a);
                if ev == 4 || (ev == 3 && ctx.mol.atoms[a].formal_charge == 1) {
                    legal = true;
                }
            }
        }
    }
    if legal && !ranks.is_empty() {
        let mut seen: SmallVec<[u32; 4]> = SmallVec::new();
        for &(o, b) in &ctx.nbrs[a] {
            nbrs.push((ranks[o], b));
            if !ctx.bond_affects_atom_chirality(b, a) {
                continue;
            }
            if seen.contains(&ranks[o]) {
                dupes = true;
                break;
            }
            seen.push(ranks[o]);
        }
    }
    (legal, dupes, nbrs)
}

/// `assignAtomChiralCodes` with `flagPossibleStereoCenters = true`.
fn assign_atom_chiral_codes(ctx: &Ctx, st: &mut LegacyStereo, ranks: &mut Vec<u32>) -> (bool, bool) {
    let mut changed = false;
    let mut unassigned = 0i32;
    for a in 0..ctx.n() {
        if st.cip_codes[a].is_some() {
            continue;
        }
        if ranks.is_empty() {
            assign_atom_cip_ranks(ctx, ranks);
        }
        let (legal, dupes, mut nbrs) = is_atom_potential_chiral_center(ctx, a, ranks);
        if legal {
            unassigned += 1;
        }
        if legal && !dupes {
            st.chirality_possible[a] = true;
        }
        let tag = st.chiral_tags[a];
        if legal && !dupes && tag.is_tetrahedral() {
            changed = true;
            unassigned -= 1;
            nbrs.sort_by_key(|&(r, _)| r);
            let probe: SmallVec<[usize; 4]> = nbrs.iter().map(|&(_, b)| b).collect();
            let reference: SmallVec<[usize; 4]> = ctx.nbrs[a].iter().map(|&(_, b)| b).collect();
            let mut odd = permutation_is_odd(&probe, &reference);
            if probe.len() == 3 && ctx.total_h[a] == 1 {
                odd = !odd;
            }
            let tag = if odd { tag.inverted() } else { tag };
            st.cip_codes[a] = Some(if tag == ChiralTag::TetrahedralCCW { CipCode::S } else { CipCode::R });
        }
    }
    (unassigned > 0, changed)
}

/// `findAtomNeighborDirHelper`: (neighbour, effective direction) pairs.
fn find_atom_neighbor_dir_helper(
    ctx: &Ctx,
    a: usize,
    ref_bond: usize,
    ranks: &[u32],
    dirs: &[BondDir],
) -> SmallVec<[(usize, BondDir); 3]> {
    let mut seen_dir = false;
    let mut out: SmallVec<[(usize, BondDir); 3]> = SmallVec::new();
    for &(o, b) in &ctx.nbrs[a] {
        if b == ref_bond {
            continue;
        }
        let mut dir = dirs[b];
        if dir.is_set() {
            seen_dir = true;
            if a != ctx.bond_begin(b) {
                dir = dir.flipped();
            }
        }
        out.push((o, dir));
    }
    if !seen_dir {
        out.clear();
    } else if out.len() == 2 && ranks[out[0].0] == ranks[out[1].0] {
        out.clear();
    } else if !out[0].1.is_set() {
        if out.len() > 1 {
            out[0].1 = if out[1].1 == BondDir::EndDownRight { BondDir::EndUpRight } else { BondDir::EndDownRight };
        }
    } else if out.len() > 1 && !out[1].1.is_set() {
        out[1].1 = if out[0].1 == BondDir::EndDownRight { BondDir::EndUpRight } else { BondDir::EndDownRight };
    }
    out
}

/// `assignBondStereoCodes`.
fn assign_bond_stereo_codes(
    ctx: &Ctx,
    st: &mut LegacyStereo,
    ranks: &mut Vec<u32>,
    dirs: &mut [BondDir],
) -> (bool, bool) {
    let mut assigned = false;
    let mut unassigned = 0i32;
    let mut to_clear: Vec<usize> = Vec::new();
    for b in 0..ctx.mol.num_bonds() {
        if ctx.btype(b) != BondType::Double || st.bond_stereo[b] != BondStereo::None {
            continue;
        }
        if ranks.is_empty() {
            assign_atom_cip_ranks(ctx, ranks);
        }
        st.stereo_atoms[b] = None;
        if !ctx.should_detect_double_bond_stereo(b) {
            continue;
        }
        let (beg, end) = (ctx.bond_begin(b), ctx.bond_end(b));
        if !matches!(ctx.degree(beg), 2 | 3) || !matches!(ctx.degree(end), 2 | 3) {
            continue;
        }
        unassigned += 1;
        let bn = find_atom_neighbor_dir_helper(ctx, beg, b, ranks, dirs);
        let en = find_atom_neighbor_dir_helper(ctx, end, b, ranks, dirs);
        if bn.is_empty() || en.is_empty() {
            continue;
        }
        let pick = |v: &SmallVec<[(usize, BondDir); 3]>| {
            if v.len() == 1 || ranks[v[0].0] > ranks[v[1].0] { v[0] } else { v[1] }
        };
        let (beg_nbr, beg_dir) = pick(&bn);
        let (end_nbr, end_dir) = pick(&en);
        let conflict_beg = bn.len() == 2 && bn[0].1 == bn[1].1;
        let conflict_end = en.len() == 2 && en[0].1 == en[1].1;
        if conflict_beg || conflict_end {
            st.bond_stereo[b] = BondStereo::None;
            assigned = true;
            if conflict_beg {
                to_clear.extend(bn.iter().filter_map(|&(o, _)| ctx.bond_between(o, beg)));
            }
            if conflict_end {
                to_clear.extend(en.iter().filter_map(|&(o, _)| ctx.bond_between(o, end)));
            }
        } else {
            st.stereo_atoms[b] = Some((beg_nbr, end_nbr));
            st.bond_stereo[b] = if beg_dir == end_dir { BondStereo::Z } else { BondStereo::E };
            assigned = true;
        }
        unassigned -= 1;
    }
    for b in to_clear {
        dirs[b] = BondDir::None;
    }
    (unassigned > 0, assigned)
}

/// `atomIsCandidateForRingStereochem` (uses the final legacy ranks).
fn atom_is_candidate_for_ring_stereochem(ctx: &Ctx, a: usize, ranks: &[u32]) -> bool {
    if ctx.num_atom_rings(a) == 0 {
        return false;
    }
    if ctx.anum(a) == 7
        && ctx.total_degree(a) == 3
        && !ctx.is_atom_in_ring_of_size(a, 3)
        && !ctx.is_bridgehead(a)
    {
        return false;
    }
    let mut non_ring: SmallVec<[usize; 4]> = SmallVec::new();
    let mut ring_nbrs = 0usize;
    let mut ring_nbr_ranks: SmallVec<[u32; 4]> = SmallVec::new();
    for &(o, b) in &ctx.nbrs[a] {
        if ctx.num_bond_rings(b) == 0 {
            non_ring.push(o);
        } else {
            ring_nbrs += 1;
            if !ring_nbr_ranks.contains(&ranks[o]) {
                ring_nbr_ranks.push(ranks[o]);
            }
        }
    }
    match non_ring.len() {
        // non-ring neighbours differ AND ring neighbours tie (RDKit #8956)
        2 => ranks[non_ring[0]] != ranks[non_ring[1]] && ring_nbrs != ring_nbr_ranks.len(),
        1 => ring_nbrs > ring_nbr_ranks.len(),
        0 => (ring_nbrs == 4 && ring_nbr_ranks.len() == 3) || (ring_nbrs == 3 && ring_nbr_ranks.len() == 2),
        _ => false,
    }
}

/// `findChiralAtomSpecialCases`: returns (possibleSpecialCases, has `_ringStereoAtoms`).
fn find_chiral_atom_special_cases(ctx: &Ctx, st: &LegacyStereo, ranks: &[u32]) -> (Vec<bool>, Vec<bool>) {
    let n = ctx.n();
    let mut special = vec![false; n];
    let mut has_rsa = vec![false; n];
    let mut cand_cache: Vec<Option<bool>> = vec![None; n];
    let mut cand = |a: usize| -> bool {
        if let Some(c) = cand_cache[a] {
            return c;
        }
        let c = atom_is_candidate_for_ring_stereochem(ctx, a, ranks);
        cand_cache[a] = Some(c);
        c
    };
    let mut atoms_seen = vec![false; n];
    let mut atoms_used = vec![false; n];
    let mut bonds_seen = vec![false; ctx.mol.num_bonds()];
    for a in 0..n {
        if atoms_seen[a] {
            continue;
        }
        if st.chiral_tags[a] == ChiralTag::Unspecified
            || st.cip_codes[a].is_some()
            || ctx.num_atom_rings(a) == 0
            || !cand(a)
        {
            continue;
        }
        let mut next: VecDeque<usize> = VecDeque::new();
        for &(o, b) in &ctx.nbrs[a] {
            if !bonds_seen[b] {
                bonds_seen[b] = true;
                if ctx.num_bond_rings(b) > 0 && !atoms_seen[o] {
                    next.push_back(o);
                    atoms_used[o] = true;
                }
            }
        }
        let mut found = has_rsa[a] && !next.is_empty();
        while let Some(r) = next.pop_front() {
            atoms_seen[r] = true;
            if st.chiral_tags[r] != ChiralTag::Unspecified && st.cip_codes[r].is_none() && cand(r) {
                found = true;
                has_rsa[r] = true;
                special[r] = true;
                special[a] = true;
            }
            for &(o, b) in &ctx.nbrs[r] {
                if !bonds_seen[b] {
                    bonds_seen[b] = true;
                    if ctx.num_bond_rings(b) > 0 && !atoms_seen[o] && !atoms_used[o] {
                        next.push_back(o);
                        atoms_used[o] = true;
                    }
                }
            }
        }
        if found {
            has_rsa[a] = true;
        } else {
            special[a] = false;
        }
        atoms_seen[a] = true;
    }
    (special, has_rsa)
}

/// RDKit legacy `assignStereochemistry(mol, cleanIt=true, force=true,
/// flagPossibleStereoCenters=true)` - what `MolFromSmiles` runs.
pub fn legacy_stereo_perception(mol: &Molecule) -> LegacyStereo {
    let ctx = Ctx::new(mol);
    legacy_impl(&ctx)
}

fn legacy_impl(ctx: &Ctx) -> LegacyStereo {
    let mol = ctx.mol;
    let n = ctx.n();
    let nb = mol.num_bonds();
    let mut st = LegacyStereo {
        chirality_possible: vec![false; n],
        cip_codes: vec![None; n],
        chiral_tags: mol.atoms.iter().map(|a| a.chiral_tag()).collect(),
        bond_stereo: vec![BondStereo::None; nb],
        stereo_atoms: vec![None; nb],
        cip_ranks: Vec::new(),
    };
    let mut dirs: Vec<BondDir> = mol.bonds.iter().map(bond_direction).collect();

    // Only run the perception when something could be stereo (RDKit 2026.03).
    let mut has_stereo_atoms = false;
    let mut has_potential_atoms = false;
    for a in 0..n {
        if !has_stereo_atoms && st.chiral_tags[a] != ChiralTag::Unspecified {
            has_stereo_atoms = true;
        } else if !has_potential_atoms {
            has_potential_atoms = is_atom_potential_chiral_center(ctx, a, &[]).0;
        }
    }
    let mut has_stereo_bonds = false;
    let mut has_potential_bonds = false;
    for b in 0..nb {
        let t = ctx.btype(b);
        if matches!(t, BondType::Double | BondType::Aromatic) && !ctx.should_detect_double_bond_stereo(b) {
            continue;
        }
        if !has_stereo_bonds && t == BondType::Double {
            let (beg, end) = (ctx.bond_begin(b), ctx.bond_end(b));
            let specified =
                ctx.nbrs[beg].iter().chain(ctx.nbrs[end].iter()).any(|&(_, nb)| dirs[nb].is_set());
            has_stereo_bonds = specified;
            if !has_potential_bonds && !specified {
                has_potential_bonds = true;
            }
        }
    }

    let mut ranks: Vec<u32> = Vec::new();
    let mut keep_going =
        has_stereo_atoms || has_stereo_bonds || has_potential_atoms || has_potential_bonds;
    while keep_going {
        let changed_atoms = if has_stereo_atoms || has_potential_atoms {
            let (h, c) = assign_atom_chiral_codes(ctx, &mut st, &mut ranks);
            has_stereo_atoms = h;
            c
        } else {
            false
        };
        let changed_bonds = if has_stereo_bonds || has_potential_bonds {
            let (h, c) = assign_bond_stereo_codes(ctx, &mut st, &mut ranks, &mut dirs);
            has_stereo_bonds = h;
            c
        } else {
            false
        };
        keep_going = (has_stereo_atoms || has_stereo_bonds) && (changed_atoms || changed_bonds);
        if keep_going {
            rerank_atoms(ctx, &st, &mut ranks);
        }
    }

    // cleanIt: remove chiral tags from atoms that are not stereocentres
    if n > 0 {
        let (special, has_rsa) = find_chiral_atom_special_cases(ctx, &st, &ranks);
        for a in 0..n {
            let tag = st.chiral_tags[a];
            if tag != ChiralTag::Unspecified
                && tag != ChiralTag::Other
                && st.cip_codes[a].is_none()
                && (!special[a] || !has_rsa[a])
            {
                st.chiral_tags[a] = ChiralTag::Unspecified;
            }
        }
    }
    st.cip_ranks = ranks;
    st
}

/// [`legacy_stereo_perception`] computed once per molecule and shared through
/// the calculation context (StereoCenters, SPS).
pub fn legacy_stereo_perception_cached<'a>(
    mol: &Molecule,
    cctx: &'a mut crate::desc::context::CalculationContext,
) -> &'a LegacyStereo {
    let key = crate::desc::context::molecule_cache_key(mol);
    if cctx.legacy_stereo_key != Some(key) || cctx.legacy_stereo.is_none() {
        cctx.legacy_stereo = Some(legacy_stereo_perception(mol));
        cctx.legacy_stereo_key = Some(key);
    }
    cctx.legacy_stereo.as_ref().expect("legacy stereo cached")
}

/// [`atom_stereo_center_counts`] reusing the perception cached in `cctx`.
pub fn atom_stereo_center_counts_with_context(
    mol: &Molecule,
    cctx: &mut crate::desc::context::CalculationContext,
) -> (u32, u32) {
    stereo_center_counts(legacy_stereo_perception_cached(mol, cctx))
}

fn stereo_center_counts(st: &LegacyStereo) -> (u32, u32) {
    let total = st.chirality_possible.iter().filter(|&&p| p).count() as u32;
    let unspecified = st
        .chirality_possible
        .iter()
        .zip(&st.chiral_tags)
        .filter(|&(&p, &t)| p && t == ChiralTag::Unspecified)
        .count() as u32;
    (total, unspecified)
}

/// `(NumAtomStereoCenters, NumUnspecifiedAtomStereoCenters)` from a single
/// perception pass.
pub fn atom_stereo_center_counts(mol: &Molecule) -> (u32, u32) {
    stereo_center_counts(&legacy_stereo_perception(mol))
}

/// RDKit `Descriptors::numAtomStereoCenters` for a molecule from SMILES.
pub fn num_atom_stereo_centers(mol: &Molecule) -> u32 {
    atom_stereo_center_counts(mol).0
}

/// RDKit `Descriptors::numUnspecifiedAtomStereoCenters` for a molecule from SMILES.
pub fn num_unspecified_atom_stereo_centers(mol: &Molecule) -> u32 {
    atom_stereo_center_counts(mol).1
}

// ---------------------------------------------------------------------------
// FindPotentialStereoBonds (legacy, cleanIt = false)
// ---------------------------------------------------------------------------

/// RDKit `MolOps::findPotentialStereoBonds(mol, cleanIt=false)` applied to the
/// legacy perception result: non-ring double bonds that could be cis/trans
/// and carry no stereo yet become [`BondStereo::Any`].
pub fn find_potential_stereo_bonds(mol: &Molecule, st: &mut LegacyStereo) {
    let ctx = Ctx::new(mol);
    find_potential_stereo_bonds_impl(&ctx, st);
}

fn find_potential_stereo_bonds_impl(ctx: &Ctx, st: &mut LegacyStereo) {
    for b in 0..ctx.mol.num_bonds() {
        if ctx.btype(b) != BondType::Double || ctx.num_bond_rings(b) > 0 {
            continue;
        }
        let stereo = st.bond_stereo[b];
        if !(stereo == BondStereo::None || (stereo == BondStereo::Any && st.stereo_atoms[b].is_none())) {
            continue;
        }
        st.bond_stereo[b] = BondStereo::None;
        let (beg, end) = (ctx.bond_begin(b), ctx.bond_end(b));
        if !matches!(ctx.degree(beg), 2 | 3) || !matches!(ctx.degree(end), 2 | 3) {
            continue;
        }
        if st.cip_ranks.is_empty() {
            assign_atom_cip_ranks(ctx, &mut st.cip_ranks);
        }
        let ranks = &st.cip_ranks;
        let helper = |a: usize| -> SmallVec<[usize; 2]> {
            ctx.nbrs[a]
                .iter()
                .filter(|&&(_, nb)| nb != b && matches!(ctx.btype(nb), BondType::Single | BondType::Aromatic))
                .map(|&(o, _)| o)
                .collect()
        };
        let bn = helper(beg);
        let en = helper(end);
        if bn.is_empty() || en.is_empty() {
            continue;
        }
        let higher = |v: &SmallVec<[usize; 2]>| if ranks[v[0]] > ranks[v[1]] { v[0] } else { v[1] };
        let atoms = match (bn.len(), en.len()) {
            (2, 2) => {
                if ranks[bn[0]] != ranks[bn[1]] && ranks[en[0]] != ranks[en[1]] {
                    Some((higher(&bn), higher(&en)))
                } else {
                    None
                }
            }
            (2, _) => (ranks[bn[0]] != ranks[bn[1]]).then(|| (higher(&bn), en[0])),
            (_, 2) => (ranks[en[0]] != ranks[en[1]]).then(|| (bn[0], higher(&en))),
            _ => Some((bn[0], en[0])),
        };
        if let Some(sa) = atoms {
            if st.stereo_atoms[b].is_none() {
                st.stereo_atoms[b] = Some(sa);
            }
            st.bond_stereo[b] = BondStereo::Any;
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical ranking (new_canon: rankFragmentAtoms without tie breaking)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BondHolder {
    sym: usize, // index into bond symbols
    btype: u8,  // RDKit Bond::BondType value
    nbr_idx: usize,
    nbr_class: i64,
}

fn rdkit_bond_type_code(t: BondType) -> u8 {
    match t {
        BondType::Single => 1,
        BondType::Double => 2,
        BondType::Triple => 3,
        BondType::Aromatic => 12,
        BondType::Dative => 17,
        BondType::Unknown => 0,
    }
}

struct Canon {
    /// Dense ranks of the atom / bond symbols (same order as the strings).
    atom_syms: Vec<u32>,
    bond_syms: Vec<u32>,
    index: Vec<i64>,
    degree: Vec<usize>,
    nbr_ids: Vec<SmallVec<[usize; 4]>>,
    bonds: Vec<Vec<BondHolder>>,
    neighbor_num: Vec<Vec<i32>>,
    revisited: Vec<Vec<i32>>,
    /// Bumped whenever `index` changes; an atom's bond list is re-sorted only
    /// when its `sorted_at` epoch is stale (sorting is deterministic, so the
    /// result is the same as re-sorting on every comparison).
    epoch: u64,
    sorted_at: Vec<u64>,
}

impl Canon {
    fn bh_compare(&self, x: &BondHolder, y: &BondHolder) -> std::cmp::Ordering {
        self.bond_syms[x.sym]
            .cmp(&self.bond_syms[y.sym])
            .then(x.btype.cmp(&y.btype))
            .then(x.nbr_class.cmp(&y.nbr_class))
    }

    fn update_nbr_index(&mut self, i: usize) {
        if self.sorted_at[i] == self.epoch {
            return;
        }
        self.sorted_at[i] = self.epoch;
        let mut bonds = std::mem::take(&mut self.bonds[i]);
        for b in bonds.iter_mut() {
            b.nbr_class = self.index[b.nbr_idx];
        }
        bonds.sort_by(|x, y| self.bh_compare(y, x));
        self.bonds[i] = bonds;
    }

    fn compare_bonds(&mut self, i: usize, j: usize) -> i32 {
        self.update_nbr_index(i);
        self.update_nbr_index(j);
        let (bi, bj) = (&self.bonds[i], &self.bonds[j]);
        for (x, y) in bi.iter().zip(bj.iter()) {
            match self.bh_compare(x, y) {
                std::cmp::Ordering::Less => return -1,
                std::cmp::Ordering::Greater => return 1,
                std::cmp::Ordering::Equal => {}
            }
        }
        ord_to_i32(bi.len().cmp(&bj.len()))
    }

    /// `AtomCompareFunctor` (with symbols, `df_useNbrs = true`).
    fn atom_compare(&mut self, i: usize, j: usize) -> i32 {
        let c = self.index[i].cmp(&self.index[j]).then(self.degree[i].cmp(&self.degree[j]));
        if c != std::cmp::Ordering::Equal {
            return ord_to_i32(c);
        }
        let c = self.atom_syms[i].cmp(&self.atom_syms[j]);
        if c != std::cmp::Ordering::Equal {
            return ord_to_i32(c);
        }
        self.compare_bonds(i, j)
    }

    /// `SpecialSymmetryAtomCompareFunctor`.
    fn special_compare(&mut self, i: usize, j: usize) -> i32 {
        let c = self.neighbor_num[i].cmp(&self.neighbor_num[j]).then(self.revisited[i].cmp(&self.revisited[j]));
        if c != std::cmp::Ordering::Equal {
            return ord_to_i32(c);
        }
        self.compare_bonds(i, j)
    }

    fn compare(&mut self, special: bool, i: usize, j: usize) -> i32 {
        if special { self.special_compare(i, j) } else { self.atom_compare(i, j) }
    }
}

fn ord_to_i32(o: std::cmp::Ordering) -> i32 {
    match o {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Port of RDKit's `hanoi` merge sort. `buf[base..base+nel]` is sorted, with
/// `buf[temp..temp+nel]` as scratch. Returns true when the result is in temp.
#[allow(clippy::too_many_arguments)]
fn hanoi(
    buf: &mut [usize],
    base: usize,
    nel: usize,
    temp: usize,
    count: &mut [i64],
    changed: &[i32],
    canon: &mut Canon,
    special: bool,
) -> bool {
    if nel == 1 {
        count[buf[base]] = 1;
        return false;
    } else if nel == 2 {
        let (n1, n2) = (buf[base], buf[base + 1]);
        let stat = if changed[n1] != 0 || changed[n2] != 0 { canon.compare(special, n1, n2) } else { 0 };
        if stat == 0 {
            count[n1] = 2;
            count[n2] = 0;
        } else if stat < 0 {
            count[n1] = 1;
            count[n2] = 1;
        } else {
            count[n1] = 1;
            count[n2] = 1;
            buf[base] = n2;
            buf[base + 1] = n1;
        }
        return false;
    }
    let mut n1 = nel / 2;
    let mut n2 = nel - n1;
    let (b1, t1, b2, t2) = (base, temp, base + n1, temp + n1);
    let result;
    let mut ptr;
    let mut s1;
    let mut s2;
    if hanoi(buf, b1, n1, t1, count, changed, canon, special) {
        s2 = if hanoi(buf, b2, n2, t2, count, changed, canon, special) { t2 } else { b2 };
        result = false;
        ptr = base;
        s1 = t1;
    } else {
        s2 = if hanoi(buf, b2, n2, t2, count, changed, canon, special) { t2 } else { b2 };
        result = true;
        ptr = temp;
        s1 = b1;
    }
    loop {
        let (a1, a2) = (buf[s1], buf[s2]);
        let stat = if changed[a1] != 0 || changed[a2] != 0 { canon.compare(special, a1, a2) } else { 0 };
        let len1 = count[a1] as usize;
        let len2 = count[a2] as usize;
        if stat == 0 {
            count[a1] = (len1 + len2) as i64;
            count[a2] = 0;
            buf.copy_within(s1..s1 + len1, ptr);
            ptr += len1;
            n1 -= len1;
            if n1 == 0 {
                if ptr != s2 {
                    buf.copy_within(s2..s2 + n2, ptr);
                }
                return result;
            }
            s1 += len1;
            buf.copy_within(s2..s2 + len2, ptr);
            ptr += len2;
            n2 -= len2;
            if n2 == 0 {
                buf.copy_within(s1..s1 + n1, ptr);
                return result;
            }
            s2 += len2;
        } else if stat < 0 {
            buf.copy_within(s1..s1 + len1, ptr);
            ptr += len1;
            n1 -= len1;
            if n1 == 0 {
                if ptr != s2 {
                    buf.copy_within(s2..s2 + n2, ptr);
                }
                return result;
            }
            s1 += len1;
        } else {
            buf.copy_within(s2..s2 + len2, ptr);
            ptr += len2;
            n2 -= len2;
            if n2 == 0 {
                buf.copy_within(s1..s1 + n1, ptr);
                return result;
            }
            s2 += len2;
        }
    }
}

fn hanoisort(
    order: &mut [usize],
    start: usize,
    len: usize,
    count: &mut [i64],
    changed: &[i32],
    canon: &mut Canon,
    special: bool,
) {
    let mut buf = vec![0usize; 2 * len];
    buf[..len].copy_from_slice(&order[start..start + len]);
    if hanoi(&mut buf, 0, len, len, count, changed, canon, special) {
        order[start..start + len].copy_from_slice(&buf[len..2 * len]);
    } else {
        order[start..start + len].copy_from_slice(&buf[..len]);
    }
}

fn activate_partitions(order: &[usize], count: &[i64], next: &mut [i64], changed: &mut [i32]) -> i64 {
    let n = order.len();
    let mut activeset: i64 = -1;
    next.iter_mut().for_each(|x| *x = -2);
    let mut i = 0;
    while i < n {
        let j = order[i];
        if count[j] > 1 {
            next[j] = activeset;
            activeset = j as i64;
            i += count[j] as usize;
        } else {
            i += 1;
        }
    }
    changed.iter_mut().for_each(|x| *x = 1);
    activeset
}

#[allow(clippy::too_many_arguments)]
fn refine_partitions(
    canon: &mut Canon,
    special: bool,
    order: &mut [usize],
    count: &mut [i64],
    mut activeset: i64,
    next: &mut [i64],
    changed: &mut [i32],
    touched: &mut [bool],
) {
    let n = order.len();
    let mut symclass: i64 = 0;
    while activeset != -1 {
        let partition = activeset as usize;
        activeset = next[partition];
        next[partition] = -2;
        let len = count[partition] as usize;
        let offset = canon.index[partition] as usize;
        hanoisort(order, offset, len, count, changed, canon, special);
        for k in 0..len {
            changed[order[offset + k]] = 0;
        }
        let first = count[order[offset]] as usize;
        for i in first..len {
            let index = order[offset + i];
            if count[index] != 0 {
                symclass = (offset + i) as i64;
            }
            canon.index[index] = symclass;
            canon.epoch += 1;
            for &nb in &canon.nbr_ids[index] {
                changed[nb] = 1;
            }
        }
        let first = count[order[offset]] as usize;
        for i in first..len {
            let index = order[offset + i];
            for &nb in &canon.nbr_ids[index] {
                touched[canon.index[nb] as usize] = true;
            }
        }
        for ii in 0..n {
            if touched[ii] {
                let p = order[ii];
                if count[p] > 1 && next[p] == -2 {
                    next[p] = activeset;
                    activeset = p as i64;
                }
                touched[ii] = false;
            }
        }
    }
}

/// `compareRingAtomsConcerningNumNeighbors`.
fn compare_ring_atoms_concerning_num_neighbors(ctx: &Ctx, canon: &mut Canon) {
    let n = ctx.n();
    let in_ring = |a: usize| ctx.num_atom_rings(a) > 0;
    for idx in 0..n {
        if !in_ring(idx) {
            continue;
        }
        let mut neighbors: VecDeque<usize> = VecDeque::from([idx]);
        let mut visited = vec![false; n];
        let mut last_level = vec![false; n];
        let mut current_level = vec![false; n];
        let mut revisited = vec![0i32; n];
        let mut out_num: Vec<i32> = Vec::new();
        let mut out_rev: Vec<i32> = Vec::new();
        while !neighbors.is_empty() {
            let mut num_level_nbrs = 0i32;
            let mut next_level: Vec<usize> = Vec::new();
            while let Some(nidx) = neighbors.pop_front() {
                if !in_ring(nidx) {
                    continue;
                }
                last_level[nidx] = true;
                visited[nidx] = true;
                for &j in &canon.nbr_ids[nidx] {
                    if !visited[j] {
                        current_level[j] = true;
                        num_level_nbrs += 1;
                        visited[j] = true;
                        next_level.push(j);
                    }
                }
            }
            for i in 0..n {
                if current_level[i] {
                    for &k in &canon.nbr_ids[i] {
                        if current_level[k] || last_level[k] {
                            revisited[k] += 1;
                        }
                    }
                }
            }
            for i in 0..n {
                last_level[i] = current_level[i];
                current_level[i] = false;
            }
            let mut tmp: Vec<i32> = revisited.iter().copied().filter(|&r| r > 0).collect();
            tmp.sort_unstable();
            tmp.push(-1);
            out_rev.extend(tmp);
            revisited.iter_mut().for_each(|r| *r = 0);
            out_num.push(num_level_nbrs);
            out_num.push(-1);
            neighbors.extend(next_level);
        }
        canon.neighbor_num[idx] = out_num;
        canon.revisited[idx] = out_rev;
    }
}

/// `Canon::rankFragmentAtoms` (all atoms/bonds in play, no tie breaking, no
/// chirality/isotopes/atom maps) with atom and bond symbols.
fn rank_fragment_atoms(ctx: &Ctx, atom_syms: &[String], bond_syms: &[String]) -> Vec<u32> {
    let n = ctx.n();
    if n == 0 {
        return Vec::new();
    }
    // symbols only ever take part in comparisons: compare their dense ranks
    let sym_ranks = |syms: &[String]| -> Vec<u32> {
        let mut ranks = vec![0u32; syms.len()];
        if syms.iter().all(|s| s.len() <= 16) {
            // big-endian, zero padded: integer order = byte-string order (the
            // symbols contain no NUL bytes, so a prefix still sorts first)
            let mut keys: Vec<(u128, u32)> = syms
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let mut b = [0u8; 16];
                    b[..s.len()].copy_from_slice(s.as_bytes());
                    (u128::from_be_bytes(b), i as u32)
                })
                .collect();
            keys.sort_unstable();
            let mut rank = 0u32;
            for k in 0..keys.len() {
                if k > 0 && keys[k].0 != keys[k - 1].0 {
                    rank += 1;
                }
                ranks[keys[k].1 as usize] = rank;
            }
            return ranks;
        }
        let mut idx: Vec<usize> = (0..syms.len()).collect();
        idx.sort_unstable_by(|&a, &b| syms[a].cmp(&syms[b]));
        let mut rank = 0u32;
        for k in 0..idx.len() {
            if k > 0 && syms[idx[k]] != syms[idx[k - 1]] {
                rank += 1;
            }
            ranks[idx[k]] = rank;
        }
        ranks
    };
    let mut canon = Canon {
        atom_syms: sym_ranks(atom_syms),
        bond_syms: sym_ranks(bond_syms),
        index: vec![0; n],
        degree: (0..n).map(|a| ctx.degree(a)).collect(),
        nbr_ids: vec![SmallVec::new(); n],
        bonds: vec![Vec::new(); n],
        neighbor_num: vec![Vec::new(); n],
        revisited: vec![Vec::new(); n],
        epoch: 1,
        sorted_at: vec![0; n],
    };
    for b in 0..ctx.mol.num_bonds() {
        let (beg, end) = (ctx.bond_begin(b), ctx.bond_end(b));
        let bt = rdkit_bond_type_code(ctx.btype(b));
        canon.nbr_ids[beg].push(end);
        canon.nbr_ids[end].push(beg);
        canon.bonds[beg].push(BondHolder { sym: b, btype: bt, nbr_idx: end, nbr_class: 0 });
        canon.bonds[end].push(BondHolder { sym: b, btype: bt, nbr_idx: beg, nbr_class: 0 });
    }
    for a in 0..n {
        let mut bonds = std::mem::take(&mut canon.bonds[a]);
        bonds.sort_by(|x, y| canon.bh_compare(y, x));
        canon.bonds[a] = bonds;
    }
    // CreateSinglePartition
    let mut order: Vec<usize> = (0..n).collect();
    let mut count = vec![0i64; n];
    count[0] = n as i64;
    let mut next = vec![-2i64; n];
    let mut changed = vec![1i32; n];
    let mut touched = vec![false; n];
    let activeset = activate_partitions(&order, &count, &mut next, &mut changed);
    refine_partitions(&mut canon, false, &mut order, &mut count, activeset, &mut next, &mut changed, &mut touched);

    let ties = count.contains(&0);
    let mut sym_ring_atoms = 0i64;
    let mut ring_atoms = 0i64;
    let mut branching = false;
    for i in 0..n {
        let a = order[i];
        if ctx.num_atom_rings(a) > 0 {
            if count[a] > 2 {
                sym_ring_atoms += count[a];
            }
            ring_atoms += 1;
            if ctx.num_atom_rings(a) > 1 && count[a] > 1 {
                branching = true;
            }
        }
    }
    if ties && ring_atoms > 0 && (sym_ring_atoms as f32) / (ring_atoms as f32) > 0.5 && branching {
        compare_ring_atoms_concerning_num_neighbors(ctx, &mut canon);
        let activeset = activate_partitions(&order, &count, &mut next, &mut changed);
        refine_partitions(&mut canon, true, &mut order, &mut count, activeset, &mut next, &mut changed, &mut touched);
    }
    canon.index.iter().map(|&i| i as u32).collect()
}

// ---------------------------------------------------------------------------
// New-style FindPotentialStereo (FindStereo.cpp)
// ---------------------------------------------------------------------------

const NOATOM: usize = usize::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Specified {
    Unspecified,
    Specified,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Descriptor {
    None,
    TetCW,
    TetCCW,
    Cis,
    Trans,
}

struct AtomInfo {
    tetrahedral: bool,
    specified: Specified,
    descriptor: Descriptor,
    controlling: SmallVec<[usize; 4]>,
}

struct BondInfo {
    specified: Specified,
    descriptor: Descriptor,
    controlling: [usize; 4],
}

struct NewStereo<'a, 'b> {
    ctx: &'b Ctx<'a>,
    tags: &'b [ChiralTag],
    bond_stereo: &'b [BondStereo],
    stereo_atoms: &'b [Option<(usize, usize)>],
}

impl NewStereo<'_, '_> {
    fn compare_symbol(&self, a: usize) -> String {
        let atom = &self.ctx.mol.atoms[a];
        let sym = ELEMENT_SYMBOLS.get(atom.atomic_number as usize).copied().unwrap_or("*");
        format!("{}{}{}", atom.isotope, sym, atom.formal_charge)
    }

    fn bond_symbol(&self, b: usize) -> String {
        let bond = &self.ctx.mol.bonds[b];
        if bond.is_aromatic() {
            return ":".to_string();
        }
        match bond.bond_type {
            BondType::Single => "-",
            BondType::Double => "=",
            BondType::Triple => "#",
            BondType::Aromatic => ":",
            _ => "?",
        }
        .to_string()
    }

    fn is_potential_nontetrahedral(&self, a: usize) -> bool {
        let tnz = self.ctx.nonzero_degree(a) + self.ctx.total_h[a];
        let z = self.ctx.anum(a);
        if !(2..=6).contains(&tnz) || (z < 12 && z != 4) {
            return false;
        }
        match self.tags[a] {
            ChiralTag::Other => true,
            ChiralTag::Unspecified => tnz >= 4,
            _ => false,
        }
    }

    fn is_potential_tetrahedral(&self, a: usize) -> bool {
        let ctx = self.ctx;
        let nz = ctx.nonzero_degree(a);
        let h = ctx.total_h[a];
        let z = ctx.anum(a);
        if nz + h > 4 {
            return false;
        }
        if nz == 4 {
            return true;
        }
        if nz <= 1 {
            return false;
        }
        if nz < 3 && z != 15 && z != 33 {
            return false;
        }
        if z == 15 || z == 33 {
            return true;
        }
        if nz == 3 {
            if h == 1 {
                return !ctx.has_protium_neighbor(a);
            }
            if (z == 16 || z == 34) && {
                let ev = ctx.explicit_valence(a);
                ev == 4 || (ev == 3 && ctx.mol.atoms[a].formal_charge == 1)
            } {
                return true;
            }
            if z == 7 {
                return ctx.mol.atoms[a].hybridization == Hybridization::SP3
                    && !ctx.has_conjugated_bond(a)
                    && (ctx.is_atom_in_ring_of_size(a, 3) || ctx.is_bridgehead(a));
            }
            return false;
        }
        false
    }

    fn atom_info(&self, a: usize) -> AtomInfo {
        let orig: SmallVec<[usize; 4]> = self.ctx.nbrs[a].iter().map(|&(o, _)| o).collect();
        let mut controlling = orig.clone();
        controlling.sort_unstable();
        let mut info = AtomInfo {
            tetrahedral: true,
            specified: Specified::Unspecified,
            descriptor: Descriptor::None,
            controlling,
        };
        let tag = self.tags[a];
        if tag.is_tetrahedral() {
            info.specified = Specified::Specified;
            let tag = if permutation_is_odd(&orig, &info.controlling) { tag.inverted() } else { tag };
            info.descriptor = if tag == ChiralTag::TetrahedralCCW { Descriptor::TetCCW } else { Descriptor::TetCW };
        } else if self.is_potential_nontetrahedral(a) {
            if tag == ChiralTag::Unspecified {
                info.tetrahedral = !matches!(self.ctx.total_degree(a), 5 | 6);
            } else {
                // @SP / @TB / @OH (assumed specified)
                info.tetrahedral = false;
                info.specified = Specified::Specified;
            }
        }
        info
    }

    fn is_potential_stereo_bond(&self, b: usize) -> bool {
        let ctx = self.ctx;
        if ctx.btype(b) != BondType::Double {
            return false;
        }
        let (beg, end) = (ctx.bond_begin(b), ctx.bond_end(b));
        let (bd, ed) = (ctx.total_degree(beg), ctx.total_degree(end));
        if bd > 1 && bd < 4 && ed > 1 && ed < 4 && ctx.total_h_with_nbrs(beg) < 2 && ctx.total_h_with_nbrs(end) < 2 {
            !ctx.ri.bond_rings.iter().any(|r| r.len() < 8 && r.iter().any(|&x| x as usize == b))
        } else {
            false
        }
    }

    fn bond_info(&self, b: usize) -> BondInfo {
        let ctx = self.ctx;
        let (beg, end) = (ctx.bond_begin(b), ctx.bond_end(b));
        let mut c: SmallVec<[usize; 4]> = SmallVec::new();
        for a in [beg, end] {
            for &(o, nb) in &ctx.nbrs[a] {
                if nb != b {
                    c.push(o);
                }
            }
            for _ in ctx.degree(a)..3 {
                c.push(NOATOM);
            }
        }
        let controlling = [c[0], c[1], c[2], c[3]];
        let mut info = BondInfo { specified: Specified::Unspecified, descriptor: Descriptor::None, controlling };
        match self.bond_stereo[b] {
            BondStereo::Any => info.specified = Specified::Unknown,
            BondStereo::E | BondStereo::Z => {
                info.specified = Specified::Specified;
                let mut cis = self.bond_stereo[b] == BondStereo::Z;
                if let Some((s0, s1)) = self.stereo_atoms[b] {
                    let first_at_begin = s0 == controlling[0];
                    let first_at_end = s1 == controlling[2];
                    if first_at_begin ^ first_at_end {
                        cis = !cis;
                    }
                }
                info.descriptor = if cis { Descriptor::Cis } else { Descriptor::Trans };
            }
            BondStereo::None => {}
        }
        info
    }

    #[allow(clippy::too_many_arguments)]
    fn flag_ring_stereo(
        &self,
        prsa: &mut [u32],
        prsb: &mut [u32],
        known_atoms: &[bool],
        possible_atoms: &[bool],
        known_bonds: &[bool],
        possible_bonds: &[bool],
    ) {
        let ctx = self.ctx;
        let mut in_ring = vec![false; ctx.n()];
        for (ridx, aring) in ctx.ri.atom_rings.iter().enumerate() {
            let bring = &ctx.ri.bond_rings[ridx];
            let sz = aring.len();
            let half = sz / 2 + sz % 2;
            let mut n_here = 0u32;
            in_ring.iter_mut().for_each(|x| *x = false);
            for ai in 0..sz {
                let aidx = aring[ai] as usize;
                if !known_atoms[aidx] && !possible_atoms[aidx] {
                    continue;
                }
                for div in [2usize, 3] {
                    if sz % div != 0 {
                        continue;
                    }
                    let inc = sz / div;
                    let mut by_bond = 0usize;
                    let mut by_atom = 0usize;
                    let mut k = inc;
                    while k < sz {
                        let other = aring[(ai + k) % sz] as usize;
                        for &(_, nb) in &ctx.nbrs[other] {
                            if (known_bonds[nb] || possible_bonds[nb]) && !bring.iter().any(|&x| x as usize == nb) {
                                by_bond += 1;
                                break;
                            }
                        }
                        if by_bond == 0 && (known_atoms[other] || possible_atoms[other]) {
                            by_atom += 1;
                        }
                        k += inc;
                    }
                    if by_bond == div - 1 || by_atom == div - 1 {
                        n_here += 1 + by_bond as u32;
                        let mut k = 0;
                        while k < sz {
                            in_ring[aring[(ai + k) % sz] as usize] = true;
                            k += inc;
                        }
                    }
                }
                if ctx.num_atom_rings(aidx) > 1 {
                    let mut prev = aidx;
                    for step in 1..=half {
                        let other = aring[(ai + step) % sz] as usize;
                        let Some(bnd) = ctx.bond_between(prev, other) else { break };
                        if ctx.num_bond_rings(bnd) < 2 {
                            break;
                        }
                        if known_atoms[other] || possible_atoms[other] {
                            n_here += 2;
                            in_ring[aidx] = true;
                            in_ring[other] = true;
                            break;
                        }
                        prev = other;
                    }
                }
            }
            if n_here > 1 {
                for &a in aring {
                    if in_ring[a as usize] {
                        prsa[a as usize] += 1;
                    }
                }
                for &b in bring {
                    prsb[b as usize] += 1;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn controlling_atoms_dupes(
        &self,
        b: usize,
        c1: usize,
        c2: usize,
        ranks: &[u32],
        possible_atoms: &[bool],
        known_atoms: &[bool],
        possible_bonds: &[bool],
        known_bonds: &[bool],
    ) -> bool {
        let ctx = self.ctx;
        if ranks[c1] != ranks[c2] {
            return false;
        }
        let m1 = ctx.atom_members(c1);
        let m2 = ctx.atom_members(c2);
        for r in m1.iter().filter(|r| m2.contains(r)) {
            let ring = &ctx.ri.atom_rings[*r];
            if ring.len() % 2 == 1 {
                continue;
            }
            for bond_end in [ctx.bond_begin(b), ctx.bond_end(b)] {
                if let Some(pos) = ring.iter().position(|&x| x as usize == bond_end) {
                    let opp = ring[(pos + ring.len() / 2) % ring.len()] as usize;
                    if possible_atoms[opp] || known_atoms[opp] {
                        return false;
                    }
                    if ctx.degree(opp) == 3 {
                        for &(o, nb) in &ctx.nbrs[opp] {
                            if !ring.iter().any(|&x| x as usize == o) && (possible_bonds[nb] || known_bonds[nb]) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }
}

/// Atoms reported as tetrahedral stereo centres by RDKit's new-style
/// `FindPotentialStereo(mol, cleanIt=false, flagPossible=true)` - i.e.
/// `Chem.FindMolChiralCenters(mol, includeUnassigned=True,
/// useLegacyImplementation=False)` - evaluated on a molecule whose chiral
/// tags and bond stereo are given by `st`.
pub fn find_potential_tetrahedral_centers(mol: &Molecule, st: &LegacyStereo) -> Vec<usize> {
    let ctx = Ctx::new(mol);
    potential_tetrahedral_impl(&ctx, st)
}

fn potential_tetrahedral_impl(ctx: &Ctx, st: &LegacyStereo) -> Vec<usize> {
    let ns = NewStereo { ctx, tags: &st.chiral_tags, bond_stereo: &st.bond_stereo, stereo_atoms: &st.stereo_atoms };
    let n = ctx.n();
    let nb = ctx.mol.num_bonds();
    let mut known_atoms = vec![false; n];
    let mut possible_atoms = vec![false; n];
    let mut atom_syms: Vec<String> = Vec::with_capacity(n);
    for a in 0..n {
        let mut sym = ns.compare_symbol(a);
        if ns.is_potential_tetrahedral(a) || ns.is_potential_nontetrahedral(a) {
            let info = ns.atom_info(a);
            match info.specified {
                Specified::Unknown => {
                    known_atoms[a] = true;
                    sym += &a.to_string();
                }
                Specified::Specified => {
                    known_atoms[a] = true;
                    sym += match info.descriptor {
                        Descriptor::TetCCW => "_CCW",
                        Descriptor::TetCW => "_CW",
                        _ => "_STEREO",
                    };
                }
                Specified::Unspecified => {
                    possible_atoms[a] = true;
                    sym += &format!("_{a}");
                }
            }
        }
        atom_syms.push(sym);
    }
    let mut known_bonds = vec![false; nb];
    let mut possible_bonds = vec![false; nb];
    let mut bond_syms: Vec<String> = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut sym = ns.bond_symbol(b);
        if ns.is_potential_stereo_bond(b) {
            let info = ns.bond_info(b);
            match info.specified {
                Specified::Unknown => {
                    known_bonds[b] = true;
                    sym += &format!("_{b}");
                }
                Specified::Specified => {
                    known_bonds[b] = true;
                    sym += match info.descriptor {
                        Descriptor::Cis => "_cis",
                        Descriptor::Trans => "_trans",
                        _ => "_STEREO",
                    };
                }
                Specified::Unspecified => {
                    possible_bonds[b] = true;
                    sym += &format!("_{b}");
                }
            }
        }
        bond_syms.push(sym);
    }
    if !known_atoms.iter().chain(&possible_atoms).chain(&known_bonds).chain(&possible_bonds).any(|&x| x) {
        // no stereo candidates: the rounds below would rank the atoms once and
        // report nothing
        return Vec::new();
    }
    let orig_possible_atoms = possible_atoms.clone();
    let orig_possible_bonds = possible_bonds.clone();
    let mut prsa = vec![0u32; n];
    let mut prsb = vec![0u32; nb];
    ns.flag_ring_stereo(&mut prsa, &mut prsb, &known_atoms, &possible_atoms, &known_bonds, &possible_bonds);

    let mut fixed_atoms = vec![false; n];
    let mut fixed_bonds = vec![false; nb];
    let mut res: Vec<(usize, bool)> = Vec::new();

    let run_rounds = |atom_syms: &mut Vec<String>,
                          bond_syms: &mut Vec<String>,
                          possible_atoms: &mut Vec<bool>,
                          known_atoms: &Vec<bool>,
                          possible_bonds: &mut Vec<bool>,
                          known_bonds: &Vec<bool>,
                          fixed_atoms: &mut Vec<bool>,
                          fixed_bonds: &mut Vec<bool>,
                          prsa: &mut Vec<u32>,
                          prsb: &mut Vec<u32>,
                          res: &mut Vec<(usize, bool)>| {
        let mut need_another = true;
        while need_another {
            res.clear();
            let ranks = rank_fragment_atoms(ctx, atom_syms, bond_syms);
            need_another = update_atoms(&ns, &ranks, atom_syms, possible_atoms, known_atoms, fixed_atoms, prsa, prsb, res);
            need_another |= update_bonds(
                &ns,
                &ranks,
                bond_syms,
                possible_atoms,
                possible_bonds,
                known_atoms,
                known_bonds,
                fixed_bonds,
            );
        }
    };
    run_rounds(
        &mut atom_syms,
        &mut bond_syms,
        &mut possible_atoms,
        &known_atoms,
        &mut possible_bonds,
        &known_bonds,
        &mut fixed_atoms,
        &mut fixed_bonds,
        &mut prsa,
        &mut prsb,
        &mut res,
    );
    if possible_atoms != orig_possible_atoms || possible_bonds != orig_possible_bonds {
        possible_atoms = orig_possible_atoms;
        for i in 0..n {
            if !fixed_atoms[i] && known_atoms[i] {
                possible_atoms[i] = true;
                known_atoms[i] = false;
            }
            if possible_atoms[i] {
                atom_syms[i] += &format!("_{i}");
            }
        }
        possible_bonds = orig_possible_bonds;
        for i in 0..nb {
            if !fixed_bonds[i] && known_bonds[i] {
                possible_bonds[i] = true;
                known_bonds[i] = false;
            }
            if possible_bonds[i] {
                bond_syms[i] += &format!("_{i}");
            }
        }
        ns.flag_ring_stereo(&mut prsa, &mut prsb, &known_atoms, &possible_atoms, &known_bonds, &possible_bonds);
        run_rounds(
            &mut atom_syms,
            &mut bond_syms,
            &mut possible_atoms,
            &known_atoms,
            &mut possible_bonds,
            &known_bonds,
            &mut fixed_atoms,
            &mut fixed_bonds,
            &mut prsa,
            &mut prsb,
            &mut res,
        );
    }
    res.into_iter().filter(|&(_, tet)| tet).map(|(a, _)| a).collect()
}

#[allow(clippy::too_many_arguments)]
fn update_atoms(
    ns: &NewStereo,
    ranks: &[u32],
    atom_syms: &mut [String],
    possible: &mut [bool],
    known: &[bool],
    fixed: &mut [bool],
    prsa: &mut [u32],
    prsb: &mut [u32],
    res: &mut Vec<(usize, bool)>,
) -> bool {
    let ctx = ns.ctx;
    let mut need_another = false;
    for a in 0..ctx.n() {
        if !(known[a] || possible[a]) {
            continue;
        }
        let info = ns.atom_info(a);
        if fixed[a] {
            res.push((a, info.tetrahedral));
            continue;
        }
        let mut nbrs: SmallVec<[usize; 4]> = SmallVec::new();
        let mut dupe = false;
        if info.tetrahedral {
            for &nbr in &info.controlling {
                let rnk = ranks[nbr] as usize;
                if nbrs.contains(&rnk) {
                    if prsa[a] > 0 {
                        match ctx.bond_between(a, nbr) {
                            Some(bnd) if prsb[bnd] > 0 => {}
                            _ => {
                                dupe = true;
                                break;
                            }
                        }
                    } else {
                        dupe = true;
                        break;
                    }
                } else {
                    nbrs.push(rnk);
                }
            }
        }
        if !dupe {
            let mut acs = atom_syms[a].clone();
            if !possible[a] {
                if info.tetrahedral {
                    let mut sorted = nbrs.clone();
                    sorted.sort_unstable();
                    let mut desc = info.descriptor;
                    if permutation_is_odd(&nbrs, &sorted) {
                        desc = match desc {
                            Descriptor::TetCCW => Descriptor::TetCW,
                            Descriptor::TetCW => Descriptor::TetCCW,
                            d => d,
                        };
                    }
                    match desc {
                        Descriptor::TetCW => acs = ns.compare_symbol(a) + "_CW",
                        Descriptor::TetCCW => acs = ns.compare_symbol(a) + "_CCW",
                        _ => {}
                    }
                }
                fixed[a] = true;
            }
            if atom_syms[a] != acs {
                atom_syms[a] = acs;
                need_another = true;
            }
            res.push((a, info.tetrahedral));
        } else {
            // only do another round if we change anything here
            need_another |= possible[a];
            possible[a] = false;
            atom_syms[a] = ns.compare_symbol(a);
            if prsa[a] > 0 {
                prsa[a] = 0;
                need_another = true;
                for (ridx, aring) in ctx.ri.atom_rings.iter().enumerate() {
                    let mut n_here = 0;
                    for &ra in aring {
                        // un-fix ring atoms so they are rechecked
                        fixed[ra as usize] = false;
                        n_here += (prsa[ra as usize] > 0) as u32;
                    }
                    if n_here <= 1 {
                        if n_here == 1
                            && let Some(&ra) = aring.iter().find(|&&ra| prsa[ra as usize] > 0) {
                                prsa[ra as usize] -= 1;
                            }
                        for &rb in &ctx.ri.bond_rings[ridx] {
                            let rb = rb as usize;
                            if prsb[rb] > 0 {
                                prsb[rb] -= 1;
                            }
                        }
                    }
                }
            }
        }
    }
    need_another
}

#[allow(clippy::too_many_arguments)]
fn update_bonds(
    ns: &NewStereo,
    ranks: &[u32],
    bond_syms: &mut [String],
    possible_atoms: &[bool],
    possible_bonds: &mut [bool],
    known_atoms: &[bool],
    known_bonds: &[bool],
    fixed_bonds: &mut [bool],
) -> bool {
    let ctx = ns.ctx;
    let mut need_another = false;
    for b in 0..ctx.mol.num_bonds() {
        if !(known_bonds[b] || possible_bonds[b]) {
            continue;
        }
        let mut info = ns.bond_info(b);
        let c = &mut info.controlling;
        if (c[0] == NOATOM && c[1] == NOATOM) || (c[2] == NOATOM && c[3] == NOATOM) {
            fixed_bonds[b] = true;
        }
        if fixed_bonds[b] {
            continue;
        }
        let mut dupe = false;
        let mut needs_swap = false;
        for (i, j) in [(0usize, 1usize), (2, 3)] {
            if c[i] != NOATOM && c[j] != NOATOM {
                if ns.controlling_atoms_dupes(
                    b,
                    c[i],
                    c[j],
                    ranks,
                    possible_atoms,
                    known_atoms,
                    possible_bonds,
                    known_bonds,
                ) {
                    dupe = true;
                } else if ranks[c[i]] < ranks[c[j]] {
                    c.swap(i, j);
                    needs_swap = !needs_swap;
                }
            }
        }
        if !dupe {
            if needs_swap {
                info.descriptor = match info.descriptor {
                    Descriptor::Cis => Descriptor::Trans,
                    Descriptor::Trans => Descriptor::Cis,
                    d => d,
                };
            }
            let mut gbs = bond_syms[b].clone();
            match info.specified {
                Specified::Specified => match info.descriptor {
                    Descriptor::Cis => gbs += "_cis",
                    Descriptor::Trans => gbs += "_trans",
                    _ => {}
                },
                Specified::Unknown => gbs += "_unk",
                Specified::Unspecified => {}
            }
            if bond_syms[b] != gbs {
                bond_syms[b] = gbs;
                need_another = true;
            }
            if !possible_bonds[b] {
                fixed_bonds[b] = true;
            }
        } else if possible_bonds[b] {
            possible_bonds[b] = false;
            bond_syms[b] = ns.bond_symbol(b);
            need_another = true;
        }
    }
    need_another
}

// ---------------------------------------------------------------------------
// SpacialScore (rdkit/Chem/SpacialScore.py)
// ---------------------------------------------------------------------------

/// Absolute spacial score `SPS(mol, normalize=False)`.
pub fn spacial_score_unnormalized(mol: &Molecule) -> u64 {
    let ctx = Ctx::new(mol);
    let st = legacy_impl(&ctx);
    spacial_score_from_legacy(&ctx, st)
}

/// [`spacial_score`] reusing the legacy perception cached in `cctx`.
pub fn spacial_score_with_context(mol: &Molecule, cctx: &mut crate::desc::context::CalculationContext) -> f64 {
    let heavy = mol.atoms.iter().filter(|a| a.atomic_number > 1).count();
    if heavy == 0 {
        return f64::NAN;
    }
    let st = legacy_stereo_perception_cached(mol, cctx).clone();
    spacial_score_from_legacy(&Ctx::new(mol), st) as f64 / heavy as f64
}

fn spacial_score_from_legacy(ctx: &Ctx, mut st: LegacyStereo) -> u64 {
    let mol = ctx.mol;
    find_potential_stereo_bonds_impl(ctx, &mut st);
    let centers = potential_tetrahedral_impl(ctx, &st);
    let n = ctx.n();
    let mut stereo = vec![1u64; n];
    for b in 0..mol.num_bonds() {
        if ctx.btype(b) == BondType::Double && st.bond_stereo[b] != BondStereo::None {
            stereo[ctx.bond_begin(b)] = 2;
            stereo[ctx.bond_end(b)] = 2;
        }
    }
    for a in centers {
        stereo[a] = 2;
    }
    let mut score = 0u64;
    for a in 0..n {
        let atom = &mol.atoms[a];
        let hyb = match atom.hybridization {
            Hybridization::SP => 1,
            Hybridization::SP2 => 2,
            Hybridization::SP3 => 3,
            _ => 4,
        };
        let ring = if atom.is_aromatic() {
            1
        } else if ctx.num_atom_rings(a) > 0 {
            2
        } else {
            1
        };
        let deg = ctx.degree(a) as u64;
        score += hyb * stereo[a] * ring * deg * deg;
    }
    score
}

/// RDKit `Descriptors.SPS` (normalised spacial score, nSPS): the absolute
/// score divided by the number of heavy atoms. Returns NaN for molecules
/// without heavy atoms (RDKit raises `ZeroDivisionError`).
pub fn spacial_score(mol: &Molecule) -> f64 {
    let heavy = mol.atoms.iter().filter(|a| a.atomic_number > 1).count();
    if heavy == 0 {
        return f64::NAN;
    }
    spacial_score_unnormalized(mol) as f64 / heavy as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn counts(smi: &str) -> (u32, u32) {
        let m = parse_smiles(smi).unwrap();
        (num_atom_stereo_centers(&m), num_unspecified_atom_stereo_centers(&m))
    }

    #[test]
    fn stereo_center_counts() {
        assert_eq!(counts("CC(O)CC"), (1, 1));
        assert_eq!(counts("C[C@H](O)CC"), (1, 0));
        assert_eq!(counts("OC1CCC(O)CC1"), (0, 0));
        assert_eq!(counts("O[C@H]1CC[C@@H](O)CC1"), (0, 0));
        assert_eq!(counts("N1(C)CC1C"), (2, 2));
        assert_eq!(counts("CS(=O)CC"), (1, 1));
        assert_eq!(counts("C1CCC2CCCCC2C1"), (0, 0));
        assert_eq!(counts("C1CC(C)C(C)C(C)C1"), (2, 2));
        assert_eq!(counts("CCC"), (0, 0));
        // pseudo-asymmetric centre resolved by R/S of its neighbours
        assert_eq!(counts("C[C@H](O)C(O)[C@@H](C)O"), (3, 1));
        assert_eq!(counts("C[C@H](O)C(O)[C@H](C)O"), (2, 0));
    }

    #[test]
    fn stereo_cip_codes() {
        let m = parse_smiles("[C@H](Cl)(F)Br").unwrap();
        assert_eq!(legacy_stereo_perception(&m).cip_codes[0], Some(CipCode::R));
        let m = parse_smiles("[C@@H](Cl)(F)Br").unwrap();
        assert_eq!(legacy_stereo_perception(&m).cip_codes[0], Some(CipCode::S));
        // RDKit-verified ring-closure / branch quirks
        for (smi, idx, code) in [
            ("F[C@]1(Br)CCOC1", 1, CipCode::S),
            ("F[C@@](Br)1CCOC1", 1, CipCode::S),
            ("[C@@H]1(F)CCOC1", 0, CipCode::R),
            ("C[S@](=O)CC", 1, CipCode::R),
            ("C[S@@+]([O-])CC", 1, CipCode::S),
            ("C[N@]1CC1C", 1, CipCode::S),
        ] {
            let m = parse_smiles(smi).unwrap();
            assert_eq!(legacy_stereo_perception(&m).cip_codes[idx], Some(code), "{smi}");
        }
    }

    #[test]
    fn stereo_double_bonds() {
        let m = parse_smiles("C/C=C/C").unwrap();
        assert_eq!(legacy_stereo_perception(&m).bond_stereo[1], BondStereo::E);
        let m = parse_smiles("C/C=C\\C").unwrap();
        assert_eq!(legacy_stereo_perception(&m).bond_stereo[1], BondStereo::Z);
    }

    #[test]
    fn stereo_spacial_score() {
        let close = |smi: &str, want: f64| {
            let got = spacial_score(&parse_smiles(smi).unwrap());
            assert!((got - want).abs() < 1e-12, "{smi}: {got} != {want}");
        };
        close("CC(O)CC", 15.0);
        close("C[C@H](O)CC", 15.0);
        close("OC1CCC(O)CC1", 39.75);
        close("N1(C)CC1C", 49.2);
        close("CS(=O)CC", 14.8);
        close("C1CCC2CCCCC2C1", 40.8);
        close("CC=CC", 9.5);
        close("C/C=C/C", 9.5);
        close("C1CC(C)C(C)C(C)C1", 45.0);
        assert!(spacial_score(&parse_smiles("[H][H]").unwrap()).is_nan());
    }
}
