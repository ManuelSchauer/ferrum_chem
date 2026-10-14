//! CDK-style molecule perception for the CDK-ported electronic descriptors.
//!
//! FerrumChem perceives molecules RDKit-style. Several CDK descriptors
//! (`BCUTDescriptor`, `AutocorrelationDescriptorPolarizability`/`Charge`,
//! `APol`/`BPol`) depend on CDK's own perception, which is reproduced here
//! locally (the shared perception code stays untouched):
//!
//! - a Kekulé structure (CDK keeps Kekulé bond orders; aromatic bonds of the
//!   RDKit model are re-kekulized),
//! - CDK atom types (`CDKAtomTypeMatcher`, reduced to the main-group elements
//!   C, N, O, S, P, Si, H and the halogens) and the implicit hydrogen counts
//!   `CDKHydrogenAdder` derives from them,
//! - `Aromaticity.cdkLegacy()` (`ElectronDonation.cdk()` electron counts from
//!   the atom types, `Cycles.cdkAromaticSet()`: all cycles of fused systems with
//!   at most three rings, the minimum cycle basis otherwise).
//!
//! Sources: `CDKAtomTypeMatcher`, `CDKHydrogenAdder`, `AtomTypeModel`,
//! `Aromaticity`, `Cycles` (CDK 2.13).

use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// Hybridization of a CDK atom type (only the states relevant for aromaticity).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Hyb {
    Sp1,
    Sp2,
    Sp3,
    Planar3,
    /// sp3d1, sp3d2, s or unset.
    Other,
}

/// The parts of a CDK atom type (`cdk-atom-types.owl`) used here.
#[derive(Debug)]
pub(crate) struct CdkType {
    pub name: &'static str,
    pub hyb: Hyb,
    /// Formal neighbour count (number of `formalBondType` entries).
    pub nb: u8,
    /// Maximum formal bond order (0 when the type lists no bonds).
    pub max_order: u8,
    /// Pi bond count derived from the formal bond types.
    pub pi: u8,
    pub charge: i8,
    /// `ElectronDonation.cdk()` contribution override (`AtomTypeModel` map).
    pub donation: Option<i8>,
}

macro_rules! cdk_types {
    ($($id:ident = ($name:expr, $hyb:ident, $nb:expr, $max:expr, $pi:expr, $q:expr, $don:expr);)*) => {
        $(pub(crate) const $id: CdkType = CdkType {
            name: $name, hyb: Hyb::$hyb, nb: $nb, max_order: $max, pi: $pi, charge: $q, donation: $don,
        };)*
    };
}

cdk_types! {
    C_SP3 = ("C.sp3", Sp3, 4, 1, 0, 0, None);
    C_SP = ("C.sp", Sp1, 2, 3, 2, 0, None);
    C_ALLENE = ("C.allene", Sp1, 2, 2, 2, 0, None);
    C_SP2 = ("C.sp2", Sp2, 3, 2, 1, 0, Some(1));
    C_PLUS_PLANAR = ("C.plus.planar", Planar3, 3, 1, 0, 1, None);
    C_PLUS_SP2 = ("C.plus.sp2", Sp2, 2, 2, 1, 1, None);
    C_PLUS_SP1 = ("C.plus.sp1", Sp1, 1, 3, 2, 1, None);
    C_MINUS_SP3 = ("C.minus.sp3", Sp3, 3, 1, 0, -1, None);
    C_MINUS_PLANAR = ("C.minus.planar", Planar3, 3, 1, 0, -1, Some(2));
    C_MINUS_SP2 = ("C.minus.sp2", Sp2, 2, 2, 1, -1, None);
    C_MINUS_SP1 = ("C.minus.sp1", Sp1, 1, 3, 2, -1, None);
    O_SP3 = ("O.sp3", Sp3, 2, 1, 0, 0, Some(2));
    O_SP2 = ("O.sp2", Sp2, 1, 2, 1, 0, None);
    O_SP2_CO2 = ("O.sp2.co2", Sp2, 1, 2, 1, 0, None);
    O_PLANAR3 = ("O.planar3", Planar3, 2, 1, 0, 0, Some(2));
    O_PLUS_SP2 = ("O.plus.sp2", Sp2, 2, 2, 1, 1, None);
    O_PLUS_SP1 = ("O.plus.sp1", Sp1, 1, 3, 2, 1, None);
    O_PLUS = ("O.plus", Sp3, 3, 1, 0, 1, None);
    O_MINUS = ("O.minus", Sp3, 1, 1, 0, -1, None);
    O_MINUS_CO2 = ("O.minus.co2", Sp3, 1, 1, 0, -1, None);
    O_MINUS2 = ("O.minus2", Sp3, 0, 0, 0, -2, None);
    N_SP3 = ("N.sp3", Sp3, 3, 1, 0, 0, Some(2));
    N_SP2 = ("N.sp2", Sp2, 2, 2, 1, 0, None);
    N_SP2_3 = ("N.sp2.3", Sp2, 3, 2, 2, 0, Some(1));
    N_SP1_2 = ("N.sp1.2", Sp1, 2, 3, 3, 0, None);
    N_PLANAR3 = ("N.planar3", Planar3, 3, 1, 0, 0, Some(2));
    N_AMIDE = ("N.amide", Sp2, 3, 1, 0, 0, Some(2));
    N_OXIDE = ("N.oxide", Sp2, 4, 2, 1, 0, None);
    N_THIOAMIDE = ("N.thioamide", Sp2, 3, 1, 0, 0, Some(2));
    N_SP1 = ("N.sp1", Sp1, 1, 3, 2, 0, None);
    N_PLUS = ("N.plus", Sp3, 4, 1, 0, 1, None);
    N_PLUS_SP2 = ("N.plus.sp2", Sp2, 3, 2, 1, 1, None);
    N_PLUS_SP1 = ("N.plus.sp1", Sp1, 2, 3, 2, 1, None);
    N_MINUS_SP3 = ("N.minus.sp3", Sp3, 2, 1, 0, -1, None);
    N_MINUS_SP2 = ("N.minus.sp2", Sp2, 1, 2, 1, -1, None);
    N_MINUS_PLANAR3 = ("N.minus.planar3", Planar3, 2, 1, 0, -1, Some(2));
    N_NITRO = ("N.nitro", Planar3, 3, 2, 2, 0, None);
    S_2 = ("S.2", Sp2, 1, 2, 1, 0, Some(2));
    S_3 = ("S.3", Sp3, 2, 1, 0, 0, Some(2));
    S_SP3_4 = ("S.sp3.4", Sp3, 4, 2, 2, 0, None);
    S_SP3D1 = ("S.sp3d1", Other, 5, 2, 1, 0, None);
    S_INYL_2 = ("S.inyl.2", Sp2, 2, 3, 2, 0, None);
    S_OXIDE = ("S.oxide", Planar3, 2, 2, 2, 0, None);
    S_PLUS = ("S.plus", Sp2, 2, 2, 1, 1, None);
    S_PLANAR3 = ("S.planar3", Planar3, 2, 1, 0, 0, Some(2));
    S_MINUS = ("S.minus", Sp3, 1, 1, 0, -1, None);
    S_2MINUS = ("S.2minus", Other, 0, 0, 0, -2, None);
    S_THIONYL = ("S.thionyl", Sp3, 4, 2, 2, 0, None);
    S_ONYL = ("S.onyl", Sp3, 4, 2, 2, 0, None);
    S_ONYL_CHARGED = ("S.onyl.charged", Sp3, 4, 2, 2, 2, None);
    S_INYL = ("S.inyl", Sp2, 3, 2, 1, 0, None);
    S_INYL_CHARGED = ("S.inyl.charged", Sp2, 3, 1, 0, 1, None);
    S_TRIOXIDE = ("S.trioxide", Sp2, 3, 2, 3, 0, None);
    S_OCTAHEDRAL = ("S.octahedral", Other, 6, 1, 0, 0, None);
    S_ANYL = ("S.anyl", Other, 4, 1, 0, 0, None);
    P_ATE = ("P.ate", Sp3, 4, 2, 1, 0, None);
    P_ATE_CHARGED = ("P.ate.charged", Sp3, 4, 1, 0, 1, None);
    P_INE = ("P.ine", Sp3, 3, 1, 0, 0, Some(2));
    P_ANIUM = ("P.anium", Sp2, 3, 2, 1, 1, None);
    P_SP1_PLUS = ("P.sp1.plus", Sp1, 2, 2, 2, 1, None);
    P_IRANE = ("P.irane", Planar3, 2, 2, 1, 0, None);
    P_ANE = ("P.ane", Other, 5, 1, 0, 0, None);
    P_IDE = ("P.ide", Sp1, 1, 3, 2, 0, None);
    AS_PLUS = ("As.plus", Sp3, 4, 1, 0, 1, None);
    AS = ("As", Sp3, 3, 1, 0, 0, None);
    AS_PLANAR3 = ("As.planar3", Other, 3, 1, 0, 0, None);
    AS_5 = ("As.5", Sp3, 4, 2, 1, 0, None);
    AS_2 = ("As.2", Sp2, 2, 2, 1, 0, None);
    AS_3PLUS = ("As.3plus", Other, 0, 0, 0, 3, None);
    AS_MINUS = ("As.minus", Other, 6, 1, 0, -1, None);
}

/// Atom data needed by the CDK-ported descriptors.
pub(crate) struct CdkView {
    /// Kekulé bond order per bond (1, 2, 3).
    pub order: Vec<u8>,
    /// Formal charges (RDKit clean-up of N/P/halogen oxides reverted).
    pub charge: Vec<i8>,
    /// CDK implicit hydrogen count (after `CDKHydrogenAdder`).
    pub h: Vec<u8>,
    /// Maximum Kekulé order of the bonds in the molecule graph (0: no bonds).
    pub max_order: Vec<u8>,
    /// Matched CDK atom type (`None`: not a typed main-group atom).
    pub types: Vec<Option<&'static CdkType>>,
    /// The atom has a CDK atom type with a lone pair count (`LonePairElectronChecker`
    /// works) once all hydrogens are explicit atoms, as `BCUTDescriptor` retypes
    /// the molecule after `convertImplicitToExplicitHydrogens`; false for `X`
    /// (unmatched) atoms and elements outside the reduced matcher.
    pub lp_typed: Vec<bool>,
    /// `Aromaticity.cdkLegacy()` flags (computed on demand).
    pub arom_atom: Vec<bool>,
    pub arom_bond: Vec<bool>,
}

#[inline]
fn other(mol: &Molecule, bi: usize, a: usize) -> usize {
    let b = &mol.bonds[bi];
    if b.source as usize == a {
        b.target as usize
    } else {
        b.source as usize
    }
}

/// Kekulé bond orders: aromatic bonds become single/double so that every
/// aromatic atom that needs one gets exactly one double bond.
pub(crate) fn kekule_orders(mol: &Molecule) -> Vec<u8> {
    let nb = mol.num_bonds();
    let n = mol.num_atoms();
    let mut order = vec![1u8; nb];
    let mut any_arom = false;
    for (i, b) in mol.bonds.iter().enumerate() {
        order[i] = match b.bond_type {
            BondType::Double => 2,
            BondType::Triple => 3,
            BondType::Aromatic => {
                any_arom = true;
                1
            }
            _ => 1,
        };
    }
    if !any_arom {
        return order;
    }
    // Atoms that need a double bond among their aromatic bonds.
    let mut need = vec![false; n];
    for a in 0..n {
        let mut base = 0i32;
        let mut n_arom = 0;
        for &bi in &mol.adjacency[a] {
            let b = &mol.bonds[bi as usize];
            base += match b.bond_type {
                BondType::Aromatic => {
                    n_arom += 1;
                    1
                }
                BondType::Single => 1,
                BondType::Double => 2,
                BondType::Triple => 3,
                BondType::Dative => (b.target as usize == a) as i32,
                BondType::Unknown => 0,
            };
        }
        if n_arom == 0 {
            continue;
        }
        let atom = &mol.atoms[a];
        base += atom.total_h() as i32;
        let tv = crate::algo::sanitize::total_valence(mol, a);
        need[a] = tv - base >= 1;
    }
    // Matching by backtracking (most constrained atom first).
    let arom_bonds: Vec<Vec<usize>> = (0..n)
        .map(|a| {
            mol.adjacency[a]
                .iter()
                .map(|&b| b as usize)
                .filter(|&b| mol.bonds[b].bond_type == BondType::Aromatic)
                .collect()
        })
        .collect();
    let mut matched = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut budget = 200_000usize;
    fn solve(
        mol: &Molecule,
        need: &[bool],
        matched: &mut [bool],
        arom_bonds: &[Vec<usize>],
        stack: &mut Vec<usize>,
        budget: &mut usize,
    ) -> bool {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        // pick the open atom with fewest options
        let mut best: Option<(usize, usize)> = None;
        for a in 0..need.len() {
            if !need[a] || matched[a] {
                continue;
            }
            let opts = arom_bonds[a]
                .iter()
                .filter(|&&b| {
                    let o = other(mol, b, a);
                    need[o] && !matched[o]
                })
                .count();
            if opts == 0 {
                return false;
            }
            if best.is_none_or(|(_, c)| opts < c) {
                best = Some((a, opts));
                if opts == 1 {
                    break;
                }
            }
        }
        let Some((a, _)) = best else { return true };
        for &b in &arom_bonds[a] {
            let o = other(mol, b, a);
            if !need[o] || matched[o] {
                continue;
            }
            matched[a] = true;
            matched[o] = true;
            stack.push(b);
            if solve(mol, need, matched, arom_bonds, stack, budget) {
                return true;
            }
            stack.pop();
            matched[a] = false;
            matched[o] = false;
        }
        false
    }
    if solve(
        mol,
        &need,
        &mut matched,
        &arom_bonds,
        &mut stack,
        &mut budget,
    ) {
        for b in stack {
            order[b] = 2;
        }
    } else {
        // Greedy fallback (should not happen for sanitized molecules).
        let mut used = vec![false; n];
        for (bi, b) in mol.bonds.iter().enumerate() {
            let (u, v) = (b.source as usize, b.target as usize);
            if b.bond_type == BondType::Aromatic && need[u] && need[v] && !used[u] && !used[v] {
                used[u] = true;
                used[v] = true;
                order[bi] = 2;
            }
        }
    }
    order
}

/// Neighbour-environment queries on the Kekulé graph used by the matcher.
struct Env<'a> {
    mol: &'a Molecule,
    order: &'a [u8],
    charge: &'a [i8],
    /// `Some(h)`: `h[a]` hydrogens of atom `a` are explicit neighbour atoms
    /// (`convertImplicitToExplicitHydrogens`), the rest of its `total_h` (if
    /// any) stays implicit.
    explicit_h: Option<&'a [u8]>,
}

impl Env<'_> {
    /// Connected bond count (explicit hydrogen atoms included).
    fn deg(&self, a: usize) -> usize {
        self.mol.adjacency[a].len() + self.added_h(a)
    }
    /// Explicit hydrogen neighbours that are not graph atoms of `mol`.
    fn added_h(&self, a: usize) -> usize {
        self.explicit_h.map_or(0, |h| h[a] as usize)
    }
    /// Implicit hydrogen count.
    fn implicit_h(&self, a: usize) -> usize {
        (self.mol.atoms[a].total_h() as usize).saturating_sub(self.added_h(a))
    }
    /// `getMaximumBondOrder(connectedBonds)`: SINGLE when there are no bonds.
    fn max_order(&self, a: usize) -> u8 {
        self.mol.adjacency[a]
            .iter()
            .map(|&b| self.order[b as usize])
            .max()
            .unwrap_or(1)
            .max(1)
    }
    fn count_order(&self, a: usize, ord: u8, elem: Option<u8>) -> usize {
        self.mol.adjacency[a]
            .iter()
            .filter(|&&b| {
                self.order[b as usize] == ord
                    && elem.is_none_or(|z| {
                        self.mol.atoms[other(self.mol, b as usize, a)].atomic_number == z
                    })
            })
            .count()
    }
    fn has_double(&self, a: usize) -> bool {
        self.count_order(a, 2, None) > 0
    }
    /// `atLeastTwoNeighborsAreSp2`.
    fn both_neighbors_sp2(&self, a: usize) -> bool {
        let mut count = 0;
        for &b in &self.mol.adjacency[a] {
            let b = b as usize;
            let o = other(self.mol, b, a);
            if self.order[b] == 2 || self.has_double(o) {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
        false
    }
    fn ring(&self, a: usize) -> bool {
        self.mol.atoms[a].is_in_ring()
    }
    /// `isAcceptable` (REQUIRE_NOTHING mode, CDK types define no valency).
    fn ok(&self, a: usize, t: &'static CdkType) -> Option<&'static CdkType> {
        if self.deg(a) + self.implicit_h(a) > t.nb as usize {
            return None;
        }
        if t.max_order > 0
            && self.mol.adjacency[a]
                .iter()
                .any(|&b| self.order[b as usize] > t.max_order)
        {
            return None;
        }
        if self.charge[a] != t.charge {
            return None;
        }
        Some(t)
    }
    /// `isAmide` / `isThioAmide`: a carbon neighbour with exactly one double bond to `z`.
    fn is_amide(&self, a: usize, z: u8) -> bool {
        self.mol.adjacency[a].iter().any(|&b| {
            let o = other(self.mol, b as usize, a);
            self.mol.atoms[o].atomic_number == 6 && self.count_order(o, 2, Some(z)) == 1
        })
    }
    fn is_carboxylate(&self, a: usize) -> bool {
        if self.deg(a) != 1 || self.mol.adjacency[a].len() != 1 {
            return false;
        }
        let c = other(self.mol, self.mol.adjacency[a][0] as usize, a);
        if self.mol.atoms[c].atomic_number != 6 || self.deg(c) < 2 {
            return false;
        }
        let (mut n_o, mut n_neg, mut n_dbl) = (0, 0, 0);
        for &b in &self.mol.adjacency[c] {
            let o = other(self.mol, b as usize, c);
            if self.mol.atoms[o].atomic_number == 8 {
                n_o += 1;
                let ord = self.order[b as usize];
                if ord == 1 && self.charge[o] == -1 {
                    n_neg += 1;
                } else if ord == 2 {
                    n_dbl += 1;
                }
            }
        }
        n_o == 2 && n_neg == 1 && n_dbl == 1
    }
    /// `isSingleHeteroAtom` (aromatic neighbours and their aromatic neighbours are carbons).
    fn is_single_hetero(&self, a: usize) -> bool {
        for &b in &self.mol.adjacency[a] {
            if !self.mol.bonds[b as usize].is_aromatic() {
                continue;
            }
            let o = other(self.mol, b as usize, a);
            if self.mol.atoms[o].atomic_number != 6 {
                return false;
            }
            for &b2 in &self.mol.adjacency[o] {
                let o2 = other(self.mol, b2 as usize, o);
                if o2 != a
                    && self.mol.bonds[b2 as usize].is_aromatic()
                    && self.mol.atoms[o2].atomic_number != 6
                {
                    return false;
                }
            }
        }
        true
    }

    /// CDK atom type of a C, N, O, P, S or As atom (`None`: `X` or another element).
    fn atom_type(&self, a: usize) -> Option<&'static CdkType> {
        match self.mol.atoms[a].atomic_number {
            6 => self.carbon(a),
            7 => self.nitrogen(a),
            8 => self.oxygen(a),
            15 => self.phosphorus(a),
            16 => self.sulphur(a),
            33 => self.arsenic(a),
            _ => None,
        }
    }

    fn carbon(&self, a: usize) -> Option<&'static CdkType> {
        let atom = &self.mol.atoms[a];
        let deg = self.deg(a);
        let maxo = self.max_order(a);
        match self.charge[a] {
            0 => {}
            1 => {
                return if deg == 0 {
                    self.ok(a, &C_PLUS_PLANAR)
                } else {
                    match maxo {
                        3 => self.ok(a, &C_PLUS_SP1),
                        2 => self.ok(a, &C_PLUS_SP2),
                        1 => self.ok(a, &C_PLUS_PLANAR),
                        _ => None,
                    }
                };
            }
            -1 => {
                if maxo == 1 && deg <= 3 {
                    if self.both_neighbors_sp2(a) && self.ring(a)
                        && let Some(t) = self.ok(a, &C_MINUS_PLANAR) {
                            return Some(t);
                        }
                    return self.ok(a, &C_MINUS_SP3);
                } else if maxo == 2 && deg <= 3 {
                    return self.ok(a, &C_MINUS_SP2);
                } else if maxo == 3 && deg <= 1 {
                    return self.ok(a, &C_MINUS_SP1);
                }
                return None;
            }
            _ => return None,
        }
        if atom.is_aromatic() {
            // An alkyne carbon that RDKit puts in an aromatic ring (a cyclic
            // dehydroannulene written in Kekulé form) is not aromatic for
            // CDK's SMILES parser and types as `C.sp`.
            if maxo == 3 {
                return self.ok(a, &C_SP);
            }
            return self.ok(a, &C_SP2);
        }
        if deg > 4 {
            return None;
        }
        match maxo {
            3 => self.ok(a, &C_SP),
            2 => match self.count_order(a, 2, None) {
                2 => self.ok(a, &C_ALLENE),
                1 => self.ok(a, &C_SP2),
                _ => None,
            },
            1 => {
                if self.mol.adjacency[a]
                    .iter()
                    .any(|&b| self.mol.bonds[b as usize].is_aromatic())
                    && let Some(t) = self.ok(a, &C_SP2) {
                        return Some(t);
                    }
                self.ok(a, &C_SP3)
            }
            _ => None,
        }
    }

    fn oxygen(&self, a: usize) -> Option<&'static CdkType> {
        let deg = self.deg(a);
        match self.charge[a] {
            0 => {}
            -1 if deg <= 1 => {
                return if self.is_carboxylate(a) {
                    self.ok(a, &O_MINUS_CO2)
                } else {
                    self.ok(a, &O_MINUS)
                };
            }
            -2 if deg == 0 => return self.ok(a, &O_MINUS2),
            1 => {
                if deg == 0
                    && let Some(t) = self.ok(a, &O_PLUS) {
                        return Some(t);
                    }
                return match self.max_order(a) {
                    2 => self.ok(a, &O_PLUS_SP2),
                    3 => self.ok(a, &O_PLUS_SP1),
                    _ => self.ok(a, &O_PLUS),
                };
            }
            _ => return None,
        }
        if deg > 2 {
            return None;
        }
        if deg == 0 {
            return self.ok(a, &O_SP3);
        }
        match self.max_order(a) {
            2 => {
                if self.is_carboxylate(a) {
                    self.ok(a, &O_SP2_CO2)
                } else {
                    self.ok(a, &O_SP2)
                }
            }
            1 => {
                let n_h = self.added_h(a)
                    + self.mol.adjacency[a]
                        .iter()
                        .filter(|&&b| self.mol.atoms[other(self.mol, b as usize, a)].atomic_number == 1)
                        .count();
                if deg - n_h == 2 && self.both_neighbors_sp2(a) && self.ring(a)
                    && let Some(t) = self.ok(a, &O_PLANAR3) {
                        return Some(t);
                    }
                self.ok(a, &O_SP3)
            }
            _ => None,
        }
    }

    fn nitrogen(&self, a: usize) -> Option<&'static CdkType> {
        let deg = self.deg(a);
        let maxo = self.max_order(a);
        match self.charge[a] {
            0 => {}
            1 => {
                return match maxo {
                    1 => self.ok(a, &N_PLUS),
                    2 => match self.count_order(a, 2, None) {
                        1 => self.ok(a, &N_PLUS_SP2),
                        2 => self.ok(a, &N_PLUS_SP1),
                        _ => None,
                    },
                    3 if deg == 2 => self.ok(a, &N_PLUS_SP1),
                    _ => None,
                };
            }
            -1 => {
                if maxo == 1 {
                    if deg >= 2 && self.both_neighbors_sp2(a) && self.ring(a) {
                        return self.ok(a, &N_MINUS_PLANAR3);
                    } else if deg <= 2 {
                        return self.ok(a, &N_MINUS_SP3);
                    }
                } else if maxo == 2 && deg <= 1 {
                    return self.ok(a, &N_MINUS_SP2);
                }
                return None;
            }
            _ => return None,
        }
        if deg > 3 {
            if deg == 4 && self.count_order(a, 2, None) == 1 {
                return self.ok(a, &N_OXIDE);
            }
            return None;
        }
        if deg == 0 {
            return self.ok(a, &N_SP3);
        }
        match maxo {
            1 => {
                if self.is_amide(a, 8) {
                    if let Some(t) = self.ok(a, &N_AMIDE) {
                        return Some(t);
                    }
                } else if self.is_amide(a, 16)
                    && let Some(t) = self.ok(a, &N_THIOAMIDE) {
                        return Some(t);
                    }
                match deg {
                    2 => {
                        let arom = self.added_h(a) == 0
                            && self.mol.adjacency[a]
                                .iter()
                                .all(|&b| self.mol.bonds[b as usize].is_aromatic());
                        if arom {
                            match self.implicit_h(a) {
                                0 => {
                                    if self.is_single_hetero(a) {
                                        self.ok(a, &N_PLANAR3)
                                    } else {
                                        self.ok(a, &N_SP2)
                                    }
                                }
                                1 => self.ok(a, &N_PLANAR3),
                                _ => None,
                            }
                        } else if self.both_neighbors_sp2(a) && self.ring(a) {
                            self.ok(a, &N_PLANAR3)
                        } else {
                            self.ok(a, &N_SP3)
                        }
                    }
                    3 => {
                        if self.both_neighbors_sp2(a) && self.ring(a)
                            && let Some(t) = self.ok(a, &N_PLANAR3) {
                                return Some(t);
                            }
                        self.ok(a, &N_SP3)
                    }
                    _ => self.ok(a, &N_SP3),
                }
            }
            2 => {
                if deg == 3 && self.count_order(a, 2, Some(8)) == 2 {
                    if let Some(t) = self.ok(a, &N_NITRO) {
                        return Some(t);
                    }
                } else if deg == 3 && self.has_double(a)
                    && let Some(t) = self.ok(a, &N_SP2_3) {
                        return Some(t);
                    }
                self.ok(a, &N_SP2)
            }
            3 => {
                if deg > 1 {
                    self.ok(a, &N_SP1_2)
                } else {
                    self.ok(a, &N_SP1)
                }
            }
            _ => None,
        }
    }

    fn sulphur(&self, a: usize) -> Option<&'static CdkType> {
        let deg = self.deg(a);
        let maxo = self.max_order(a);
        let q = self.charge[a];
        if q != 0 {
            return match (q, deg) {
                (-1, 1) => self.ok(a, &S_MINUS),
                (1, 2) => self.ok(a, &S_PLUS),
                (1, 3) => self.ok(a, &S_INYL_CHARGED),
                (2, 4) => self.ok(a, &S_ONYL_CHARGED),
                (-2, 0) => self.ok(a, &S_2MINUS),
                _ => None,
            };
        }
        let dbl = self.count_order(a, 2, None);
        match deg {
            0 => self.ok(a, &S_3),
            1 => {
                let o = self.order[self.mol.adjacency[a][0] as usize];
                match o {
                    2 => self.ok(a, &S_2),
                    1 => self.ok(a, &S_3),
                    _ => None,
                }
            }
            2 => {
                if self.both_neighbors_sp2(a) && self.ring(a) {
                    if dbl == 2 {
                        self.ok(a, &S_INYL_2)
                    } else {
                        self.ok(a, &S_PLANAR3)
                    }
                } else if self.count_order(a, 2, Some(8)) == 2 {
                    self.ok(a, &S_OXIDE)
                } else if dbl == 2 {
                    self.ok(a, &S_INYL_2)
                } else {
                    self.ok(a, &S_3)
                }
            }
            3 => match dbl {
                1 => self.ok(a, &S_INYL),
                3 => self.ok(a, &S_TRIOXIDE),
                0 => self.ok(a, &S_ANYL),
                _ => None,
            },
            4 => {
                let d_o = self.count_order(a, 2, Some(8));
                let d_n = self.count_order(a, 2, Some(7));
                let d_s = self.count_order(a, 2, Some(16));
                if d_o + d_n == 2 {
                    self.ok(a, &S_ONYL)
                } else if d_s == 1 && d_o == 1 {
                    self.ok(a, &S_THIONYL)
                } else if maxo == 1 {
                    self.ok(a, &S_ANYL)
                } else if d_o == 1 && dbl == 1 {
                    self.ok(a, &S_SP3D1)
                } else if dbl == 2 && maxo == 2 {
                    self.ok(a, &S_SP3_4)
                } else {
                    None
                }
            }
            5 => match maxo {
                2 => self.ok(a, &S_SP3D1),
                1 => self.ok(a, &S_OCTAHEDRAL),
                _ => None,
            },
            6 if maxo == 1 => self.ok(a, &S_OCTAHEDRAL),
            _ => None,
        }
    }

    fn arsenic(&self, a: usize) -> Option<&'static CdkType> {
        let deg = self.deg(a);
        match self.charge[a] {
            1 if deg <= 4 => self.ok(a, &AS_PLUS),
            0 => {
                if deg == 4
                    && let Some(t) = self.ok(a, &AS_5) {
                        return Some(t);
                    }
                if deg == 2 {
                    match self.max_order(a) {
                        2 => {
                            if let Some(t) = self.ok(a, &AS_2) {
                                return Some(t);
                            }
                        }
                        1 => {
                            if let Some(t) = self.ok(a, &AS_PLANAR3) {
                                return Some(t);
                            }
                        }
                        _ => {}
                    }
                }
                self.ok(a, &AS)
            }
            3 => self.ok(a, &AS_3PLUS),
            -1 => self.ok(a, &AS_MINUS),
            _ => None,
        }
    }

    fn phosphorus(&self, a: usize) -> Option<&'static CdkType> {
        let deg = self.deg(a);
        let q = self.charge[a];
        let dbl = self.count_order(a, 2, None);
        match deg {
            0 if q == 0 => self.ok(a, &P_INE),
            1 if q == 0 => self.ok(a, &P_IDE),
            3 => {
                if q == 1 {
                    self.ok(a, &P_ANIUM)
                } else if dbl == 1 {
                    self.ok(a, &P_ATE)
                } else {
                    self.ok(a, &P_INE)
                }
            }
            2 => match self.max_order(a) {
                2 => {
                    if q == 1 {
                        self.ok(a, &P_SP1_PLUS)
                    } else {
                        self.ok(a, &P_IRANE)
                    }
                }
                1 => self.ok(a, &P_INE),
                _ => None,
            },
            4 => {
                if q == 1 && dbl == 0 {
                    self.ok(a, &P_ATE_CHARGED)
                } else if dbl == 1 {
                    self.ok(a, &P_ATE)
                } else {
                    None
                }
            }
            5 if q == 0 => self.ok(a, &P_ANE),
            _ => None,
        }
    }
}

/// Reverts RDKit's `cleanUp` of hypervalent N, P and halogen oxides
/// (`N(=O)=O` -> `[N+](=O)[O-]`, `Cl(=O)(=O)(=O)O` -> `[Cl+3]([O-])...`), which
/// CDK keeps as written. Atoms from the SMILES organic subset cannot carry a
/// charge, so charged non-bracket atoms identify clean-up results.
fn revert_cleanup(mol: &Molecule, order: &mut [u8], charge: &mut [i8]) {
    for a in 0..mol.num_atoms() {
        let atom = &mol.atoms[a];
        let z = atom.atomic_number;
        if atom.formal_charge <= 0 || atom.no_implicit_h() || !matches!(z, 7 | 15 | 17 | 35 | 53) {
            continue;
        }
        let mut restored = 0i8;
        for &bi in &mol.adjacency[a] {
            let bi = bi as usize;
            let o = other(mol, bi, a);
            let nb = &mol.atoms[o];
            if charge[o] != -1 || nb.no_implicit_h() {
                continue;
            }
            if nb.atomic_number == 8 && order[bi] == 1 {
                order[bi] = 2;
            } else if z == 7 && nb.atomic_number == 7 && order[bi] == 2 {
                order[bi] = 3;
            } else {
                continue;
            }
            charge[o] = 0;
            restored += 1;
            if z == 7 || z == 15 {
                break;
            }
        }
        charge[a] -= restored;
    }
}

/// Lone-pair typing status of atoms outside the reduced matcher (H, halogens, Si).
fn simple_typed(mol: &Molecule, order: &[u8], charge: &[i8], a: usize) -> bool {
    let atom = &mol.atoms[a];
    let deg = mol.adjacency[a].len() + atom.total_h() as usize;
    match (atom.atomic_number, charge[a]) {
        // X.plus.sp2 / X.plus.sp3 (`isAcceptable`: neighbours + H within 1 / 2);
        // a bare `[Br+]` has no maximum bond order and stays untyped
        (9 | 17 | 35 | 53, 1) => match mol.adjacency[a].iter().map(|&b| order[b as usize]).max() {
            Some(2) => deg <= 1,
            Some(1) => deg <= 2,
            None => deg <= 2 && (atom.atomic_number != 35 || deg > 0),
            _ => false,
        },
        (1, -1..=1) => deg <= 1,
        (9 | 17 | 35 | 53, 0) => deg <= 1 || (atom.atomic_number != 9 && deg <= 4),
        (9 | 17 | 35 | 53, -1) => deg == 0,
        // Cl.perchlorate.charged (`[O-][Cl+3]([O-])([O-])O`)
        (17, 3) => {
            deg == 4
                && mol.adjacency[a]
                    .iter()
                    .all(|&b| mol.bonds[b as usize].bond_type == BondType::Single)
        }
        (14, 0) => deg <= 4,
        _ => false,
    }
}

impl CdkView {
    /// Builds the CDK view: Kekulé orders, atom types and hydrogen counts.
    pub(crate) fn new(mol: &Molecule) -> Self {
        let n = mol.num_atoms();
        let mut order = kekule_orders(mol);
        let mut charge: Vec<i8> = mol.atoms.iter().map(|a| a.formal_charge).collect();
        revert_cleanup(mol, &mut order, &mut charge);
        // hydrogen atoms of the input that RDKit folded into their neighbour
        // (`[H][S+](C)C`) are atoms in CDK and count as neighbours for typing
        let removed_h: Option<Vec<u8>> = (!mol.removed_hydrogens().is_empty()).then(|| {
            let mut h = vec![0u8; n];
            for r in mol.removed_hydrogens() {
                h[r.atom as usize] += 1;
            }
            h
        });
        let env = Env {
            mol,
            order: &order,
            charge: &charge,
            explicit_h: removed_h.as_deref(),
        };
        let mut types = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        let mut max_order = Vec::with_capacity(n);
        for a in 0..n {
            let atom = &mol.atoms[a];
            let t = env.atom_type(a);
            let deg = mol.adjacency[a].len();
            h.push(match t {
                Some(t) => (t.nb as usize).saturating_sub(deg) as u8,
                None => atom.total_h(),
            });
            types.push(t);
            max_order.push(
                mol.adjacency[a]
                    .iter()
                    .map(|&b| order[b as usize])
                    .max()
                    .unwrap_or(0),
            );
        }
        // BCUTDescriptor retypes the molecule with explicit hydrogen atoms:
        // e.g. the untyped `[SH](=O)(C)O` (implicit H) becomes `S.sp3d1`, while
        // `[PH](C)(C)C` stays untyped.
        let explicit = Env {
            explicit_h: Some(&h),
            ..env
        };
        let lp_typed = (0..n)
            .map(|a| {
                let t = if h[a] > 0 { explicit.atom_type(a) } else { types[a] };
                t.is_some_and(|t| t.name != "As.3plus" && t.name != "As.minus") || simple_typed(mol, &order, &charge, a)
            })
            .collect();
        Self {
            order,
            charge,
            h,
            max_order,
            types,
            lp_typed,
            arom_atom: vec![false; n],
            arom_bond: vec![false; mol.num_bonds()],
        }
    }

    /// Runs `f` with the view of `mol` after `perceive_legacy_aromaticity`,
    /// cached in the calculation context (keyed by `molecule_cache_key`) so
    /// that the descriptors using it build and perceive it once per molecule.
    pub(crate) fn with_legacy_cached<R>(
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
        f: impl FnOnce(&CdkView, &mut crate::desc::context::CalculationContext) -> R,
    ) -> R {
        let key = crate::desc::context::molecule_cache_key(mol);
        let view = match ctx.cdk_legacy_view.take() {
            Some((k, view)) if k == key => view,
            _ => {
                let mut view = Self::new(mol);
                view.perceive_legacy_aromaticity(mol);
                view
            }
        };
        let r = f(&view, ctx);
        ctx.cdk_legacy_view = Some((key, view));
        r
    }

    /// `ElectronDonation.cdk()` contributions (`AtomTypeModel`, exocyclic pi bonds excluded).
    pub(crate) fn electrons(&self, mol: &Molecule) -> Vec<i32> {
        let n = mol.num_atoms();
        let mut e = vec![-1i32; n];
        for a in 0..n {
            if !mol.atoms[a].is_in_ring() {
                continue;
            }
            let Some(t) = self.types[a] else { continue };
            let v = t.donation.map_or(t.pi as i32, |d| d as i32);
            e[a] = match t.hyb {
                Hyb::Sp2 | Hyb::Planar3 => v,
                Hyb::Sp3 => {
                    if v == 0 {
                        -1
                    } else {
                        v
                    }
                }
                _ => -1,
            };
        }
        for (bi, b) in mol.bonds.iter().enumerate() {
            if self.order[bi] >= 2 && !b.is_in_ring() {
                let (u, v) = (b.source as usize, b.target as usize);
                let tn = |x: usize| self.types[x].map_or("", |t| t.name);
                if (tn(u) == "N.sp2.3" && tn(v) == "O.sp2")
                    || (tn(u) == "O.sp2" && tn(v) == "N.sp2.3")
                {
                    continue;
                }
                e[u] = -1;
                e[v] = -1;
            }
        }
        e
    }

    /// Applies `Aromaticity.cdkLegacy()` (sets `arom_atom` / `arom_bond`).
    pub(crate) fn perceive_legacy_aromaticity(&mut self, mol: &Molecule) {
        let e = self.electrons(mol);
        let n = mol.num_atoms();
        // Subgraph of contributing atoms: bonds between two such atoms.
        let sub_bonds: Vec<usize> = (0..mol.num_bonds())
            .filter(|&b| {
                let bd = &mol.bonds[b];
                bd.is_in_ring() && e[bd.source as usize] >= 0 && e[bd.target as usize] >= 0
            })
            .collect();
        if sub_bonds.len() < 3 {
            return;
        }
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for &b in &sub_bonds {
            let bd = &mol.bonds[b];
            let (u, v) = (bd.source as usize, bd.target as usize);
            adj[u].push((v, b));
            adj[v].push((u, b));
        }
        for block in biconnected_blocks(n, &adj) {
            if block.len() < 3 {
                continue;
            }
            let mut verts: Vec<usize> = block
                .iter()
                .flat_map(|&b| [mol.bonds[b].source as usize, mol.bonds[b].target as usize])
                .collect();
            verts.sort_unstable();
            verts.dedup();
            let r = block.len() + 1 - verts.len();
            let cycles = if r <= 3 {
                all_cycles(mol, &block, &verts)
            } else {
                min_cycle_basis(mol, &block, &verts, r)
            };
            for cyc in cycles {
                // cyc: bond indices of a simple cycle
                let mut sum = 0i32;
                let mut vs: Vec<usize> = cyc
                    .iter()
                    .flat_map(|&b| [mol.bonds[b].source as usize, mol.bonds[b].target as usize])
                    .collect();
                vs.sort_unstable();
                vs.dedup();
                for &v in &vs {
                    sum += e[v];
                }
                if (sum - 2) % 4 == 0 {
                    for &b in &cyc {
                        self.arom_bond[b] = true;
                        self.arom_atom[mol.bonds[b].source as usize] = true;
                        self.arom_atom[mol.bonds[b].target as usize] = true;
                    }
                }
            }
        }
    }
}

/// Biconnected components (as bond lists) of the graph `adj` (vertex -> (nbr, bond)).
fn biconnected_blocks(n: usize, adj: &[Vec<(usize, usize)>]) -> Vec<Vec<usize>> {
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut time = 0usize;
    let mut blocks = Vec::new();
    let mut estack: Vec<usize> = Vec::new();
    for root in 0..n {
        if disc[root] != usize::MAX || adj[root].is_empty() {
            continue;
        }
        // iterative DFS: (vertex, parent bond, next neighbour index)
        let mut st: Vec<(usize, usize, usize)> = vec![(root, usize::MAX, 0)];
        disc[root] = time;
        low[root] = time;
        time += 1;
        while let Some(&mut (v, pb, ref mut idx)) = st.last_mut() {
            if *idx < adj[v].len() {
                let (w, b) = adj[v][*idx];
                *idx += 1;
                if b == pb {
                    continue;
                }
                if disc[w] == usize::MAX {
                    estack.push(b);
                    disc[w] = time;
                    low[w] = time;
                    time += 1;
                    st.push((w, b, 0));
                } else if disc[w] < disc[v] {
                    estack.push(b);
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                st.pop();
                if let Some(&(u, _, _)) = st.last() {
                    low[u] = low[u].min(low[v]);
                    if low[v] >= disc[u] {
                        let mut comp = Vec::new();
                        while let Some(e) = estack.pop() {
                            comp.push(e);
                            if e == pb {
                                break;
                            }
                        }
                        blocks.push(comp);
                    }
                }
            }
        }
    }
    blocks
}

/// Edge bitset helpers.
fn bits_of(cyc: &[usize], index: &[usize], words: usize) -> Vec<u64> {
    let mut v = vec![0u64; words];
    for &b in cyc {
        let i = index[b];
        v[i / 64] ^= 1 << (i % 64);
    }
    v
}

/// Fundamental cycles of a connected block (bond lists), via a BFS spanning tree.
fn fundamental_cycles(mol: &Molecule, block: &[usize], verts: &[usize]) -> Vec<Vec<usize>> {
    let pos = |v: usize| verts.binary_search(&v).unwrap();
    let nv = verts.len();
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];
    for &b in block {
        let (u, v) = (
            pos(mol.bonds[b].source as usize),
            pos(mol.bonds[b].target as usize),
        );
        adj[u].push((v, b));
        adj[v].push((u, b));
    }
    let mut parent = vec![(usize::MAX, usize::MAX); nv];
    let mut depth = vec![usize::MAX; nv];
    depth[0] = 0;
    let mut queue = std::collections::VecDeque::from([0usize]);
    let mut tree = std::collections::HashSet::new();
    while let Some(u) = queue.pop_front() {
        for &(w, b) in &adj[u] {
            if depth[w] == usize::MAX {
                depth[w] = depth[u] + 1;
                parent[w] = (u, b);
                tree.insert(b);
                queue.push_back(w);
            }
        }
    }
    let mut out = Vec::new();
    for &b in block {
        if tree.contains(&b) {
            continue;
        }
        let (mut x, mut y) = (
            pos(mol.bonds[b].source as usize),
            pos(mol.bonds[b].target as usize),
        );
        let mut cyc = vec![b];
        while depth[x] > depth[y] {
            cyc.push(parent[x].1);
            x = parent[x].0;
        }
        while depth[y] > depth[x] {
            cyc.push(parent[y].1);
            y = parent[y].0;
        }
        while x != y {
            cyc.push(parent[x].1);
            x = parent[x].0;
            cyc.push(parent[y].1);
            y = parent[y].0;
        }
        out.push(cyc);
    }
    out
}

/// Whether an edge set forms one simple cycle (connected, all degrees 2).
fn is_simple_cycle(mol: &Molecule, edges: &[usize]) -> bool {
    let mut deg: std::collections::HashMap<usize, u8> = std::collections::HashMap::new();
    for &b in edges {
        *deg.entry(mol.bonds[b].source as usize).or_default() += 1;
        *deg.entry(mol.bonds[b].target as usize).or_default() += 1;
    }
    if deg.values().any(|&d| d != 2) {
        return false;
    }
    // connected: walk
    let start = mol.bonds[edges[0]].source as usize;
    let mut seen = vec![false; edges.len()];
    let mut cur = start;
    let mut count = 0;
    loop {
        let next = edges.iter().enumerate().find(|&(i, &b)| {
            !seen[i] && (mol.bonds[b].source as usize == cur || mol.bonds[b].target as usize == cur)
        });
        match next {
            Some((i, &b)) => {
                seen[i] = true;
                count += 1;
                cur = other(mol, b, cur);
            }
            None => break,
        }
    }
    count == edges.len() && cur == start
}

/// All simple cycles of a block with at most three independent cycles
/// (every non-empty combination of fundamental cycles that is a simple cycle).
fn all_cycles(mol: &Molecule, block: &[usize], verts: &[usize]) -> Vec<Vec<usize>> {
    let basis = fundamental_cycles(mol, block, verts);
    let r = basis.len();
    let mut out = Vec::new();
    for mask in 1u32..(1 << r) {
        let mut set: Vec<usize> = Vec::new();
        for (i, c) in basis.iter().enumerate() {
            if mask & (1 << i) != 0 {
                for &b in c {
                    if let Some(p) = set.iter().position(|&x| x == b) {
                        set.swap_remove(p);
                    } else {
                        set.push(b);
                    }
                }
            }
        }
        if !set.is_empty() && is_simple_cycle(mol, &set) {
            out.push(set);
        }
    }
    out
}

/// A minimum cycle basis (Horton candidates + GF(2) elimination).
fn min_cycle_basis(mol: &Molecule, block: &[usize], verts: &[usize], r: usize) -> Vec<Vec<usize>> {
    let pos = |v: usize| verts.binary_search(&v).unwrap();
    let nv = verts.len();
    let mut index = vec![usize::MAX; mol.num_bonds()];
    for (i, &b) in block.iter().enumerate() {
        index[b] = i;
    }
    let words = block.len().div_ceil(64);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];
    for &b in block {
        let (u, v) = (
            pos(mol.bonds[b].source as usize),
            pos(mol.bonds[b].target as usize),
        );
        adj[u].push((v, b));
        adj[v].push((u, b));
    }
    let mut cands: Vec<Vec<usize>> = Vec::new();
    let mut parent = vec![(usize::MAX, usize::MAX); nv];
    let mut depth = vec![usize::MAX; nv];
    for root in 0..nv {
        depth.fill(usize::MAX);
        depth[root] = 0;
        parent[root] = (usize::MAX, usize::MAX);
        let mut queue = std::collections::VecDeque::from([root]);
        let mut first = vec![usize::MAX; nv]; // first branch vertex from root
        while let Some(u) = queue.pop_front() {
            for &(w, b) in &adj[u] {
                if depth[w] == usize::MAX {
                    depth[w] = depth[u] + 1;
                    parent[w] = (u, b);
                    first[w] = if u == root { w } else { first[u] };
                    queue.push_back(w);
                }
            }
        }
        for &b in block {
            let (x, y) = (
                pos(mol.bonds[b].source as usize),
                pos(mol.bonds[b].target as usize),
            );
            if parent[x].1 == b || parent[y].1 == b {
                continue;
            }
            // paths root->x and root->y must only share root
            if x != root && y != root && first[x] == first[y] {
                continue;
            }
            let mut cyc = vec![b];
            let mut t = x;
            while t != root {
                cyc.push(parent[t].1);
                t = parent[t].0;
            }
            let mut t = y;
            while t != root {
                cyc.push(parent[t].1);
                t = parent[t].0;
            }
            cands.push(cyc);
        }
    }
    cands.sort_by_key(|c| c.len());
    let mut basis_rows: Vec<(Vec<u64>, usize)> = Vec::new(); // (row, pivot)
    let mut out = Vec::new();
    for c in cands {
        if out.len() == r {
            break;
        }
        let mut row = bits_of(&c, &index, words);
        for (brow, piv) in &basis_rows {
            if row[piv / 64] >> (piv % 64) & 1 == 1 {
                for k in 0..words {
                    row[k] ^= brow[k];
                }
            }
        }
        if let Some(piv) = (0..block.len()).find(|&i| row[i / 64] >> (i % 64) & 1 == 1) {
            // keep rows reduced on their pivots
            for (brow, _) in basis_rows.iter_mut() {
                if brow[piv / 64] >> (piv % 64) & 1 == 1 {
                    for k in 0..words {
                        brow[k] ^= row[k];
                    }
                }
            }
            basis_rows.push((row, piv));
            out.push(c);
        }
    }
    out
}

/// CDK element masses (`Isotopes.getNaturalMass`, `getMajorIsotope().getExactMass()`), by atomic number.
pub(crate) const NATURAL_MASS: [f64; 119] = [
    f64::NAN,
    1.0079407538257898,
    4.002601902393709,
    6.9400376247955,
    9.0121822,
    10.8110280884,
    12.010735896788,
    14.006703207035999,
    15.9994049275694,
    18.99840322,
    20.180046383575,
    22.98976928,
    24.305051612423,
    26.98153863,
    28.085384754933294,
    30.97376163,
    32.066084993038,
    35.452937578184,
    39.94767671084131,
    39.098301124462765,
    40.0780226519885,
    44.955911900000004,
    47.86674924515999,
    50.9414674975,
    51.99613307061901,
    54.9380451,
    55.845145623526996,
    58.933195,
    58.693351562447994,
    63.54564009360001,
    65.39556265692,
    69.723065622084,
    72.61275825015,
    74.9215965,
    78.9593882401,
    79.90352797085,
    83.79932499156999,
    85.467663596857,
    87.6166441049,
    88.9058483,
    91.22364760672002,
    92.9063781,
    95.93129211008002,
    f64::NAN,
    101.06494544279002,
    102.90550400000001,
    106.41532878129999,
    107.86815084455,
    112.41155220052,
    114.81808572199999,
    118.71010831536998,
    121.75978613256999,
    127.60312784714,
    126.904473,
    131.29248059059,
    132.9054519,
    137.32689183471098,
    138.90545398283,
    140.11572621391997,
    140.9076528,
    144.23613137220002,
    f64::NAN,
    150.36634852618,
    151.96437047419,
    157.25212221543998,
    158.9253468,
    162.49703350552002,
    164.9303221,
    167.25630413192002,
    168.9342133,
    173.0376955879,
    174.96672138554996,
    178.48497214162,
    180.94787573628,
    183.84177810241,
    186.2067066106,
    190.22486304559004,
    192.21605641479997,
    195.077807880313,
    196.96656870000004,
    200.5991666178,
    204.383332426508,
    207.21690766300003,
    208.9803987,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    232.03805529999997,
    231.035884,
    238.0289182792545,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];

pub(crate) const MAJOR_ISOTOPE_MASS: [f64; 119] = [
    f64::NAN,
    1.007825032,
    4.002603254,
    7.01600455,
    9.0121822,
    11.0093054,
    12.0,
    14.003074,
    15.99491462,
    18.99840322,
    19.99244018,
    22.98976928,
    23.9850417,
    26.98153863,
    27.97692653,
    30.97376163,
    31.972071,
    34.96885268,
    39.96238312,
    38.96370668,
    39.96259098,
    44.9559119,
    47.9479463,
    50.9439595,
    51.9405075,
    54.9380451,
    55.9349375,
    58.933195,
    57.9353429,
    62.9295975,
    63.9291422,
    68.9255736,
    73.9211778,
    74.9215965,
    79.9165213,
    78.9183371,
    83.911507,
    84.91178974,
    87.9056121,
    88.9058483,
    89.9047044,
    92.9063781,
    97.9054082,
    f64::NAN,
    101.9043493,
    102.905504,
    105.903486,
    106.905097,
    113.9033585,
    114.903878,
    119.9021947,
    120.9038157,
    129.9062244,
    126.904473,
    131.9041535,
    132.9054519,
    137.9052472,
    138.9063533,
    139.9054387,
    140.9076528,
    141.9077233,
    f64::NAN,
    151.9197324,
    152.9212303,
    157.9241039,
    158.9253468,
    163.9291748,
    164.9303221,
    165.9302931,
    168.9342133,
    173.9388621,
    174.9407718,
    179.94655,
    180.9479958,
    183.9509312,
    186.9557531,
    191.9614807,
    192.9629264,
    194.9647911,
    196.9665687,
    201.970643,
    204.9744275,
    207.9766521,
    208.9803987,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    232.0380553,
    231.035884,
    238.0507882,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];
