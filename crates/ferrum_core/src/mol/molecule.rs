//! Molecular model module: Molecule.
//!
//! This file defines core molecular graph data structures or element metadata
//! used throughout the library. These types are intentionally compact and
//! thread-safe to support high-throughput descriptor and ML workloads.
//!
//! Learning focus:
//! - Data-oriented Rust struct design for scientific computing.
//! - Graph-centric chemistry representations.

use crate::mol::{atom::Atom, bond::Bond};
use crate::types::{AtomIdx, BondIdx};
use smallvec::SmallVec;
use std::sync::OnceLock;

/// Properties derived from the molecule graph, each computed at most once
/// (`OnceLock`: thread-safe, computed by the first caller, read-only after).
/// They are pure functions of the molecule, so descriptors can fill them
/// through `&Molecule` and several threads may share one molecule.
#[derive(Debug, Default, Clone)]
struct MoleculeCache {
    gasteiger_charges: OnceLock<Vec<f64>>,
    labute_contribs: OnceLock<CachedLabuteContribs>,
    crippen_contribs: OnceLock<(Vec<f64>, Vec<f64>)>,
    crippen_totals: OnceLock<(f64, f64)>,
    hbond_donor_count: OnceLock<u32>,
    hbond_acceptor_count: OnceLock<u32>,
    rotatable_bond_count: OnceLock<u32>,
    tpsa: OnceLock<f64>,
    min_cycle_basis_bonds: OnceLock<Vec<smallvec::SmallVec<[u16; 8]>>>,
}

#[derive(Debug, Clone)]
/// Implements CachedLabuteContribs.
struct CachedLabuteContribs {
    atom_vsa: Vec<f64>,
    h_contrib: f64,
}

/// Molecular graph representation.
///
/// # Thread Safety
/// `Molecule` is `Send + Sync`, making it safe to share across threads.
/// This is critical for parallel batch processing with Rayon.
/// All internal data structures (Vec, SmallVec) are thread-safe.
#[derive(Debug, Default, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    // Adjazenzliste: Für jedes Atom eine Liste von Bindungs-Indizes.
    // SmallVec optimiert: Keine Heap-Allocation bei < 4 Bindungen.
    pub adjacency: Vec<SmallVec<[BondIdx; 4]>>,

    // 3D conformer data
    pub conformers: Vec<crate::mol::conformer::Conformer>,

    /// Ring perception result (RDKit symmetrized SSSR): set during
    /// sanitization, computed on first use for molecules built by hand.
    ring_info: OnceLock<Box<RingInfo>>,

    /// Derived properties, computed on first use (see [`MoleculeCache`]).
    cache: MoleculeCache,

    /// Hydrogen atoms of the input that sanitization folded into their
    /// neighbour's explicit H count (see [`RemovedHydrogen`]); `None` (no
    /// allocation) when there were none. Not a cache: it describes the input
    /// and survives [`invalidate_cached_properties`](Self::invalidate_cached_properties).
    removed_h: Option<Box<RemovedHydrogens>>,
}

/// The removed-hydrogen record of a molecule (one allocation, boxed so that
/// molecules without removed hydrogens only pay for a null pointer).
pub(crate) type RemovedHydrogens = SmallVec<[RemovedHydrogen; 4]>;

/// A hydrogen atom of the input that `sanitize::remove_hs` (RDKit `removeHs`)
/// folded into the explicit H count of its neighbour (`[H]C`, `[H][C@](C)(N)O`,
/// `[H]n1cccc1`). RDKit forgets such hydrogens; CDK keeps them as atoms, so the
/// CDK views restore them from this record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemovedHydrogen {
    /// The atom the hydrogen was bonded to (index in the sanitized molecule).
    pub atom: AtomIdx,
    /// Formal charge of the hydrogen atom.
    pub charge: i8,
    /// Direction of its bond (`/`: `Some(true)`, `\`: `Some(false)`), seen
    /// from `atom` as the bond's begin atom (double-bond stereo).
    pub up: Option<bool>,
}

/// Ring information as perceived by RDKit's symmetrized SSSR.
#[derive(Debug, Default, Clone)]
pub struct RingInfo {
    /// Atom rings in traversal order (consecutive atoms are bonded).
    pub atom_rings: Vec<SmallVec<[AtomIdx; 8]>>,
    /// Bond rings, aligned with `atom_rings` (`bond_rings[i][k]` joins
    /// `atom_rings[i][k]` and `atom_rings[i][k + 1]`).
    pub bond_rings: Vec<SmallVec<[BondIdx; 8]>>,
    /// Number of rings each atom is a member of.
    pub atom_ring_counts: Vec<u8>,
    /// Number of rings each bond is a member of.
    pub bond_ring_counts: Vec<u8>,
}

impl RingInfo {
    pub fn num_rings(&self) -> usize {
        self.atom_rings.len()
    }
    pub fn num_atom_rings(&self, atom: usize) -> usize {
        self.atom_ring_counts.get(atom).copied().unwrap_or(0) as usize
    }
    pub fn num_bond_rings(&self, bond: usize) -> usize {
        self.bond_ring_counts.get(bond).copied().unwrap_or(0) as usize
    }
}

// Static assertion: Molecule must be Send + Sync for parallel processing
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Molecule>;
};

impl Molecule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n_atoms: usize) -> Self {
        Self {
            atoms: Vec::with_capacity(n_atoms),
            bonds: Vec::with_capacity(n_atoms),
            adjacency: Vec::with_capacity(n_atoms),
            conformers: Vec::new(),
            ring_info: OnceLock::new(),
            cache: MoleculeCache::default(),
            removed_h: None,
        }
    }

    /// Constructs parts between representations.
    pub(crate) fn from_parts(
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        adjacency: Vec<SmallVec<[BondIdx; 4]>>,
    ) -> Self {
        Self {
            atoms,
            bonds,
            adjacency,
            conformers: Vec::new(),
            ring_info: OnceLock::new(),
            cache: MoleculeCache::default(),
            removed_h: None,
        }
    }

    /// Hydrogen atoms of the input that sanitization folded into their
    /// neighbours, in input order (empty for most molecules).
    pub fn removed_hydrogens(&self) -> &[RemovedHydrogen] {
        self.removed_h.as_deref().map_or(&[], |list| list.as_slice())
    }

    /// Number of hydrogen atoms of the input folded into `atom`.
    pub fn num_removed_hydrogens(&self, atom: usize) -> usize {
        self.removed_hydrogens().iter().filter(|h| h.atom as usize == atom).count()
    }

    pub(crate) fn set_removed_hydrogens(&mut self, list: RemovedHydrogens) {
        self.removed_h = if list.is_empty() { None } else { Some(Box::new(list)) };
    }

    pub(crate) fn take_removed_hydrogens(&mut self) -> RemovedHydrogens {
        self.removed_h.take().map_or_else(RemovedHydrogens::new, |b| *b)
    }

    /// Renumbers the atoms of the removed-hydrogen record after atoms were
    /// deleted or reordered (`new_idx[old]`, `u16::MAX`: deleted, which drops
    /// the hydrogens of that atom).
    pub(crate) fn remap_removed_hydrogens(&mut self, new_idx: &[u16]) {
        let mut list = self.take_removed_hydrogens();
        list.retain_mut(|h| {
            h.atom = new_idx.get(h.atom as usize).copied().unwrap_or(u16::MAX);
            h.atom != u16::MAX
        });
        self.set_removed_hydrogens(list);
    }

    /// RDKit `getTotalNumHs(true)`: implicit + explicit H plus hydrogen-atom neighbours.
    pub fn total_hs_with_neighbors(&self, atom: usize) -> u32 {
        let h_nbrs = self.adjacency[atom]
            .iter()
            .filter(|&&b| {
                let bond = &self.bonds[b as usize];
                let other = if bond.source as usize == atom { bond.target } else { bond.source };
                self.atoms[other as usize].atomic_number == 1
            })
            .count() as u32;
        self.atoms[atom].total_h() as u32 + h_nbrs
    }

    /// Ring information (RDKit symmetrized SSSR). Set by sanitization; for
    /// molecules built by hand it is computed on first use. Always `Some`.
    pub fn ring_info(&self) -> Option<&RingInfo> {
        Some(self.ring_info.get_or_init(|| Box::new(crate::algo::rings::compute_ring_info(self))))
    }

    pub(crate) fn set_ring_info(&mut self, info: RingInfo) {
        self.ring_info = OnceLock::from(Box::new(info));
    }

    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }
    pub fn num_bonds(&self) -> usize {
        self.bonds.len()
    }

    pub fn get_degree(&self, atom: AtomIdx) -> usize {
        self.adjacency[atom as usize].len()
    }

    pub fn invalidate_cached_properties(&mut self) {
        self.cache = MoleculeCache::default();
        self.ring_info = OnceLock::new();
    }

    // --- Derived-property cache: `*_or_init` computes on first use ---

    pub(crate) fn gasteiger_charges_or_init(&self, f: impl FnOnce() -> Vec<f64>) -> &[f64] {
        self.cache.gasteiger_charges.get_or_init(f)
    }

    pub(crate) fn cached_gasteiger_charges(&self) -> Option<&[f64]> {
        self.cache.gasteiger_charges.get().map(Vec::as_slice)
    }

    pub(crate) fn labute_contribs_or_init(&self, f: impl FnOnce() -> (Vec<f64>, f64)) -> (&[f64], f64) {
        let c = self.cache.labute_contribs.get_or_init(|| {
            let (atom_vsa, h_contrib) = f();
            CachedLabuteContribs { atom_vsa, h_contrib }
        });
        (&c.atom_vsa, c.h_contrib)
    }

    pub(crate) fn cached_labute_contribs(&self) -> Option<(&[f64], f64)> {
        self.cache.labute_contribs.get().map(|c| (c.atom_vsa.as_slice(), c.h_contrib))
    }

    pub(crate) fn crippen_contribs_or_init(&self, f: impl FnOnce() -> (Vec<f64>, Vec<f64>)) -> (&[f64], &[f64]) {
        let (logp, mr) = self.cache.crippen_contribs.get_or_init(f);
        (logp, mr)
    }

    pub(crate) fn cached_crippen_contribs(&self) -> Option<(&[f64], &[f64])> {
        self.cache.crippen_contribs.get().map(|(l, m)| (l.as_slice(), m.as_slice()))
    }

    pub(crate) fn crippen_totals_or_init(&self, f: impl FnOnce() -> (f64, f64)) -> (f64, f64) {
        *self.cache.crippen_totals.get_or_init(f)
    }

    pub(crate) fn cached_crippen_totals(&self) -> Option<(f64, f64)> {
        self.cache.crippen_totals.get().copied()
    }

    pub(crate) fn hbond_donor_count_or_init(&self, f: impl FnOnce() -> u32) -> u32 {
        *self.cache.hbond_donor_count.get_or_init(f)
    }

    pub(crate) fn hbond_acceptor_count_or_init(&self, f: impl FnOnce() -> u32) -> u32 {
        *self.cache.hbond_acceptor_count.get_or_init(f)
    }

    pub(crate) fn rotatable_bond_count_or_init(&self, f: impl FnOnce() -> u32) -> u32 {
        *self.cache.rotatable_bond_count.get_or_init(f)
    }

    pub(crate) fn tpsa_or_init(&self, f: impl FnOnce() -> f64) -> f64 {
        *self.cache.tpsa.get_or_init(f)
    }

    pub(crate) fn min_cycle_basis_bonds_or_init(
        &self,
        f: impl FnOnce() -> Vec<smallvec::SmallVec<[u16; 8]>>,
    ) -> &[smallvec::SmallVec<[u16; 8]>] {
        self.cache.min_cycle_basis_bonds.get_or_init(f)
    }

}

#[cfg(test)]
mod tests {
    use super::RemovedHydrogen;
    use crate::desc::context::molecule_cache_key;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn removed_hydrogens_are_recorded() {
        let h = |smiles: &str| parse_smiles(smiles).unwrap().removed_hydrogens().to_vec();
        assert!(h("CCO").is_empty());
        assert!(h("c1cc[nH]c1").is_empty());
        // isotopic H, H defining double-bond stereo, H2 stay atoms
        assert!(h("[2H]C").is_empty());
        assert!(h("[H]/N=C/C").is_empty());
        assert!(h("[H][H]").is_empty());
        let rh = |atom, charge, up| RemovedHydrogen { atom, charge, up };
        assert_eq!(h("[H][C@](C)(N)O"), [rh(0, 0, None)]);
        assert_eq!(h("[H]n1cccc1"), [rh(0, 0, None)]);
        assert_eq!(h("C[H]"), [rh(0, 0, None)]);
        assert_eq!(h("[H]O[H]"), [rh(0, 0, None), rh(0, 0, None)]);
        // the direction is seen from the heavy atom
        assert_eq!(h("[H]/C(C)=C/C"), [rh(0, 0, Some(false))]);
        assert_eq!(h("C/C=C(C)/[H]"), [rh(2, 0, Some(true))]);
        assert_eq!(h("[H-].[H]C"), [rh(1, 0, None)]);
    }

    #[test]
    fn removed_hydrogens_change_the_cache_key() {
        let key = |smiles: &str| molecule_cache_key(&parse_smiles(smiles).unwrap());
        assert_ne!(key("[H]n1cccc1"), key("c1cc[nH]c1"));
        assert_eq!(key("[H]n1cccc1"), key("[H]n1cccc1"));
        let mol = parse_smiles("[H]n1cccc1").unwrap();
        let mut copy = mol.clone();
        copy.invalidate_cached_properties();
        assert_eq!(copy.removed_hydrogens(), mol.removed_hydrogens());
    }
}
