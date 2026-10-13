//! Molecular model module: Molecule.
//!
//! This file defines core molecular graph data structures or element metadata
//! used throughout the library. These types are intentionally compact and
//! thread-safe to support high-throughput descriptor and ML workloads.
//!
//! Learning focus:
//! - Data-oriented Rust struct design for scientific computing.
//! - Graph-centric chemistry representations.

//! ## File guide: Molecule
//!
//! This file belongs to FerrumChem's **molecular graph model layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct MoleculeCache`
//! - `struct CachedLabuteContribs`
//! - `struct Molecule`
//! - `fn new`
//! - `fn with_capacity`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Molecule` encodes graph primitives with compact integer indices to keep memory predictable on large CSV batches.
//! - `Molecule` expects topology invariants (bond endpoints, adjacency symmetry) to hold after all mutating operations.
//! - Bitflag/state fields in `Molecule` are treated as cached derivations and must be invalidated after structural edits.
//! - Specific scope anchor: `crates/ferrum_core/src/mol/molecule.rs` defines molecule behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Graph-based cheminformatics data-model conventions for atom/bond/molecule containers.
//! - IUPAC element/valence background relevant to `Molecule` typed fields.
//! - Rust ownership/borrowing patterns for safe high-performance molecular graph manipulation.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/mol/molecule.rs`.

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
///
/// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
/// Mol-layer note: core graph data structure component.
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
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Molecule>;
};

impl Molecule {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn new() -> Self {
        Self::default()
    }

    /// Implements capacity.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn with_capacity(n_atoms: usize) -> Self {
        Self {
            atoms: Vec::with_capacity(n_atoms),
            bonds: Vec::with_capacity(n_atoms),
            adjacency: Vec::with_capacity(n_atoms),
            conformers: Vec::new(),
            ring_info: OnceLock::new(),
            cache: MoleculeCache::default(),
        }
    }

    /// Constructs parts between representations.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
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
        }
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

    /// Implements atoms.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }
    /// Implements bonds.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn num_bonds(&self) -> usize {
        self.bonds.len()
    }

    /// Returns degree.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn get_degree(&self, atom: AtomIdx) -> usize {
        self.adjacency[atom as usize].len()
    }

    /// Implements cached properties.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
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
