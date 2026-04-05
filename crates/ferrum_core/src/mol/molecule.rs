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

#[derive(Debug, Default, Clone)]
struct MoleculeCache {
    gasteiger_charges: Option<Vec<f64>>,
    labute_contribs: Option<CachedLabuteContribs>,
    crippen_contribs: Option<(Vec<f64>, Vec<f64>)>,
    crippen_totals: Option<(f64, f64)>,
    hbond_donor_count: Option<u32>,
    hbond_acceptor_count: Option<u32>,
    rotatable_bond_count: Option<u32>,
    tpsa: Option<f64>,
    balaban_bo_dist: Option<Vec<f64>>,
    min_cycle_basis_bonds: Option<Vec<smallvec::SmallVec<[usize; 8]>>>,
    symmetrized_sssr_bonds: Option<Vec<smallvec::SmallVec<[usize; 8]>>>,
    unweighted_dist_matrix: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
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

    // Lazy-initialized cache for expensive properties
    cache: Option<Box<MoleculeCache>>,
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
            cache: None,
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
            cache: None,
        }
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
        self.cache = None;
    }

    /// Implements mut.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn cache_mut(&mut self) -> &mut MoleculeCache {
        if self.cache.is_none() {
            self.cache = Some(Box::new(MoleculeCache::default()));
        }
        self.cache.as_mut().unwrap()
    }

    // --- Cache Accessors ---

    /// Implements gasteiger charges.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_gasteiger_charges(&self) -> Option<&[f64]> {
        self.cache.as_deref()?.gasteiger_charges.as_deref()
    }

    /// Implements gasteiger charges.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_gasteiger_charges(&mut self, charges: Vec<f64>) {
        self.cache_mut().gasteiger_charges = Some(charges);
    }

    /// Implements labute contribs.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_labute_contribs(&self) -> Option<(&[f64], f64)> {
        let contribs = self.cache.as_deref()?.labute_contribs.as_ref()?;
        Some((&contribs.atom_vsa, contribs.h_contrib))
    }

    /// Implements labute contribs.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_labute_contribs(&mut self, atom_vsa: Vec<f64>, h_contrib: f64) {
        self.cache_mut().labute_contribs = Some(CachedLabuteContribs {
            atom_vsa,
            h_contrib,
        });
    }

    /// Implements crippen contribs.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_crippen_contribs(&self) -> Option<(&[f64], &[f64])> {
        let (logp, mr) = self.cache.as_deref()?.crippen_contribs.as_ref()?;
        Some((logp, mr))
    }

    /// Implements crippen contribs.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_crippen_contribs(&mut self, logp: Vec<f64>, mr: Vec<f64>) {
        self.cache_mut().crippen_contribs = Some((logp, mr));
    }

    /// Implements crippen totals.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_crippen_totals(&self) -> Option<(f64, f64)> {
        self.cache.as_deref()?.crippen_totals
    }

    /// Implements crippen totals.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_crippen_totals(&mut self, logp: f64, mr: f64) {
        self.cache_mut().crippen_totals = Some((logp, mr));
    }

    /// Implements hbond donor count.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_hbond_donor_count(&self) -> Option<u32> {
        self.cache.as_deref()?.hbond_donor_count
    }

    /// Implements hbond donor count.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_hbond_donor_count(&mut self, count: u32) {
        self.cache_mut().hbond_donor_count = Some(count);
    }

    /// Implements hbond acceptor count.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_hbond_acceptor_count(&self) -> Option<u32> {
        self.cache.as_deref()?.hbond_acceptor_count
    }

    /// Implements hbond acceptor count.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_hbond_acceptor_count(&mut self, count: u32) {
        self.cache_mut().hbond_acceptor_count = Some(count);
    }

    /// Implements rotatable bond count.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_rotatable_bond_count(&self) -> Option<u32> {
        self.cache.as_deref()?.rotatable_bond_count
    }

    /// Implements rotatable bond count.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_rotatable_bond_count(&mut self, count: u32) {
        self.cache_mut().rotatable_bond_count = Some(count);
    }

    /// Implements tpsa.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_tpsa(&self) -> Option<f64> {
        self.cache.as_deref()?.tpsa
    }

    /// Implements tpsa.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_tpsa(&mut self, tpsa: f64) {
        self.cache_mut().tpsa = Some(tpsa);
    }

    /// Implements balaban bo dist.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_balaban_bo_dist(&self) -> Option<&Vec<f64>> {
        self.cache.as_deref()?.balaban_bo_dist.as_ref()
    }

    /// Implements balaban bo dist.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_balaban_bo_dist(&mut self, dist: Vec<f64>) {
        self.cache_mut().balaban_bo_dist = Some(dist);
    }

    /// Implements min cycle basis bonds.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_min_cycle_basis_bonds(
        &self,
    ) -> Option<&Vec<smallvec::SmallVec<[usize; 8]>>> {
        self.cache.as_deref()?.min_cycle_basis_bonds.as_ref()
    }

    /// Implements min cycle basis bonds.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_min_cycle_basis_bonds(
        &mut self,
        rings: Vec<smallvec::SmallVec<[usize; 8]>>,
    ) {
        self.cache_mut().min_cycle_basis_bonds = Some(rings);
    }

    /// Implements symmetrized sssr bonds.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_symmetrized_sssr_bonds(
        &self,
    ) -> Option<&Vec<smallvec::SmallVec<[usize; 8]>>> {
        self.cache.as_deref()?.symmetrized_sssr_bonds.as_ref()
    }

    /// Implements symmetrized sssr bonds.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_symmetrized_sssr_bonds(
        &mut self,
        rings: Vec<smallvec::SmallVec<[usize; 8]>>,
    ) {
        self.cache_mut().symmetrized_sssr_bonds = Some(rings);
    }

    /// Implements unweighted dist matrix.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn cached_unweighted_dist_matrix(&self) -> Option<&[u8]> {
        self.cache.as_deref()?.unweighted_dist_matrix.as_deref()
    }

    /// Implements unweighted dist matrix.
    ///
    /// Function context: `crates/ferrum_core/src/mol/molecule.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub(crate) fn store_unweighted_dist_matrix(&mut self, matrix: Vec<u8>) {
        self.cache_mut().unweighted_dist_matrix = Some(matrix);
    }
}
