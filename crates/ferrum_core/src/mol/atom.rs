//! Molecular model module: Atom.
//!
//! This file defines core molecular graph data structures or element metadata
//! used throughout the library. These types are intentionally compact and
//! thread-safe to support high-throughput descriptor and ML workloads.
//!
//! Learning focus:
//! - Data-oriented Rust struct design for scientific computing.
//! - Graph-centric chemistry representations.


//! ## File guide: Atom
//!
//! This file belongs to FerrumChem's **molecular graph model layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct Atom`
//! - `fn new`
//! - `fn is_aromatic`
//! - `fn set_aromatic`
//! - `fn is_in_ring`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Atom` encodes graph primitives with compact integer indices to keep memory predictable on large CSV batches.
//! - `Atom` expects topology invariants (bond endpoints, adjacency symmetry) to hold after all mutating operations.
//! - Bitflag/state fields in `Atom` are treated as cached derivations and must be invalidated after structural edits.
//! - Specific scope anchor: `crates/ferrum_core/src/mol/atom.rs` defines atom behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Graph-based cheminformatics data-model conventions for atom/bond/molecule containers.
//! - IUPAC element/valence background relevant to `Atom` typed fields.
//! - Rust ownership/borrowing patterns for safe high-performance molecular graph manipulation.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/mol/atom.rs`.

use crate::types::Hybridization;

/// Atom representation in a molecular graph.
///
/// # Thread Safety
/// `Atom` is `Copy + Send + Sync`, making it trivially thread-safe.
/// The struct is designed to be compact (12 bytes) and cache-friendly
/// for high-performance parallel processing.
#[derive(Debug, Clone, Copy)]
pub struct Atom {
    pub atomic_number: u8,
    pub formal_charge: i8,
    pub num_explicit_h: u8,
    pub num_implicit_h: u8, // Calculated implicit hydrogens
    pub isotope: u16,
    pub hybridization: Hybridization,
    pub flags: u8, // Bit 0: Aromatic, Bit 1: Ring
}

// Static assertion: Atom must be Send + Sync
const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Atom>;
};

impl Atom {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn new(element: u8) -> Self {
        Self {
            atomic_number: element,
            formal_charge: 0,
            num_explicit_h: 0,
            num_implicit_h: 0,
            isotope: 0,
            hybridization: Hybridization::Unspecified,
            flags: 0,
        }
    }

    /// Checks whether aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn is_aromatic(&self) -> bool {
        self.flags & 1 != 0
    }
    /// Sets aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn set_aromatic(&mut self, val: bool) {
        if val {
            self.flags |= 1;
        } else {
            self.flags &= !1;
        }
    }

    /// Checks whether in ring.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn is_in_ring(&self) -> bool {
        self.flags & 0b10 != 0
    }
    /// Sets in ring.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn set_in_ring(&mut self, val: bool) {
        if val {
            self.flags |= 0b10;
        } else {
            self.flags &= !0b10;
        }
    }

    /// Whether this atom has no implicit H (bracket atom with explicit specification)
    pub fn no_implicit_h(&self) -> bool {
        self.flags & 0b100 != 0
    }
    /// Sets no implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn set_no_implicit_h(&mut self, val: bool) {
        if val {
            self.flags |= 0b100;
        } else {
            self.flags &= !0b100;
        }
    }

    /// Total hydrogen count (explicit + implicit)
    pub fn total_h(&self) -> u8 {
        self.num_explicit_h + self.num_implicit_h
    }
}
