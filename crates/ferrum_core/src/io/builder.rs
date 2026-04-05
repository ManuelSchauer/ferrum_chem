//! I/O module: Builder.
//!
//! This module handles external data interchange (SMILES/SMARTS/CSV/SDF) and
//! conversion into FerrumChem graph types. It is the boundary between textual
//! chemical formats and internal, typed molecular structures.
//!
//! Learning focus:
//! - Parser/state-machine techniques in Rust.
//! - Error-aware ingestion pipelines for large datasets.

//! ## File guide: Builder
//!
//! This file belongs to FerrumChem's **input/output parsing layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `trait MoleculeBuilder`
//! - `fn add_atom`
//! - `fn add_bond`
//! - `fn set_charge`
//! - `fn set_isotope`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Builder` treats textual chemistry formats as untrusted input and prefers explicit parse errors over silent repair.
//! - `Builder` must preserve deterministic tokenization so repeated reads produce identical molecule/query graphs.
//! - Format-specific edge cases in `Builder` are validated against toolkit-compatible semantics where possible.
//! - Specific scope anchor: `crates/ferrum_core/src/io/builder.rs` defines builder behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - SMILES/SMARTS/CTfile family specifications relevant to `Builder` parser or reader behavior.
//! - RDKit parsing behavior used as external compatibility reference for `Builder`.
//! - Rust parser/state-machine patterns for reliable streaming and error reporting.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/io/builder.rs`.

use crate::mol::atom::Atom;
use crate::mol::bond::Bond;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use smallvec::SmallVec;

/// Abstraktion für den Aufbau eines Moleküls.
pub trait MoleculeBuilder {
    /// Implements atom.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn add_atom(&mut self, element: u8) -> AtomIdx;
    /// Implements bond.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn add_bond(&mut self, u: AtomIdx, v: AtomIdx, order: BondType);
    /// Sets charge.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_charge(&mut self, atom: AtomIdx, charge: i8);
    /// Sets isotope.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_isotope(&mut self, atom: AtomIdx, iso: u16);
    /// Sets explicit h.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_explicit_h(&mut self, atom: AtomIdx, count: u8);
    /// Sets aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_aromatic(&mut self, atom: AtomIdx, aromatic: bool);
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn finish(self) -> Molecule;
}

// Implementierung direkt auf dem Molecule Struct
impl MoleculeBuilder for Molecule {
    #[inline]
    /// Implements atom.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn add_atom(&mut self, element: u8) -> AtomIdx {
        self.invalidate_cached_properties();
        let idx = self.atoms.len() as AtomIdx;
        self.atoms.push(Atom::new(element));
        self.adjacency.push(SmallVec::new());
        idx
    }

    #[inline]
    /// Implements bond.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn add_bond(&mut self, u: AtomIdx, v: AtomIdx, order: BondType) {
        self.invalidate_cached_properties();
        let bond_idx = self.bonds.len() as u16;
        self.bonds.push(Bond::new(u, v, order));

        // Add to adjacency lists
        self.adjacency[u as usize].push(bond_idx);
        self.adjacency[v as usize].push(bond_idx);
    }

    #[inline]
    /// Sets charge.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_charge(&mut self, atom: AtomIdx, charge: i8) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].formal_charge = charge;
    }

    #[inline]
    /// Sets isotope.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_isotope(&mut self, atom: AtomIdx, iso: u16) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].isotope = iso;
    }

    #[inline]
    /// Sets explicit h.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_explicit_h(&mut self, atom: AtomIdx, count: u8) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].num_explicit_h = count;
    }

    #[inline]
    /// Sets aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn set_aromatic(&mut self, atom: AtomIdx, aromatic: bool) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].set_aromatic(aromatic);
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/io/builder.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn finish(self) -> Molecule {
        self
    }
}
