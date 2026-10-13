//! Molecular model module: Bond.
//!
//! This file defines core molecular graph data structures or element metadata
//! used throughout the library. These types are intentionally compact and
//! thread-safe to support high-throughput descriptor and ML workloads.
//!
//! Learning focus:
//! - Data-oriented Rust struct design for scientific computing.
//! - Graph-centric chemistry representations.

//! ## File guide: Bond
//!
//! This file belongs to FerrumChem's **molecular graph model layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct Bond`
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
//! - `Bond` encodes graph primitives with compact integer indices to keep memory predictable on large CSV batches.
//! - `Bond` expects topology invariants (bond endpoints, adjacency symmetry) to hold after all mutating operations.
//! - Bitflag/state fields in `Bond` are treated as cached derivations and must be invalidated after structural edits.
//! - Specific scope anchor: `crates/ferrum_core/src/mol/bond.rs` defines bond behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Graph-based cheminformatics data-model conventions for atom/bond/molecule containers.
//! - IUPAC element/valence background relevant to `Bond` typed fields.
//! - Rust ownership/borrowing patterns for safe high-performance molecular graph manipulation.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/mol/bond.rs`.

use crate::types::{AtomIdx, BondType};

/// Bond representation in a molecular graph.
///
/// # Thread Safety
/// `Bond` is `Copy + Send + Sync`, making it trivially thread-safe.
/// The struct is compact (6 bytes) for efficient parallel processing.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Bond {
    pub source: AtomIdx,
    pub target: AtomIdx,
    pub bond_type: BondType,
    pub flags: u8, // Bit 0: Aromatic, Bit 1: Ring
}

#[cfg(test)]
mod tests {
    use super::Bond;
    use std::mem::{align_of, size_of};

    #[test]
    /// Runs regression test for bond layout size.
    ///
    /// Function context: `crates/ferrum_core/src/mol/bond.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn test_bond_layout_size() {
        assert_eq!(size_of::<Bond>(), 6);
    }

    #[test]
    /// Runs regression test for bond layout alignment.
    ///
    /// Function context: `crates/ferrum_core/src/mol/bond.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn test_bond_layout_alignment() {
        assert_eq!(align_of::<Bond>(), 2);
    }
}

// Static assertion: Bond must be Send + Sync
const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/mol/bond.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Bond>;
};

impl Bond {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/mol/bond.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn new(u: AtomIdx, v: AtomIdx, t: BondType) -> Self {
        Self {
            source: u,
            target: v,
            bond_type: t,
            flags: 0,
        }
    }

    /// Check if this bond is aromatic (bit 0 of flags).
    /// Note: Aromatic bonds may have bond_type Single or Double after kekulization.
    pub fn is_aromatic(&self) -> bool {
        self.flags & 0b01 != 0
    }

    /// Sets aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/mol/bond.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn set_aromatic(&mut self, val: bool) {
        if val {
            self.flags |= 0b01;
        } else {
            self.flags &= !0b01;
        }
    }

    /// Checks whether in ring.
    ///
    /// Function context: `crates/ferrum_core/src/mol/bond.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    /// Bond carries a SMILES direction marker `/` or `\\` (bit 3).
    pub fn has_direction(&self) -> bool {
        self.flags & 0b1000 != 0
    }

    pub fn set_has_direction(&mut self, val: bool) {
        if val {
            self.flags |= 0b1000;
        } else {
            self.flags &= !0b1000;
        }
    }

    /// Conjugation flag as assigned by RDKit's `setConjugation` (bit 2).
    pub fn is_conjugated(&self) -> bool {
        self.flags & 0b100 != 0
    }

    pub fn set_conjugated(&mut self, val: bool) {
        if val {
            self.flags |= 0b100;
        } else {
            self.flags &= !0b100;
        }
    }

    pub fn is_in_ring(&self) -> bool {
        self.flags & 0b10 != 0
    }
    /// Sets in ring.
    ///
    /// Function context: `crates/ferrum_core/src/mol/bond.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    pub fn set_in_ring(&mut self, val: bool) {
        if val {
            self.flags |= 0b10;
        } else {
            self.flags &= !0b10;
        }
    }
}
