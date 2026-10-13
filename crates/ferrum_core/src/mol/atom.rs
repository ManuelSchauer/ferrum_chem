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
/// The struct is designed to be compact (8 bytes) and cache-friendly
/// for high-performance parallel processing.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Atom {
    pub atomic_number: u8,
    pub formal_charge: i8,
    pub num_explicit_h: u8,
    pub num_implicit_h: u8, // Calculated implicit hydrogens
    pub isotope: u16,
    pub hybridization: Hybridization,
    pub flags: u8, // Bit 0: aromatic, 1: ring, 2: no implicit H, 3-5: radicals, 6-7: chiral tag
}

#[cfg(test)]
mod tests {
    use super::Atom;
    use std::mem::{align_of, size_of};

    #[test]
    /// Runs regression test for atom layout size.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn test_atom_layout_size() {
        assert_eq!(size_of::<Atom>(), 8);
    }

    #[test]
    /// Runs regression test for atom layout alignment.
    ///
    /// Function context: `crates/ferrum_core/src/mol/atom.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn test_atom_layout_alignment() {
        assert_eq!(align_of::<Atom>(), 2);
    }
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
    /// Number of radical electrons (stored in flag bits 3..=5).
    pub fn num_radicals(&self) -> u8 {
        (self.flags >> 3) & 0b111
    }

    pub fn set_num_radicals(&mut self, n: u8) {
        self.flags = (self.flags & !0b0011_1000) | ((n.min(7)) << 3);
    }

    pub fn total_h(&self) -> u8 {
        self.num_explicit_h + self.num_implicit_h
    }

    /// Tetrahedral chirality tag (stored in flag bits 6..=7).
    ///
    /// The tag is relative to the order of the atom's bonds in
    /// `Molecule::adjacency`, with an implicit H (or lone pair) taken to be
    /// the last neighbour - the same convention RDKit uses for its bond order.
    pub fn chiral_tag(&self) -> ChiralTag {
        match (self.flags >> 6) & 0b11 {
            1 => ChiralTag::TetrahedralCW,
            2 => ChiralTag::TetrahedralCCW,
            3 => ChiralTag::Other,
            _ => ChiralTag::Unspecified,
        }
    }

    /// Sets the chirality tag (flag bits 6..=7).
    pub fn set_chiral_tag(&mut self, tag: ChiralTag) {
        let bits = match tag {
            ChiralTag::Unspecified => 0,
            ChiralTag::TetrahedralCW => 1,
            ChiralTag::TetrahedralCCW => 2,
            ChiralTag::Other => 3,
        };
        self.flags = (self.flags & 0b0011_1111) | (bits << 6);
    }
}

/// Atom chirality tag (RDKit `Atom::ChiralType`, reduced to what SMILES
/// input produces). `Other` covers `@AL`, `@SP`, `@TB` and `@OH` classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChiralTag {
    Unspecified,
    /// `@@` (RDKit `CHI_TETRAHEDRAL_CW`).
    TetrahedralCW,
    /// `@` (RDKit `CHI_TETRAHEDRAL_CCW`).
    TetrahedralCCW,
    Other,
}

impl ChiralTag {
    /// `true` for CW/CCW tetrahedral tags.
    pub fn is_tetrahedral(self) -> bool {
        matches!(self, ChiralTag::TetrahedralCW | ChiralTag::TetrahedralCCW)
    }

    /// The opposite tetrahedral tag (other tags are returned unchanged).
    pub fn inverted(self) -> ChiralTag {
        match self {
            ChiralTag::TetrahedralCW => ChiralTag::TetrahedralCCW,
            ChiralTag::TetrahedralCCW => ChiralTag::TetrahedralCW,
            t => t,
        }
    }
}
