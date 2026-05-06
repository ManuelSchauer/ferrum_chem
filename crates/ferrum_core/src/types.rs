//! ## File guide: Types
//!
//! This file belongs to FerrumChem's **library support layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `type AtomIdx`
//! - `type BondIdx`
//! - `type Real`
//! - `enum BondType`
//! - `enum Hybridization`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Types` coordinates module boundaries and public API exports without introducing side-effectful global state.
//! - Top-level contracts in `Types` are designed for deterministic behavior across single-thread and Rayon batch contexts.
//! - `Types` documentation acts as the learning index and should stay synchronized with implementation changes.
//! - Specific scope anchor: `crates/ferrum_core/src/types.rs` defines types behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust crate architecture and API-design references informing `Types` public surface.
//! - FerrumChem module-level docs and integration tests linked from `Types`.
//! - Cheminformatics/ML pipeline references summarized by `Types` entry docs.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/types.rs`.

/// Index type for atoms (max 65,535 atoms per molecule).
pub type AtomIdx = u16;

/// Index type for bonds.
pub type BondIdx = u16;

/// Floating point type for descriptors (switchable f32/f64).
pub type Real = f32;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
    Dative,
    Unknown,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hybridization {
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
    Unspecified,
}
