//! # Molecular graph data model
//!
//! Core types representing molecular structure.
//!
//! ## Submodules
//!
//! - `atom`: per-atom fields (element, charge, H counts, aromatic/ring flags)
//! - `bond`: bond endpoints, bond type, aromatic/ring flags
//! - `molecule`: owning graph container + adjacency and cached properties
//! - `elements`: periodic-table constants and element helper functions
//!
//! ## Design intent
//!
//! These types are compact, explicit, and thread-safe, which makes them a good
//! study target for data-oriented Rust in scientific/graph workloads.
//!
//! ## File guide: Molecular Model Registry
//!
//! This file is the map for FerrumChem's core graph data structures.
//!
//! Key symbols in this file:
//! - `mod atom`
//! - `mod bond`
//! - `mod molecule`
//! - `mod elements`
//!
//! Suggested reading workflow:
//! 1. Start with `atom` and `bond`.
//! 2. Read `molecule` to understand adjacency and caches.
//! 3. Use `elements` to see periodic table assumptions/constants.

//!
//! ### Algorithmic assumptions for this file
//! - `Mod` encodes graph primitives with compact integer indices to keep memory predictable on large CSV batches.
//! - `Mod` expects topology invariants (bond endpoints, adjacency symmetry) to hold after all mutating operations.
//! - Bitflag/state fields in `Mod` are treated as cached derivations and must be invalidated after structural edits.
//! - Specific scope anchor: `crates/ferrum_core/src/mol/mod.rs` defines mod behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Graph-based cheminformatics data-model conventions for atom/bond/molecule containers.
//! - IUPAC element/valence background relevant to `Mod` typed fields.
//! - Rust ownership/borrowing patterns for safe high-performance molecular graph manipulation.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/mol/mod.rs`.

pub mod atom;
pub mod bond;
pub mod elements;
pub mod molecule;
