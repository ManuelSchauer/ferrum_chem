//! # Core chemistry algorithms
//!
//! This module groups the low-level graph and chemistry algorithms used
//! throughout FerrumChem.
//!
//! ## What lives here
//!
//! - `aromaticity`: aromaticity perception
//! - `rings`: cycle/ring detection and cached ring helpers
//! - `iso`: substructure matching (SMARTS/VF2-style matching logic)
//! - `valency`: implicit-H assignment and valence-related utilities
//! - `canon`: canonicalization support routines
//! - `components`, `paths`: graph utility traversals
//! - `standardize`: molecule normalization pipeline entry points
//!
//! ## Learning notes
//!
//! Read these modules to understand how cheminformatics descriptors are built on
//! top of graph algorithms: most descriptor code depends on these routines for
//! ring membership, aromatic flags, connectivity, and matching behavior.
//!
//! ## File guide: Algorithm Module Registry
//!
//! This file is the module map for chemistry algorithms. It does not implement
//! chemistry itself, but it defines where each algorithm family lives.
//!
//! Key symbols in this file:
//! - `mod aromaticity`
//! - `mod rings`
//! - `mod iso`
//! - `mod valency`
//! - `mod standardize`
//!
//! Suggested reading workflow:
//! 1. Start with `valency` and `rings`.
//! 2. Continue to `aromaticity`.
//! 3. Read `iso` for SMARTS matching internals.
//! 4. End with `standardize` to see orchestration of these primitives.

//!
//! ### Algorithmic assumptions for this file
//! - `Mod` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Mod` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Mod` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/mod.rs` defines mod behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Mod` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Mod` decisions.
//! - FerrumChem regression tests that pin `Mod` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/mod.rs`.

pub mod aromaticity;
pub mod canon;
pub mod components;
pub mod distance_matrix;
pub mod electronic;
pub mod iso;
pub mod kekulize;
pub mod paths;
pub mod rings;
pub mod standardize;
pub mod valency;
