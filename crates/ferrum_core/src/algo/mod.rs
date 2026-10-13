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

pub mod canon;
pub mod components;
pub mod distance_matrix;
pub mod electronic;
pub mod iso;
pub mod linalg;
pub mod paths;
pub mod rings;
pub mod sanitize;
pub mod sssr;
pub mod stereo;
