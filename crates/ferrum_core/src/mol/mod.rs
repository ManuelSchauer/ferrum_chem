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

pub mod atom;
pub mod bond;
pub mod conformer;
pub mod elements;
pub mod molecule;
pub mod rdkit_data;
