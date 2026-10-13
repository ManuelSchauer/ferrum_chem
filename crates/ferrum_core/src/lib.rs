//! # FerrumChem Core (`ferrum_core`)
//!
//! `ferrum_core` is the main library crate for FerrumChem: a pure-Rust
//! cheminformatics engine focused on performance, memory efficiency,
//! and reproducibility.
//!
//! This crate is intentionally structured as a modular cheminformatics stack:
//!
//! 1. **Molecular graph model** (`mol`)
//!    - compact `Atom`, `Bond`, and `Molecule` types
//!    - adjacency-list graph representation tuned for batch workloads
//! 2. **I/O and parsing** (`io`)
//!    - SMILES / SMARTS parsers
//!    - CSV ingestion for large datasets
//! 3. **Core chemistry algorithms** (`algo`)
//!    - aromaticity, ring finding, substructure matching, valence assignment
//! 4. **Descriptor system** (`desc`)
//!    - 70+ descriptor and fingerprint families via a typed API
//!
//! ## Practical data flow
//!
//! Typical use follows:
//!
//! - parse molecules (`io::smiles::parse_smiles`)
//! - compute descriptors (`desc::Descriptor`)
//! - convert to vectors / flat batch matrices (`desc::build_ml_vector` / batch APIs)

pub mod algo;
pub mod desc;
pub mod error;
pub mod io;
pub mod mol;
pub mod types;

// Convenience re-exports for common entry-point types.
pub use mol::atom::Atom;
pub use mol::molecule::Molecule;
