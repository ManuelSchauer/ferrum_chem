//! # FerrumChem Core (`ferrum_core`)
//!
//! `ferrum_core` is the main library crate for FerrumChem: a pure-Rust
//! cheminformatics and QSAR engine focused on performance, memory efficiency,
//! and reproducibility.
//!
//! This crate is intentionally structured as a learning-friendly stack where
//! each layer has a clear responsibility:
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
//! 5. **Machine learning pipeline** (`ml`)
//!    - scaling, PCA, CV, models, and applicability-domain tooling
//!
//! ## Practical data flow
//!
//! Typical use follows:
//!
//! - parse molecules (`io::smiles::parse_smiles`)
//! - compute descriptors (`desc::Descriptor`)
//! - convert to vectors (`desc::build_ml_vector` / batch APIs)
//! - train/predict (`ml` pipeline modules)
//!
//! ## Why this crate is useful for learning
//!
//! - demonstrates **data-oriented Rust design** for scientific code
//! - shows how cheminformatics concepts map to graph algorithms
//! - keeps APIs strongly typed to reduce hidden runtime errors
//!
//! ## File guide: Ferrum Core Library Root
//!
//! This file is the top-level entry point for the crate and defines the public
//! module boundaries (`algo`, `io`, `mol`, `desc`, `ml`) exposed to downstream
//! users.
//!
//! Key symbols in this file:
//! - `mod algo`
//! - `mod io`
//! - `mod mol`
//! - `mod desc`
//! - `mod ml`
//!
//! Suggested reading workflow:
//! 1. Read this file to understand crate architecture.
//! 2. Follow `mol` and `io` for data model + ingestion.
//! 3. Continue into `algo` and `desc` for chemistry feature logic.
//! 4. Finish with `ml` for end-to-end QSAR modeling flow.

//!
//! ### Algorithmic assumptions for this file
//! - `Lib` coordinates module boundaries and public API exports without introducing side-effectful global state.
//! - Top-level contracts in `Lib` are designed for deterministic behavior across single-thread and Rayon batch contexts.
//! - `Lib` documentation acts as the learning index and should stay synchronized with implementation changes.
//! - Specific scope anchor: `crates/ferrum_core/src/lib.rs` defines lib behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust crate architecture and API-design references informing `Lib` public surface.
//! - FerrumChem module-level docs and integration tests linked from `Lib`.
//! - Cheminformatics/ML pipeline references summarized by `Lib` entry docs.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/lib.rs`.

pub mod algo;
pub mod desc;
pub mod error;
pub mod io;
pub mod ml;
pub mod mol;
pub mod types;

// Convenience re-exports for common entry-point types.
pub use mol::atom::Atom;
pub use mol::molecule::Molecule;
