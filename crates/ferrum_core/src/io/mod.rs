//! # Input / output and parsers
//!
//! This module is the boundary between external representations and internal
//! FerrumChem graph types.
//!
//! ## Submodules
//!
//! - `smiles`: SMILES parser with automatic chemistry-prep steps
//! - `smarts`: SMARTS parser + query graph primitives
//! - `csv_reader`: high-throughput CSV ingestion for dataset workflows
//! - `sdf`: SDF support (currently a stub/not fully implemented)
//!
//! ## Learning notes
//!
//! The parsers in this module are useful references for:
//!
//! - state-machine style parsing in Rust
//! - explicit error propagation using crate-level `Result<T, FerrumError>`
//! - converting textual chemistry into adjacency-list molecular graphs
//!
//! ## File guide: I/O Module Registry
//!
//! This file defines parser and ingestion module boundaries.
//!
//! Key symbols in this file:
//! - `mod smiles`
//! - `mod smarts`
//! - `mod csv_reader`
//! - `mod sdf`
//!
//! Suggested reading workflow:
//! 1. `smiles` for molecule construction and sanitization flow.
//! 2. `smarts` for query representation and matching predicates.
//! 3. `csv_reader` for dataset-scale ingestion patterns.
//! 4. `sdf` to see currently deferred surfaces.

//!
//! ### Algorithmic assumptions for this file
//! - `Mod` treats textual chemistry formats as untrusted input and prefers explicit parse errors over silent repair.
//! - `Mod` must preserve deterministic tokenization so repeated reads produce identical molecule/query graphs.
//! - Format-specific edge cases in `Mod` are validated against toolkit-compatible semantics where possible.
//! - Specific scope anchor: `crates/ferrum_core/src/io/mod.rs` defines mod behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - SMILES/SMARTS/CTfile family specifications relevant to `Mod` parser or reader behavior.
//! - RDKit parsing behavior used as external compatibility reference for `Mod`.
//! - Rust parser/state-machine patterns for reliable streaming and error reporting.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/io/mod.rs`.

mod builder;
pub mod csv_reader;
pub mod sdf;
pub mod smarts;
pub mod smiles;
pub mod smiles_writer;
