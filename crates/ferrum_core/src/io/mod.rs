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

mod builder;
pub mod csv_reader;
pub mod sdf;
pub mod smarts;
pub mod smiles;
pub mod smiles_writer;
pub mod stream;
