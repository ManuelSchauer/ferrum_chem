//! Shared error types for ferrum_core.
//!
//! Centralized error enums keep parser, chemistry, descriptor, and ML failures
//! explicit and composable. This module defines the canonical Result surface
//! used by public APIs.

//! ## File guide: Error
//!
//! This file belongs to FerrumChem's **library support layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `enum FerrumError`
//! - `type Result`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Error` coordinates module boundaries and public API exports without introducing side-effectful global state.
//! - Top-level contracts in `Error` are designed for deterministic behavior across single-thread and Rayon batch contexts.
//! - `Error` documentation acts as the learning index and should stay synchronized with implementation changes.
//! - Specific scope anchor: `crates/ferrum_core/src/error.rs` defines error behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust crate architecture and API-design references informing `Error` public surface.
//! - FerrumChem module-level docs and integration tests linked from `Error`.
//! - Cheminformatics/ML pipeline references summarized by `Error` entry docs.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/error.rs`.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FerrumError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parsing Error: {0}")]
    ParseError(String),

    #[error("Chemistry Error: {0}")]
    ChemError(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Canonical `Result` alias used across ferrum_core.
///
/// Keeping a crate-wide alias makes function signatures concise while ensuring
/// all public APIs converge on `FerrumError`.
pub type Result<T> = std::result::Result<T, FerrumError>;
