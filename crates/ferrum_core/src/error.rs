//! Shared error types for ferrum_core.
//!
//! Centralized error enums keep parser, chemistry, descriptor, and ML failures
//! explicit and composable. This module defines the canonical Result surface
//! used by public APIs.

use thiserror::Error;

#[derive(Error, Debug)]
/// Implements FerrumError.
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
