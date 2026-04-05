//! I/O module: Sdf.
//!
//! This module handles external data interchange (SMILES/SMARTS/CSV/SDF) and
//! conversion into FerrumChem graph types. It is the boundary between textual
//! chemical formats and internal, typed molecular structures.
//!
//! Learning focus:
//! - Parser/state-machine techniques in Rust.
//! - Error-aware ingestion pipelines for large datasets.

//! ## File guide: Sdf
//!
//! This file belongs to FerrumChem's **input/output parsing layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn parse_sdf`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Sdf` treats textual chemistry formats as untrusted input and prefers explicit parse errors over silent repair.
//! - `Sdf` must preserve deterministic tokenization so repeated reads produce identical molecule/query graphs.
//! - Format-specific edge cases in `Sdf` are validated against toolkit-compatible semantics where possible.
//! - Specific scope anchor: `crates/ferrum_core/src/io/sdf.rs` defines sdf behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - SMILES/SMARTS/CTfile family specifications relevant to `Sdf` parser or reader behavior.
//! - RDKit parsing behavior used as external compatibility reference for `Sdf`.
//! - Rust parser/state-machine patterns for reliable streaming and error reporting.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/io/sdf.rs`.

use crate::error::Result;
use crate::mol::molecule::Molecule;
use std::path::Path;

/// Parst eine SDF Datei und gibt einen Vektor von Molekülen zurück.
pub fn parse_sdf(path: &Path) -> Result<Vec<Molecule>> {
    // Hier V2000 Block Parsing implementieren
    todo!("Implement SDF parser")
}
