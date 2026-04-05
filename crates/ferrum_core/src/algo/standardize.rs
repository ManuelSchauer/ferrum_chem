//! Standardize algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.

//! ## File guide: Standardize
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn process`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Standardize` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Standardize` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Standardize` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/standardize.rs` defines standardize behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Standardize` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Standardize` decisions.
//! - FerrumChem regression tests that pin `Standardize` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/standardize.rs`.

use super::{aromaticity, canon, components, rings, valency};
use crate::error::Result;
use crate::mol::molecule::Molecule;

/// Die Pipeline of Truth.
pub fn process(mut mol: Molecule) -> Result<Molecule> {
    // 1. Cleaning
    components::keep_largest_fragment(&mut mol);

    // 2. Topology
    valency::assign_implicit_hydrogens(&mut mol)?;

    // 3. Perception
    rings::find_sssr(&mut mol);
    aromaticity::apply_hueckel(&mut mol);

    // 4. Canonicalization
    let order = canon::rank_atoms(&mol);
    let sorted = canon::reorder_molecule(mol, &order);

    Ok(sorted)
}
