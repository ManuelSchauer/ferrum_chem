//! Descriptor module: Zagreb.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.


//! ## File guide: Zagreb
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct ZagrebIndices`
//! - `fn name`
//! - `fn calculate`
//! - `mod tests`
//! - `fn calculate_zagreb`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Zagreb` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Zagreb` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/zagreb.rs` defines zagreb behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Zagreb` workflows.
//! - Repository unit and parity tests covering `Zagreb` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/zagreb.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Zagreb Indices: Degree-based topological descriptors.
///
/// The Zagreb indices are among the oldest and most studied molecular descriptors,
/// introduced by Gutman and Trinajstić in 1972. They are based on vertex degrees
/// in the molecular graph.
///
/// # Indices
/// - **M1 (First Zagreb Index)**: Sum of squared degrees — M1 = Σᵢ deg(i)²
/// - **M2 (Second Zagreb Index)**: Sum of degree products over edges — M2 = Σ(i,j)∈E deg(i) × deg(j)
///
/// # Properties
/// - Correlate with molecular branching
/// - Higher values indicate more branched structures
///
/// Returns FloatVec with 2 values: [M1, M2]
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct ZagrebIndices;

impl Descriptor for ZagrebIndices {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "ZagrebIndices"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let n = mol.num_atoms();

        if n == 0 {
            return DescriptorValue::FloatVec(vec![0.0, 0.0]);
        }

        let mut m1: u64 = 0;
        for atom_idx in 0..n {
            let degree = mol.get_degree(atom_idx as u16);
            m1 += (degree * degree) as u64;
        }

        let mut m2: u64 = 0;
        for bond in &mol.bonds {
            let deg_u = mol.get_degree(bond.source);
            let deg_v = mol.get_degree(bond.target);
            m2 += (deg_u * deg_v) as u64;
        }

        DescriptorValue::FloatVec(vec![m1 as f32, m2 as f32])
    }
}

// Static assertion: ZagrebIndices must be Send + Sync
const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ZagrebIndices>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Calculates zagreb.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate_zagreb(smiles: &str) -> (f32, f32) {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        let v = ZagrebIndices.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_float_vec().to_vec();
        (v[0], v[1])
    }

    #[test]
    /// Runs regression test for zagreb methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_methane() {
        let (m1, m2) = calculate_zagreb("C");
        assert_eq!(m1, 0.0);
        assert_eq!(m2, 0.0);
    }

    #[test]
    /// Runs regression test for zagreb ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_ethane() {
        let (m1, m2) = calculate_zagreb("CC");
        assert_eq!(m1, 2.0);
        assert_eq!(m2, 1.0);
    }

    #[test]
    /// Runs regression test for zagreb propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_propane() {
        let (m1, m2) = calculate_zagreb("CCC");
        assert_eq!(m1, 6.0);
        assert_eq!(m2, 4.0);
    }

    #[test]
    /// Runs regression test for zagreb isobutane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_isobutane() {
        let (m1, m2) = calculate_zagreb("CC(C)C");
        assert_eq!(m1, 12.0);
        assert_eq!(m2, 9.0);
    }

    #[test]
    /// Runs regression test for zagreb butane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_butane() {
        let (m1, m2) = calculate_zagreb("CCCC");
        assert_eq!(m1, 10.0);
        assert_eq!(m2, 8.0);
    }

    #[test]
    /// Runs regression test for zagreb benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_benzene() {
        let (m1, m2) = calculate_zagreb("c1ccccc1");
        assert_eq!(m1, 24.0);
        assert_eq!(m2, 24.0);
    }

    #[test]
    /// Runs regression test for zagreb cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_cyclohexane() {
        let (m1, m2) = calculate_zagreb("C1CCCCC1");
        assert_eq!(m1, 24.0);
        assert_eq!(m2, 24.0);
    }

    #[test]
    /// Runs regression test for zagreb naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_naphthalene() {
        let (m1, m2) = calculate_zagreb("c1ccc2ccccc2c1");
        assert!(m1 > 24.0, "Naphthalene M1 should be > benzene");
        assert!(m2 > 24.0, "Naphthalene M2 should be > benzene");
    }

    #[test]
    /// Runs regression test for zagreb branched vs linear.
    ///
    /// Function context: `crates/ferrum_core/src/desc/zagreb.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_zagreb_branched_vs_linear() {
        let (m1_linear, _) = calculate_zagreb("CCCC");
        let (m1_branched, _) = calculate_zagreb("CC(C)C");
        assert!(m1_branched > m1_linear);
    }
}
