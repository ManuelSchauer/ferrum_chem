//! Lipinski's Rule of Five for drug-likeness prediction.
//!
//! The Rule of Five states that poor absorption or permeation is more likely when:
//! - Molecular weight > 500 Da
//! - LogP > 5
//! - Number of hydrogen bond donors > 5
//! - Number of hydrogen bond acceptors > 10
//!
//! Reference: Lipinski, C. A., et al. (1997). Experimental and computational approaches
//! to estimate solubility and permeability in drug discovery and development settings.
//! Advanced Drug Delivery Reviews, 23(1-3), 3-25.

//! ## File guide: Lipinski
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct LipinskiViolations`
//! - `fn name`
//! - `fn calculate`
//! - `mod tests`
//! - `fn parse_and_prepare`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Lipinski` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Lipinski` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/lipinski.rs` defines lipinski behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Lipinski` workflows.
//! - Repository unit and parity tests covering `Lipinski` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/lipinski.rs`.

use super::hbond::{compute_hbond_acceptors_cached, compute_hbond_donors_cached};
use super::logp::compute_logp_cached;
use super::typed::{Descriptor, DescriptorValue};
use crate::desc::molecular_weight::compute_molecular_weight;
use crate::mol::molecule::Molecule;

/// Lipinski Rule of Five violations count.
///
/// Returns Int(u32) with number of rule violations (0-4).
/// - 0 violations = drug-like
/// - 1-2 violations = borderline
/// - 3-4 violations = not drug-like
///
/// Rules:
/// 1. MW ≤ 500 Da
/// 2. LogP ≤ 5
/// 3. H-bond donors ≤ 5
/// 4. H-bond acceptors ≤ 10
pub struct LipinskiViolations;

impl Descriptor for LipinskiViolations {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "LipinskiViolations"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mw = compute_molecular_weight(mol);
        let logp = compute_logp_cached(mol).0;
        let hbd = compute_hbond_donors_cached(mol);
        let hba = compute_hbond_acceptors_cached(mol);

        let mut violations = 0u32;

        if mw > 500.0 {
            violations += 1;
        }
        if logp > 5.0 {
            violations += 1;
        }
        if hbd > 5 {
            violations += 1;
        }
        if hba > 10 {
            violations += 1;
        }

        DescriptorValue::Int(violations)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<LipinskiViolations>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Parses and prepare.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse_and_prepare(smiles: &str) -> Molecule {
        let mol = parse_smiles(smiles).unwrap();
        mol
    }

    #[test]
    /// Runs regression test for lipinski aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_lipinski_aspirin() {
        let mut mol = parse_and_prepare("CC(=O)Oc1ccccc1C(=O)O");
        let desc = LipinskiViolations;
        let value = desc.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        assert_eq!(
            value,
            DescriptorValue::Int(0),
            "Aspirin should have 0 Lipinski violations"
        );
    }

    #[test]
    /// Runs regression test for lipinski caffeine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_lipinski_caffeine() {
        let mut mol = parse_and_prepare("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        let desc = LipinskiViolations;
        let value = desc.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        assert_eq!(
            value,
            DescriptorValue::Int(0),
            "Caffeine should have 0 Lipinski violations"
        );
    }

    #[test]
    /// Runs regression test for lipinski ibuprofen.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_lipinski_ibuprofen() {
        let mut mol = parse_and_prepare("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
        let desc = LipinskiViolations;
        let value = desc.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        assert_eq!(
            value,
            DescriptorValue::Int(0),
            "Ibuprofen should have 0 Lipinski violations"
        );
    }

    #[test]
    /// Runs regression test for lipinski ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_lipinski_ethanol() {
        let mut mol = parse_and_prepare("CCO");
        let desc = LipinskiViolations;
        let value = desc.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        assert_eq!(
            value,
            DescriptorValue::Int(0),
            "Ethanol should have 0 violations"
        );
    }

    #[test]
    /// Runs regression test for lipinski descriptor name.
    ///
    /// Function context: `crates/ferrum_core/src/desc/lipinski.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_lipinski_descriptor_name() {
        let desc = LipinskiViolations;
        assert_eq!(desc.name(), "LipinskiViolations");
    }
}
