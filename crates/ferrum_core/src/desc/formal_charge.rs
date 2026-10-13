//! Descriptor module: Formal Charge.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Formal Charge
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct FormalCharge`
//! - `fn name`
//! - `fn calculate`
//! - `mod tests`
//! - `fn test_formal_charge_neutral`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Formal Charge` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Formal Charge` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/formal_charge.rs` defines formal charge behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Formal Charge` workflows.
//! - Repository unit and parity tests covering `Formal Charge` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/formal_charge.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Calculates the total formal charge of the molecule.
///
/// The formal charge is the sum of all individual atom formal charges.
/// Most neutral molecules have a formal charge of 0.
pub struct FormalCharge;

impl Descriptor for FormalCharge {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "FormalCharge"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let total: i32 = mol.atoms.iter().map(|atom| atom.formal_charge as i32).sum();
        DescriptorValue::Float(total as f64)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FormalCharge>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for formal charge neutral.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_formal_charge_neutral() {
        let mut mol = parse_smiles("CCO").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
    }

    #[test]
    /// Runs regression test for formal charge positive.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_formal_charge_positive() {
        let mut mol = parse_smiles("[NH4+]").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(1.0)
        );
    }

    #[test]
    /// Runs regression test for formal charge negative.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_formal_charge_negative() {
        let mut mol = parse_smiles("[OH-]").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(-1.0)
        );
    }

    #[test]
    /// Runs regression test for formal charge doubly charged.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_formal_charge_doubly_charged() {
        let mut mol = parse_smiles("[O-2]").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(-2.0)
        );
    }

    #[test]
    /// Runs regression test for formal charge benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_formal_charge_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
    }

    #[test]
    /// Runs regression test for formal charge carboxylate.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_formal_charge_carboxylate() {
        let mut mol = parse_smiles("CC([O-])=O").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(-1.0)
        );
    }

    #[test]
    /// Runs regression test for formal charge zwitterion.
    ///
    /// Function context: `crates/ferrum_core/src/desc/formal_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_formal_charge_zwitterion() {
        let mut mol = parse_smiles("[NH3+]CC([O-])=O").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
    }
}
