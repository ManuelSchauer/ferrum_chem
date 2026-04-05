//! Descriptor module: Gasteiger Charge.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Gasteiger Charge
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn compute_charges`
//! - `fn compute_charges_cached`
//! - `struct MaxPartialCharge`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Gasteiger Charge` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Gasteiger Charge` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/gasteiger_charge.rs` defines gasteiger charge behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Gasteiger Charge` workflows.
//! - Repository unit and parity tests covering `Gasteiger Charge` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/gasteiger_charge.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Compute Gasteiger partial charges for the molecule.
pub fn compute_charges(mol: &Molecule) -> Vec<f64> {
    crate::algo::electronic::compute_gasteiger_charges(mol)
}

/// Computes charges cached.
///
/// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_charges_cached(mol: &mut Molecule) -> Vec<f64> {
    crate::algo::electronic::compute_gasteiger_charges_cached(mol)
}

/// Maximum partial charge across all atoms.
pub struct MaxPartialCharge;

impl Descriptor for MaxPartialCharge {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "MaxPartialCharge"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let max = charges.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        DescriptorValue::Float(if max.is_finite() { max as f32 } else { 0.0 })
    }
}

/// Minimum partial charge across all atoms.
pub struct MinPartialCharge;

impl Descriptor for MinPartialCharge {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "MinPartialCharge"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let min = charges.iter().copied().fold(f64::INFINITY, f64::min);
        DescriptorValue::Float(if min.is_finite() { min as f32 } else { 0.0 })
    }
}

/// Maximum absolute partial charge across all atoms.
pub struct MaxAbsPartialCharge;

impl Descriptor for MaxAbsPartialCharge {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "MaxAbsPartialCharge"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let max_abs = charges
            .iter()
            .map(|c| c.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        DescriptorValue::Float(if max_abs.is_finite() {
            max_abs as f32
        } else {
            0.0
        })
    }
}

/// Minimum absolute partial charge across all atoms.
pub struct MinAbsPartialCharge;

impl Descriptor for MinAbsPartialCharge {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "MinAbsPartialCharge"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let min_abs = charges
            .iter()
            .map(|c| c.abs())
            .fold(f64::INFINITY, f64::min);
        DescriptorValue::Float(if min_abs.is_finite() {
            min_abs as f32
        } else {
            0.0
        })
    }
}

/// Per-atom Gasteiger partial charges as a vector.
pub struct GasteigerCharges;

impl Descriptor for GasteigerCharges {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "GasteigerCharges"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        DescriptorValue::FloatVec(charges.iter().map(|&c| c as f32).collect())
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MaxPartialCharge>;
    let _ = assert_send_sync::<MinPartialCharge>;
    let _ = assert_send_sync::<MaxAbsPartialCharge>;
    let _ = assert_send_sync::<MinAbsPartialCharge>;
    let _ = assert_send_sync::<GasteigerCharges>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).expect("valid SMILES")
    }

    /// Implements float.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_float(desc: &dyn Descriptor, mol: &mut Molecule) -> f32 {
        match desc.calculate(mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => v,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for max partial charge formaldehyde.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_max_partial_charge_formaldehyde() {
        let mut mol = parse("C=O");
        let max = calc_float(&MaxPartialCharge, &mut mol);
        assert!(
            max > 0.0,
            "max charge on C=O should be positive (the C), got {max}"
        );
    }

    #[test]
    /// Runs regression test for min partial charge formaldehyde.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_min_partial_charge_formaldehyde() {
        let mut mol = parse("C=O");
        let min = calc_float(&MinPartialCharge, &mut mol);
        assert!(
            min < 0.0,
            "min charge on C=O should be negative (the O), got {min}"
        );
    }

    #[test]
    /// Runs regression test for max abs partial charge.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_max_abs_partial_charge() {
        let mut mol = parse("C=O");
        let max_abs = calc_float(&MaxAbsPartialCharge, &mut mol);
        let min = calc_float(&MinPartialCharge, &mut mol);
        // O should have largest |charge|
        assert!(
            (max_abs - min.abs()).abs() < 1e-6 || max_abs >= min.abs(),
            "max abs should be >= |min|"
        );
        assert!(max_abs > 0.0);
    }

    #[test]
    /// Runs regression test for min abs partial charge.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_min_abs_partial_charge() {
        let mut mol = parse("C=O");
        let min_abs = calc_float(&MinAbsPartialCharge, &mut mol);
        let max_abs = calc_float(&MaxAbsPartialCharge, &mut mol);
        assert!(
            min_abs <= max_abs,
            "min abs {min_abs} should be <= max abs {max_abs}"
        );
    }

    #[test]
    /// Runs regression test for charges vector length.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_charges_vector_length() {
        let mut mol = parse("CCO");
        match GasteigerCharges.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => {
                assert_eq!(
                    v.len(),
                    mol.atoms.len(),
                    "vector length should equal num_atoms"
                );
            }
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for acetic acid charges.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acetic_acid_charges() {
        // CC(=O)O — carbonyl C positive, oxygens negative
        let mut mol = parse("CC(=O)O");
        let charges = compute_charges(&mol);
        // Atom 1 is the carbonyl carbon (bonded to both O atoms)
        assert!(
            charges[1] > 0.0,
            "carbonyl C should be positive, got {}",
            charges[1]
        );
        // Atom 2 is =O, Atom 3 is -OH — both should be negative
        assert!(
            charges[2] < 0.0,
            "=O should be negative, got {}",
            charges[2]
        );
        assert!(
            charges[3] < 0.0,
            "-OH oxygen should be negative, got {}",
            charges[3]
        );
    }

    #[test]
    /// Runs regression test for benzene symmetry.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_benzene_symmetry() {
        let mut mol = parse("c1ccccc1");
        let max = calc_float(&MaxPartialCharge, &mut mol);
        let min = calc_float(&MinPartialCharge, &mut mol);
        assert!(
            (max - min).abs() < 0.01,
            "benzene should be symmetric: max={max}, min={min}"
        );
    }

    #[test]
    /// Runs regression test for empty molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_empty_molecule() {
        let mut mol = Molecule::default();
        assert_eq!(
            MaxPartialCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            MinPartialCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            MaxAbsPartialCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            MinAbsPartialCharge.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            GasteigerCharges.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::FloatVec(vec![])
        );
    }

    #[test]
    /// Runs regression test for ethanol oxygen negative.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gasteiger_charge.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ethanol_oxygen_negative() {
        let mut mol = parse("CCO");
        let min = calc_float(&MinPartialCharge, &mut mol);
        assert!(
            min < 0.0,
            "O in ethanol should give negative min charge, got {min}"
        );
    }
}
