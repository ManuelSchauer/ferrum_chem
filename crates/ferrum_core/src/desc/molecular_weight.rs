//! Descriptor module: Molecular Weight.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.


//! ## File guide: Molecular Weight
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn compute_molecular_weight`
//! - `fn compute_molecular_weight_f64`
//! - `struct MolecularWeight`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Molecular Weight` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Molecular Weight` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/molecular_weight.rs` defines molecular weight behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Molecular Weight` workflows.
//! - Repository unit and parity tests covering `Molecular Weight` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/molecular_weight.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::elements::ATOMIC_MASSES;
use crate::mol::molecule::Molecule;

/// Compute the molecular weight for a molecule.
///
/// Standalone function for callers that need the raw `f32` value
/// without going through the [`Descriptor`] trait.
pub fn compute_molecular_weight(mol: &Molecule) -> f32 {
    compute_molecular_weight_f64(mol) as f32
}

/// High-precision molecular weight calculation.
///
/// Uses `f64` accumulation to avoid drift on large molecules where `f32`
/// summation can exceed tight comparison tolerances.
pub fn compute_molecular_weight_f64(mol: &Molecule) -> f64 {
    let mut total_mass: f64 = 0.0;

    for atom in &mol.atoms {
        let mass = if atom.isotope > 0 {
            atom.isotope as f64
        } else {
            ATOMIC_MASSES[atom.atomic_number as usize] as f64
        };

        total_mass += mass;

        if atom.num_explicit_h > 0 {
            total_mass += (ATOMIC_MASSES[1] as f64) * (atom.num_explicit_h as f64);
        }

        if atom.num_implicit_h > 0 {
            total_mass += (ATOMIC_MASSES[1] as f64) * (atom.num_implicit_h as f64);
        }
    }

    total_mass
}

/// Molecular weight descriptor.
///
/// Returns `DescriptorValue::Float(f32)`.
///
/// # Example
///
/// ```
/// use ferrum_core::desc::molecular_weight::MolecularWeight;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mut mol = parse_smiles("CCO").unwrap();
/// let desc = MolecularWeight;
/// let value = desc.calculate(&mut mol, &mut ferrum_core::desc::context::CalculationContext::new());
///
/// match value {
///     ferrum_core::desc::typed::DescriptorValue::Float64(mw) => {
///         assert!((mw - 46.07).abs() < 0.1); // Ethanol ≈ 46 Da
///     }
///     _ => panic!("Expected Float64"),
/// }
/// ```
pub struct MolecularWeight;

impl Descriptor for MolecularWeight {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "MolecularWeight"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Float64(compute_molecular_weight_f64(mol))
    }
}

// Thread safety
const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MolecularWeight>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for molecular weight water.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_molecular_weight_water() {
        // Water: H2O = 2*1.008 + 15.999 = 18.015
        let mut mol = parse_smiles("O").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => assert!(((v as f64) - 18.015).abs() < 0.01),
            DescriptorValue::Float64(v) => assert!((v - 18.015).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    /// Runs regression test for molecular weight methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_molecular_weight_methane() {
        // Methane: CH4
        let mut mol = parse_smiles("C").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => assert!(((v as f64) - 16.043).abs() < 0.01),
            DescriptorValue::Float64(v) => assert!((v - 16.043).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    /// Runs regression test for molecular weight methane explicit h.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_molecular_weight_methane_explicit_h() {
        // Methane with explicit H: [CH4]
        let mut mol = parse_smiles("[CH4]").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => assert!(((v as f64) - 16.043).abs() < 0.01),
            DescriptorValue::Float64(v) => assert!((v - 16.043).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    /// Runs regression test for molecular weight benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_molecular_weight_benzene() {
        // Benzene: C6H6
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => assert!(((v as f64) - 78.114).abs() < 0.01),
            DescriptorValue::Float64(v) => assert!((v - 78.114).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    /// Runs regression test for molecular weight ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_molecular_weight_ethanol() {
        // Ethanol: C2H6O = 2*12.011 + 6*1.008 + 15.999 = 46.069
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => assert!(((v as f64) - 46.069).abs() < 0.01),
            DescriptorValue::Float64(v) => assert!((v - 46.069).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    /// Runs regression test for molecular weight isotope.
    ///
    /// Function context: `crates/ferrum_core/src/desc/molecular_weight.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_molecular_weight_isotope() {
        // Heavy water: [2H]O[2H]
        let mut mol = parse_smiles("[2H]O[2H]").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => assert!(((v as f64) - 19.999).abs() < 0.01),
            DescriptorValue::Float64(v) => assert!((v - 19.999).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }
}
