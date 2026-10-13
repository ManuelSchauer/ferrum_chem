//! Descriptor module: Polarizability.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Polarizability
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `POLARIZABILITY` (CDK APol/BPol table)
//! - `struct APol`
//! - `fn name`
//! - `fn calculate`
//! - `struct BPol`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Polarizability` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Polarizability` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/polarizability.rs` defines polarizability behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Polarizability` workflows.
//! - Repository unit and parity tests covering `Polarizability` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/polarizability.rs`.

use super::Descriptor;
use super::cdk_perception::CdkView;
use crate::desc::DescriptorValue;
use crate::mol::molecule::Molecule;

/// Atomic polarizabilities (Å^3) indexed by atomic number (0..=102), exactly
/// as in CDK's `APolDescriptor` / `BPolDescriptor`.
const POLARIZABILITY: [f64; 103] = [
    0.0, 0.666793, 0.204956, 24.3, 5.6, 3.03, 1.76, 1.1, 0.802, 0.557, 0.3956, 23.6, 10.6, 6.8,
    5.38, 3.63, 2.9, 2.18, 1.6411, 43.4, 22.8, 17.8, 14.6, 12.4, 11.6, 9.4, 8.4, 7.5, 6.8, 6.1,
    7.1, 8.12, 6.07, 4.31, 3.77, 3.05, 2.4844, 47.3, 27.6, 22.7, 17.9, 15.7, 12.8, 11.4, 9.6, 8.6,
    4.8, 7.2, 7.2, 10.2, 7.7, 6.6, 5.5, 5.35, 4.044, 59.6, 39.7, 31.1, 29.6, 28.2, 31.4, 30.1,
    28.8, 27.7, 23.5, 25.5, 24.5, 23.6, 22.7, 21.8, 21.0, 21.9, 16.2, 13.1, 11.1, 9.7, 8.5, 7.6,
    6.5, 5.8, 5.7, 7.6, 6.8, 7.4, 6.8, 6.0, 5.3, 48.7, 38.3, 32.1, 32.1, 25.4, 27.4, 24.8, 24.5,
    23.3, 23.0, 22.7, 20.5, 19.7, 23.8, 18.2, 17.5,
];

/// Polarizability of an element; NaN where CDK has no value (CDK then reports an error).
#[inline]
fn atom_pol(atomic_number: u8) -> f64 {
    match atomic_number {
        1..=102 => POLARIZABILITY[atomic_number as usize],
        _ => f64::NAN,
    }
}

/// APol (CDK `APolDescriptor`): sum of atomic polarizabilities including the
/// implicit hydrogens CDK assigns (`CDKHydrogenAdder` after atom typing).
pub struct APol;

impl Descriptor for APol {
    fn name(&self) -> &str {
        "APol"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let h_pol = POLARIZABILITY[1];
        let sum = CdkView::with_legacy_cached(mol, ctx, |view, _| {
            let mut sum = 0.0f64;
            for (atom, &nh) in mol.atoms.iter().zip(&view.h) {
                sum += atom_pol(atom.atomic_number);
                sum += h_pol * nh as f64;
            }
            sum
        });
        DescriptorValue::Float(sum)
    }
}

/// BPol (CDK `BPolDescriptor`): sum over bonds (including the bonds to
/// implicit hydrogens) of the absolute polarizability difference of the atoms.
pub struct BPol;

impl Descriptor for BPol {
    fn name(&self) -> &str {
        "BPol"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let h = CdkView::with_legacy_cached(mol, ctx, |view, _| view.h.clone());
        let mut sum = 0.0f64;
        for bond in &mol.bonds {
            let p_src = atom_pol(mol.atoms[bond.source as usize].atomic_number);
            let p_tgt = atom_pol(mol.atoms[bond.target as usize].atomic_number);
            sum += (p_src - p_tgt).abs();
        }
        let h_pol = POLARIZABILITY[1];
        for (atom, &nh) in mol.atoms.iter().zip(&h) {
            sum += (atom_pol(atom.atomic_number) - h_pol).abs() * nh as f64;
        }
        DescriptorValue::Float(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc(desc: &dyn Descriptor, smiles: &str) -> f64 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match desc.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Float(v) => v,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    // Expected values: CDK 2.13 (tests/data/cdk_parity_reference.csv).

    #[test]
    fn test_apol_methane() {
        // C 1.76 + 4 H 0.666793
        assert!((calc(&APol, "C") - 4.427172).abs() < 1e-9);
    }

    #[test]
    fn test_apol_benzene() {
        let expected = 6.0 * 1.76 + 6.0 * 0.666793;
        assert!((calc(&APol, "c1ccccc1") - expected).abs() < 1e-9);
    }

    #[test]
    fn test_bpol_ethanol() {
        // C-C 0, C-O 0.958, 5 C-H, 1 O-H
        let expected = 0.958 + 5.0 * (1.76 - 0.666793) + (0.802 - 0.666793f64).abs();
        assert!((calc(&BPol, "CCO") - expected).abs() < 1e-9);
    }

    #[test]
    fn test_bpol_methane_matches_cdk() {
        assert!((calc(&BPol, "[H]C([H])([H])[H]") - 4.372828).abs() < 1e-9);
    }

    #[test]
    fn test_salt() {
        // [Na+].[Cl-]: 23.6 + 2.18
        assert!((calc(&APol, "[Na+].[Cl-]") - 25.78).abs() < 1e-9);
        assert_eq!(calc(&BPol, "[Na+].[Cl-]"), 0.0);
    }

    #[test]
    fn test_empty_molecule() {
        let mut mol = Molecule::default();
        let mut ctx = crate::desc::context::CalculationContext::new();
        assert!(
            matches!(APol.calculate(&mut mol, &mut ctx), DescriptorValue::Float(v) if v == 0.0)
        );
        assert!(
            matches!(BPol.calculate(&mut mol, &mut ctx), DescriptorValue::Float(v) if v == 0.0)
        );
    }
}
