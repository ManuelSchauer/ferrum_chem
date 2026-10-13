//! Descriptor module: Gasteiger Charge.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Compute Gasteiger partial charges for the molecule.
pub fn compute_charges(mol: &Molecule) -> Vec<f64> {
    crate::algo::electronic::compute_gasteiger_charges(mol)
}

fn compute_charges_cached(mol: &Molecule) -> Vec<f64> {
    crate::algo::electronic::compute_gasteiger_charges_cached(mol)
}

/// RDKit `Descriptors._ChargeDescriptors`: `(minChg, maxChg)`.
///
/// Reproduces Python's `min(chg, cur)` / `max(chg, cur)` semantics exactly,
/// including how NaN charges (elements without Gasteiger parameters)
/// propagate: `min(a, b)` returns `a` unless `b < a`.
pub(crate) fn rdkit_charge_extremes(mol: &Molecule) -> (f64, f64) {
    let mut min_chg = 500.0f64;
    let mut max_chg = -500.0f64;
    for chg in compute_charges_cached(mol) {
        min_chg = if min_chg < chg { min_chg } else { chg };
        max_chg = if max_chg > chg { max_chg } else { chg };
    }
    (min_chg, max_chg)
}

/// Python `min(a, b)`.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Python `max(a, b)`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Maximum partial charge across all atoms.
pub struct MaxPartialCharge;

impl Descriptor for MaxPartialCharge {
    fn name(&self) -> &str {
        "MaxPartialCharge"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let (_, max) = rdkit_charge_extremes(mol);
        DescriptorValue::Float(max)
    }
}

/// Minimum partial charge across all atoms.
pub struct MinPartialCharge;

impl Descriptor for MinPartialCharge {
    fn name(&self) -> &str {
        "MinPartialCharge"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let (min, _) = rdkit_charge_extremes(mol);
        DescriptorValue::Float(min)
    }
}

/// Maximum absolute partial charge across all atoms.
pub struct MaxAbsPartialCharge;

impl Descriptor for MaxAbsPartialCharge {
    fn name(&self) -> &str {
        "MaxAbsPartialCharge"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let (min, max) = rdkit_charge_extremes(mol);
        DescriptorValue::Float(py_max(min.abs(), max.abs()))
    }
}

/// Minimum absolute partial charge across all atoms.
pub struct MinAbsPartialCharge;

impl Descriptor for MinAbsPartialCharge {
    fn name(&self) -> &str {
        "MinAbsPartialCharge"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let (min, max) = rdkit_charge_extremes(mol);
        DescriptorValue::Float(py_min(min.abs(), max.abs()))
    }
}

/// Per-atom Gasteiger partial charges as a vector.
pub struct GasteigerCharges;

impl Descriptor for GasteigerCharges {
    fn name(&self) -> &str {
        "GasteigerCharges"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        DescriptorValue::FloatVec(charges.to_vec())
    }
}

const _: () = {
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

    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).expect("valid SMILES")
    }

    fn calc_float(desc: &dyn Descriptor, mol: &Molecule) -> f64 {
        match desc.calculate(mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => v,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_max_partial_charge_formaldehyde() {
        let mol = parse("C=O");
        let max = calc_float(&MaxPartialCharge, &mol);
        assert!(
            max > 0.0,
            "max charge on C=O should be positive (the C), got {max}"
        );
    }

    #[test]
    fn test_min_partial_charge_formaldehyde() {
        let mol = parse("C=O");
        let min = calc_float(&MinPartialCharge, &mol);
        assert!(
            min < 0.0,
            "min charge on C=O should be negative (the O), got {min}"
        );
    }

    #[test]
    fn test_max_abs_partial_charge() {
        let mol = parse("C=O");
        let max_abs = calc_float(&MaxAbsPartialCharge, &mol);
        let min = calc_float(&MinPartialCharge, &mol);
        // O should have largest |charge|
        assert!(
            (max_abs - min.abs()).abs() < 1e-6 || max_abs >= min.abs(),
            "max abs should be >= |min|"
        );
        assert!(max_abs > 0.0);
    }

    #[test]
    fn test_min_abs_partial_charge() {
        let mol = parse("C=O");
        let min_abs = calc_float(&MinAbsPartialCharge, &mol);
        let max_abs = calc_float(&MaxAbsPartialCharge, &mol);
        assert!(
            min_abs <= max_abs,
            "min abs {min_abs} should be <= max abs {max_abs}"
        );
    }

    #[test]
    fn test_charges_vector_length() {
        let mol = parse("CCO");
        match GasteigerCharges.calculate(
            &mol,
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
    fn test_acetic_acid_charges() {
        // CC(=O)O — carbonyl C positive, oxygens negative
        let mol = parse("CC(=O)O");
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
    fn test_benzene_symmetry() {
        let mol = parse("c1ccccc1");
        let max = calc_float(&MaxPartialCharge, &mol);
        let min = calc_float(&MinPartialCharge, &mol);
        assert!(
            (max - min).abs() < 0.01,
            "benzene should be symmetric: max={max}, min={min}"
        );
    }

    #[test]
    fn test_empty_molecule() {
        // RDKit's _ChargeDescriptors starts from (500, -500) and returns those
        // values unchanged for a molecule without atoms.
        let mol = Molecule::default();
        let calc = |d: &dyn Descriptor| d.calculate(&mol, &mut crate::desc::context::CalculationContext::new());
        assert_eq!(calc(&MaxPartialCharge), DescriptorValue::Float(-500.0));
        assert_eq!(calc(&MinPartialCharge), DescriptorValue::Float(500.0));
        assert_eq!(calc(&MaxAbsPartialCharge), DescriptorValue::Float(500.0));
        assert_eq!(calc(&MinAbsPartialCharge), DescriptorValue::Float(500.0));
    }

    #[test]
    fn test_ethanol_oxygen_negative() {
        let mol = parse("CCO");
        let min = calc_float(&MinPartialCharge, &mol);
        assert!(
            min < 0.0,
            "O in ethanol should give negative min charge, got {min}"
        );
    }
}
