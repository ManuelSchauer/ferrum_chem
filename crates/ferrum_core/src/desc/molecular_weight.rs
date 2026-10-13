//! Descriptor module: Molecular Weight.
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

/// Compute the molecular weight for a molecule.
///
/// Standalone function for callers that need the raw `f64` value
/// without going through the [`Descriptor`] trait.
pub fn compute_molecular_weight(mol: &Molecule) -> f64 {
    compute_molecular_weight_f64(mol)
}

/// High-precision molecular weight calculation.
///
/// Uses `f64` accumulation to avoid drift on large molecules where `f64`
/// summation can exceed tight comparison tolerances.
pub fn compute_molecular_weight_f64(mol: &Molecule) -> f64 {
    // RDKit calcAMW: sum of Atom::getMass() plus getTotalNumHs() * H weight.
    let hydrogen_mass = crate::mol::rdkit_data::ELEMENTS[1].mass;
    mol.atoms
        .iter()
        .map(|atom| {
            crate::mol::rdkit_data::atom_mass(atom.atomic_number, atom.isotope)
                + atom.total_h() as f64 * hydrogen_mass
        })
        .sum()
}

/// Molecular weight descriptor.
///
/// Returns `DescriptorValue::Float(f64)`.
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
///     ferrum_core::desc::typed::DescriptorValue::Float(mw) => {
///         assert!((mw - 46.07).abs() < 0.1); // Ethanol ≈ 46 Da
///     }
///     _ => panic!("Expected Float"),
/// }
/// ```
pub struct MolecularWeight;

impl Descriptor for MolecularWeight {
    fn name(&self) -> &str {
        "MolecularWeight"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Float(compute_molecular_weight_f64(mol))
    }
}

// Thread safety
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MolecularWeight>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_molecular_weight_water() {
        // Water: H2O = 2*1.008 + 15.999 = 18.015
        let mol = parse_smiles("O").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Float(v) => assert!((v - 18.015).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_methane() {
        // Methane: CH4
        let mol = parse_smiles("C").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Float(v) => assert!((v - 16.043).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_methane_explicit_h() {
        // Methane with explicit H: [CH4]
        let mol = parse_smiles("[CH4]").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Float(v) => assert!((v - 16.043).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_benzene() {
        // Benzene: C6H6
        let mol = parse_smiles("c1ccccc1").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Float(v) => assert!((v - 78.114).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_ethanol() {
        // Ethanol: C2H6O = 2*12.011 + 6*1.008 + 15.999 = 46.069
        let mol = parse_smiles("CCO").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Float(v) => assert!((v - 46.069).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_isotope() {
        // Heavy water: [2H]O[2H]
        let mol = parse_smiles("[2H]O[2H]").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            // RDKit: 15.999 + 2 * 2.014101778 (exact deuterium mass)
            DescriptorValue::Float(v) => assert!((v - 20.027203556).abs() < 1e-9),
            _ => panic!("Expected Float"),
        }
    }
}
