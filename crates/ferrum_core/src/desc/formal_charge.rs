//! Descriptor module: Formal Charge.
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

/// Calculates the total formal charge of the molecule.
///
/// The formal charge is the sum of all individual atom formal charges.
/// Most neutral molecules have a formal charge of 0.
pub struct FormalCharge;

impl Descriptor for FormalCharge {
    fn name(&self) -> &str {
        "FormalCharge"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let total: i32 = mol.atoms.iter().map(|atom| atom.formal_charge as i32).sum();
        DescriptorValue::Float(total as f64)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FormalCharge>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_formal_charge_neutral() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
    }

    #[test]
    fn test_formal_charge_positive() {
        let mol = parse_smiles("[NH4+]").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(1.0)
        );
    }

    #[test]
    fn test_formal_charge_negative() {
        let mol = parse_smiles("[OH-]").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(-1.0)
        );
    }

    #[test]
    fn test_formal_charge_doubly_charged() {
        let mol = parse_smiles("[O-2]").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(-2.0)
        );
    }

    #[test]
    fn test_formal_charge_benzene() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
    }

    #[test]
    fn test_formal_charge_carboxylate() {
        let mol = parse_smiles("CC([O-])=O").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(-1.0)
        );
    }

    #[test]
    fn test_formal_charge_zwitterion() {
        let mol = parse_smiles("[NH3+]CC([O-])=O").unwrap();
        assert_eq!(
            FormalCharge.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Float(0.0)
        );
    }
}
