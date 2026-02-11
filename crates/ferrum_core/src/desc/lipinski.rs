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

use super::typed::{DescriptorValue, Descriptor};
use super::logp::compute_logp;
use super::hbond::{compute_hbond_donors, compute_hbond_acceptors};
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
    fn name(&self) -> &str {
        "LipinskiViolations"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mw = compute_molecular_weight(mol);
        let logp = compute_logp(mol).0;
        let hbd = compute_hbond_donors(mol);
        let hba = compute_hbond_acceptors(mol);

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
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<LipinskiViolations>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::valency::assign_implicit_hydrogens;

    fn parse_and_prepare(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        mol
    }

    #[test]
    fn test_lipinski_aspirin() {
        let mut mol = parse_and_prepare("CC(=O)Oc1ccccc1C(=O)O");
        let desc = LipinskiViolations;
        let value = desc.calculate(&mut mol);
        assert_eq!(value, DescriptorValue::Int(0), "Aspirin should have 0 Lipinski violations");
    }

    #[test]
    fn test_lipinski_caffeine() {
        let mut mol = parse_and_prepare("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        let desc = LipinskiViolations;
        let value = desc.calculate(&mut mol);
        assert_eq!(value, DescriptorValue::Int(0), "Caffeine should have 0 Lipinski violations");
    }

    #[test]
    fn test_lipinski_ibuprofen() {
        let mut mol = parse_and_prepare("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
        let desc = LipinskiViolations;
        let value = desc.calculate(&mut mol);
        assert_eq!(value, DescriptorValue::Int(0), "Ibuprofen should have 0 Lipinski violations");
    }

    #[test]
    fn test_lipinski_ethanol() {
        let mut mol = parse_and_prepare("CCO");
        let desc = LipinskiViolations;
        let value = desc.calculate(&mut mol);
        assert_eq!(value, DescriptorValue::Int(0), "Ethanol should have 0 violations");
    }

    #[test]
    fn test_lipinski_descriptor_name() {
        let desc = LipinskiViolations;
        assert_eq!(desc.name(), "LipinskiViolations");
    }
}
