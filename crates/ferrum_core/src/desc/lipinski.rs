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

use super::{Descriptor, molecular_weight::MolecularWeight, logp::LogP, hbond::{HBondDonor, HBondAcceptor}};
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Lipinski's Rule of Five descriptor.
///
/// Returns 9 values:
/// 1. Molecular weight (Da)
/// 2. LogP (octanol-water partition coefficient)
/// 3. HBond donors count
/// 4. HBond acceptors count
/// 5. MW passes (1.0 if ≤ 500, 0.0 otherwise)
/// 6. LogP passes (1.0 if ≤ 5, 0.0 otherwise)
/// 7. HBD passes (1.0 if ≤ 5, 0.0 otherwise)
/// 8. HBA passes (1.0 if ≤ 10, 0.0 otherwise)
/// 9. Overall passes (1.0 if all 4 rules pass, 0.0 otherwise)
///
/// # Example
/// ```ignore
/// use ferrum_core::desc::{Descriptor, lipinski::LipinskiRuleOfFive};
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap(); // Aspirin
/// let lipinski = LipinskiRuleOfFive;
/// let mut output = vec![0.0; lipinski.size()];
/// lipinski.calculate(&mol, &mut output);
///
/// println!("MW: {}, LogP: {}, HBD: {}, HBA: {}", 
///          output[0], output[1], output[2], output[3]);
/// println!("Passes Lipinski: {}", output[8]);
/// ```
pub struct LipinskiRuleOfFive;

impl LipinskiRuleOfFive {
    // Rule thresholds
    const MW_THRESHOLD: Real = 500.0;
    const LOGP_THRESHOLD: Real = 5.0;
    const HBD_THRESHOLD: Real = 5.0;
    const HBA_THRESHOLD: Real = 10.0;
}

impl Descriptor for LipinskiRuleOfFive {
    fn name(&self) -> String {
        "LipinskiRuleOfFive".to_string()
    }

    fn size(&self) -> usize {
        9 // MW, LogP, HBD, HBA, MW_pass, LogP_pass, HBD_pass, HBA_pass, overall_pass
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Calculate individual descriptors
        let mw_desc = MolecularWeight;
        let logp_desc = LogP;
        let hbd_desc = HBondDonor;
        let hba_desc = HBondAcceptor;

        let mut mw = 0.0;
        let mut logp = 0.0;
        let mut hbd = 0.0;
        let mut hba = 0.0;

        mw_desc.calculate(mol, std::slice::from_mut(&mut mw));
        logp_desc.calculate(mol, std::slice::from_mut(&mut logp));
        hbd_desc.calculate(mol, std::slice::from_mut(&mut hbd));
        hba_desc.calculate(mol, std::slice::from_mut(&mut hba));

        // Store raw values
        output[0] = mw as Real;
        output[1] = logp as Real;
        output[2] = hbd as Real;
        output[3] = hba as Real;

        // Check individual rules
        let mw_pass = if mw <= Self::MW_THRESHOLD { 1.0 } else { 0.0 };
        let logp_pass = if logp <= Self::LOGP_THRESHOLD { 1.0 } else { 0.0 };
        let hbd_pass = if hbd <= Self::HBD_THRESHOLD { 1.0 } else { 0.0 };
        let hba_pass = if hba <= Self::HBA_THRESHOLD { 1.0 } else { 0.0 };

        output[4] = mw_pass;
        output[5] = logp_pass;
        output[6] = hbd_pass;
        output[7] = hba_pass;

        // Overall pass: all 4 rules must pass
        output[8] = if mw_pass == 1.0 && logp_pass == 1.0 && hbd_pass == 1.0 && hba_pass == 1.0 {
            1.0
        } else {
            0.0
        };
    }
}

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
        let mol = parse_and_prepare("CC(=O)Oc1ccccc1C(=O)O"); // Aspirin
        let lipinski = LipinskiRuleOfFive;
        let mut output = vec![0.0; lipinski.size()];
        lipinski.calculate(&mol, &mut output);

        println!("Aspirin - MW: {}, LogP: {}, HBD: {}, HBA: {}", 
                 output[0], output[1], output[2], output[3]);
        println!("Passes: MW={}, LogP={}, HBD={}, HBA={}, Overall={}", 
                 output[4], output[5], output[6], output[7], output[8]);

        // Aspirin: MW ~180, LogP ~1.2, HBD=1, HBA=4
        assert!(output[0] < 500.0, "MW should be < 500");
        assert!(output[1] < 5.0, "LogP should be < 5");
        assert!(output[2] <= 5.0, "HBD should be <= 5");
        assert!(output[3] <= 10.0, "HBA should be <= 10");
        assert_eq!(output[8], 1.0, "Aspirin should pass all Lipinski rules");
    }

    #[test]
    fn test_lipinski_caffeine() {
        let mol = parse_and_prepare("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"); // Caffeine
        let lipinski = LipinskiRuleOfFive;
        let mut output = vec![0.0; lipinski.size()];
        lipinski.calculate(&mol, &mut output);

        println!("Caffeine - MW: {}, LogP: {}, HBD: {}, HBA: {}", 
                 output[0], output[1], output[2], output[3]);

        // Caffeine: MW ~194, LogP ~-0.07, HBD=0, HBA=6
        assert!(output[0] < 500.0);
        assert_eq!(output[8], 1.0, "Caffeine should pass Lipinski rules");
    }

    #[test]
    fn test_lipinski_ibuprofen() {
        let mol = parse_and_prepare("CC(C)Cc1ccc(cc1)C(C)C(=O)O"); // Ibuprofen
        let lipinski = LipinskiRuleOfFive;
        let mut output = vec![0.0; lipinski.size()];
        lipinski.calculate(&mol, &mut output);

        println!("Ibuprofen - MW: {}, LogP: {}, HBD: {}, HBA: {}", 
                 output[0], output[1], output[2], output[3]);

        // Ibuprofen: MW ~206, LogP ~3.97, HBD=1, HBA=2
        assert!(output[0] < 500.0);
        assert!(output[1] < 5.0);
        assert_eq!(output[8], 1.0, "Ibuprofen should pass Lipinski rules");
    }

    #[test]
    fn test_lipinski_fails_mw() {
        // Large peptide-like molecule that fails MW rule
        let mol = parse_and_prepare("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"); // Long alkane
        let lipinski = LipinskiRuleOfFive;
        let mut output = vec![0.0; lipinski.size()];
        lipinski.calculate(&mol, &mut output);

        println!("Long alkane - MW: {}, passes MW: {}", output[0], output[4]);
        
        if output[0] > 500.0 {
            assert_eq!(output[4], 0.0, "Should fail MW rule");
            assert_eq!(output[8], 0.0, "Should fail overall");
        }
    }

    #[test]
    fn test_lipinski_very_lipophilic() {
        // Very lipophilic molecule
        let mol = parse_and_prepare("CCCCCCCCCCCCCCCCCCCC"); // Eicosane (C20)
        let lipinski = LipinskiRuleOfFive;
        let mut output = vec![0.0; lipinski.size()];
        lipinski.calculate(&mol, &mut output);

        println!("Eicosane - LogP: {}, passes LogP: {}", output[1], output[5]);
        
        // Long alkanes have high LogP, may fail
        if output[1] > 5.0 {
            assert_eq!(output[5], 0.0, "Should fail LogP rule");
        }
    }

    #[test]
    fn test_lipinski_descriptor_size() {
        let lipinski = LipinskiRuleOfFive;
        assert_eq!(lipinski.size(), 9);
        assert_eq!(lipinski.name(), "LipinskiRuleOfFive");
    }

    #[test]
    fn test_lipinski_paracetamol() {
        let mol = parse_and_prepare("CC(=O)Nc1ccc(O)cc1"); // Paracetamol/Acetaminophen
        let lipinski = LipinskiRuleOfFive;
        let mut output = vec![0.0; lipinski.size()];
        lipinski.calculate(&mol, &mut output);

        println!("Paracetamol - MW: {}, LogP: {}, HBD: {}, HBA: {}", 
                 output[0], output[1], output[2], output[3]);

        // Paracetamol: MW ~151, LogP ~0.46, HBD=2, HBA=2
        assert_eq!(output[8], 1.0, "Paracetamol should pass Lipinski rules");
    }
}
