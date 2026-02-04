use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Counts hydrogen bond donor groups in the molecule.
/// 
/// A hydrogen bond donor is defined as a nitrogen or oxygen atom
/// with at least one hydrogen attached (N-H or O-H groups).
/// 
/// This follows the Lipinski Rule of Five definition where
/// the number of H-bond donors should be ≤ 5 for drug-like molecules.
/// 
/// **Note**: This implementation counts all N-H and O-H groups, which matches
/// RDKit's Lipinski descriptor for drug-like molecules. For very simple molecules
/// like standalone water (O) or ammonia (N), there may be differences due to
/// context-specific handling in RDKit.
/// 
/// # Examples
/// ```
/// use ferrum_core::desc::{Descriptor, hbond::HBondDonor};
/// use ferrum_core::io::smiles::parse_smiles;
/// 
/// let mol = parse_smiles("CCO").unwrap(); // Ethanol
/// let desc = HBondDonor;
/// let mut result = vec![0.0];
/// desc.calculate(&mol, &mut result);
/// assert_eq!(result[0], 1.0); // One -OH group
/// ```
pub struct HBondDonor;

impl Descriptor for HBondDonor {
    fn name(&self) -> String {
        "HBondDonor".to_string()
    }
    
    fn size(&self) -> usize {
        1
    }
    
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| {
                // N or O with at least one hydrogen
                (atom.atomic_number == 7 || atom.atomic_number == 8) &&
                atom.total_h() > 0
            })
            .count();
        
        output[0] = count as Real;
    }
}

/// Counts hydrogen bond acceptor groups in the molecule.
/// 
/// A hydrogen bond acceptor is defined as a nitrogen or oxygen atom
/// that can accept hydrogen bonds (has lone pairs).
/// 
/// This follows the Lipinski Rule of Five definition where
/// the number of H-bond acceptors should be ≤ 10 for drug-like molecules.
/// 
/// # Examples
/// ```
/// use ferrum_core::desc::{Descriptor, hbond::HBondAcceptor};
/// use ferrum_core::io::smiles::parse_smiles;
/// 
/// let mol = parse_smiles("CC(=O)C").unwrap(); // Acetone
/// let desc = HBondAcceptor;
/// let mut result = vec![0.0];
/// desc.calculate(&mol, &mut result);
/// assert_eq!(result[0], 1.0); // One C=O oxygen
/// ```
pub struct HBondAcceptor;

impl Descriptor for HBondAcceptor {
    fn name(&self) -> String {
        "HBondAcceptor".to_string()
    }
    
    fn size(&self) -> usize {
        1
    }
    
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| {
                // N or O atoms (potential acceptors)
                atom.atomic_number == 7 || atom.atomic_number == 8
            })
            .count();
        
        output[0] = count as Real;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    
    // HBondDonor Tests
    
    #[test]
    fn test_hbond_donor_water() {
        let mol = parse_smiles("O").unwrap(); // Water: H2O
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // O-H
    }
    
    #[test]
    fn test_hbond_donor_ammonia() {
        let mol = parse_smiles("N").unwrap(); // Ammonia: NH3
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // N-H
    }
    
    #[test]
    fn test_hbond_donor_ethanol() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol: C2H6O
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One -OH group
    }
    
    #[test]
    fn test_hbond_donor_urea() {
        let mol = parse_smiles("NC(=O)N").unwrap(); // Urea
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 2.0); // Two -NH2 groups
    }
    
    #[test]
    fn test_hbond_donor_methane() {
        let mol = parse_smiles("C").unwrap(); // Methane: CH4
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // No N-H or O-H
    }
    
    #[test]
    fn test_hbond_donor_acetone() {
        let mol = parse_smiles("CC(=O)C").unwrap(); // Acetone
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // C=O has no hydrogen on oxygen
    }
    
    #[test]
    fn test_hbond_donor_ethylamine() {
        let mol = parse_smiles("CCN").unwrap(); // Ethylamine
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // -NH2 group
    }
    
    #[test]
    fn test_hbond_donor_glycine() {
        let mol = parse_smiles("NCC(=O)O").unwrap(); // Glycine
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 2.0); // -NH2 and -COOH
    }
    
    #[test]
    fn test_hbond_donor_phenol() {
        let mol = parse_smiles("c1ccc(cc1)O").unwrap(); // Phenol
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One -OH (matches RDKit)
    }
    
    #[test]
    fn test_hbond_donor_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap(); // Aspirin
        let desc = HBondDonor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // Carboxylic acid OH (matches RDKit)
    }
    
    // HBondAcceptor Tests
    
    #[test]
    fn test_hbond_acceptor_water() {
        let mol = parse_smiles("O").unwrap(); // Water
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One oxygen
    }
    
    #[test]
    fn test_hbond_acceptor_ammonia() {
        let mol = parse_smiles("N").unwrap(); // Ammonia
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One nitrogen
    }
    
    #[test]
    fn test_hbond_acceptor_ethanol() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One oxygen
    }
    
    #[test]
    fn test_hbond_acceptor_acetone() {
        let mol = parse_smiles("CC(=O)C").unwrap(); // Acetone
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One C=O oxygen
    }
    
    #[test]
    fn test_hbond_acceptor_urea() {
        let mol = parse_smiles("NC(=O)N").unwrap(); // Urea
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 3.0); // 2 N + 1 O
    }
    
    #[test]
    fn test_hbond_acceptor_methane() {
        let mol = parse_smiles("C").unwrap(); // Methane
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // No N or O
    }
    
    #[test]
    fn test_hbond_acceptor_diethyl_ether() {
        let mol = parse_smiles("CCOCC").unwrap(); // Diethyl ether
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One oxygen
    }
    
    #[test]
    fn test_hbond_acceptor_glycine() {
        let mol = parse_smiles("NCC(=O)O").unwrap(); // Glycine
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 3.0); // 1 N + 2 O (carbonyl and hydroxyl)
    }
    
    #[test]
    fn test_hbond_acceptor_phenol() {
        let mol = parse_smiles("c1ccc(cc1)O").unwrap(); // Phenol
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // One O (matches RDKit)
    }
    
    #[test]
    fn test_hbond_acceptor_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap(); // Aspirin
        let desc = HBondAcceptor;
        let mut result = vec![0.0];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 4.0); // 4 oxygens - actually RDKit reports 3
        // Note: RDKit uses more sophisticated SMARTS patterns
        // Our simple count gives 4, RDKit gives 3 (excludes ester O)
    }
}
