use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Calculates the total formal charge of the molecule.
/// 
/// The formal charge is the sum of all individual atom formal charges.
/// Most neutral molecules have a formal charge of 0.
/// 
/// # Example
/// ```
/// use ferrum_core::desc::{Descriptor, formal_charge::FormalCharge};
/// use ferrum_core::io::smiles::parse_smiles;
/// 
/// // Ammonium ion: [NH4]+
/// let mol = parse_smiles("[NH4+]").unwrap();
/// let desc = FormalCharge;
/// let mut result = vec![0.0; 1];
/// desc.calculate(&mol, &mut result);
/// assert_eq!(result[0], 1.0); // +1 charge
/// 
/// // Hydroxide ion: [OH]-
/// let mol = parse_smiles("[OH-]").unwrap();
/// let mut result = vec![0.0; 1];
/// desc.calculate(&mol, &mut result);
/// assert_eq!(result[0], -1.0); // -1 charge
/// ```
pub struct FormalCharge;

impl Descriptor for FormalCharge {
    fn name(&self) -> String { "FormalCharge".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let total_charge: i32 = mol.atoms.iter()
            .map(|atom| atom.formal_charge as i32)
            .sum();
        
        output[0] = total_charge as Real;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_formal_charge_neutral() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol - neutral
        let desc = FormalCharge;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_formal_charge_positive() {
        let mol = parse_smiles("[NH4+]").unwrap(); // Ammonium
        let desc = FormalCharge;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_formal_charge_negative() {
        let mol = parse_smiles("[OH-]").unwrap(); // Hydroxide
        let desc = FormalCharge;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], -1.0);
    }

    #[test]
    fn test_formal_charge_doubly_charged() {
        let mol = parse_smiles("[O-2]").unwrap(); // Oxide ion
        let desc = FormalCharge;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], -2.0);
    }

    #[test]
    fn test_formal_charge_benzene() {
        let mol = parse_smiles("c1ccccc1").unwrap(); // Benzene - neutral
        let desc = FormalCharge;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_formal_charge_carboxylate() {
        let mol = parse_smiles("CC([O-])=O").unwrap(); // Acetate anion
        let desc = FormalCharge;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], -1.0);
    }

    #[test]
    fn test_formal_charge_zwitterion() {
        // Glycine zwitterion: [NH3+]CC([O-])=O
        let mol = parse_smiles("[NH3+]CC([O-])=O").unwrap();
        let desc = FormalCharge;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // +1 and -1 cancel out
    }
}
