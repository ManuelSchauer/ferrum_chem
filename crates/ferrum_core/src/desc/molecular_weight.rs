use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::mol::elements::ATOMIC_MASSES;
use crate::types::Real;

/// Berechnet das Molekulargewicht (MW).
pub struct MolecularWeight;

impl Descriptor for MolecularWeight {
    fn name(&self) -> String {
        "MolecularWeight".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut total_mass: f32 = 0.0;

        for atom in &mol.atoms {
            // Use isotope-specific mass if specified, otherwise standard atomic mass
            let mass = if atom.isotope > 0 {
                // For isotopes, use the isotope number as approximate mass
                // In a production system, you'd have a separate isotope mass table
                atom.isotope as f32
            } else {
                // Standard atomic mass from lookup table
                ATOMIC_MASSES[atom.atomic_number as usize]
            };

            total_mass += mass;

            // Add explicit hydrogens
            if atom.num_explicit_h > 0 {
                total_mass += ATOMIC_MASSES[1] * (atom.num_explicit_h as f32);
            }

            // Add implicit hydrogens
            if atom.num_implicit_h > 0 {
                total_mass += ATOMIC_MASSES[1] * (atom.num_implicit_h as f32);
            }
        }

        output[0] = total_mass as Real;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_molecular_weight_water() {
        // Water: H2O = 2*1.008 + 15.999 = 18.015
        let mol = parse_smiles("O").unwrap();

        let desc = MolecularWeight;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);

        // O + 2H = 15.999 + 2*1.008 = 18.015
        assert!((result[0] - 18.015).abs() < 0.01);
    }

    #[test]
    fn test_molecular_weight_methane() {
        // Methane: CH4
        let mol = parse_smiles("C").unwrap();

        let desc = MolecularWeight;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);

        // C + 4H: 12.011 + 4*1.008 = 16.043
        assert!((result[0] - 16.043).abs() < 0.01);
    }

    #[test]
    fn test_molecular_weight_methane_explicit_h() {
        // Methane with explicit H: [CH4]
        let mol = parse_smiles("[CH4]").unwrap();

        let desc = MolecularWeight;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);

        // C: 12.011 + 4*H: 4*1.008 = 16.043
        assert!((result[0] - 16.043).abs() < 0.01);
    }

    #[test]
    fn test_molecular_weight_benzene() {
        // Benzene: C6H6
        let mol = parse_smiles("c1ccccc1").unwrap();

        let desc = MolecularWeight;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);

        // 6*C + 6*H: 6*12.011 + 6*1.008 = 78.114
        assert!((result[0] - 78.114).abs() < 0.01);
    }

    #[test]
    fn test_molecular_weight_ethanol() {
        // Ethanol: C2H6O = 2*12.011 + 6*1.008 + 15.999 = 46.069
        let mol = parse_smiles("CCO").unwrap();

        let desc = MolecularWeight;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);

        assert!((result[0] - 46.069).abs() < 0.01);
    }

    #[test]
    fn test_molecular_weight_isotope() {
        // Heavy water: [2H]O[2H]
        let mol = parse_smiles("[2H]O[2H]").unwrap();

        let desc = MolecularWeight;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);

        // 2*2 + 15.999 = 19.999
        assert!((result[0] - 19.999).abs() < 0.01);
    }
}
