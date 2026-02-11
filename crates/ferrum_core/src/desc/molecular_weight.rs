use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::mol::elements::ATOMIC_MASSES;

/// Compute the molecular weight for a molecule.
///
/// Standalone function for callers that need the raw `f32` value
/// without going through the [`Descriptor`] trait.
pub fn compute_molecular_weight(mol: &Molecule) -> f32 {
    let mut total_mass: f32 = 0.0;

    for atom in &mol.atoms {
        let mass = if atom.isotope > 0 {
            atom.isotope as f32
        } else {
            ATOMIC_MASSES[atom.atomic_number as usize]
        };

        total_mass += mass;

        if atom.num_explicit_h > 0 {
            total_mass += ATOMIC_MASSES[1] * (atom.num_explicit_h as f32);
        }

        if atom.num_implicit_h > 0 {
            total_mass += ATOMIC_MASSES[1] * (atom.num_implicit_h as f32);
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
/// let value = desc.calculate(&mut mol);
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

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_molecular_weight(mol))
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
        let mut mol = parse_smiles("O").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol) {
            DescriptorValue::Float(v) => assert!((v - 18.015).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_methane() {
        // Methane: CH4
        let mut mol = parse_smiles("C").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol) {
            DescriptorValue::Float(v) => assert!((v - 16.043).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_methane_explicit_h() {
        // Methane with explicit H: [CH4]
        let mut mol = parse_smiles("[CH4]").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol) {
            DescriptorValue::Float(v) => assert!((v - 16.043).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_benzene() {
        // Benzene: C6H6
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol) {
            DescriptorValue::Float(v) => assert!((v - 78.114).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_ethanol() {
        // Ethanol: C2H6O = 2*12.011 + 6*1.008 + 15.999 = 46.069
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol) {
            DescriptorValue::Float(v) => assert!((v - 46.069).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_molecular_weight_isotope() {
        // Heavy water: [2H]O[2H]
        let mut mol = parse_smiles("[2H]O[2H]").unwrap();
        let desc = MolecularWeight;
        match desc.calculate(&mut mol) {
            DescriptorValue::Float(v) => assert!((v - 19.999).abs() < 0.01),
            _ => panic!("Expected Float"),
        }
    }
}
