use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Counts the total number of atoms in the molecule.
/// Includes all atoms (heavy atoms and hydrogens).
pub struct AtomCount;

impl Descriptor for AtomCount {
    fn name(&self) -> String { "AtomCount".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        output[0] = mol.num_atoms() as Real;
    }
}

/// Counts only heavy atoms (non-hydrogen atoms).
/// Heavy atoms are any atom with atomic number > 1.
pub struct HeavyAtomCount;

impl Descriptor for HeavyAtomCount {
    fn name(&self) -> String { "HeavyAtomCount".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| atom.atomic_number > 1)
            .count();
        output[0] = count as Real;
    }
}

/// Counts atoms of a specific element type.
/// 
/// # Example
/// ```
/// use ferrum_core::desc::{Descriptor, atom_count::SingleAtomCount};
/// use ferrum_core::io::smiles::parse_smiles;
/// 
/// let mol = parse_smiles("CCO").unwrap(); // Ethanol
/// let carbon_counter = SingleAtomCount::new("C");
/// let mut result = vec![0.0; 1];
/// carbon_counter.calculate(&mol, &mut result);
/// assert_eq!(result[0], 2.0); // 2 carbons
/// ```
pub struct SingleAtomCount {
    element_symbol: String,
    atomic_number: u8,
}

impl SingleAtomCount {
    /// Creates a new counter for the specified element symbol.
    /// 
    /// # Arguments
    /// * `element` - Element symbol (e.g., "C", "N", "O")
    pub fn new(element: &str) -> Self {
        let atomic_number = crate::mol::elements::symbol_to_atomic_number(element)
            .unwrap_or(0);
        Self {
            element_symbol: element.to_string(),
            atomic_number,
        }
    }
}

impl Descriptor for SingleAtomCount {
    fn name(&self) -> String { 
        format!("{}Count", self.element_symbol)
    }
    
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| atom.atomic_number == self.atomic_number)
            .count();
        output[0] = count as Real;
    }
}

/// Counts aromatic atoms in the molecule.
/// Requires aromaticity perception to be run first.
pub struct AromaticAtomCount;

impl Descriptor for AromaticAtomCount {
    fn name(&self) -> String { "AromaticAtomCount".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| atom.is_aromatic())
            .count();
        output[0] = count as Real;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::aromaticity::perceive_aromaticity;

    #[test]
    fn test_atom_count() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol: C2H6O
        let desc = AtomCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 3.0); // C, C, O
    }

    #[test]
    fn test_heavy_atom_count() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol
        let desc = HeavyAtomCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 3.0); // C, C, O (no H counted)
    }

    #[test]
    fn test_single_atom_count_carbon() {
        let mol = parse_smiles("CCO").unwrap();
        let desc = SingleAtomCount::new("C");
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 2.0);
    }

    #[test]
    fn test_single_atom_count_oxygen() {
        let mol = parse_smiles("CCO").unwrap();
        let desc = SingleAtomCount::new("O");
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_single_atom_count_nitrogen() {
        let mol = parse_smiles("CCN").unwrap();
        let desc = SingleAtomCount::new("N");
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_aromatic_atom_count() {
        let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticAtomCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 6.0); // All 6 carbons aromatic
    }

    #[test]
    fn test_aromatic_atom_count_non_aromatic() {
        let mut mol = parse_smiles("CCCC").unwrap(); // Butane
        perceive_aromaticity(&mut mol);
        
        let desc = AromaticAtomCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // No aromatic atoms
    }
}
