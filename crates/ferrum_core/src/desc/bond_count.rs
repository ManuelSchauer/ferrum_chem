use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::{BondType, Real};

/// Counts bonds in the molecule.
/// 
/// Can return either:
/// - Total bond count only (size = 1)
/// - Detailed breakdown: total, single, double, triple (size = 4)
/// 
/// # Example
/// ```
/// use ferrum_core::desc::{Descriptor, bond_count::BondCountDescriptor};
/// use ferrum_core::io::smiles::parse_smiles;
/// 
/// let mol = parse_smiles("C=C").unwrap(); // Ethene
/// 
/// // Simple count
/// let desc = BondCountDescriptor::new(false);
/// let mut result = vec![0.0; 1];
/// desc.calculate(&mol, &mut result);
/// assert_eq!(result[0], 1.0); // 1 bond total
/// 
/// // Detailed count
/// let desc = BondCountDescriptor::new(true);
/// let mut result = vec![0.0; 4];
/// desc.calculate(&mol, &mut result);
/// assert_eq!(result[0], 1.0); // 1 bond total
/// assert_eq!(result[1], 0.0); // 0 single bonds
/// assert_eq!(result[2], 1.0); // 1 double bond
/// assert_eq!(result[3], 0.0); // 0 triple bonds
/// ```
pub struct BondCountDescriptor {
    include_detailed: bool,
}

impl BondCountDescriptor {
    /// Creates a new bond count descriptor.
    /// 
    /// # Arguments
    /// * `include_detailed` - If true, returns [total, single, double, triple]
    ///                       If false, returns only total count
    pub fn new(include_detailed: bool) -> Self {
        Self { include_detailed }
    }
}

impl Descriptor for BondCountDescriptor {
    fn name(&self) -> String { 
        if self.include_detailed {
            "BondCount_Detailed".to_string()
        } else {
            "BondCount".to_string()
        }
    }
    
    fn size(&self) -> usize { 
        if self.include_detailed { 4 } else { 1 }
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let total = mol.num_bonds() as Real;
        
        if !self.include_detailed {
            output[0] = total;
            return;
        }
        
        // Detailed breakdown
        let mut single = 0;
        let mut double = 0;
        let mut triple = 0;
        
        for bond in &mol.bonds {
            match bond.bond_type {
                BondType::Single => single += 1,
                BondType::Double => double += 1,
                BondType::Triple => triple += 1,
                _ => {} // Aromatic, Dative, Unknown not counted separately
            }
        }
        
        output[0] = total;
        output[1] = single as Real;
        output[2] = double as Real;
        output[3] = triple as Real;
    }
}

/// Counts rotatable bonds in the molecule.
/// 
/// A rotatable bond is defined as:
/// - Single bond (not aromatic)
/// - Not in a ring
/// - Not terminal (both atoms must have degree > 1)
/// - Not attached to terminal atoms (hydrogens don't count)
pub struct RotatableBondCount;

impl Descriptor for RotatableBondCount {
    fn name(&self) -> String { "RotatableBondCount".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut count = 0;
        
        for bond in &mol.bonds {
            // Must be single bond
            if bond.bond_type != BondType::Single {
                continue;
            }
            
            // Must not be in ring
            if bond.is_in_ring() {
                continue;
            }
            
            // Both atoms must have degree > 1 (not terminal)
            let source_degree = mol.get_degree(bond.source);
            let target_degree = mol.get_degree(bond.target);
            
            if source_degree > 1 && target_degree > 1 {
                count += 1;
            }
        }
        
        output[0] = count as Real;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::rings::find_sssr;

    #[test]
    fn test_bond_count_simple() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol: C-C-O
        let desc = BondCountDescriptor::new(false);
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 2.0); // 2 bonds
    }

    #[test]
    fn test_bond_count_detailed_single() {
        let mol = parse_smiles("CCCC").unwrap(); // Butane
        let desc = BondCountDescriptor::new(true);
        let mut result = vec![0.0; 4];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 3.0); // Total: 3 bonds
        assert_eq!(result[1], 3.0); // Single: 3
        assert_eq!(result[2], 0.0); // Double: 0
        assert_eq!(result[3], 0.0); // Triple: 0
    }

    #[test]
    fn test_bond_count_detailed_double() {
        let mol = parse_smiles("C=C").unwrap(); // Ethene
        let desc = BondCountDescriptor::new(true);
        let mut result = vec![0.0; 4];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // Total: 1
        assert_eq!(result[1], 0.0); // Single: 0
        assert_eq!(result[2], 1.0); // Double: 1
        assert_eq!(result[3], 0.0); // Triple: 0
    }

    #[test]
    fn test_bond_count_detailed_triple() {
        let mol = parse_smiles("C#N").unwrap(); // HCN
        let desc = BondCountDescriptor::new(true);
        let mut result = vec![0.0; 4];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // Total: 1
        assert_eq!(result[1], 0.0); // Single: 0
        assert_eq!(result[2], 0.0); // Double: 0
        assert_eq!(result[3], 1.0); // Triple: 1
    }

    #[test]
    fn test_bond_count_detailed_mixed() {
        let mol = parse_smiles("CC=C").unwrap(); // Propene: C-C=C
        let desc = BondCountDescriptor::new(true);
        let mut result = vec![0.0; 4];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 2.0); // Total: 2
        assert_eq!(result[1], 1.0); // Single: 1
        assert_eq!(result[2], 1.0); // Double: 1
        assert_eq!(result[3], 0.0); // Triple: 0
    }

    #[test]
    fn test_rotatable_bond_count_ethane() {
        let mol = parse_smiles("CC").unwrap(); // Ethane
        let desc = RotatableBondCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // Terminal atoms, no rotatable bonds
    }

    #[test]
    fn test_rotatable_bond_count_propane() {
        let mol = parse_smiles("CCC").unwrap(); // Propane
        let desc = RotatableBondCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // Both end carbons are terminal
    }

    #[test]
    fn test_rotatable_bond_count_butane() {
        let mol = parse_smiles("CCCC").unwrap(); // Butane
        let desc = RotatableBondCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 1.0); // Central C-C bond is rotatable
    }

    #[test]
    fn test_rotatable_bond_count_ethene() {
        let mol = parse_smiles("C=C").unwrap(); // Ethene
        let desc = RotatableBondCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // Double bond is not rotatable
    }

    #[test]
    fn test_rotatable_bond_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene
        find_sssr(&mut mol); // Set ring flags
        
        let desc = RotatableBondCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // Ring bonds not rotatable
    }

    #[test]
    fn test_rotatable_bond_count_toluene() {
        let mut mol = parse_smiles("Cc1ccccc1").unwrap(); // Toluene
        find_sssr(&mut mol);
        
        let desc = RotatableBondCount;
        let mut result = vec![0.0; 1];
        desc.calculate(&mol, &mut result);
        assert_eq!(result[0], 0.0); // Methyl is terminal, C-Ph bond not rotatable
    }
}
