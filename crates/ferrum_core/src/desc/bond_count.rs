use super::typed::{DescriptorValue, Descriptor};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

// ---------------------------------------------------------------------------
// Public computation functions
// ---------------------------------------------------------------------------

/// Compute total bond count.
pub fn compute_bond_count(mol: &Molecule) -> u32 {
    mol.num_bonds() as u32
}

/// Compute rotatable bond count.
pub fn compute_rotatable_bond_count(mol: &Molecule) -> u32 {
    let mut count = 0u32;
    
    for bond in &mol.bonds {
        if bond.bond_type != BondType::Single {
            continue;
        }
        if bond.is_in_ring() {
            continue;
        }
        let source_degree = mol.get_degree(bond.source);
        let target_degree = mol.get_degree(bond.target);
        
        if source_degree > 1 && target_degree > 1 {
            count += 1;
        }
    }
    
    count
}

// ---------------------------------------------------------------------------
// Typed Descriptors
// ---------------------------------------------------------------------------

/// Total bond count.
///
/// Returns Int(u32) with total number of bonds.
pub struct BondCount;

impl Descriptor for BondCount {
    fn name(&self) -> &str {
        "BondCount"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(compute_bond_count(mol))
    }
}

/// Detailed bond counts.
///
/// Returns IntVec with [total, single, double, triple, aromatic].
pub struct BondCountDetailed;

impl Descriptor for BondCountDetailed {
    fn name(&self) -> &str {
        "BondCountDetailed"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut counts = vec![0u32; 5]; // total, single, double, triple, aromatic
        
        counts[0] = mol.num_bonds() as u32;
        
        for bond in &mol.bonds {
            match bond.bond_type {
                BondType::Single => counts[1] += 1,
                BondType::Double => counts[2] += 1,
                BondType::Triple => counts[3] += 1,
                BondType::Aromatic => counts[4] += 1,
                _ => {}
            }
        }
        
        DescriptorValue::IntVec(counts)
    }
}

/// Rotatable bond count.
///
/// Returns Int(u32) with number of rotatable bonds.
/// Rotatable bonds are single, non-ring, non-terminal bonds.
pub struct RotatableBondCount;

impl Descriptor for RotatableBondCount {
    fn name(&self) -> &str {
        "RotatableBondCount"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(compute_rotatable_bond_count(mol))
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BondCount>;
    let _ = assert_send_sync::<BondCountDetailed>;
    let _ = assert_send_sync::<RotatableBondCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::rings::find_sssr;

    #[test]
    fn test_bond_count_simple() {
        let mut mol = parse_smiles("CCO").unwrap();
        assert_eq!(BondCount.calculate(&mut mol).as_int(), 2);
    }

    #[test]
    fn test_bond_count_detailed_single() {
        let mut mol = parse_smiles("CCCC").unwrap();
        let v = BondCountDetailed.calculate(&mut mol).as_int_vec().to_vec();
        assert_eq!(v[0], 3); // Total
        assert_eq!(v[1], 3); // Single
        assert_eq!(v[2], 0); // Double
        assert_eq!(v[3], 0); // Triple
    }

    #[test]
    fn test_bond_count_detailed_double() {
        let mut mol = parse_smiles("C=C").unwrap();
        let v = BondCountDetailed.calculate(&mut mol).as_int_vec().to_vec();
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 1);
        assert_eq!(v[3], 0);
    }

    #[test]
    fn test_bond_count_detailed_triple() {
        let mut mol = parse_smiles("C#N").unwrap();
        let v = BondCountDetailed.calculate(&mut mol).as_int_vec().to_vec();
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 0);
        assert_eq!(v[3], 1);
    }

    #[test]
    fn test_bond_count_detailed_mixed() {
        let mut mol = parse_smiles("CC=C").unwrap();
        let v = BondCountDetailed.calculate(&mut mol).as_int_vec().to_vec();
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 1);
        assert_eq!(v[3], 0);
    }

    #[test]
    fn test_rotatable_bond_count_ethane() {
        let mut mol = parse_smiles("CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_propane() {
        let mut mol = parse_smiles("CCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_butane() {
        let mut mol = parse_smiles("CCCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_ethene() {
        let mut mol = parse_smiles("C=C").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        find_sssr(&mut mol);
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_toluene() {
        let mut mol = parse_smiles("Cc1ccccc1").unwrap();
        find_sssr(&mut mol);
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 0);
    }
    
    #[test]
    fn test_rotatable_bond_count_pentane() {
        let mut mol = parse_smiles("CCCCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 2);
    }
    
    #[test]
    fn test_rotatable_bond_count_ethanol() {
        let mut mol = parse_smiles("CCO").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 0);
    }
    
    #[test]
    fn test_rotatable_bond_count_propanol() {
        let mut mol = parse_smiles("CCCO").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol).as_int(), 1);
    }

    #[test]
    fn test_compute_bond_count() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_bond_count(&mol), 2);
    }

    #[test]
    fn test_compute_rotatable_bond_count() {
        let mol = parse_smiles("CCCC").unwrap();
        assert_eq!(compute_rotatable_bond_count(&mol), 1);
    }
}
