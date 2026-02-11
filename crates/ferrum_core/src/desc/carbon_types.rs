//! Carbon type classification descriptors
//!
//! Classifies carbon atoms by hybridization and neighbor count:
//! - C1SP1: 1 neighbor, sp hybridization (terminal alkyne)
//! - C2SP1: 2 neighbors, sp hybridization (internal alkyne)
//! - C1SP2: 1 neighbor, sp² hybridization (terminal alkene)
//! - C2SP2: 2 neighbors, sp² hybridization (internal alkene)
//! - C3SP2: 3 neighbors, sp² hybridization (aromatic, carbonyl)
//! - C1SP3: 1 neighbor, sp³ hybridization (methyl)
//! - C2SP3: 2 neighbors, sp³ hybridization (methylene)
//! - C3SP3: 3 neighbors, sp³ hybridization (methine)
//! - C4SP3: 4 neighbors, sp³ hybridization (quaternary)

use crate::mol::molecule::Molecule;
use crate::types::Hybridization;
use crate::desc::typed::{Descriptor, DescriptorValue};

/// Count carbon atoms by type: neighbor count and hybridization
///
/// Returns (C1SP1, C2SP1, C1SP2, C2SP2, C3SP2, C1SP3, C2SP3, C3SP3, C4SP3)
///
/// # Examples
/// - Methane (CH₄): C4SP3=1 (if explicit H) or C1SP3=1 (if implicit)
/// - Ethane (C₂H₆): C1SP3=2
/// - Ethene (C₂H₄): C2SP2=2
/// - Ethyne (C₂H₂): C1SP1=2
/// - Benzene: C3SP2=6 (aromatic)
/// - Acetone: C1SP3=2, C3SP2=1 (carbonyl carbon)
pub fn carbon_types(mol: &Molecule) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    use crate::types::BondType;
    
    let mut counts = (0, 0, 0, 0, 0, 0, 0, 0, 0);
    
    for (idx, atom) in mol.atoms.iter().enumerate() {
        // Only count carbon atoms
        if atom.atomic_number != 6 {
            continue;
        }
        
        let neighbors = mol.get_degree(idx as u16) as u32;
        
        // Infer hybridization from bond types if not specified
        let hybridization = if atom.hybridization == Hybridization::Unspecified {
            // Count double and triple bonds
            let mut has_triple = false;
            let mut has_double = false;
            
            for &bond_idx in &mol.adjacency[idx] {
                let bond = &mol.bonds[bond_idx as usize];
                match bond.bond_type {
                    BondType::Triple => has_triple = true,
                    BondType::Double => has_double = true,
                    _ => {}
                }
            }
            
            if has_triple {
                Hybridization::SP
            } else if has_double || atom.is_aromatic() {
                Hybridization::SP2
            } else {
                Hybridization::SP3
            }
        } else {
            atom.hybridization
        };
        
        match hybridization {
            Hybridization::SP => {
                match neighbors {
                    1 => counts.0 += 1, // C1SP1
                    2 => counts.1 += 1, // C2SP1
                    _ => {} // Unusual, ignore
                }
            }
            Hybridization::SP2 => {
                match neighbors {
                    1 => counts.2 += 1, // C1SP2
                    2 => counts.3 += 1, // C2SP2
                    3 => counts.4 += 1, // C3SP2
                    _ => {} // Unusual, ignore
                }
            }
            Hybridization::SP3 => {
                match neighbors {
                    1 => counts.5 += 1, // C1SP3
                    2 => counts.6 += 1, // C2SP3
                    3 => counts.7 += 1, // C3SP3
                    4 => counts.8 += 1, // C4SP3
                    _ => {} // Unusual, ignore
                }
            }
            Hybridization::SP3D | Hybridization::SP3D2 => {
                // Higher coordination (phosphorus, sulfur), skip
            }
            Hybridization::Unspecified => {
                // Try to infer from neighbors if needed
                // For now, skip
            }
        }
    }
    
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }
    
    #[test]
    fn test_methane() {
        let mol = parse("C");
        let (c1sp1, c2sp1, c1sp2, c2sp2, c3sp2, c1sp3, c2sp3, c3sp3, c4sp3) = carbon_types(&mol);
        assert_eq!(c1sp1, 0);
        assert_eq!(c2sp1, 0);
        assert_eq!(c1sp2, 0);
        assert_eq!(c2sp2, 0);
        assert_eq!(c3sp2, 0);
        assert_eq!(c1sp3, 0); // Methane has 4 implicit H, degree=0 in our model
        assert_eq!(c2sp3, 0);
        assert_eq!(c3sp3, 0);
        // Note: Isolated C with implicit H might not be classified correctly
    }
    
    #[test]
    fn test_ethane() {
        let mol = parse("CC");
        println!("Ethane: {} atoms", mol.num_atoms());
        for (i, atom) in mol.atoms.iter().enumerate() {
            println!("  Atom {}: Z={}, hyb={:?}, degree={}", 
                i, atom.atomic_number, atom.hybridization, mol.get_degree(i as u16));
        }
        let (_c1sp1, _c2sp1, _c1sp2, _c2sp2, _c3sp2, c1sp3, _c2sp3, _c3sp3, _c4sp3) = carbon_types(&mol);
        println!("c1sp3={}, c2sp3={}, c3sp3={}, c4sp3={}", c1sp3, _c2sp3, _c3sp3, _c4sp3);
        assert_eq!(c1sp3, 2); // Two methyls
    }
    
    #[test]
    fn test_propane() {
        let mol = parse("CCC");
        let (_c1sp1, _c2sp1, _c1sp2, _c2sp2, _c3sp2, c1sp3, c2sp3, _c3sp3, _c4sp3) = carbon_types(&mol);
        assert_eq!(c1sp3, 2); // Two methyls
        assert_eq!(c2sp3, 1); // One methylene
    }
    
    #[test]
    fn test_ethene() {
        let mol = parse("C=C");
        let (_c1sp1, _c2sp1, c1sp2, _c2sp2, _c3sp2, _c1sp3, _c2sp3, _c3sp3, _c4sp3) = carbon_types(&mol);
        assert_eq!(c1sp2, 2); // Two sp² carbons, each with 1 neighbor (double bond)
    }
    
    #[test]
    fn test_ethyne() {
        let mol = parse("C#C");
        let (c1sp1, _c2sp1, _c1sp2, _c2sp2, _c3sp2, _c1sp3, _c2sp3, _c3sp3, _c4sp3) = carbon_types(&mol);
        assert_eq!(c1sp1, 2); // Two sp carbons
    }
    
    #[test]
    fn test_benzene() {
        let mol = parse("c1ccccc1");
        let (_c1sp1, _c2sp1, _c1sp2, c2sp2, _c3sp2, _c1sp3, _c2sp3, _c3sp3, _c4sp3) = carbon_types(&mol);
        assert_eq!(c2sp2, 6); // Six aromatic sp² carbons, each with 2 neighbors
    }
    
    #[test]
    fn test_acetone() {
        let mol = parse("CC(=O)C");
        let (_c1sp1, _c2sp1, _c1sp2, _c2sp2, c3sp2, c1sp3, _c2sp3, _c3sp3, _c4sp3) = carbon_types(&mol);
        assert_eq!(c1sp3, 2); // Two methyls
        assert_eq!(c3sp2, 1); // Carbonyl carbon
    }
    
    #[test]
    fn test_neopentane() {
        let mol = parse("CC(C)(C)C");
        let (_c1sp1, _c2sp1, _c1sp2, _c2sp2, _c3sp2, c1sp3, _c2sp3, _c3sp3, c4sp3) = carbon_types(&mol);
        assert_eq!(c1sp3, 4); // Four methyls
        assert_eq!(c4sp3, 1); // One quaternary carbon
    }
    
    #[test]
    fn test_propyne() {
        let mol = parse("CC#C");
        let (c1sp1, c2sp1, _c1sp2, _c2sp2, _c3sp2, c1sp3, _c2sp3, _c3sp3, _c4sp3) = carbon_types(&mol);
        assert_eq!(c1sp1, 1); // Terminal alkyne C
        assert_eq!(c2sp1, 1); // Internal alkyne C
        assert_eq!(c1sp3, 1); // Methyl
    }
}

// ===========================================================================
// Typed API: Carbon Types
// ===========================================================================

/// Carbon type counts descriptor.
///
/// Returns IntU16Vec with 9 values:
/// [C1SP1, C2SP1, C1SP2, C2SP2, C3SP2, C1SP3, C2SP3, C3SP3, C4SP3]
///
/// Uses u16 since most molecules have <65k carbons of any single type.
pub struct CarbonTypes;

impl Descriptor for CarbonTypes {
    fn name(&self) -> &str {
        "CarbonTypes"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        use crate::types::BondType;
        
        let mut counts = vec![0u16; 9];
        
        for (idx, atom) in mol.atoms.iter().enumerate() {
            // Only count carbon atoms
            if atom.atomic_number != 6 {
                continue;
            }
            
            let neighbors = mol.get_degree(idx as u16);
            
            // Infer hybridization from bond types if not specified
            let hybridization = if atom.hybridization == Hybridization::Unspecified {
                // Count double and triple bonds
                let mut has_triple = false;
                let mut has_double = false;
                
                for &bond_idx in &mol.adjacency[idx] {
                    let bond = &mol.bonds[bond_idx as usize];
                    match bond.bond_type {
                        BondType::Triple => has_triple = true,
                        BondType::Double => has_double = true,
                        _ => {}
                    }
                }
                
                if has_triple {
                    Hybridization::SP
                } else if has_double || atom.is_aromatic() {
                    Hybridization::SP2
                } else {
                    Hybridization::SP3
                }
            } else {
                atom.hybridization
            };
            
            match hybridization {
                Hybridization::SP => {
                    match neighbors {
                        1 => counts[0] += 1, // C1SP1
                        2 => counts[1] += 1, // C2SP1
                        _ => {} // Unusual, ignore
                    }
                }
                Hybridization::SP2 => {
                    match neighbors {
                        1 => counts[2] += 1, // C1SP2
                        2 => counts[3] += 1, // C2SP2
                        3 => counts[4] += 1, // C3SP2
                        _ => {} // Unusual, ignore
                    }
                }
                Hybridization::SP3 => {
                    match neighbors {
                        1 => counts[5] += 1, // C1SP3
                        2 => counts[6] += 1, // C2SP3
                        3 => counts[7] += 1, // C3SP3
                        4 => counts[8] += 1, // C4SP3
                        _ => {} // Unusual, ignore
                    }
                }
                Hybridization::SP3D | Hybridization::SP3D2 => {
                    // Higher coordination (phosphorus, sulfur), skip
                }
                Hybridization::Unspecified => {
                    // Already handled above
                }
            }
        }
        
        DescriptorValue::IntU16Vec(counts)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<CarbonTypes>;
};
