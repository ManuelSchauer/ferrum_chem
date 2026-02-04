use crate::mol::molecule::Molecule;
use crate::mol::elements::get_valences;
use crate::types::BondType;
use crate::error::{Result, FerrumError};

/// Calculates and assigns implicit hydrogens based on valence rules.
///
/// # Thread Safety
/// This function is thread-safe - it only mutates the molecule passed to it.
/// Each molecule can be processed independently in parallel.
///
/// # Algorithm
/// For each atom:
/// 1. Calculate bond order sum (connectivity)
/// 2. Look up possible valences for the element
/// 3. Find the smallest valence >= connectivity
/// 4. Implicit H = valence - connectivity - formal_charge - explicit_h
///
/// # Valence Rules
/// - C: 4 (sp3, sp2, sp)
/// - N: 3, 5 (amine vs nitro)
/// - O: 2 (alcohol, ether, carbonyl)
/// - S: 2, 4, 6 (sulfide, sulfoxide, sulfone)
/// - P: 3, 5 (phosphine vs phosphate)
/// - Halogens: 1
///
/// # Example
/// ```ignore
/// let mut mol = parse_smiles("CCO").unwrap();
/// assign_implicit_hydrogens(&mut mol)?;
/// // C: 3H, C: 2H, O: 1H
/// ```
pub fn assign_implicit_hydrogens(mol: &mut Molecule) -> Result<()> {
    for atom_idx in 0..mol.atoms.len() {
        let atom = &mol.atoms[atom_idx];

        // Skip if explicit hydrogens are already specified
        if atom.num_explicit_h > 0 {
            mol.atoms[atom_idx].num_implicit_h = 0;
            continue;
        }

        // Get possible valences for this element
        let valences = get_valences(atom.atomic_number);
        if valences.is_empty() {
            // Unknown element or metal - assume no implicit H
            mol.atoms[atom_idx].num_implicit_h = 0;
            continue;
        }

        // Calculate bond order sum (connectivity)
        let connectivity = calculate_connectivity(mol, atom_idx as u16);

        // Adjust for formal charge
        // Positive charge reduces valence, negative increases
        let charge_adjustment = atom.formal_charge as i16;

        // Find the smallest valence that can accommodate the connectivity
        let mut best_valence: Option<i16> = None;
        for &v in valences {
            let adjusted = (v as i16) - charge_adjustment;
            if adjusted >= connectivity as i16 {
                if best_valence.is_none() || adjusted < best_valence.unwrap() {
                    best_valence = Some(adjusted);
                }
            }
        }

        if let Some(valence) = best_valence {
            let implicit_h = valence - (connectivity as i16);
            if implicit_h >= 0 {
                mol.atoms[atom_idx].num_implicit_h = implicit_h as u8;
            } else {
                // Valence exceeded - this might be an error or unusual structure
                mol.atoms[atom_idx].num_implicit_h = 0;
            }
        } else {
            // No suitable valence found - might be hypervalent or error
            mol.atoms[atom_idx].num_implicit_h = 0;
        }
    }

    Ok(())
}

/// Calculate the sum of bond orders for an atom.
///
/// # Thread Safety
/// Read-only function, safe to call concurrently on different molecules.
///
/// # Aromatic Bonds
/// Aromatic bonds are special - they contribute 1.5 to connectivity in reality,
/// but we use integer math. For organic aromatic atoms (C, N, O in rings),
/// we count 2 aromatic bonds as contributing 3 to connectivity.
///
/// Note: In SMILES, aromatic atoms are marked with lowercase letters (c, n, o, etc.)
/// but bonds between them might be marked as Single. We check both the bond type
/// and whether both atoms are aromatic.
fn calculate_connectivity(mol: &Molecule, atom_idx: u16) -> u8 {
    let mut connectivity: u8 = 0;
    let mut aromatic_bonds: u8 = 0;

    let atom = &mol.atoms[atom_idx as usize];
    let is_atom_aromatic = atom.is_aromatic();

    // Sum bond orders from adjacency list
    for &bond_idx in &mol.adjacency[atom_idx as usize] {
        let bond = &mol.bonds[bond_idx as usize];

        // Determine the other atom in this bond
        let other_idx = if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        };
        let other_atom = &mol.atoms[other_idx as usize];

        // Check if this is an aromatic bond
        // Either explicitly marked as Aromatic, or both atoms are aromatic
        let is_aromatic_bond = bond.bond_type == BondType::Aromatic ||
                               (is_atom_aromatic && other_atom.is_aromatic());

        let bond_order = match bond.bond_type {
            BondType::Single if is_aromatic_bond => {
                // Single bond between two aromatic atoms is actually aromatic
                aromatic_bonds += 1;
                0
            },
            BondType::Single => 1,
            BondType::Double => 2,
            BondType::Triple => 3,
            BondType::Aromatic => {
                // Count aromatic bonds separately
                aromatic_bonds += 1;
                0  // Don't add to connectivity yet
            },
            BondType::Dative => 0,  // Coordinate bond, doesn't count
            BondType::Unknown => 1,  // Assume single
        };

        connectivity += bond_order;
    }

    // For aromatic systems: each aromatic bond contributes 1.5
    // So 2 aromatic bonds = 3 connectivity
    // We use: connectivity = aromatic_bonds + aromatic_bonds/2
    connectivity += aromatic_bonds + (aromatic_bonds / 2);

    connectivity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_methane_implicit_h() {
        // Methane: C (should get 4 implicit H)
        let mut mol = parse_smiles("C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].atomic_number, 6);  // Carbon
        assert_eq!(mol.atoms[0].num_implicit_h, 4); // CH4
    }

    #[test]
    fn test_ethane_implicit_h() {
        // Ethane: CC
        let mut mol = parse_smiles("CC").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each carbon has 1 bond to the other carbon, so 3 H each
        assert_eq!(mol.atoms[0].num_implicit_h, 3); // CH3
        assert_eq!(mol.atoms[1].num_implicit_h, 3); // CH3
    }

    #[test]
    fn test_water_implicit_h() {
        // Water: O
        let mut mol = parse_smiles("O").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].atomic_number, 8);  // Oxygen
        assert_eq!(mol.atoms[0].num_implicit_h, 2); // H2O
    }

    #[test]
    fn test_ammonia_implicit_h() {
        // Ammonia: N
        let mut mol = parse_smiles("N").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].atomic_number, 7);  // Nitrogen
        assert_eq!(mol.atoms[0].num_implicit_h, 3); // NH3
    }

    #[test]
    fn test_ethene_implicit_h() {
        // Ethene: C=C
        let mut mol = parse_smiles("C=C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each carbon has 1 double bond (order 2), so 2 H each
        assert_eq!(mol.atoms[0].num_implicit_h, 2); // CH2
        assert_eq!(mol.atoms[1].num_implicit_h, 2); // CH2
    }

    #[test]
    fn test_acetylene_implicit_h() {
        // Acetylene: C#C
        let mut mol = parse_smiles("C#C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each carbon has 1 triple bond (order 3), so 1 H each
        assert_eq!(mol.atoms[0].num_implicit_h, 1); // CH
        assert_eq!(mol.atoms[1].num_implicit_h, 1); // CH
    }

    #[test]
    fn test_charged_ammonium() {
        // Ammonium: [NH4+]
        let mut mol = parse_smiles("[NH4+]").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Already has explicit H, so should have 0 implicit
        assert_eq!(mol.atoms[0].num_implicit_h, 0);
        assert_eq!(mol.atoms[0].num_explicit_h, 4);
    }

    #[test]
    fn test_benzene_implicit_h() {
        // Benzene: c1ccccc1
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        // Each aromatic carbon has 2 aromatic bonds, so 1 H each
        for atom in &mol.atoms {
            assert_eq!(atom.num_implicit_h, 1);
        }
    }

    #[test]
    fn test_ethanol_implicit_h() {
        // Ethanol: CCO
        let mut mol = parse_smiles("CCO").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();

        assert_eq!(mol.atoms[0].num_implicit_h, 3); // CH3
        assert_eq!(mol.atoms[1].num_implicit_h, 2); // CH2
        assert_eq!(mol.atoms[2].num_implicit_h, 1); // OH
    }
}