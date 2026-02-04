use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use crate::algo::rings::find_sssr;

/// Perceive aromaticity using the Hückel 4n+2 rule.
///
/// # Thread Safety
/// This function is thread-safe - it only mutates the molecule passed to it.
/// Each molecule can be processed independently in parallel.
///
/// # Algorithm
/// For each ring detected (SSSR):
/// 1. Check if ring has conjugated system (alternating/aromatic bonds)
/// 2. Count π electrons according to element rules
/// 3. Apply Hückel rule: aromatic if π electrons = 4n+2 (n=0,1,2,...)
/// 4. Set aromatic flags on atoms and bonds in aromatic rings
///
/// # π Electron Counting Rules
/// - C (sp2): 1 π electron
/// - N (pyridine-like, no H): 1 π electron
/// - N (pyrrole-like, with H): 2 π electrons (lone pair)
/// - O, S: 2 π electrons (lone pair)
/// - Other elements: 0 (not considered for aromaticity)
///
/// # Example
/// ```ignore
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// perceive_aromaticity(&mut mol);
/// assert!(mol.atoms[0].is_aromatic());  // Benzene carbon
/// ```
pub fn perceive_aromaticity(mol: &mut Molecule) {
    // First, detect rings (SSSR) - now returns all SSSR rings
    let rings = find_sssr(mol);

    for ring in rings {
        // Now check all rings, not just 6-membered
        if is_ring_aromatic(mol, &ring) {
            mark_ring_aromatic(mol, &ring);
        }
    }
}

/// Check if a ring is aromatic using Hückel's rule.
///
/// # Thread Safety
/// Read-only function, safe to call concurrently.
fn is_ring_aromatic(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    // Typically aromatic rings are 5, 6, or 7 membered, but Hückel's rule applies generally
    if ring.len() < 3 {
        return false;
    }

    // Check if ring has conjugated system
    if !is_conjugated(mol, ring) {
        return false;
    }

    // Count π electrons
    let pi_electrons = count_pi_electrons(mol, ring);

    // Apply Hückel rule: 4n+2 π electrons (n = 0, 1, 2, ...)
    // Common cases: 2, 6, 10, 14, ...
    is_huckel_aromatic(pi_electrons)
}

/// Check if ring has a conjugated system (alternating or aromatic bonds).
fn is_conjugated(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    // Check bonds in the ring
    let mut aromatic_or_double_bonds = 0;

    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];

        // Find bond between these atoms
        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            let bond = &mol.bonds[bond_idx];
            match bond.bond_type {
                BondType::Double | BondType::Aromatic => {
                    aromatic_or_double_bonds += 1;
                }
                BondType::Single => {
                    // Single bonds are okay in conjugated systems
                }
                _ => return false, // Triple bonds break conjugation
            }
        } else {
            return false; // No bond found
        }
    }

    // For conjugation, we need roughly half the bonds to be double/aromatic
    // This works for rings of any size:
    // - 5-membered: need at least 2 (e.g., pyrrole, furan)
    // - 6-membered: need at least 3 (e.g., benzene)
    // - 7-membered: need at least 3 (e.g., tropylium)
    let min_required = (ring.len() + 1) / 2;
    aromatic_or_double_bonds >= min_required
}

/// Find bond between two atoms.
fn find_bond_between(mol: &Molecule, from: AtomIdx, to: AtomIdx) -> Option<usize> {
    for &bond_idx in &mol.adjacency[from as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        if (bond.source == from && bond.target == to) ||
           (bond.source == to && bond.target == from) {
            return Some(bond_idx as usize);
        }
    }
    None
}

/// Count π electrons in a ring according to aromaticity rules.
fn count_pi_electrons(mol: &Molecule, ring: &[AtomIdx]) -> u8 {
    let mut pi_electrons = 0;

    for &atom_idx in ring {
        let atom = &mol.atoms[atom_idx as usize];
        let element = atom.atomic_number;

        match element {
            6 => {
                // Carbon: contributes 1 π electron if sp2
                pi_electrons += 1;
            }
            7 => {
                // Nitrogen: 1 if pyridine-like (no H), 2 if pyrrole-like (with H)
                if atom.total_h() > 0 {
                    pi_electrons += 2; // Pyrrole-like (lone pair)
                } else {
                    pi_electrons += 1; // Pyridine-like
                }
            }
            8 | 16 => {
                // Oxygen or Sulfur: contributes 2 π electrons (lone pair)
                pi_electrons += 2;
            }
            _ => {
                // Other elements: not considered for simple aromaticity
                return 0;
            }
        }
    }

    pi_electrons
}

/// Check if number of π electrons satisfies Hückel's 4n+2 rule.
fn is_huckel_aromatic(pi_electrons: u8) -> bool {
    // 4n+2 for n = 0, 1, 2, ... gives: 2, 6, 10, 14, 18, ...
    // Most common for 6-membered rings: 6 π electrons
    matches!(pi_electrons, 2 | 6 | 10 | 14 | 18)
}

/// Mark all atoms and bonds in a ring as aromatic.
fn mark_ring_aromatic(mol: &mut Molecule, ring: &[AtomIdx]) {
    // Mark atoms
    for &atom_idx in ring {
        mol.atoms[atom_idx as usize].set_aromatic(true);
    }

    // Mark bonds
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];

        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            mol.bonds[bond_idx].bond_type = BondType::Aromatic;
        }
    }
}

// Alias for backwards compatibility
pub fn apply_hueckel(mol: &mut Molecule) {
    perceive_aromaticity(mol);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_benzene_aromatic() {
        // Benzene: c1ccccc1 (6 π electrons)
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);

        // All 6 carbons should be aromatic
        for atom in &mol.atoms {
            assert!(atom.is_aromatic(), "Benzene atoms should be aromatic");
        }
    }

    #[test]
    fn test_pyridine_aromatic() {
        // Pyridine: c1ccncc1 (6 π electrons: 5 C + 1 N)
        let mut mol = parse_smiles("c1ccncc1").unwrap();
        perceive_aromaticity(&mut mol);

        // All atoms should be aromatic
        for atom in &mol.atoms {
            assert!(atom.is_aromatic(), "Pyridine atoms should be aromatic");
        }
    }

    #[test]
    fn test_cyclohexane_not_aromatic() {
        // Cyclohexane: C1CCCCC1 (0 π electrons, all single bonds)
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        perceive_aromaticity(&mut mol);

        // No atoms should be aromatic
        for atom in &mol.atoms {
            assert!(!atom.is_aromatic(), "Cyclohexane should not be aromatic");
        }
    }

    #[test]
    fn test_toluene_aromatic() {
        // Toluene: Cc1ccccc1 (benzene ring with methyl)
        let mut mol = parse_smiles("Cc1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);

        // Benzene carbons should be aromatic (indices 1-6)
        for i in 1..7 {
            assert!(mol.atoms[i].is_aromatic(), "Benzene ring should be aromatic");
        }

        // Methyl carbon should NOT be aromatic
        assert!(!mol.atoms[0].is_aromatic(), "Methyl carbon should not be aromatic");
    }

    #[test]
    fn test_huckel_rule() {
        // Test Hückel 4n+2 rule directly
        assert!(is_huckel_aromatic(2));   // n=0: 4(0)+2 = 2
        assert!(is_huckel_aromatic(6));   // n=1: 4(1)+2 = 6 (benzene)
        assert!(is_huckel_aromatic(10));  // n=2: 4(2)+2 = 10
        assert!(is_huckel_aromatic(14));  // n=3: 4(3)+2 = 14

        assert!(!is_huckel_aromatic(4));  // 4n rule (anti-aromatic)
        assert!(!is_huckel_aromatic(8));  // 4n rule
        assert!(!is_huckel_aromatic(0));  // No π electrons
    }
    
    #[test]
    fn test_pyrrole_5_ring_aromatic() {
        // Pyrrole: [nH]1cccc1 (6 π electrons in 5-membered ring)
        // N contributes 2 π electrons (lone pair), C contributes 1 each
        let mut mol = parse_smiles("[nH]1cccc1").unwrap();
        perceive_aromaticity(&mut mol);
        
        // All 5 atoms should be aromatic
        for atom in &mol.atoms {
            assert!(atom.is_aromatic(), "Pyrrole atoms should be aromatic (5-ring with 6π e-)");
        }
    }
    
    #[test]
    fn test_furan_5_ring_aromatic() {
        // Furan: o1cccc1 (6 π electrons in 5-membered ring)
        // O contributes 2 π electrons (lone pair), C contributes 1 each
        let mut mol = parse_smiles("o1cccc1").unwrap();
        perceive_aromaticity(&mut mol);
        
        // All 5 atoms should be aromatic
        for atom in &mol.atoms {
            assert!(atom.is_aromatic(), "Furan atoms should be aromatic (5-ring with 6π e-)");
        }
    }
    
    #[test]
    fn test_cyclopentadiene_not_aromatic() {
        // Cyclopentadiene: C1=CC=CC1 (4 π electrons - anti-aromatic!)
        // This should NOT be aromatic (fails 4n+2 rule, has 4π which is 4n)
        let mut mol = parse_smiles("C1=CC=CC1").unwrap();
        perceive_aromaticity(&mut mol);
        
        // No atoms should be aromatic (4π electrons, not 4n+2)
        for atom in &mol.atoms {
            assert!(!atom.is_aromatic(), "Cyclopentadiene should not be aromatic (4π e-)");
        }
    }
}