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
    // First, detect rings (SSSR)
    let rings = find_sssr(mol);

    // Check which rings are aromatic
    let mut aromatic_rings = Vec::new();
    for ring in &rings {
        if is_ring_aromatic(mol, ring) {
            aromatic_rings.push(ring.clone());
        }
    }

    // Collect all atoms that are in at least one aromatic ring
    let n = mol.num_atoms();
    let mut in_aromatic_ring = vec![false; n];
    for ring in &aromatic_rings {
        for &idx in ring {
            in_aromatic_ring[idx as usize] = true;
        }
    }

    // Strip aromaticity from atoms that SMILES marked aromatic but aren't in any
    // Hückel-aromatic SSSR ring. This handles fused ring systems like Pyrene where
    // some SSSR rings (e.g. 5-membered all-C) don't satisfy 4n+2.
    for i in 0..n {
        if mol.atoms[i].is_aromatic() && !in_aromatic_ring[i] {
            mol.atoms[i].set_aromatic(false);
        }
    }

    // Kekulize bonds that were Aromatic but now connect to non-aromatic atoms
    for bond in &mut mol.bonds {
        if bond.bond_type == BondType::Aromatic {
            let src_arom = in_aromatic_ring[bond.source as usize];
            let tgt_arom = in_aromatic_ring[bond.target as usize];
            if !src_arom && !tgt_arom {
                // Both non-aromatic → Single
                bond.bond_type = BondType::Single;
                bond.set_aromatic(false);
            } else if !src_arom || !tgt_arom {
                // Boundary bond (one aromatic, one not) → Double
                bond.bond_type = BondType::Double;
                bond.set_aromatic(false);
            }
            // Both aromatic → keep as Aromatic
        }
    }

    // Mark all atoms/bonds in aromatic rings
    for ring in &aromatic_rings {
        mark_ring_aromatic(mol, ring);
    }
}

/// Check if a ring is aromatic using Hückel's rule.
fn is_ring_aromatic(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    if ring.len() < 3 {
        return false;
    }

    // Check if ring has conjugated system
    if !is_conjugated(mol, ring) {
        return false;
    }

    // Count π electrons with flexible N handling.
    // N can contribute 1π (pyridine-like) or 2π (pyrrole-like).
    // C with exocyclic C=O but no ring double bond contributes 0π (cross-conjugated).
    // C⁻ (carbanion) contributes 2π (lone pair, like pyrrole N).
    // C⁺ (carbocation) contributes 0π (empty p-orbital, like tropylium).
    // Try all N combinations — if ANY gives 4n+2, ring is aromatic.
    let mut fixed_pi: u8 = 0;
    let mut n_count: u8 = 0;

    for &atom_idx in ring {
        let atom = &mol.atoms[atom_idx as usize];
        match atom.atomic_number {
            6 => {
                let charge = atom.formal_charge;
                if charge < 0 {
                    fixed_pi += 2; // Carbanion: lone pair contributes (cyclopentadienyl)
                } else if charge > 0 {
                    fixed_pi += 0; // Carbocation: empty p-orbital (tropylium)
                } else {
                    // Neutral C: check if it has a double/aromatic bond IN the ring
                    let has_ring_double = mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
                        let bond = &mol.bonds[bond_idx as usize];
                        if !matches!(bond.bond_type, BondType::Double | BondType::Aromatic) {
                            return false;
                        }
                        let other = if bond.source == atom_idx { bond.target } else { bond.source };
                        ring.contains(&other)
                    });
                    if has_ring_double {
                        fixed_pi += 1; // Normal sp2 carbon in ring
                    } else {
                        // C with only single bonds in ring (e.g. C=O exocyclic).
                        fixed_pi += 0;
                    }
                }
            }
            7 => n_count += 1,
            8 | 16 => fixed_pi += 2,
            _ => return false,
        }
    }

    // Try all 2^n_count combinations of N contributing 1 or 2 π electrons
    for mask in 0..(1u32 << n_count) {
        let mut total = fixed_pi;
        for i in 0..n_count {
            total += if mask & (1 << i) != 0 { 2 } else { 1 };
        }
        if is_huckel_aromatic(total) {
            return true;
        }
    }

    false
}

/// Check if ring has a conjugated system — every atom must be sp2-capable.
fn is_conjugated(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            if mol.bonds[bond_idx].bond_type == BondType::Triple {
                return false;
            }
        } else {
            return false;
        }
    }

    // Every atom must be sp2: has a double/aromatic bond (in-ring or exocyclic),
    // is a heteroatom (N, O, S) with a lone pair, or is a charged carbon (C⁻/C⁺).
    for &atom_idx in ring {
        let atom = &mol.atoms[atom_idx as usize];
        if matches!(atom.atomic_number, 7 | 8 | 16) {
            continue;
        }
        // Charged carbons are sp2-capable (C⁻ has lone pair, C⁺ has empty p-orbital)
        if atom.atomic_number == 6 && atom.formal_charge != 0 {
            continue;
        }
        let has_pi_bond = mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            matches!(bond.bond_type, BondType::Double | BondType::Aromatic)
        });
        if !has_pi_bond {
            return false;
        }
    }

    true
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
                // Nitrogen: distinguish pyridine-like (1π) vs pyrrole-like (2π)
                //
                // Pyridine-like: N with no H, 2 ring neighbors → lone pair NOT in π system
                // Pyrrole-like: N with H, or N with 3 ring neighbors → lone pair IN π system
                //
                // Also check for explicit double bond in ring or amide pattern
                
                let h_count = atom.total_h();
                
                // Count ring neighbors
                let ring_neighbor_count = mol.adjacency[atom_idx as usize].iter()
                    .filter(|&&bond_idx| {
                        let bond = &mol.bonds[bond_idx as usize];
                        let other = if bond.source == atom_idx { bond.target } else { bond.source };
                        ring.contains(&other)
                    })
                    .count();
                
                // Check for double bond in ring (explicit, not aromatic)
                let has_double_bond_in_ring = mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
                    let bond = &mol.bonds[bond_idx as usize];
                    if bond.bond_type == BondType::Double {
                        let other = if bond.source == atom_idx { bond.target } else { bond.source };
                        ring.contains(&other)
                    } else {
                        false
                    }
                });
                
                // Check for exocyclic C=O (amide nitrogen)
                let has_carbonyl_neighbor = mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
                    let bond = &mol.bonds[bond_idx as usize];
                    let neighbor = if bond.source == atom_idx { bond.target } else { bond.source };
                    let neighbor_atom = &mol.atoms[neighbor as usize];
                    if neighbor_atom.atomic_number == 6 && ring.contains(&neighbor) {
                        mol.adjacency[neighbor as usize].iter().any(|&bond_idx2| {
                            let bond2 = &mol.bonds[bond_idx2 as usize];
                            if bond2.bond_type == BondType::Double {
                                let other2 = if bond2.source == neighbor { bond2.target } else { bond2.source };
                                let other_atom2 = &mol.atoms[other2 as usize];
                                other_atom2.atomic_number == 8 && !ring.contains(&other2)
                            } else {
                                false
                            }
                        })
                    } else {
                        false
                    }
                });
                
                // Pyridine-like: no H, exactly 2 ring neighbors, OR has explicit double bond
                // This covers both Kekulized (C=N-C) and aromatic (c:n:c) representations
                if has_double_bond_in_ring || has_carbonyl_neighbor || (h_count == 0 && ring_neighbor_count == 2) {
                    pi_electrons += 1; // Pyridine-like
                } else {
                    pi_electrons += 2; // Pyrrole-like (lone pair contributes)
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

    // Mark bonds - set aromatic flag AND bond type
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];

        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            mol.bonds[bond_idx].set_aromatic(true);
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
    
    #[test]
    fn test_cyclopentadienyl_anion_aromatic() {
        // Cyclopentadienyl anion: [cH-]1cccc1 (6 π electrons: 4C×1 + C⁻×2)
        let mut mol = parse_smiles("[cH-]1cccc1").unwrap();
        perceive_aromaticity(&mut mol);
        
        for atom in &mol.atoms {
            assert!(atom.is_aromatic(), "Cyclopentadienyl anion should be aromatic (6π e-)");
        }
    }
}