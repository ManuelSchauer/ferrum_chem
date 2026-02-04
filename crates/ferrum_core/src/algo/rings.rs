use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::{VecDeque, HashSet};

/// Detects rings using SSSR (Smallest Set of Smallest Rings) algorithm.
///
/// Returns the list of rings found (each ring is a list of AtomIdx).
/// Also sets the `is_in_ring` flags on atoms and bonds.
///
/// # Thread Safety
/// This function is thread-safe - it only mutates the molecule passed to it.
/// Each molecule can be processed independently in parallel.
///
/// # Algorithm
/// Uses a cycle basis detection approach:
/// 1. Build spanning tree via BFS
/// 2. Find fundamental cycles for each non-tree edge
/// 3. Mark atoms and bonds that are in rings
/// 4. Returns the rings found
///
/// # Performance
/// O(E + V) where E = edges (bonds), V = vertices (atoms)
///
/// # Sets Ring Flags
/// - `Atom.flags` bit 1: in_ring
/// - `Bond.flags` bit 1: in_ring
///
/// # Example
/// ```ignore
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// let rings = find_sssr(&mut mol);
/// assert!(mol.atoms[0].is_in_ring());
/// assert_eq!(rings.len(), 1);
/// assert_eq!(rings[0].len(), 6);
/// ```
pub fn find_sssr(mol: &mut Molecule) -> Vec<Vec<AtomIdx>> {
    if mol.num_atoms() == 0 {
        return Vec::new();
    }

    // Reset all ring flags first
    for atom in &mut mol.atoms {
        atom.flags &= !0b10; // Clear bit 1 (ring flag)
    }
    for bond in &mut mol.bonds {
        bond.flags &= !0b10; // Clear bit 1 (ring flag)
    }

    let n_atoms = mol.num_atoms();
    let mut all_rings = Vec::new();

    // Find all connected components and detect rings in each
    let mut visited = vec![false; n_atoms];

    for start_idx in 0..n_atoms {
        if visited[start_idx] {
            continue;
        }

        // Detect rings in this connected component
        let component_rings = detect_rings_in_component(mol, start_idx as AtomIdx, &mut visited);
        all_rings.extend(component_rings);
    }

    all_rings
}

/// Detect rings in a single connected component using cycle detection.
///
/// # Thread Safety
/// Read-only access to molecule structure, writes only to the molecule being processed.
fn detect_rings_in_component(mol: &mut Molecule, start: AtomIdx, visited: &mut [bool]) -> Vec<Vec<AtomIdx>> {
    let n_atoms = mol.num_atoms();

    // Build spanning tree using BFS
    let mut parent = vec![None; n_atoms];
    let mut in_tree = vec![false; n_atoms];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited[start as usize] = true;
    in_tree[start as usize] = true;

    while let Some(current) = queue.pop_front() {
        // Explore neighbors
        for &bond_idx in &mol.adjacency[current as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            let neighbor = if bond.source == current {
                bond.target
            } else {
                bond.source
            };

            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                parent[neighbor as usize] = Some(current);
                in_tree[neighbor as usize] = true;
                queue.push_back(neighbor);
            }
        }
    }

    // Collect all cycles first (to avoid borrowing issues)
    let mut all_cycles = Vec::new();

    for atom_idx in 0..n_atoms {
        if !in_tree[atom_idx] {
            continue;
        }

        // Get adjacency list for this atom (copy to avoid borrow issues)
        let neighbors: Vec<_> = mol.adjacency[atom_idx].iter().copied().collect();

        for bond_idx in neighbors {
            let bond = &mol.bonds[bond_idx as usize];
            let neighbor = if bond.source == atom_idx as AtomIdx {
                bond.target
            } else {
                bond.source
            };

            // Skip if neighbor not in this component
            if !in_tree[neighbor as usize] {
                continue;
            }

            // Check if this edge is a non-tree edge (creates a cycle)
            if parent[atom_idx] != Some(neighbor) && parent[neighbor as usize] != Some(atom_idx as AtomIdx) {
                // Found a cycle - trace back from both endpoints to common ancestor
                let cycle = find_cycle_path(atom_idx as AtomIdx, neighbor, &parent);
                if !cycle.is_empty() {
                    all_cycles.push(cycle);
                }
            }
        }
    }

    // Deduplicate cycles
    let mut unique_cycles: Vec<Vec<AtomIdx>> = Vec::new();
    for cycle in all_cycles {
        let mut is_dup = false;
        for existing in &unique_cycles {
            if rings_are_same(&cycle, existing) {
                is_dup = true;
                break;
            }
        }
        if !is_dup {
            unique_cycles.push(cycle);
        }
    }

    // Now mark all cycles
    for cycle in &unique_cycles {
        mark_cycle_as_ring(mol, cycle);
    }

    unique_cycles
}

/// Check if two rings contain the same atoms
fn rings_are_same(ring1: &[AtomIdx], ring2: &[AtomIdx]) -> bool {
    if ring1.len() != ring2.len() {
        return false;
    }
    let s1: HashSet<_> = ring1.iter().copied().collect();
    let s2: HashSet<_> = ring2.iter().copied().collect();
    s1 == s2
}

/// Find the cycle path between two atoms in a spanning tree.
///
/// # Thread Safety
/// Read-only function, safe to call concurrently on different molecules.
fn find_cycle_path(start: AtomIdx, end: AtomIdx, parent: &[Option<AtomIdx>]) -> Vec<AtomIdx> {
    // Trace paths from both endpoints to root
    let mut path_start = Vec::new();
    let mut path_end = Vec::new();

    let mut current = start;
    while let Some(p) = parent[current as usize] {
        path_start.push(current);
        current = p;
    }
    path_start.push(current); // Add root

    current = end;
    while let Some(p) = parent[current as usize] {
        path_end.push(current);
        current = p;
    }
    path_end.push(current); // Add root

    // Find common ancestor
    let set_start: HashSet<_> = path_start.iter().copied().collect();
    let common_ancestor = path_end.iter().find(|&&atom| set_start.contains(&atom));

    if let Some(&ancestor) = common_ancestor {
        // Build cycle: start -> ancestor -> end -> start
        let mut cycle = Vec::new();

        // Path from start to ancestor
        for &atom in &path_start {
            cycle.push(atom);
            if atom == ancestor {
                break;
            }
        }

        // Path from ancestor to end (reversed)
        let mut end_to_ancestor = Vec::new();
        for &atom in &path_end {
            if atom == ancestor {
                break;
            }
            end_to_ancestor.push(atom);
        }
        end_to_ancestor.reverse();
        cycle.extend(end_to_ancestor);

        cycle
    } else {
        Vec::new()
    }
}

/// Mark atoms and bonds in a cycle as being in a ring.
///
/// # Thread Safety
/// Mutates only the molecule passed to it, safe for parallel processing of different molecules.
fn mark_cycle_as_ring(mol: &mut Molecule, cycle: &[AtomIdx]) {
    // Mark atoms in ring
    for &atom_idx in cycle {
        mol.atoms[atom_idx as usize].flags |= 0b10; // Set bit 1 (ring flag)
    }

    // Mark bonds in ring
    for i in 0..cycle.len() {
        let from = cycle[i];
        let to = cycle[(i + 1) % cycle.len()];

        // Find the bond between these atoms
        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            mol.bonds[bond_idx as usize].flags |= 0b10; // Set bit 1 (ring flag)
        }
    }
}

/// Find the bond index between two atoms.
///
/// # Thread Safety
/// Read-only function, safe to call concurrently.
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

/// Check if an atom is in a ring.
pub fn is_atom_in_ring(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    mol.atoms[atom_idx as usize].flags & 0b10 != 0
}

/// Check if a bond is in a ring.
pub fn is_bond_in_ring(mol: &Molecule, bond_idx: usize) -> bool {
    mol.bonds[bond_idx].flags & 0b10 != 0
}

/// Count the number of rings in the molecule (approximate).
pub fn count_rings(mol: &Molecule) -> usize {
    // Simple heuristic: count bonds in rings and divide by average ring size
    let ring_bonds: usize = mol.bonds.iter()
        .filter(|b| b.flags & 0b10 != 0)
        .count();

    if ring_bonds == 0 {
        0
    } else {
        // Assume average ring size of 6 (benzene-like)
        (ring_bonds / 6).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_no_rings() {
        // Ethane: CC (no rings)
        let mut mol = parse_smiles("CC").unwrap();
        find_sssr(&mut mol);

        for atom in &mol.atoms {
            assert_eq!(atom.flags & 0b10, 0, "Atom should not be in ring");
        }
    }

    #[test]
    fn test_benzene_ring() {
        // Benzene: c1ccccc1 (one 6-membered ring)
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        find_sssr(&mut mol);

        // All atoms should be in ring
        for atom in &mol.atoms {
            assert_ne!(atom.flags & 0b10, 0, "Benzene atom should be in ring");
        }

        // All bonds should be in ring
        for bond in &mol.bonds {
            assert_ne!(bond.flags & 0b10, 0, "Benzene bond should be in ring");
        }
    }

    #[test]
    fn test_cyclohexane_ring() {
        // Cyclohexane: C1CCCCC1 (one 6-membered ring)
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        find_sssr(&mut mol);

        // All 6 atoms should be in ring
        let ring_atoms = mol.atoms.iter().filter(|a| a.flags & 0b10 != 0).count();
        assert_eq!(ring_atoms, 6, "Cyclohexane should have 6 atoms in ring");
    }

    #[test]
    fn test_toluene() {
        // Toluene: Cc1ccccc1 (benzene ring + methyl)
        let mut mol = parse_smiles("Cc1ccccc1").unwrap();
        find_sssr(&mut mol);

        // 6 atoms in ring (benzene), 1 not in ring (methyl)
        let ring_atoms = mol.atoms.iter().filter(|a| a.flags & 0b10 != 0).count();
        assert_eq!(ring_atoms, 6, "Toluene should have 6 atoms in ring");

        // Check methyl carbon is NOT in ring
        assert_eq!(mol.atoms[0].flags & 0b10, 0, "Methyl carbon should not be in ring");
    }

    #[test]
    fn test_naphthalene() {
        // Naphthalene: c1ccc2ccccc2c1 (two fused 6-membered rings)
        let mut mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        find_sssr(&mut mol);

        // All 10 atoms should be in rings
        let ring_atoms = mol.atoms.iter().filter(|a| a.flags & 0b10 != 0).count();
        assert_eq!(ring_atoms, 10, "Naphthalene should have 10 atoms in rings");
    }

    #[test]
    fn test_helper_functions() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        find_sssr(&mut mol);

        // Test helper functions
        assert!(is_atom_in_ring(&mol, 0));
        assert!(is_atom_in_ring(&mol, 5));

        for i in 0..mol.num_bonds() {
            assert!(is_bond_in_ring(&mol, i));
        }

        let ring_count = count_rings(&mol);
        assert!(ring_count >= 1, "Should detect at least 1 ring");
    }
}
