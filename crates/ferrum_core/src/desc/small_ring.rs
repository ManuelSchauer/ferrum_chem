use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use crate::desc::typed::{Descriptor, DescriptorValue};
use std::collections::HashSet;

/// Small ring descriptor - analyzes ring structure in molecules.
///
/// Returns IntU16Vec with 11 values:
/// - [0]: Total small rings (3-9 members)
/// - [1]: Aromatic rings
/// - [2]: Ring blocks (distinct ring systems)
/// - [3]: Aromatic blocks
/// - [4-10]: Individual ring sizes (3, 4, 5, 6, 7, 8, 9)
///
/// Based on CDK's SmallRingDescriptor implementation.
///
/// # Thread Safety
/// Thread-safe: stateless and uses only immutable references.
pub struct SmallRingDescriptor;

impl Descriptor for SmallRingDescriptor {
    fn name(&self) -> &str {
        "SmallRingDescriptor"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut counts = vec![0u16; 11];
        
        // Find all small rings (3-9 members)
        let rings = find_small_rings(mol, 3, 9);
        
        // Count total small rings
        counts[0] = rings.len() as u16;
        
        // Count rings by size
        let mut size_counts = [0u16; 7]; // indices 0-6 for sizes 3-9
        for ring in &rings {
            let size = ring.len();
            if size >= 3 && size <= 9 {
                size_counts[size - 3] += 1;
            }
        }
        
        // Set individual size counts (indices 4-10)
        for i in 0..7 {
            counts[4 + i] = size_counts[i];
        }
        
        // Count aromatic rings
        let arom_ring_count = count_aromatic_rings(mol, &rings);
        counts[1] = arom_ring_count as u16;
        
        // Count ring blocks (distinct ring systems)
        let ring_blocks = count_ring_blocks(mol);
        counts[2] = ring_blocks as u16;
        
        // Count aromatically connected components
        let arom_blocks = count_aromatic_components(mol);
        counts[3] = arom_blocks as u16;
        
        DescriptorValue::IntU16Vec(counts)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<SmallRingDescriptor>;
};

/// Find all small rings in a molecule (sizes min_size to max_size).
fn find_small_rings(mol: &Molecule, min_size: usize, max_size: usize) -> Vec<Vec<AtomIdx>> {
    let n_atoms = mol.num_atoms();
    if n_atoms == 0 {
        return Vec::new();
    }
    
    // Mark atoms in rings using ring blocks
    let ring_blocks = identify_ring_blocks(mol);
    
    let mut rings = Vec::new();
    
    // Search for rings of each size
    for ring_size in min_size..=max_size {
        for start_atom in 0..n_atoms {
            let start_idx = start_atom as AtomIdx;
            
            // Skip if not in a ring
            if ring_blocks[start_atom] == 0 {
                continue;
            }
            
            // Try to find a ring starting from this atom
            let mut path = vec![start_idx];
            let block_id = ring_blocks[start_atom];
            
            find_rings_recursive(
                mol,
                &mut path,
                ring_size,
                start_idx,
                block_id,
                &ring_blocks,
                &mut rings,
            );
        }
    }
    
    // Deduplicate rings
    deduplicate_rings(rings)
}

/// Recursively search for rings of a specific size.
fn find_rings_recursive(
    mol: &Molecule,
    path: &mut Vec<AtomIdx>,
    target_size: usize,
    start_atom: AtomIdx,
    block_id: usize,
    ring_blocks: &[usize],
    rings: &mut Vec<Vec<AtomIdx>>,
) {
    let current = *path.last().unwrap();
    
    // If we've reached target size, check if we can close the ring
    if path.len() == target_size {
        // Check if current atom is connected to start
        for &bond_idx in &mol.adjacency[current as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            let neighbor = if bond.source == current {
                bond.target
            } else {
                bond.source
            };
            
            if neighbor == start_atom {
                // Found a ring!
                rings.push(path.clone());
                return;
            }
        }
        return;
    }
    
    // Not at target size yet, explore neighbors
    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let neighbor = if bond.source == current {
            bond.target
        } else {
            bond.source
        };
        
        // Skip if not in same ring block
        if ring_blocks[neighbor as usize] != block_id {
            continue;
        }
        
        // Skip if already in path (except we allow start atom at the end)
        if path.len() < target_size && path.contains(&neighbor) {
            continue;
        }
        
        // Add to path and recurse
        path.push(neighbor);
        find_rings_recursive(
            mol,
            path,
            target_size,
            start_atom,
            block_id,
            ring_blocks,
            rings,
        );
        path.pop();
    }
}

/// Identify which atoms belong to which ring blocks (ring systems).
/// Returns a vector where 0 = not in ring, 1..N = ring block ID.
fn identify_ring_blocks(mol: &Molecule) -> Vec<usize> {
    let n_atoms = mol.num_atoms();
    let mut ring_blocks = vec![0; n_atoms];
    let mut visited = vec![false; n_atoms];
    
    // Use DFS to find cycles and mark ring blocks
    let mut path = Vec::new();
    let mut next_block_id = 1;
    
    for start in 0..n_atoms {
        if visited[start] {
            continue;
        }
        
        path.clear();
        path.push(start as AtomIdx);
        visited[start] = true;
        
        dfs_mark_rings(
            mol,
            &mut path,
            &mut visited,
            &mut ring_blocks,
            &mut next_block_id,
        );
    }
    
    ring_blocks
}

/// DFS helper to mark ring blocks.
fn dfs_mark_rings(
    mol: &Molecule,
    path: &mut Vec<AtomIdx>,
    visited: &mut [bool],
    ring_blocks: &mut [usize],
    next_block_id: &mut usize,
) {
    let current = *path.last().unwrap();
    
    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let neighbor = if bond.source == current {
            bond.target
        } else {
            bond.source
        };
        
        if path.len() >= 2 && neighbor == path[path.len() - 2] {
            // Skip back edge to parent
            continue;
        }
        
        if visited[neighbor as usize] {
            // Found a cycle - mark all atoms in the cycle
            if let Some(pos) = path.iter().position(|&a| a == neighbor) {
                let block_id = *next_block_id;
                *next_block_id += 1;
                
                for &atom in &path[pos..] {
                    if ring_blocks[atom as usize] == 0 {
                        ring_blocks[atom as usize] = block_id;
                    } else {
                        // Merge ring blocks
                        let old_id = ring_blocks[atom as usize];
                        for rb in ring_blocks.iter_mut() {
                            if *rb == old_id {
                                *rb = block_id;
                            }
                        }
                    }
                }
            }
        } else {
            // Continue DFS
            visited[neighbor as usize] = true;
            path.push(neighbor);
            dfs_mark_rings(mol, path, visited, ring_blocks, next_block_id);
            path.pop();
        }
    }
}

/// Count the number of distinct ring blocks (ring systems).
fn count_ring_blocks(mol: &Molecule) -> usize {
    let ring_blocks = identify_ring_blocks(mol);
    let unique_blocks: HashSet<usize> = ring_blocks.iter()
        .filter(|&&id| id > 0)
        .copied()
        .collect();
    unique_blocks.len()
}

/// Count aromatic rings.
fn count_aromatic_rings(mol: &Molecule, rings: &[Vec<AtomIdx>]) -> usize {
    rings.iter()
        .filter(|ring| is_ring_aromatic(mol, ring))
        .count()
}

/// Check if a ring is aromatic (all bonds in the ring are aromatic).
fn is_ring_aromatic(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    if ring.is_empty() {
        return false;
    }
    
    for i in 0..ring.len() {
        let atom1 = ring[i];
        let atom2 = ring[(i + 1) % ring.len()];
        
        // Find bond between these atoms
        let mut bond_found = false;
        let mut bond_aromatic = false;
        
        for &bond_idx in &mol.adjacency[atom1 as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            if (bond.source == atom1 && bond.target == atom2) ||
               (bond.source == atom2 && bond.target == atom1) {
                bond_found = true;
                bond_aromatic = bond.bond_type == crate::types::BondType::Aromatic;
                break;
            }
        }
        
        if !bond_found || !bond_aromatic {
            return false;
        }
    }
    
    true
}

/// Count aromatically connected components.
fn count_aromatic_components(mol: &Molecule) -> usize {
    let n_atoms = mol.num_atoms();
    let mut visited = vec![false; n_atoms];
    let mut component_count = 0;
    
    for start in 0..n_atoms {
        if visited[start] || !mol.atoms[start].is_aromatic() {
            continue;
        }
        
        // Found a new aromatic component
        component_count += 1;
        
        // BFS to mark all atoms in this aromatic component
        let mut queue = vec![start as AtomIdx];
        visited[start] = true;
        
        while let Some(current) = queue.pop() {
            for &bond_idx in &mol.adjacency[current as usize] {
                let bond = &mol.bonds[bond_idx as usize];
                
                // Only follow aromatic bonds
                if bond.bond_type != crate::types::BondType::Aromatic {
                    continue;
                }
                
                let neighbor = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                };
                
                if !visited[neighbor as usize] && mol.atoms[neighbor as usize].is_aromatic() {
                    visited[neighbor as usize] = true;
                    queue.push(neighbor);
                }
            }
        }
    }
    
    component_count
}

/// Deduplicate rings (rings with same atoms but different starting points).
fn deduplicate_rings(mut rings: Vec<Vec<AtomIdx>>) -> Vec<Vec<AtomIdx>> {
    let mut unique = Vec::new();
    let mut seen_sets: Vec<HashSet<AtomIdx>> = Vec::new();
    
    for ring in rings.drain(..) {
        let ring_set: HashSet<AtomIdx> = ring.iter().copied().collect();
        
        // Check if we've seen this set of atoms before
        if !seen_sets.iter().any(|s| s == &ring_set) {
            seen_sets.push(ring_set);
            unique.push(ring);
        }
    }
    
    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::valency::assign_implicit_hydrogens;
    use crate::algo::rings::find_sssr;
    use crate::algo::aromaticity::perceive_aromaticity;
    
    fn prepare_molecule(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        find_sssr(&mut mol);
        perceive_aromaticity(&mut mol);
        mol
    }

    fn calc_small_ring(smiles: &str) -> Vec<u16> {
        let mut mol = prepare_molecule(smiles);
        match SmallRingDescriptor.calculate(&mut mol) {
            DescriptorValue::IntU16Vec(v) => v,
            other => panic!("expected IntU16Vec, got {:?}", other),
        }
    }
    
    #[test]
    fn test_benzene() {
        let output = calc_small_ring("c1ccccc1");
        
        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[1], 1, "Should have 1 aromatic ring");
        assert_eq!(output[2], 1, "Should have 1 ring block");
        assert!(output[3] >= 1, "Should have at least 1 aromatic component");
        assert_eq!(output[4 + 3], 1, "Should have 1 six-membered ring (index 7)");
    }
    
    #[test]
    fn test_naphthalene() {
        let output = calc_small_ring("c1ccc2ccccc2c1");
        
        assert!(output[0] >= 2, "Should have at least 2 small rings");
        assert!(output[1] >= 2, "Should have at least 2 aromatic rings");
        assert_eq!(output[2], 1, "Should have 1 ring block (fused system)");
        assert_eq!(output[3], 1, "Should have 1 aromatic component");
    }
    
    #[test]
    fn test_cyclohexane() {
        let output = calc_small_ring("C1CCCCC1");
        
        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[1], 0, "Should have 0 aromatic rings");
        assert_eq!(output[2], 1, "Should have 1 ring block");
        assert_eq!(output[3], 0, "Should have 0 aromatic components");
        assert_eq!(output[4 + 3], 1, "Should have 1 six-membered ring");
    }
    
    #[test]
    fn test_cyclopropane() {
        let output = calc_small_ring("C1CC1");
        
        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[4], 1, "Should have 1 three-membered ring (index 4)");
    }
    
    #[test]
    fn test_no_rings() {
        let output = calc_small_ring("CCCC");
        
        for i in 0..11 {
            assert_eq!(output[i], 0, "All counts should be 0 for non-cyclic molecule");
        }
    }
}
