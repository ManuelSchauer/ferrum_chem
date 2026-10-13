//! Components algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.

use crate::mol::molecule::Molecule;
use std::collections::VecDeque;

/// Find connected components via BFS. Returns a vec of component labels (one per atom).
fn connected_components(mol: &Molecule) -> Vec<usize> {
    let n = mol.atoms.len();
    let mut comp = vec![usize::MAX; n];
    let mut label = 0;
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        comp[start] = label;
        while let Some(u) = queue.pop_front() {
            for &bi in &mol.adjacency[u] {
                let b = &mol.bonds[bi as usize];
                let v = if b.source as usize == u {
                    b.target as usize
                } else {
                    b.source as usize
                };
                if comp[v] == usize::MAX {
                    comp[v] = label;
                    queue.push_back(v);
                }
            }
        }
        label += 1;
    }
    comp
}

/// Returns the number of disconnected fragments.
pub fn num_fragments(mol: &Molecule) -> usize {
    if mol.atoms.is_empty() {
        return 0;
    }
    let comp = connected_components(mol);
    *comp.iter().max().unwrap_or(&0) + 1
}

/// Removes salts/solvents by keeping only the largest connected fragment.
pub fn keep_largest_fragment(mol: &mut Molecule) {
    mol.invalidate_cached_properties();
    if mol.atoms.len() <= 1 {
        return;
    }
    let comp = connected_components(mol);
    let n_comp = *comp.iter().max().unwrap_or(&0) + 1;
    if n_comp <= 1 {
        return;
    }
    // Find largest component by heavy atom count
    let mut sizes = vec![0usize; n_comp];
    for &c in &comp {
        sizes[c] += 1;
    }
    let keep = sizes.iter().enumerate().max_by_key(|(_, s)| **s).unwrap().0;
    // Collect atoms to keep, build index map
    let mut new_idx = vec![usize::MAX; mol.atoms.len()];
    let mut kept_atoms = Vec::new();
    for (i, &c) in comp.iter().enumerate() {
        if c == keep {
            new_idx[i] = kept_atoms.len();
            kept_atoms.push(mol.atoms[i]);
        }
    }
    // Keep only bonds where both atoms are in the kept component
    let mut kept_bonds = Vec::new();
    for b in &mol.bonds {
        let s = new_idx[b.source as usize];
        let t = new_idx[b.target as usize];
        if s != usize::MAX && t != usize::MAX {
            let mut nb = *b;
            nb.source = s as u16;
            nb.target = t as u16;
            kept_bonds.push(nb);
        }
    }
    // Rebuild adjacency
    let mut adj = vec![smallvec::smallvec![]; kept_atoms.len()];
    for (bi, b) in kept_bonds.iter().enumerate() {
        adj[b.source as usize].push(bi as u16);
        adj[b.target as usize].push(bi as u16);
    }
    mol.atoms = kept_atoms;
    mol.bonds = kept_bonds;
    mol.adjacency = adj;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_single_fragment() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(num_fragments(&mol), 1);
    }

    #[test]
    fn test_two_fragments() {
        let mol = parse_smiles("[Na+].[Cl-]").unwrap();
        assert_eq!(num_fragments(&mol), 2);
    }

    #[test]
    fn test_three_fragments() {
        let mol = parse_smiles("[Na+].O.[Cl-]").unwrap();
        assert_eq!(num_fragments(&mol), 3);
    }

    #[test]
    fn test_keep_largest_fragment() {
        let mut mol = parse_smiles("[Na+].CCCCCC.[Cl-]").unwrap();
        assert_eq!(num_fragments(&mol), 3);
        keep_largest_fragment(&mut mol);
        assert_eq!(num_fragments(&mol), 1);
        assert_eq!(mol.atoms.len(), 6); // hexane has 6 heavy atoms
    }

    #[test]
    fn test_keep_largest_no_change() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        let orig_atoms = mol.atoms.len();
        keep_largest_fragment(&mut mol);
        assert_eq!(mol.atoms.len(), orig_atoms);
    }

    #[test]
    fn test_keep_largest_salt_strip() {
        let mut mol = parse_smiles("[Na+].OC(=O)c1ccccc1").unwrap();
        keep_largest_fragment(&mut mol);
        assert_eq!(num_fragments(&mol), 1);
        // Benzoic acid: 9 heavy atoms (7C + 2O)
        assert_eq!(mol.atoms.len(), 9);
    }
}
