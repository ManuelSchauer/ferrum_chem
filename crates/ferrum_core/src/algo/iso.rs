use crate::mol::molecule::Molecule;
use crate::io::smarts::{QueryGraph, QueryAtom, QueryBond};

/// Substructure matching using VF2 algorithm.
///
/// Finds if the query pattern exists as a subgraph in the target molecule.
/// Returns true if at least one match is found.
///
/// # Example
/// ```ignore
/// let molecule = parse_smiles("CCO").unwrap();
/// let pattern = QueryGraph::single_atom(6); // Match any carbon
/// assert!(has_substruct_match(&molecule, &pattern));
/// ```
pub fn has_substruct_match(target: &Molecule, pattern: &QueryGraph) -> bool {
    find_first_match(target, pattern).is_some()
}

/// Finds the first substructure match and returns the atom mapping.
///
/// Returns a vector where index i is the pattern atom index,
/// and the value is the corresponding target atom index.
pub fn find_first_match(target: &Molecule, pattern: &QueryGraph) -> Option<Vec<u16>> {
    // Empty pattern matches everything
    if pattern.atoms.is_empty() {
        return Some(vec![]);
    }
    
    // Pattern larger than target cannot match
    if pattern.atoms.len() > target.atoms.len() {
        return None;
    }
    
    let mut state = MatchState::new(target, pattern);
    
    // Try matching starting from each atom in the target
    if backtrack(&mut state, 0) {
        Some(state.mapping.clone())
    } else {
        None
    }
}

/// Counts the number of substructure matches.
pub fn count_matches(target: &Molecule, pattern: &QueryGraph) -> usize {
    find_all_matches(target, pattern).len()
}

/// Finds all substructure matches.
///
/// Returns a vector of mappings, where each mapping is a vector
/// of target atom indices corresponding to pattern atoms.
pub fn find_all_matches(target: &Molecule, pattern: &QueryGraph) -> Vec<Vec<u16>> {
    if pattern.atoms.is_empty() {
        return vec![vec![]];
    }
    
    if pattern.atoms.len() > target.atoms.len() {
        return vec![];
    }
    
    let mut matches = Vec::new();
    let mut state = MatchState::new(target, pattern);
    
    backtrack_all(&mut state, 0, &mut matches);
    
    matches
}

/// State for VF2 algorithm.
struct MatchState<'a> {
    target: &'a Molecule,
    pattern: &'a QueryGraph,
    mapping: Vec<u16>,          // pattern_idx -> target_idx
    reverse_mapping: Vec<Option<u16>>, // target_idx -> Some(pattern_idx) or None
}

impl<'a> MatchState<'a> {
    fn new(target: &'a Molecule, pattern: &'a QueryGraph) -> Self {
        Self {
            target,
            pattern,
            mapping: vec![u16::MAX; pattern.atoms.len()],
            reverse_mapping: vec![None; target.atoms.len()],
        }
    }
    
    /// Checks if a pattern atom can be mapped to a target atom.
    fn is_feasible(&self, pattern_idx: usize, target_idx: u16) -> bool {
        // Check if target atom already mapped
        if self.reverse_mapping[target_idx as usize].is_some() {
            return false;
        }
        
        // Check atom compatibility
        if !self.pattern.atoms[pattern_idx].matches(self.target, target_idx) {
            return false;
        }
        
        // Check bond compatibility with already mapped neighbors
        for (neighbor_pattern_idx, neighbor_target_idx) in self.mapped_neighbors(pattern_idx) {
            if !self.check_bond_compatibility(pattern_idx, neighbor_pattern_idx,
                                              target_idx, neighbor_target_idx) {
                return false;
            }
        }
        
        true
    }
    
    /// Returns iterator over mapped neighbors of a pattern atom.
    fn mapped_neighbors(&self, pattern_idx: usize) -> Vec<(usize, u16)> {
        self.pattern.adjacency[pattern_idx]
            .iter()
            .filter_map(|&bond_idx| {
                let bond = &self.pattern.bonds[bond_idx as usize];
                let neighbor_idx = if bond.source == pattern_idx as u16 {
                    bond.target as usize
                } else {
                    bond.source as usize
                };
                
                let mapped_target = self.mapping[neighbor_idx];
                if mapped_target != u16::MAX {
                    Some((neighbor_idx, mapped_target))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Checks if bonds between pattern atoms match bonds between target atoms.
    fn check_bond_compatibility(&self, p1: usize, p2: usize, t1: u16, t2: u16) -> bool {
        // Find bond in pattern
        let pattern_bond_idx = self.pattern.adjacency[p1]
            .iter()
            .find(|&&bond_idx| {
                let bond = &self.pattern.bonds[bond_idx as usize];
                (bond.source == p1 as u16 && bond.target == p2 as u16) ||
                (bond.target == p1 as u16 && bond.source == p2 as u16)
            });
        
        if pattern_bond_idx.is_none() {
            return false;
        }
        
        let pattern_bond_idx = *pattern_bond_idx.unwrap();
        let pattern_bond = &self.pattern.bonds[pattern_bond_idx as usize];
        
        // Find bond in target
        let target_bond_idx = self.target.adjacency[t1 as usize]
            .iter()
            .find(|&&bond_idx| {
                let bond = &self.target.bonds[bond_idx as usize];
                (bond.source == t1 && bond.target == t2) ||
                (bond.target == t1 && bond.source == t2)
            });
        
        if target_bond_idx.is_none() {
            return false;
        }
        
        let target_bond_idx = *target_bond_idx.unwrap();
        
        // Check if bond types match
        pattern_bond.query.matches(self.target, target_bond_idx)
    }
    
    /// Adds a mapping.
    fn add_pair(&mut self, pattern_idx: usize, target_idx: u16) {
        self.mapping[pattern_idx] = target_idx;
        self.reverse_mapping[target_idx as usize] = Some(pattern_idx as u16);
    }
    
    /// Removes a mapping.
    fn remove_pair(&mut self, pattern_idx: usize) {
        let target_idx = self.mapping[pattern_idx];
        if target_idx != u16::MAX {
            self.mapping[pattern_idx] = u16::MAX;
            self.reverse_mapping[target_idx as usize] = None;
        }
    }
}

/// VF2 backtracking to find first match.
fn backtrack(state: &mut MatchState, pattern_idx: usize) -> bool {
    // All pattern atoms mapped - success!
    if pattern_idx >= state.pattern.atoms.len() {
        return true;
    }
    
    // Try mapping current pattern atom to each target atom
    for target_idx in 0..state.target.atoms.len() {
        let target_idx = target_idx as u16;
        
        if state.is_feasible(pattern_idx, target_idx) {
            state.add_pair(pattern_idx, target_idx);
            
            if backtrack(state, pattern_idx + 1) {
                return true;
            }
            
            state.remove_pair(pattern_idx);
        }
    }
    
    false
}

/// VF2 backtracking to find all matches.
fn backtrack_all(state: &mut MatchState, pattern_idx: usize, matches: &mut Vec<Vec<u16>>) {
    // All pattern atoms mapped - record this match
    if pattern_idx >= state.pattern.atoms.len() {
        matches.push(state.mapping.clone());
        return;
    }
    
    // Try mapping current pattern atom to each target atom
    for target_idx in 0..state.target.atoms.len() {
        let target_idx = target_idx as u16;
        
        if state.is_feasible(pattern_idx, target_idx) {
            state.add_pair(pattern_idx, target_idx);
            backtrack_all(state, pattern_idx + 1, matches);
            state.remove_pair(pattern_idx);
        }
    }
}

/// Helper to create simple query graphs for testing.
impl QueryGraph {
    /// Creates a query that matches a single atom of given element.
    pub fn single_atom(element: u8) -> Self {
        Self {
            atoms: vec![QueryAtom::element(element)],
            bonds: vec![],
            adjacency: vec![smallvec::SmallVec::new()],
        }
    }
    
    /// Creates a query for two bonded atoms.
    pub fn two_atoms(elem1: u8, elem2: u8, bond: QueryBond) -> Self {
        use crate::io::smarts::QueryGraphBond;
        
        Self {
            atoms: vec![
                QueryAtom::element(elem1),
                QueryAtom::element(elem2),
            ],
            bonds: vec![QueryGraphBond {
                source: 0,
                target: 1,
                query: bond,
            }],
            adjacency: vec![
                smallvec::smallvec![0],
                smallvec::smallvec![0],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::io::smarts::QueryBond;
    
    #[test]
    fn test_single_atom_match() {
        let mol = parse_smiles("CCO").unwrap();
        
        // Match carbon
        let pattern = QueryGraph::single_atom(6);
        assert!(has_substruct_match(&mol, &pattern));
        
        let matches = find_all_matches(&mol, &pattern);
        assert_eq!(matches.len(), 2); // Two carbons
        
        // Match oxygen
        let pattern = QueryGraph::single_atom(8);
        assert!(has_substruct_match(&mol, &pattern));
        
        let matches = find_all_matches(&mol, &pattern);
        assert_eq!(matches.len(), 1); // One oxygen
    }
    
    #[test]
    fn test_no_match() {
        let mol = parse_smiles("CCO").unwrap();
        
        // No nitrogen in molecule
        let pattern = QueryGraph::single_atom(7);
        assert!(!has_substruct_match(&mol, &pattern));
    }
    
    #[test]
    fn test_two_atom_match() {
        let mol = parse_smiles("CCO").unwrap();
        
        // Match C-C bond
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert!(has_substruct_match(&mol, &pattern));
        
        // Match C-O bond
        let pattern = QueryGraph::two_atoms(6, 8, QueryBond::Single);
        assert!(has_substruct_match(&mol, &pattern));
    }
    
    #[test]
    fn test_count_matches() {
        let mol = parse_smiles("CCCC").unwrap(); // Butane
        
        // Count carbons
        let pattern = QueryGraph::single_atom(6);
        assert_eq!(count_matches(&mol, &pattern), 4);
        
        // Count C-C bonds - matches both directions (C1-C2 and C2-C1)
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert_eq!(count_matches(&mol, &pattern), 6); // 3 bonds * 2 directions
    }
    
    #[test]
    fn test_double_bond_match() {
        let mol = parse_smiles("C=C").unwrap(); // Ethene
        
        // Should match double bond
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Double);
        assert!(has_substruct_match(&mol, &pattern));
        
        // Should not match single bond
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert!(!has_substruct_match(&mol, &pattern));
    }
    
    #[test]
    fn test_aromatic_pattern() {
        use crate::algo::aromaticity;
        
        let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene
        aromaticity::perceive_aromaticity(&mut mol);
        
        // Should match aromatic carbon
        let pattern = QueryGraph {
            atoms: vec![QueryAtom::aromatic()],
            bonds: vec![],
            adjacency: vec![smallvec::SmallVec::new()],
        };
        
        assert!(has_substruct_match(&mol, &pattern));
        assert_eq!(count_matches(&mol, &pattern), 6); // 6 aromatic carbons
    }
    
    #[test]
    fn test_complex_pattern() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol
        
        // Find first match returns valid mapping
        let pattern = QueryGraph::two_atoms(6, 8, QueryBond::Single);
        let mapping = find_first_match(&mol, &pattern);
        
        assert!(mapping.is_some());
        let mapping = mapping.unwrap();
        assert_eq!(mapping.len(), 2);
        
        // Verify mapped atoms are carbon and oxygen
        let target = &mol;
        assert_eq!(target.atoms[mapping[0] as usize].atomic_number, 6); // Carbon
        assert_eq!(target.atoms[mapping[1] as usize].atomic_number, 8); // Oxygen
    }
}
