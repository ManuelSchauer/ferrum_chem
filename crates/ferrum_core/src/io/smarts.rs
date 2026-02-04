use crate::error::{Result, FerrumError};
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use smallvec::SmallVec;

/// Query graph for SMARTS pattern matching.
/// Represents a molecular pattern with flexible matching criteria.
#[derive(Debug, Clone)]
pub struct QueryGraph {
    pub atoms: Vec<QueryAtom>,
    pub bonds: Vec<QueryGraphBond>,
    pub adjacency: Vec<SmallVec<[u16; 4]>>,
}

/// Bond in query graph (stores connectivity).
#[derive(Debug, Clone)]
pub struct QueryGraphBond {
    pub source: u16,
    pub target: u16,
    pub query: QueryBond,
}

/// Query atom with flexible matching primitives.
#[derive(Debug, Clone)]
pub struct QueryAtom {
    pub primitives: Vec<AtomPrimitive>,
}

/// Atom matching primitives for SMARTS queries.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomPrimitive {
    /// Match specific element by atomic number
    Element(u8),
    
    /// Match any atom (*)
    AnyAtom,
    
    /// Match any aliphatic atom (A)
    AnyAliphatic,
    
    /// Match any aromatic atom (a)
    AnyAromatic,
    
    /// Match aromatic atom of this element
    Aromatic(u8),
    
    /// Match aliphatic atom of this element
    Aliphatic(u8),
    
    /// Logical NOT
    Not(Box<AtomPrimitive>),
    
    /// Logical OR (one of these must match)
    Or(Vec<AtomPrimitive>),
    
    /// Logical AND (all must match)
    And(Vec<AtomPrimitive>),
    
    /// Hydrogen count [H0], [H1], [H2]
    HCount(u8),
    
    /// Total hydrogen count (explicit + implicit) [h0], [h1]
    TotalHCount(u8),
    
    /// Degree (number of connections) [D1], [D2], [D3]
    Degree(u8),
    
    /// Formal charge [+], [-], [+2], [-1]
    Charge(i8),
    
    /// In any ring (R)
    InRing,
    
    /// Not in ring (R0)
    NotInRing,
    
    /// In ring of specific size [r5], [r6]
    RingSize(u8),
    
    /// Valence [v3], [v4]
    Valence(u8),
    
    /// Number of ring bonds [x2], [x3]
    RingBonds(u8),
}

/// Query bond for pattern matching.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryBond {
    /// Any bond (~)
    Any,
    
    /// Single bond (-)
    Single,
    
    /// Double bond (=)
    Double,
    
    /// Triple bond (#)
    Triple,
    
    /// Aromatic bond (:)
    Aromatic,
    
    /// Single or aromatic (-,:)
    SingleOrAromatic,
    
    /// Double or aromatic (=,:)
    DoubleOrAromatic,
    
    /// Ring bond (@)
    Ring,
    
    /// Not this bond type
    Not(Box<QueryBond>),
}

impl QueryAtom {
    /// Creates a query atom that matches any atom.
    pub fn any() -> Self {
        Self {
            primitives: vec![AtomPrimitive::AnyAtom],
        }
    }
    
    /// Creates a query atom that matches a specific element.
    pub fn element(atomic_number: u8) -> Self {
        Self {
            primitives: vec![AtomPrimitive::Element(atomic_number)],
        }
    }
    
    /// Creates a query atom that matches any aromatic atom.
    pub fn aromatic() -> Self {
        Self {
            primitives: vec![AtomPrimitive::AnyAromatic],
        }
    }
    
    /// Creates a query atom that matches aromatic atoms.
    pub fn aromatic_element(atomic_number: u8) -> Self {
        Self {
            primitives: vec![AtomPrimitive::Aromatic(atomic_number)],
        }
    }
    
    /// Checks if this query atom matches a molecule atom.
    pub fn matches(&self, mol: &Molecule, atom_idx: u16) -> bool {
        let atom = &mol.atoms[atom_idx as usize];
        
        // All primitives must match
        for primitive in &self.primitives {
            if !match_primitive(primitive, mol, atom_idx, atom) {
                return false;
            }
        }
        
        true
    }
}

/// Matches a single atom primitive against a molecule atom.
fn match_primitive(
    primitive: &AtomPrimitive,
    mol: &Molecule,
    atom_idx: u16,
    atom: &crate::mol::atom::Atom,
) -> bool {
    match primitive {
        AtomPrimitive::Element(z) => atom.atomic_number == *z,
        
        AtomPrimitive::AnyAtom => true,
        
        AtomPrimitive::AnyAliphatic => !atom.is_aromatic(),
        
        AtomPrimitive::AnyAromatic => atom.is_aromatic(),
        
        AtomPrimitive::Aromatic(z) => {
            atom.atomic_number == *z && atom.is_aromatic()
        }
        
        AtomPrimitive::Aliphatic(z) => {
            atom.atomic_number == *z && !atom.is_aromatic()
        }
        
        AtomPrimitive::Not(inner) => {
            !match_primitive(inner, mol, atom_idx, atom)
        }
        
        AtomPrimitive::Or(options) => {
            options.iter().any(|p| match_primitive(p, mol, atom_idx, atom))
        }
        
        AtomPrimitive::And(requirements) => {
            requirements.iter().all(|p| match_primitive(p, mol, atom_idx, atom))
        }
        
        AtomPrimitive::HCount(n) => {
            // In SMARTS, H without explicit/implicit distinction means total H count
            atom.total_h() == *n
        }
        
        AtomPrimitive::TotalHCount(n) => {
            atom.total_h() == *n
        }
        
        AtomPrimitive::Degree(n) => {
            mol.get_degree(atom_idx) == *n as usize
        }
        
        AtomPrimitive::Charge(charge) => {
            atom.formal_charge == *charge
        }
        
        AtomPrimitive::InRing => atom.is_in_ring(),
        
        AtomPrimitive::NotInRing => !atom.is_in_ring(),
        
        AtomPrimitive::RingSize(_size) => {
            // Requires ring size calculation - simplified for now
            atom.is_in_ring()
        }
        
        AtomPrimitive::Valence(v) => {
            // Calculate valence: sum of bond orders + implicit hydrogens
            let mut valence = 0u8;
            
            // Add bond orders
            for &bond_idx in &mol.adjacency[atom_idx as usize] {
                let bond = &mol.bonds[bond_idx as usize];
                valence += match bond.bond_type {
                    crate::types::BondType::Single => 1,
                    crate::types::BondType::Double => 2,
                    crate::types::BondType::Triple => 3,
                    crate::types::BondType::Aromatic => 1, // Aromatic bonds count as 1 for valence
                    crate::types::BondType::Dative => 1,
                    crate::types::BondType::Unknown => 1,
                };
            }
            
            // Add hydrogens (implicit + explicit)
            valence += atom.total_h();
            
            valence == *v
        }
        
        AtomPrimitive::RingBonds(n) => {
            // Count bonds connected to this atom that are in rings
            let ring_bond_count = mol.adjacency[atom_idx as usize]
                .iter()
                .filter(|&&bond_idx| mol.bonds[bond_idx as usize].is_in_ring())
                .count();
            
            ring_bond_count == *n as usize
        }
    }
}

impl QueryBond {
    /// Checks if this query bond matches a molecule bond.
    pub fn matches(&self, mol: &Molecule, bond_idx: u16) -> bool {
        let bond = &mol.bonds[bond_idx as usize];
        
        match self {
            QueryBond::Any => true,
            
            QueryBond::Single => bond.bond_type == BondType::Single,
            
            QueryBond::Double => bond.bond_type == BondType::Double,
            
            QueryBond::Triple => bond.bond_type == BondType::Triple,
            
            QueryBond::Aromatic => bond.bond_type == BondType::Aromatic,
            
            QueryBond::SingleOrAromatic => {
                bond.bond_type == BondType::Single || 
                bond.bond_type == BondType::Aromatic
            }
            
            QueryBond::DoubleOrAromatic => {
                bond.bond_type == BondType::Double || 
                bond.bond_type == BondType::Aromatic
            }
            
            QueryBond::Ring => bond.is_in_ring(),
            
            QueryBond::Not(inner) => !inner.matches(mol, bond_idx),
        }
    }
}

/// Parses a SMARTS pattern into a QueryGraph.
///
/// # Example
/// ```ignore
/// let pattern = parse_smarts("[CH3]")?;  // Matches methyl group
/// let pattern = parse_smarts("c1ccccc1")?;  // Matches benzene
/// let pattern = parse_smarts("[N,O]")?;  // Matches N or O
/// ```
pub fn parse_smarts(smarts: &str) -> Result<QueryGraph> {
    let mut parser = SmartsParser::new(smarts);
    parser.parse().map_err(|e| FerrumError::ParseError(format!("SMARTS: {}", e)))
}

struct SmartsParser {
    input: Vec<char>,
    pos: usize,
    atoms: Vec<QueryAtom>,
    bonds: Vec<QueryGraphBond>,
    adjacency: Vec<SmallVec<[u16; 4]>>,
    ring_closures: std::collections::HashMap<u32, (u16, Option<QueryBond>)>,
}

impl SmartsParser {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
            atoms: Vec::new(),
            bonds: Vec::new(),
            adjacency: Vec::new(),
            ring_closures: std::collections::HashMap::new(),
        }
    }
    
    fn parse(&mut self) -> std::result::Result<QueryGraph, String> {
        if self.input.is_empty() {
            return Ok(QueryGraph {
                atoms: vec![],
                bonds: vec![],
                adjacency: vec![],
            });
        }
        
        self.parse_chain()?;
        
        // Check for unclosed ring closures
        if !self.ring_closures.is_empty() {
            return Err(format!("Unclosed ring closures"));
        }
        
        Ok(QueryGraph {
            atoms: self.atoms.clone(),
            bonds: self.bonds.clone(),
            adjacency: self.adjacency.clone(),
        })
    }
    
    fn parse_chain(&mut self) -> std::result::Result<(), String> {
        let mut prev_atom_idx: Option<u16> = None;
        let mut bond_type: Option<QueryBond> = None;
        let mut branch_stack: Vec<u16> = Vec::new(); // Stack for handling branches
        
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            
            match ch {
                '-' => { bond_type = Some(QueryBond::Single); self.pos += 1; }
                '=' => { bond_type = Some(QueryBond::Double); self.pos += 1; }
                '#' => { bond_type = Some(QueryBond::Triple); self.pos += 1; }
                ':' => { bond_type = Some(QueryBond::Aromatic); self.pos += 1; }
                '~' => { bond_type = Some(QueryBond::Any); self.pos += 1; }
                
                '[' => {
                    let atom = self.parse_bracket_atom()?;
                    let atom_idx = self.add_atom(atom);
                    
                    if let Some(prev_idx) = prev_atom_idx {
                        self.add_bond(prev_idx, atom_idx, bond_type.unwrap_or(QueryBond::Any));
                    }
                    
                    prev_atom_idx = Some(atom_idx);
                    bond_type = None;
                    self.parse_ring_closures(atom_idx)?;
                }
                
                'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'B' | 'I' | 
                'c' | 'n' | 'o' | 's' | 'p' => {
                    let (atom, consumed) = self.parse_organic_atom()?;
                    let atom_idx = self.add_atom(atom);
                    
                    if let Some(prev_idx) = prev_atom_idx {
                        self.add_bond(prev_idx, atom_idx, bond_type.unwrap_or(QueryBond::Single));
                    }
                    
                    prev_atom_idx = Some(atom_idx);
                    bond_type = None;
                    self.pos += consumed;
                    self.parse_ring_closures(atom_idx)?;
                }
                
                '*' => {
                    let atom = QueryAtom::any();
                    let atom_idx = self.add_atom(atom);
                    
                    if let Some(prev_idx) = prev_atom_idx {
                        self.add_bond(prev_idx, atom_idx, bond_type.unwrap_or(QueryBond::Any));
                    }
                    
                    prev_atom_idx = Some(atom_idx);
                    bond_type = None;
                    self.pos += 1;
                    self.parse_ring_closures(atom_idx)?;
                }
                
                'a' => {
                    // Lowercase 'a' outside brackets = any aliphatic atom
                    let atom = QueryAtom { primitives: vec![AtomPrimitive::AnyAliphatic] };
                    let atom_idx = self.add_atom(atom);
                    
                    if let Some(prev_idx) = prev_atom_idx {
                        self.add_bond(prev_idx, atom_idx, bond_type.unwrap_or(QueryBond::Single));
                    }
                    
                    prev_atom_idx = Some(atom_idx);
                    bond_type = None;
                    self.pos += 1;
                    self.parse_ring_closures(atom_idx)?;
                }
                
                'A' => {
                    // Uppercase 'A' outside brackets = any aliphatic atom
                    let atom = QueryAtom { primitives: vec![AtomPrimitive::AnyAliphatic] };
                    let atom_idx = self.add_atom(atom);
                    
                    if let Some(prev_idx) = prev_atom_idx {
                        self.add_bond(prev_idx, atom_idx, bond_type.unwrap_or(QueryBond::Single));
                    }
                    
                    prev_atom_idx = Some(atom_idx);
                    bond_type = None;
                    self.pos += 1;
                    self.parse_ring_closures(atom_idx)?;
                }
                
                '(' => {
                    // Start of branch - save current position to stack
                    self.pos += 1;
                    if let Some(current_idx) = prev_atom_idx {
                        branch_stack.push(current_idx);
                    } else {
                        return Err("Branch at start of pattern".to_string());
                    }
                    // The next atom will be bonded to the current atom
                    // prev_atom_idx stays the same (branch point)
                }
                
                ')' => {
                    // End of branch - restore previous atom from stack
                    self.pos += 1;
                    if let Some(branch_point) = branch_stack.pop() {
                        prev_atom_idx = Some(branch_point);
                        bond_type = None;
                    } else {
                        return Err("Unmatched closing parenthesis".to_string());
                    }
                }
                
                _ => {
                    return Err(format!("Unexpected character '{}'", ch));
                }
            }
        }
        
        // Check for unclosed branches
        if !branch_stack.is_empty() {
            return Err(format!("Unclosed branch parentheses"));
        }
        
        Ok(())
    }
    
    fn parse_bracket_atom(&mut self) -> std::result::Result<QueryAtom, String> {
        self.pos += 1; // Skip '['
        
        // Parse the full bracket expression with logical operators
        let primitives = self.parse_bracket_expression()?;
        
        if self.peek() != Some(']') {
            return Err("Missing ]".to_string());
        }
        self.pos += 1;
        
        Ok(QueryAtom { primitives })
    }
    
    /// Parse bracket expression with support for logical operators (,=OR ;=AND !=NOT)
    fn parse_bracket_expression(&mut self) -> std::result::Result<Vec<AtomPrimitive>, String> {
        let mut and_groups = Vec::new();
        
        while self.peek() != Some(']') && self.peek().is_some() {
            // Check if we're starting an OR group
            let mut or_group = Vec::new();
            
            // Parse primitives separated by commas (OR)
            loop {
                let primitive = self.parse_single_primitive()?;
                or_group.push(primitive);
                
                // Check what comes next
                match self.peek() {
                    Some(',') => {
                        // Comma = OR, continue this OR group
                        self.pos += 1;
                        continue;
                    }
                    Some(';') | Some('&') => {
                        // Semicolon or & = AND, finalize OR group and continue to next AND term
                        self.pos += 1;
                        break;
                    }
                    Some(']') | None => {
                        // End of expression
                        break;
                    }
                    _ => {
                        // No operator means implicit AND - finalize OR group
                        break;
                    }
                }
            }
            
            // Add the OR group (or single primitive) to AND groups
            if or_group.len() == 1 {
                and_groups.push(or_group.into_iter().next().unwrap());
            } else {
                and_groups.push(AtomPrimitive::Or(or_group));
            }
        }
        
        // Return single primitive, OR, or AND
        if and_groups.is_empty() {
            Ok(vec![AtomPrimitive::AnyAtom])
        } else if and_groups.len() == 1 {
            Ok(vec![and_groups.into_iter().next().unwrap()])
        } else {
            Ok(vec![AtomPrimitive::And(and_groups)])
        }
    }
    
    /// Parse a single primitive (element, #N, property, or !primitive)
    fn parse_single_primitive(&mut self) -> std::result::Result<AtomPrimitive, String> {
        // Handle NOT operator
        if self.peek() == Some('!') {
            self.pos += 1;
            let inner = self.parse_single_primitive()?;
            return Ok(AtomPrimitive::Not(Box::new(inner)));
        }
        
        // Check for atomic number notation (#6, #7, etc.)
        if self.peek() == Some('#') {
            self.pos += 1; // Skip '#'
            
            // Parse the atomic number
            let mut num_str = String::new();
            while self.pos < self.input.len() {
                let ch = self.input[self.pos];
                if ch.is_ascii_digit() {
                    num_str.push(ch);
                    self.pos += 1;
                } else {
                    break;
                }
            }
            
            if num_str.is_empty() {
                return Err("Expected atomic number after #".to_string());
            }
            
            let atomic_number: u8 = num_str.parse()
                .map_err(|_| format!("Invalid atomic number: {}", num_str))?;
            
            return Ok(AtomPrimitive::Element(atomic_number));
        }
        
        // Check for property (H, D, X, v, R, r, +, -)
        match self.peek() {
            Some('H') => {
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1);
                return Ok(AtomPrimitive::HCount(count));
            }
            Some('h') => {
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1);
                return Ok(AtomPrimitive::TotalHCount(count));
            }
            Some('D') => {
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1);
                return Ok(AtomPrimitive::Degree(count));
            }
            Some('X') => {
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1);
                return Ok(AtomPrimitive::Degree(count));
            }
            Some('v') => {
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1);
                return Ok(AtomPrimitive::Valence(count));
            }
            Some('x') => {
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1);
                return Ok(AtomPrimitive::RingBonds(count));
            }
            Some('R') => {
                self.pos += 1;
                if let Some(size) = self.parse_digit() {
                    if size == 0 {
                        return Ok(AtomPrimitive::NotInRing);
                    } else {
                        return Ok(AtomPrimitive::RingSize(size));
                    }
                } else {
                    return Ok(AtomPrimitive::InRing);
                }
            }
            Some('r') => {
                self.pos += 1;
                if let Some(size) = self.parse_digit() {
                    if size == 0 {
                        return Ok(AtomPrimitive::NotInRing);
                    } else {
                        return Ok(AtomPrimitive::RingSize(size));
                    }
                } else {
                    return Ok(AtomPrimitive::InRing);
                }
            }
            Some('+') => {
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1) as i8;
                return Ok(AtomPrimitive::Charge(count));
            }
            Some('-') if self.pos + 1 < self.input.len() && self.input[self.pos + 1].is_ascii_digit() => {
                // Negative charge
                self.pos += 1;
                let count = self.parse_digit().unwrap_or(1) as i8;
                return Ok(AtomPrimitive::Charge(-count));
            }
            _ => {}
        }
        
        // Parse element symbol
        let mut element = String::new();
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch == ']' || ch == 'H' || ch == 'h' || ch == 'D' || ch == '+' 
               || ch == 'X' || ch == 'x' || ch == 'v' || ch == 'R' || ch == 'r' || ch == ';' || ch == ',' 
               || ch == '!' || ch == '&' || ch == '#' {
                break;
            }
            // Allow '-' if it's part of a symbol or charge at start
            if ch == '-' && !element.is_empty() {
                break;
            }
            element.push(ch);
            self.pos += 1;
        }
        
        if element.is_empty() || element == "*" {
            Ok(AtomPrimitive::AnyAtom)
        } else if element == "A" {
            Ok(AtomPrimitive::AnyAliphatic)
        } else if element == "a" {
            Ok(AtomPrimitive::AnyAromatic)
        } else if element == "-" {
            // Standalone negative charge
            Ok(AtomPrimitive::Charge(-1))
        } else {
            let is_aromatic = element.chars().next().unwrap().is_lowercase();
            let symbol = if is_aromatic { element.to_uppercase() } else { element };
            
            let atomic_number = crate::mol::elements::symbol_to_atomic_number(&symbol)
                .ok_or_else(|| format!("Unknown element: {}", symbol))?;
            
            if is_aromatic {
                Ok(AtomPrimitive::Aromatic(atomic_number))
            } else {
                Ok(AtomPrimitive::Element(atomic_number))
            }
        }
    }
    
    fn parse_organic_atom(&mut self) -> std::result::Result<(QueryAtom, usize), String> {
        let ch = self.input[self.pos];
        let is_aromatic = ch.is_lowercase();
        
        let symbol = ch.to_string();
        let upper_symbol = if is_aromatic { symbol.to_uppercase() } else { symbol.clone() };
        
        let atomic_number = crate::mol::elements::symbol_to_atomic_number(&upper_symbol)
            .ok_or_else(|| format!("Unknown element: {}", upper_symbol))?;
        
        let atom = if is_aromatic {
            QueryAtom { primitives: vec![AtomPrimitive::Aromatic(atomic_number)] }
        } else {
            QueryAtom { primitives: vec![AtomPrimitive::Element(atomic_number)] }
        };
        
        Ok((atom, 1))
    }
    
    fn parse_ring_closures(&mut self, atom_idx: u16) -> std::result::Result<(), String> {
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            let ring_num = (self.input[self.pos] as u32) - ('0' as u32);
            self.pos += 1;
            
            if let Some((prev_idx, bond_type)) = self.ring_closures.remove(&ring_num) {
                self.add_bond(prev_idx, atom_idx, bond_type.unwrap_or(QueryBond::Single));
            } else {
                self.ring_closures.insert(ring_num, (atom_idx, None));
            }
        }
        
        Ok(())
    }
    
    fn peek(&self) -> Option<char> {
        if self.pos < self.input.len() { Some(self.input[self.pos]) } else { None }
    }
    
    fn parse_digit(&mut self) -> Option<u8> {
        if self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            let digit = (self.input[self.pos] as u8) - b'0';
            self.pos += 1;
            Some(digit)
        } else {
            None
        }
    }
    
    fn add_atom(&mut self, atom: QueryAtom) -> u16 {
        let idx = self.atoms.len() as u16;
        self.atoms.push(atom);
        self.adjacency.push(SmallVec::new());
        idx
    }
    
    fn add_bond(&mut self, source: u16, target: u16, bond: QueryBond) {
        let bond_idx = self.bonds.len() as u16;
        self.bonds.push(QueryGraphBond { source, target, query: bond });
        self.adjacency[source as usize].push(bond_idx);
        self.adjacency[target as usize].push(bond_idx);
    }
}

#[cfg(test)]
mod smarts_parser_tests {
    use super::*;
    
    #[test]
    fn test_parse_single_element() {
        let pattern = parse_smarts("[C]").unwrap();
        assert_eq!(pattern.atoms.len(), 1);
        assert_eq!(pattern.bonds.len(), 0);
    }
    
    #[test]
    fn test_parse_chain() {
        let pattern = parse_smarts("CC").unwrap();
        assert_eq!(pattern.atoms.len(), 2);
        assert_eq!(pattern.bonds.len(), 1);
    }
    
    #[test]
    fn test_parse_double_bond() {
        let pattern = parse_smarts("C=O").unwrap();
        assert_eq!(pattern.atoms.len(), 2);
        assert_eq!(pattern.bonds.len(), 1);
    }
    
    #[test]
    fn test_parse_aromatic_ring() {
        let pattern = parse_smarts("c1ccccc1").unwrap();
        assert_eq!(pattern.atoms.len(), 6);
        assert_eq!(pattern.bonds.len(), 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::algo::rings::find_sssr;
    
    #[test]
    fn test_query_atom_element() {
        let mol = parse_smiles("CCO").unwrap();
        
        let carbon_query = QueryAtom::element(6);  // Carbon
        assert!(carbon_query.matches(&mol, 0));  // First C
        assert!(carbon_query.matches(&mol, 1));  // Second C
        assert!(!carbon_query.matches(&mol, 2)); // O
        
        let oxygen_query = QueryAtom::element(8);  // Oxygen
        assert!(!oxygen_query.matches(&mol, 0));
        assert!(!oxygen_query.matches(&mol, 1));
        assert!(oxygen_query.matches(&mol, 2));
    }
    
    #[test]
    fn test_query_atom_any() {
        let mol = parse_smiles("CCO").unwrap();
        let any_query = QueryAtom::any();
        
        assert!(any_query.matches(&mol, 0));
        assert!(any_query.matches(&mol, 1));
        assert!(any_query.matches(&mol, 2));
    }
    
    #[test]
    fn test_query_atom_aromatic() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);
        
        let aromatic_query = QueryAtom::aromatic_element(6);
        assert!(aromatic_query.matches(&mol, 0));
        assert!(aromatic_query.matches(&mol, 5));
        
        // Non-aromatic molecule
        let mol2 = parse_smiles("CCCC").unwrap();
        assert!(!aromatic_query.matches(&mol2, 0));
    }
    
    #[test]
    fn test_query_atom_degree() {
        let mol = parse_smiles("CC(C)C").unwrap();  // Isopropane: C-C(C)-C
        
        let degree1_query = QueryAtom {
            primitives: vec![AtomPrimitive::Degree(1)],
        };
        assert!(degree1_query.matches(&mol, 0));  // Terminal C
        assert!(!degree1_query.matches(&mol, 1)); // Central C has degree 3
        
        let degree3_query = QueryAtom {
            primitives: vec![AtomPrimitive::Degree(3)],
        };
        assert!(degree3_query.matches(&mol, 1));  // Central C
    }
    
    #[test]
    fn test_query_atom_in_ring() {
        let mut mol = parse_smiles("C1CCCCC1").unwrap();  // Cyclohexane
        find_sssr(&mut mol);
        
        let ring_query = QueryAtom {
            primitives: vec![AtomPrimitive::InRing],
        };
        
        assert!(ring_query.matches(&mol, 0));
        assert!(ring_query.matches(&mol, 3));
        
        // Non-ring molecule
        let mol2 = parse_smiles("CCCC").unwrap();
        assert!(!ring_query.matches(&mol2, 0));
    }
    
    #[test]
    fn test_query_bond_types() {
        let mol = parse_smiles("C=C").unwrap();
        
        let single_query = QueryBond::Single;
        let double_query = QueryBond::Double;
        
        assert!(!single_query.matches(&mol, 0));
        assert!(double_query.matches(&mol, 0));
    }
}


#[cfg(test)]
mod smarts_advanced_tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::valency::assign_implicit_hydrogens;
    use crate::algo::aromaticity::perceive_aromaticity;
    
    #[test]
    fn test_or_in_brackets() {
        // Test comma-separated OR lists
        let query = parse_smarts("[#6,#7,#8]").unwrap();
        assert_eq!(query.atoms.len(), 1);
        
        // Should match C, N, or O
        let mut mol = parse_smiles("C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches C
        
        let mut mol = parse_smiles("N").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches N
        
        let mut mol = parse_smiles("O").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches O
    }
    
    #[test]
    fn test_and_in_brackets() {
        // Test implicit AND (no operator between primitives)
        let query = parse_smarts("[CH3]").unwrap();
        
        // Should match carbon with exactly 3 hydrogens
        let mut mol = parse_smiles("C").unwrap(); // CH4
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 0); // CH4 doesn't match CH3
        
        let mut mol = parse_smiles("CC").unwrap(); // 2x CH3
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 2); // both carbons are CH3
    }
    
    #[test]
    fn test_branching() {
        // Test branching patterns
        let query = parse_smarts("C(C)C").unwrap();
        assert_eq!(query.atoms.len(), 3);
        assert_eq!(query.bonds.len(), 2);
        
        // In propane (C-C-C), this pattern matches twice:
        // - First C as center (connected to middle C and... wait, it only has one neighbor)
        // Actually, both terminal carbons match: each is connected to the middle C
        // and the middle C counts as both neighbors
        let mut mol = parse_smiles("CCC").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        // Pattern C(C)C means: C connected to two other C atoms via branches
        // In linear propane, the middle C matches (has 2 C neighbors)
        assert!(matches.len() >= 1); // at least one match
    }
    
    #[test]
    fn test_complex_branching() {
        // Test [CH2](C)C pattern (CH2 connected to two carbons)
        let query = parse_smarts("[CH2](C)C").unwrap();
        assert_eq!(query.atoms.len(), 3);
        
        // In propane, the middle carbon is CH2 and connected to 2 carbons
        let mut mol = parse_smiles("CCC").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        // Middle C is CH2 with 2 C neighbors
        assert!(matches.len() >= 1);
    }
    
    #[test]
    fn test_not_operator() {
        // Test NOT operator
        let query = parse_smarts("[!#6]").unwrap();
        
        // Should match non-carbon atoms
        let mut mol = parse_smiles("CCO").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches O only
    }
    
    #[test]
    fn test_multiple_or_elements() {
        // Test [N,O,P,S]
        let query = parse_smarts("[N,O,P,S]").unwrap();
        
        let mut mol = parse_smiles("NOPS").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 4); // all 4 atoms match
    }
    
    #[test]
    fn test_aromatic_with_hcount() {
        // Test [cH] (aromatic carbon with 1 H)
        let query = parse_smarts("[cH]").unwrap();
        
        // Benzene has 6 aromatic CH
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        perceive_aromaticity(&mut mol);
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 6);
    }
    
    #[test]
    fn test_lowercase_a_outside_brackets() {
        // Test lowercase 'a' as any aliphatic
        let query = parse_smarts("a").unwrap();
        
        // Should match aliphatic atoms
        let mut mol = parse_smiles("CCC").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 3); // all 3 carbons
    }
    
    #[test]
    fn test_nested_branches() {
        // Test C(C)(C)C (carbon with 3 branches)
        let query = parse_smarts("C(C)(C)C").unwrap();
        assert_eq!(query.atoms.len(), 4);
        
        // Should match tert-butyl center in neopentane
        let mut mol = parse_smiles("CC(C)(C)C").unwrap(); // neopentane
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        // The central carbon has 4 C neighbors, pattern needs 3
        assert!(matches.len() >= 1); // at least the center matches
    }
    
    #[test]
    fn test_valence_matching() {
        // Test valence matching [Cv4] (carbon with valence 4)
        let query = parse_smarts("[Cv4]").unwrap();
        
        // Methane: C with 4 H = valence 4
        let mut mol = parse_smiles("C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // CH4 has valence 4
        
        // Ethane: each C has valence 4 (3H + 1C)
        let mut mol = parse_smiles("CC").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 2); // both carbons have valence 4
    }
    
    #[test]
    fn test_valence_with_double_bonds() {
        // Test [Cv4] with double bonds
        let query = parse_smarts("[Cv4]").unwrap();
        
        // Ethene: C=C, each C has 2H + 1 double bond = 2 + 2 = 4
        let mut mol = parse_smiles("C=C").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 2); // both carbons have valence 4
    }
    
    #[test]
    fn test_ring_bonds_count() {
        // Test [Cx2] (atom with 2 ring bonds)
        let query = parse_smarts("[Cx2]").unwrap();
        
        // Benzene: each carbon has 2 ring bonds
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        perceive_aromaticity(&mut mol);
        
        // Mark ring atoms and bonds by finding SSSR
        let _ = crate::algo::rings::find_sssr(&mut mol);
        
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 6); // all 6 carbons have 2 ring bonds
    }
    
    #[test]
    fn test_valence_nitrogen() {
        // Test [Nv3] (nitrogen with valence 3)
        let query = parse_smarts("[Nv3]").unwrap();
        
        // Ammonia: N with 3 H = valence 3
        let mut mol = parse_smiles("N").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1);
        
        // Methylamine: N with 2H + 1C = valence 3
        let mut mol = parse_smiles("CN").unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // N has valence 3
    }
}
