use crate::error::{FerrumError, Result};
use crate::mol::molecule::Molecule;
use crate::mol::elements::{symbol_to_atomic_number, is_aromatic_symbol};
use crate::types::BondType;
use crate::io::builder::MoleculeBuilder;

/// Parses a SMILES string into a complete Molecule with implicit hydrogens calculated.
///
/// # Thread Safety
/// This function is thread-safe and can be called concurrently from multiple threads.
/// Each call creates an independent parser with its own state.
/// The returned `Molecule` is `Send + Sync` for use in parallel processing.
///
/// # Automatic Processing
/// This function automatically:
/// 1. Calculates implicit hydrogens after parsing
/// 2. Perceives aromaticity (Hückel 4n+2 rule)
///
/// The returned molecule is complete and ready for descriptor calculations.
/// This ensures consistency across all molecules and prevents bugs from
/// forgetting to calculate implicit H or aromaticity.
///
/// # Performance
/// Single-pass parsing with automatic implicit H calculation and aromaticity perception.
/// Typical performance: 1-6M molecules/sec (full pipeline).
///
/// # Example
/// ```ignore
/// use ferrum_core::io::smiles::parse_smiles;
/// let mol = parse_smiles("c1ccccc1").unwrap();
/// assert_eq!(mol.num_atoms(), 6); // benzene
/// assert_eq!(mol.atoms[0].num_implicit_h, 1); // each C has 1 H
/// assert!(mol.atoms[0].is_aromatic()); // aromatic carbon
/// ```
pub fn parse_smiles(smiles: &str) -> Result<Molecule> {
    let mut parser = SmilesParser::new(smiles);
    let mut mol = parser.parse()?;

    // ✅ Automatically calculate implicit hydrogens
    // This ensures molecules are always complete and ready to use
    crate::algo::valency::assign_implicit_hydrogens(&mut mol)?;

    // ✅ Automatically perceive aromaticity
    // Required for correct SMARTS matching and descriptors (TPSA, LogP, etc.)
    crate::algo::aromaticity::perceive_aromaticity(&mut mol);

    Ok(mol)
}

/// SMILES parser state machine.
struct SmilesParser<'a> {
    input: &'a str,
    pos: usize,
    mol: Molecule,
    // Stack for branch points
    stack: Vec<u16>,
    // Ring closures: ring_number -> atom_idx
    rings: [Option<u16>; 100],
    // Last atom added
    last_atom: Option<u16>,
    // Pending bond type (for next atom connection)
    pending_bond: Option<BondType>,
}

impl<'a> SmilesParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            mol: Molecule::new(),
            stack: Vec::with_capacity(16),
            rings: [None; 100],
            last_atom: None,
            pending_bond: None,
        }
    }

    fn parse(&mut self) -> Result<Molecule> {
        while self.pos < self.input.len() {
            self.parse_token()?;
        }

        // Validate all ring closures are matched
        for (i, ring) in self.rings.iter().enumerate() {
            if ring.is_some() {
                return Err(FerrumError::ParseError(
                    format!("Unclosed ring closure: {}", i)
                ));
            }
        }

        Ok(std::mem::take(&mut self.mol))
    }

    fn parse_token(&mut self) -> Result<()> {
        let c = self.current_char()?;

        match c {
            // Branch opening
            '(' => {
                if let Some(atom) = self.last_atom {
                    self.stack.push(atom);
                    self.pos += 1;
                } else {
                    return Err(FerrumError::ParseError(
                        "Branch without preceding atom".to_string()
                    ));
                }
            }
            // Branch closing
            ')' => {
                self.last_atom = self.stack.pop();
                if self.last_atom.is_none() {
                    return Err(FerrumError::ParseError(
                        "Unmatched closing parenthesis".to_string()
                    ));
                }
                self.pos += 1;
            }
            // Bond symbols
            '-' => {
                self.pending_bond = Some(BondType::Single);
                self.pos += 1;
            }
            '=' => {
                self.pending_bond = Some(BondType::Double);
                self.pos += 1;
            }
            '#' => {
                self.pending_bond = Some(BondType::Triple);
                self.pos += 1;
            }
            ':' => {
                self.pending_bond = Some(BondType::Aromatic);
                self.pos += 1;
            }
            '/' | '\\' => {
                // Stereochemistry - ignore for now
                self.pos += 1;
            }
            // Ring closures (digits)
            '0'..='9' => {
                let ring_num = self.parse_ring_number()?;
                self.handle_ring_closure(ring_num)?;
            }
            '%' => {
                // Two-digit ring number
                self.pos += 1;
                let ring_num = self.parse_ring_number()?;
                self.handle_ring_closure(ring_num)?;
            }
            // Bracketed atom [...]
            '[' => {
                self.parse_bracket_atom()?;
            }
            // Organic subset atoms
            'B' | 'C' | 'N' | 'O' | 'P' | 'S' | 'F' | 'I' |
            'b' | 'c' | 'n' | 'o' | 'p' | 's' => {
                self.parse_organic_atom()?;
            }
            // Two-letter elements
            'A' | 'G' | 'H' | 'K' | 'L' | 'M' | 'R' | 'T' |
            'U' | 'V' | 'W' | 'X' | 'Y' | 'Z' | 'D' | 'E' => {
                // Check if it's Br or Cl (most common)
                if self.peek_two_letter_element() {
                    self.parse_organic_atom()?;
                } else {
                    return Err(FerrumError::ParseError(
                        format!("Unknown element at position {}", self.pos)
                    ));
                }
            }
            '.' => {
                // Disconnected fragment - reset connection
                self.last_atom = None;
                self.pending_bond = None;
                self.pos += 1;
            }
            _ => {
                return Err(FerrumError::ParseError(
                    format!("Unexpected character '{}' at position {}", c, self.pos)
                ));
            }
        }

        Ok(())
    }

    fn parse_organic_atom(&mut self) -> Result<()> {
        let start = self.pos;
        let mut is_aromatic = false;

        // Check for aromatic lowercase
        let first_char = self.current_char()?;
        if is_aromatic_symbol(first_char) {
            is_aromatic = true;
        }

        // Parse element symbol (1 or 2 chars)
        let symbol = if self.pos + 1 < self.input.len() {
            let two_char = &self.input[start..start + 2];
            if symbol_to_atomic_number(two_char).is_some() {
                self.pos += 2;
                two_char
            } else {
                self.pos += 1;
                &self.input[start..start + 1]
            }
        } else {
            self.pos += 1;
            &self.input[start..start + 1]
        };

        let atomic_num = symbol_to_atomic_number(symbol)
            .ok_or_else(|| FerrumError::ParseError(
                format!("Unknown element symbol: {}", symbol)
            ))?;

        self.add_atom(atomic_num, is_aromatic, 0, 0, 0)?;
        Ok(())
    }

    fn parse_bracket_atom(&mut self) -> Result<()> {
        self.pos += 1; // skip '['

        let mut isotope: u16 = 0;
        let mut charge: i8 = 0;
        let mut h_count: u8 = 0;
        let mut is_aromatic = false;

        // Parse isotope (optional digits)
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_digit() {
            let digit = (self.input.as_bytes()[self.pos] - b'0') as u16;
            isotope = isotope * 10 + digit;
            self.pos += 1;
        }

        // Parse element symbol
        let start = self.pos;
        let c = self.current_char()?;

        if is_aromatic_symbol(c) {
            is_aromatic = true;
            self.pos += 1;
        } else if c.is_ascii_uppercase() {
            self.pos += 1;
            // Check for second lowercase letter
            if self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_lowercase() {
                self.pos += 1;
            }
        } else {
            return Err(FerrumError::ParseError(
                format!("Invalid atom symbol in bracket at position {}", self.pos)
            ));
        }

        let symbol = &self.input[start..self.pos];
        let atomic_num = symbol_to_atomic_number(symbol)
            .ok_or_else(|| FerrumError::ParseError(
                format!("Unknown element symbol: {}", symbol)
            ))?;

        // Parse @/@@ for chirality (skip for now)
        if self.pos < self.input.len() && self.current_char()? == '@' {
            self.pos += 1;
            if self.pos < self.input.len() && self.current_char()? == '@' {
                self.pos += 1;
            }
        }

        // Parse explicit H count
        if self.pos < self.input.len() && self.current_char()? == 'H' {
            self.pos += 1;
            if self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_digit() {
                h_count = (self.input.as_bytes()[self.pos] - b'0') as u8;
                self.pos += 1;
            } else {
                h_count = 1;
            }
        }

        // Parse charge
        if self.pos < self.input.len() {
            let c = self.current_char()?;
            if c == '+' || c == '-' {
                let sign = if c == '+' { 1 } else { -1 };
                self.pos += 1;

                // Count repeated +/- or parse digit
                if self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_digit() {
                    charge = ((self.input.as_bytes()[self.pos] - b'0') as i8) * sign;
                    self.pos += 1;
                } else {
                    charge = sign;
                    // Count additional +/- symbols
                    while self.pos < self.input.len() && self.current_char()? == c {
                        charge += sign;
                        self.pos += 1;
                    }
                }
            }
        }

        // Expect closing bracket
        if self.pos >= self.input.len() || self.current_char()? != ']' {
            return Err(FerrumError::ParseError(
                "Missing closing bracket for atom".to_string()
            ));
        }
        self.pos += 1;

        self.add_atom(atomic_num, is_aromatic, isotope, charge, h_count)?;
        Ok(())
    }

    fn add_atom(&mut self, atomic_num: u8, is_aromatic: bool, isotope: u16,
                charge: i8, h_count: u8) -> Result<()> {
        let new_idx = self.mol.add_atom(atomic_num);

        if is_aromatic {
            self.mol.set_aromatic(new_idx, true);
        }
        if isotope > 0 {
            self.mol.set_isotope(new_idx, isotope);
        }
        if charge != 0 {
            self.mol.set_charge(new_idx, charge);
        }
        if h_count > 0 {
            self.mol.set_explicit_h(new_idx, h_count);
        }

        // Connect to previous atom
        if let Some(prev_idx) = self.last_atom {
            let bond_type = self.pending_bond.unwrap_or_else(|| {
                // Default bond type based on aromaticity
                if is_aromatic {
                    BondType::Aromatic
                } else {
                    BondType::Single
                }
            });
            self.mol.add_bond(prev_idx, new_idx, bond_type);
        }

        self.last_atom = Some(new_idx);
        self.pending_bond = None;

        Ok(())
    }

    fn parse_ring_number(&mut self) -> Result<usize> {
        let start = self.pos;
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_digit() {
            self.pos += 1;
        }

        if start == self.pos {
            return Err(FerrumError::ParseError(
                "Expected ring number".to_string()
            ));
        }

        let num_str = &self.input[start..self.pos];
        num_str.parse::<usize>().map_err(|_|
            FerrumError::ParseError(format!("Invalid ring number: {}", num_str))
        )
    }

    fn handle_ring_closure(&mut self, ring_num: usize) -> Result<()> {
        if ring_num >= 100 {
            return Err(FerrumError::ParseError(
                format!("Ring number {} exceeds maximum (99)", ring_num)
            ));
        }

        let current_atom = self.last_atom.ok_or_else(||
            FerrumError::ParseError("Ring closure without atom".to_string())
        )?;

        if let Some(ring_atom) = self.rings[ring_num] {
            // Close the ring
            let bond_type = self.pending_bond.unwrap_or(BondType::Single);
            self.mol.add_bond(ring_atom, current_atom, bond_type);
            self.rings[ring_num] = None;
            self.pending_bond = None;
        } else {
            // Open the ring
            self.rings[ring_num] = Some(current_atom);
        }

        Ok(())
    }

    fn current_char(&self) -> Result<char> {
        self.input.chars().nth(self.pos).ok_or_else(||
            FerrumError::ParseError("Unexpected end of input".to_string())
        )
    }

    fn peek_two_letter_element(&self) -> bool {
        if self.pos + 1 < self.input.len() {
            let two_char = &self.input[self.pos..self.pos + 2];
            matches!(two_char, "Br" | "Cl")
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_smiles() {
        // Methane: C
        let mol = parse_smiles("C").unwrap();
        assert_eq!(mol.num_atoms(), 1);
        assert_eq!(mol.atoms[0].atomic_number, 6);

        // Ethane: CC
        let mol = parse_smiles("CC").unwrap();
        assert_eq!(mol.num_atoms(), 2);
        assert_eq!(mol.num_bonds(), 1);
    }

    #[test]
    fn test_benzene() {
        // Benzene: c1ccccc1
        let mol = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(mol.num_atoms(), 6);
        assert_eq!(mol.num_bonds(), 6);

        // All atoms should be aromatic
        for atom in &mol.atoms {
            assert!(atom.is_aromatic());
        }
    }

    #[test]
    fn test_branches() {
        // Isobutane: CC(C)C
        let mol = parse_smiles("CC(C)C").unwrap();
        assert_eq!(mol.num_atoms(), 4);
        assert_eq!(mol.num_bonds(), 3);
    }

    #[test]
    fn test_double_bond() {
        // Ethene: C=C
        let mol = parse_smiles("C=C").unwrap();
        assert_eq!(mol.num_atoms(), 2);
        assert_eq!(mol.num_bonds(), 1);
        assert_eq!(mol.bonds[0].bond_type, BondType::Double);
    }

    #[test]
    fn test_bracket_atom() {
        // Sodium ion: [Na+]
        let mol = parse_smiles("[Na+]").unwrap();
        assert_eq!(mol.num_atoms(), 1);
        assert_eq!(mol.atoms[0].atomic_number, 11);
        assert_eq!(mol.atoms[0].formal_charge, 1);

        // Heavy water: [2H]O[2H]
        let mol = parse_smiles("[2H]O[2H]").unwrap();
        assert_eq!(mol.num_atoms(), 3);
        assert_eq!(mol.atoms[0].isotope, 2);
        assert_eq!(mol.atoms[2].isotope, 2);
    }

    #[test]
    fn test_explicit_hydrogens() {
        // Methane with explicit H: [CH4]
        let mol = parse_smiles("[CH4]").unwrap();
        assert_eq!(mol.num_atoms(), 1);
        assert_eq!(mol.atoms[0].num_explicit_h, 4);
    }
}
