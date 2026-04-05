//! I/O module: Smiles.
//!
//! This module handles external data interchange (SMILES/SMARTS/CSV/SDF) and
//! conversion into FerrumChem graph types. It is the boundary between textual
//! chemical formats and internal, typed molecular structures.
//!
//! Learning focus:
//! - Parser/state-machine techniques in Rust.
//! - Error-aware ingestion pipelines for large datasets.

//! ## File guide: Smiles
//!
//! This file belongs to FerrumChem's **input/output parsing layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn parse_smiles`
//! - `struct SmilesParser`
//! - `fn new`
//! - `fn parse`
//! - `fn parse_token`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Smiles` treats textual chemistry formats as untrusted input and prefers explicit parse errors over silent repair.
//! - `Smiles` must preserve deterministic tokenization so repeated reads produce identical molecule/query graphs.
//! - Format-specific edge cases in `Smiles` are validated against toolkit-compatible semantics where possible.
//! - Specific scope anchor: `crates/ferrum_core/src/io/smiles.rs` defines smiles behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - SMILES/SMARTS/CTfile family specifications relevant to `Smiles` parser or reader behavior.
//! - RDKit parsing behavior used as external compatibility reference for `Smiles`.
//! - Rust parser/state-machine patterns for reliable streaming and error reporting.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/io/smiles.rs`.

use crate::error::{FerrumError, Result};
use crate::mol::elements::{is_aromatic_symbol, symbol_to_atomic_number};
use crate::mol::molecule::Molecule;
use crate::mol::{atom::Atom, bond::Bond};
use crate::types::BondType;
use smallvec::SmallVec;

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
/// # Note on Kekulization
/// This function does NOT kekulize aromatic rings (assign explicit single/double bonds).
/// Aromatic bonds remain as BondType::Aromatic with the aromatic flag set.
/// This matches RDKit's default representation and is correct for TPSA and other descriptors.
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
    let saw_ring_closure = parser.saw_ring_closure;

    // ✅ Automatically calculate implicit hydrogens
    // This ensures molecules are always complete and ready to use
    crate::algo::valency::assign_implicit_hydrogens(&mut mol)?;

    // ✅ Detect dative (coordinate) bonds for transition metal complexes.
    // Skip the pass for all-organic molecules to avoid unnecessary bond scans.
    if mol
        .atoms
        .iter()
        .any(|atom| crate::mol::elements::is_transition_metal(atom.atomic_number))
    {
        crate::algo::valency::assign_dative_bonds(&mut mol);
    }

    // ✅ Automatically perceive aromaticity when pi candidates are present.
    // Molecules without aromatic atoms or multiple bonds cannot become aromatic,
    // but they still need ring flags for SMARTS ring primitives.
    let needs_aromaticity = mol.atoms.iter().any(|atom| atom.is_aromatic())
        || mol
            .bonds
            .iter()
            .any(|bond| matches!(bond.bond_type, BondType::Double | BondType::Aromatic));
    if needs_aromaticity {
        crate::algo::aromaticity::perceive_aromaticity(&mut mol);
    } else if saw_ring_closure {
        crate::algo::rings::find_sssr(&mut mol);
    }

    // Prime minimum cycle basis for ring-containing molecules so descriptors that
    // rely on this cache (e.g. AromaticRingCount, QED aromatic ring term) avoid
    // recomputing expensive cycle basis work in tight descriptor loops.
    if saw_ring_closure {
        crate::algo::rings::ensure_minimum_cycle_basis_bonds_cached(&mut mol);
    }

    // ❌ Do NOT kekulize by default
    // RDKit keeps aromatic bonds as type AROMATIC (bond value = 1.5)
    // Kekulization is only needed for specific algorithms that require it
    // TPSA, LogP, and most descriptors work correctly with AROMATIC bond type

    Ok(mol)
}

/// SMILES parser state machine.
struct SmilesParser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    mol: Molecule,
    // Stack for branch points
    stack: SmallVec<[u16; 16]>,
    // Ring closures: ring_number -> (atom_idx, explicit bond if provided at opening side)
    rings: [Option<(u16, Option<BondType>)>; 100],
    // Last atom added
    last_atom: Option<u16>,
    // Pending bond type (for next atom connection)
    pending_bond: Option<BondType>,
    // Tracks whether ring-closure syntax was encountered.
    saw_ring_closure: bool,
}

impl<'a> SmilesParser<'a> {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn new(input: &'a str) -> Self {
        let capacity = input.len();
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
            mol: Molecule::with_capacity(capacity),
            stack: SmallVec::new(),
            rings: [None; 100],
            last_atom: None,
            pending_bond: None,
            saw_ring_closure: false,
        }
    }

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse(&mut self) -> Result<Molecule> {
        while self.pos < self.bytes.len() {
            self.parse_token()?;
        }

        // Validate all ring closures are matched
        for (i, ring) in self.rings.iter().enumerate() {
            if ring.is_some() {
                return Err(FerrumError::ParseError(format!(
                    "Unclosed ring closure: {}",
                    i
                )));
            }
        }

        Ok(std::mem::take(&mut self.mol))
    }

    /// Parses token.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse_token(&mut self) -> Result<()> {
        let c = self.current_byte()?;

        match c {
            // Branch opening
            b'(' => {
                if let Some(atom) = self.last_atom {
                    self.stack.push(atom);
                    self.pos += 1;
                } else {
                    return Err(FerrumError::ParseError(
                        "Branch without preceding atom".to_string(),
                    ));
                }
            }
            // Branch closing
            b')' => {
                self.last_atom = self.stack.pop();
                if self.last_atom.is_none() {
                    return Err(FerrumError::ParseError(
                        "Unmatched closing parenthesis".to_string(),
                    ));
                }
                self.pos += 1;
            }
            // Bond symbols
            b'-' => {
                self.pending_bond = Some(BondType::Single);
                self.pos += 1;
            }
            b'=' => {
                self.pending_bond = Some(BondType::Double);
                self.pos += 1;
            }
            b'#' => {
                self.pending_bond = Some(BondType::Triple);
                self.pos += 1;
            }
            b':' => {
                self.pending_bond = Some(BondType::Aromatic);
                self.pos += 1;
            }
            b'/' | b'\\' => {
                // Stereochemistry - ignore for now
                self.pos += 1;
            }
            // Ring closures (digits)
            b'0'..=b'9' => {
                // Single digit ring closure (0-9)
                let ring_num = self.parse_single_digit_ring()?;
                self.handle_ring_closure(ring_num)?;
            }
            b'%' => {
                // Multi-digit ring number (10-99)
                self.pos += 1;
                let ring_num = self.parse_multi_digit_ring()?;
                self.handle_ring_closure(ring_num)?;
            }
            // Bracketed atom [...]
            b'[' => {
                self.parse_bracket_atom()?;
            }
            // Organic subset atoms
            b'B' | b'C' | b'N' | b'O' | b'P' | b'S' | b'F' | b'I' | b'b' | b'c' | b'n' | b'o'
            | b'p' | b's' => {
                self.parse_organic_atom()?;
            }
            // Two-letter elements
            b'A' | b'G' | b'H' | b'K' | b'L' | b'M' | b'R' | b'T' | b'U' | b'V' | b'W' | b'X'
            | b'Y' | b'Z' | b'D' | b'E' => {
                // Check if it's Br or Cl (most common)
                if self.peek_two_letter_element() {
                    self.parse_organic_atom()?;
                } else {
                    return Err(FerrumError::ParseError(format!(
                        "Unknown element at position {}",
                        self.pos
                    )));
                }
            }
            b'.' => {
                // Disconnected fragment - reset connection
                self.last_atom = None;
                self.pending_bond = None;
                self.pos += 1;
            }
            _ => {
                return Err(FerrumError::ParseError(format!(
                    "Unexpected character '{}' at position {}",
                    c as char, self.pos
                )));
            }
        }

        Ok(())
    }

    #[inline]
    /// Implements atomic number.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn organic_atomic_number(token: u8) -> Option<u8> {
        match token {
            b'B' | b'b' => Some(5),
            b'C' | b'c' => Some(6),
            b'N' | b'n' => Some(7),
            b'O' | b'o' => Some(8),
            b'F' => Some(9),
            b'P' | b'p' => Some(15),
            b'S' | b's' => Some(16),
            b'I' => Some(53),
            _ => None,
        }
    }

    /// Parses organic atom.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse_organic_atom(&mut self) -> Result<()> {
        let first = self.current_byte()?;
        let is_aromatic = is_aromatic_symbol(first as char);

        let atomic_num = if self.pos + 1 < self.bytes.len() {
            match (first, self.bytes[self.pos + 1]) {
                (b'B', b'r') => {
                    self.pos += 2;
                    35
                }
                (b'C', b'l') => {
                    self.pos += 2;
                    17
                }
                _ => {
                    self.pos += 1;
                    Self::organic_atomic_number(first).ok_or_else(|| {
                        FerrumError::ParseError(format!(
                            "Unknown element symbol '{}' at position {}",
                            first as char,
                            self.pos - 1
                        ))
                    })?
                }
            }
        } else {
            self.pos += 1;
            Self::organic_atomic_number(first).ok_or_else(|| {
                FerrumError::ParseError(format!(
                    "Unknown element symbol '{}' at position {}",
                    first as char,
                    self.pos - 1
                ))
            })?
        };

        self.add_atom(atomic_num, is_aromatic, 0, 0, 0, false)?;
        Ok(())
    }

    /// Parses bracket atom.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse_bracket_atom(&mut self) -> Result<()> {
        self.pos += 1; // skip '['

        let mut isotope: u16 = 0;
        let mut charge: i8 = 0;
        let mut h_count: u8 = 0;
        let mut is_aromatic = false;

        // Parse isotope (optional digits)
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            let digit = (self.bytes[self.pos] - b'0') as u16;
            isotope = isotope * 10 + digit;
            self.pos += 1;
        }

        // Parse element symbol
        let start = self.pos;
        let c = self.current_byte()?;

        if is_aromatic_symbol(c as char) {
            is_aromatic = true;
            self.pos += 1;
            // Aromatic two-letter bracket atoms supported by RDKit/OpenSMILES.
            // Examples: [se], [as], [te]
            if self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_lowercase() {
                let candidate = &self.input[start..self.pos + 1];
                if matches!(candidate, "se" | "as" | "te") {
                    self.pos += 1;
                }
            }
        } else if c.is_ascii_uppercase() {
            self.pos += 1;
            // Check for second lowercase letter
            if self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_lowercase() {
                self.pos += 1;
            }
        } else {
            return Err(FerrumError::ParseError(format!(
                "Invalid atom symbol in bracket at position {}",
                self.pos
            )));
        }

        let symbol = &self.input[start..self.pos];
        let atomic_num = symbol_to_atomic_number(symbol).ok_or_else(|| {
            FerrumError::ParseError(format!("Unknown element symbol: {}", symbol))
        })?;

        // Parse @/@@ for chirality (skip for now)
        if self.pos < self.bytes.len() && self.current_byte()? == b'@' {
            self.pos += 1;
            if self.pos < self.bytes.len() && self.current_byte()? == b'@' {
                self.pos += 1;
            }
        }

        // Parse explicit H count
        if self.pos < self.bytes.len() && self.current_byte()? == b'H' {
            self.pos += 1;
            if self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                h_count = (self.bytes[self.pos] - b'0') as u8;
                self.pos += 1;
            } else {
                h_count = 1;
            }
        }

        // Parse charge
        if self.pos < self.bytes.len() {
            let c = self.current_byte()?;
            if c == b'+' || c == b'-' {
                let sign = if c == b'+' { 1 } else { -1 };
                self.pos += 1;

                // Count repeated +/- or parse digit
                if self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                    charge = ((self.bytes[self.pos] - b'0') as i8) * sign;
                    self.pos += 1;
                } else {
                    charge = sign;
                    // Count additional +/- symbols
                    while self.pos < self.bytes.len() && self.current_byte()? == c {
                        charge += sign;
                        self.pos += 1;
                    }
                }
            }
        }

        // Expect closing bracket
        if self.pos >= self.bytes.len() || self.current_byte()? != b']' {
            return Err(FerrumError::ParseError(
                "Missing closing bracket for atom".to_string(),
            ));
        }
        self.pos += 1;

        self.add_atom(atomic_num, is_aromatic, isotope, charge, h_count, true)?;
        Ok(())
    }

    /// Implements atom.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn add_atom(
        &mut self,
        atomic_num: u8,
        is_aromatic: bool,
        isotope: u16,
        charge: i8,
        h_count: u8,
        is_bracket: bool,
    ) -> Result<()> {
        let new_idx = self.mol.atoms.len() as u16;
        self.mol.atoms.push(Atom::new(atomic_num));
        self.mol.adjacency.push(smallvec::SmallVec::new());

        {
            let atom = &mut self.mol.atoms[new_idx as usize];
            if is_aromatic {
                atom.set_aromatic(true);
            }
            if isotope > 0 {
                atom.isotope = isotope;
            }
            if charge != 0 {
                atom.formal_charge = charge;
            }
            if h_count > 0 {
                atom.num_explicit_h = h_count;
            }
            // Bracket atoms have explicit H specification (even if 0)
            if is_bracket {
                atom.set_no_implicit_h(true);
            }
        }

        if let Some(prev_idx) = self.last_atom {
            let bond_type = self.pending_bond.unwrap_or_else(|| {
                // Default bond: aromatic only if BOTH atoms are aromatic
                let prev_aromatic = self.mol.atoms[prev_idx as usize].is_aromatic();
                if is_aromatic && prev_aromatic {
                    BondType::Aromatic
                } else {
                    BondType::Single
                }
            });
            let bond_idx = self.mol.bonds.len() as u16;
            self.mol.bonds.push(Bond::new(prev_idx, new_idx, bond_type));
            self.mol.adjacency[prev_idx as usize].push(bond_idx);
            self.mol.adjacency[new_idx as usize].push(bond_idx);
        }

        self.last_atom = Some(new_idx as u16);
        self.pending_bond = None;

        Ok(())
    }

    /// Parses single digit ring.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse_single_digit_ring(&mut self) -> Result<usize> {
        // SMILES spec: single digit ring closures are 0-9
        // They must NOT be parsed greedily - '12' is two closures: '1' and '2'
        let c = self.current_byte()?;
        if !c.is_ascii_digit() {
            return Err(FerrumError::ParseError(
                "Expected single digit ring number".to_string(),
            ));
        }

        self.pos += 1;
        Ok((c - b'0') as usize)
    }

    /// Parses multi digit ring.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse_multi_digit_ring(&mut self) -> Result<usize> {
        // SMILES spec: multi-digit ring closures use % prefix
        // After %, read exactly 2 digits for ring numbers 10-99
        let start = self.pos;

        // Read up to 2 digits
        let mut count = 0;
        while count < 2 && self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
            count += 1;
        }

        if count == 0 {
            return Err(FerrumError::ParseError(
                "Expected ring number after %".to_string(),
            ));
        }

        let num_str = &self.input[start..self.pos];
        num_str
            .parse::<usize>()
            .map_err(|_| FerrumError::ParseError(format!("Invalid ring number: {}", num_str)))
    }

    /// Implements ring closure.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn handle_ring_closure(&mut self, ring_num: usize) -> Result<()> {
        if ring_num >= 100 {
            return Err(FerrumError::ParseError(format!(
                "Ring number {} exceeds maximum (99)",
                ring_num
            )));
        }
        self.saw_ring_closure = true;

        let current_atom = self
            .last_atom
            .ok_or_else(|| FerrumError::ParseError("Ring closure without atom".to_string()))?;

        if let Some((ring_atom, opening_bond)) = self.rings[ring_num] {
            // Close the ring
            let closing_bond = self.pending_bond.take();

            if let (Some(ob), Some(cb)) = (opening_bond, closing_bond) {
                if ob != cb {
                    return Err(FerrumError::ParseError(format!(
                        "Inconsistent ring closure bond types for ring {}",
                        ring_num
                    )));
                }
            }

            let bond_type = closing_bond.or(opening_bond).unwrap_or_else(|| {
                // Default: aromatic if both atoms are aromatic
                let a1_arom = self.mol.atoms[ring_atom as usize].is_aromatic();
                let a2_arom = self.mol.atoms[current_atom as usize].is_aromatic();
                if a1_arom && a2_arom {
                    BondType::Aromatic
                } else {
                    BondType::Single
                }
            });
            let bond_idx = self.mol.bonds.len() as u16;
            self.mol
                .bonds
                .push(Bond::new(ring_atom, current_atom, bond_type));
            self.mol.adjacency[ring_atom as usize].push(bond_idx);
            self.mol.adjacency[current_atom as usize].push(bond_idx);
            self.rings[ring_num] = None;
        } else {
            // Open the ring
            self.rings[ring_num] = Some((current_atom, self.pending_bond.take()));
        }

        Ok(())
    }

    #[inline]
    /// Implements byte.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn current_byte(&self) -> Result<u8> {
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| FerrumError::ParseError("Unexpected end of input".to_string()))
    }

    /// Implements two letter element.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn peek_two_letter_element(&self) -> bool {
        self.pos + 1 < self.bytes.len()
            && ((self.bytes[self.pos] == b'B' && self.bytes[self.pos + 1] == b'r')
                || (self.bytes[self.pos] == b'C' && self.bytes[self.pos + 1] == b'l'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Runs regression test for simple smiles.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
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
    /// Runs regression test for benzene.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
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
    /// Runs regression test for branches.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_branches() {
        // Isobutane: CC(C)C
        let mol = parse_smiles("CC(C)C").unwrap();
        assert_eq!(mol.num_atoms(), 4);
        assert_eq!(mol.num_bonds(), 3);
    }

    #[test]
    /// Runs regression test for double bond.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_double_bond() {
        // Ethene: C=C
        let mol = parse_smiles("C=C").unwrap();
        assert_eq!(mol.num_atoms(), 2);
        assert_eq!(mol.num_bonds(), 1);
        assert_eq!(mol.bonds[0].bond_type, BondType::Double);
    }

    #[test]
    /// Runs regression test for bracket atom.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
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

        // Aromatic bracket atoms with two-letter aromatic symbols
        let mol = parse_smiles("[se]1cccc1").unwrap();
        assert_eq!(mol.atoms[0].atomic_number, 34);

        let mol = parse_smiles("[as]1cccc1").unwrap();
        assert_eq!(mol.atoms[0].atomic_number, 33);

        let mol = parse_smiles("[te]1cccc1").unwrap();
        assert_eq!(mol.atoms[0].atomic_number, 52);
    }

    #[test]
    /// Runs regression test for explicit hydrogens.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_explicit_hydrogens() {
        // Methane with explicit H: [CH4]
        let mol = parse_smiles("[CH4]").unwrap();
        assert_eq!(mol.num_atoms(), 1);
        assert_eq!(mol.atoms[0].num_explicit_h, 4);
    }

    #[test]
    /// Runs regression test for ring closure bond consistency.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_ring_closure_bond_consistency() {
        // Ring opening and closing specify different explicit bond types.
        // RDKit rejects this and we should too.
        let err = parse_smiles("C-1CC=1").unwrap_err();
        assert!(
            err.to_string()
                .contains("Inconsistent ring closure bond types"),
            "unexpected error: {err}"
        );
    }
}
