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
use crate::mol::{
    atom::{Atom, ChiralTag},
    bond::Bond,
};
use crate::types::{AtomIdx, BondType};
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
    // Like RDKit, everything after the first whitespace is a name, not SMILES.
    let smiles = smiles.trim_start().split_whitespace().next().unwrap_or("");
    let mut parser = SmilesParser::new(smiles);
    let mut mol = parser.parse()?;
    adjust_chirality_for_removed_hs(&mut mol);
    // One chemistry model: RDKit's sanitization (H removal, cleanup, valence
    // check, rings, kekulization, aromaticity, conjugation, hybridization).
    crate::algo::sanitize::sanitize(&mut mol)?;
    Ok(mol)
}

/// RDKit's `removeHs` moves the bond to a removed H to the end of a chiral
/// atom's bond list before deleting it (adjusting the tag). Mirror that for
/// the hydrogens `sanitize::remove_hs` is about to remove.
fn adjust_chirality_for_removed_hs(mol: &mut Molecule) {
    let other = |mol: &Molecule, bi: usize, a: usize| {
        let b = &mol.bonds[bi];
        if b.source as usize == a { b.target as usize } else { b.source as usize }
    };
    for c in 0..mol.atoms.len() {
        if mol.atoms[c].chiral_tag() == crate::mol::atom::ChiralTag::Unspecified {
            continue;
        }
        let mut bonds: SmallVec<[usize; 6]> = mol.adjacency[c].iter().map(|&b| b as usize).collect();
        let mut h_nbrs: SmallVec<[usize; 4]> = SmallVec::new();
        for &bi in &bonds {
            let h = other(mol, bi, c);
            let atom = &mol.atoms[h];
            // same criteria as `sanitize::remove_hs`
            if atom.atomic_number != 1 || atom.isotope != 0 || mol.adjacency[h].len() != 1 {
                continue;
            }
            let z = mol.atoms[c].atomic_number;
            if z == 0 || z == 1 {
                continue;
            }
            if mol.bonds[bi].has_direction()
                && mol.adjacency[c].len() == 2
                && mol.adjacency[c]
                    .iter()
                    .any(|&b| mol.bonds[b as usize].bond_type == BondType::Double)
            {
                continue;
            }
            h_nbrs.push(h);
        }
        if h_nbrs.is_empty() {
            continue;
        }
        h_nbrs.sort_unstable();
        let mut odd = false;
        for h in h_nbrs {
            let k = bonds.iter().position(|&bi| other(mol, bi, c) == h).unwrap();
            if (bonds.len() - 1 - k) % 2 == 1 {
                odd = !odd;
            }
            bonds.remove(k);
        }
        if odd {
            let tag = mol.atoms[c].chiral_tag().inverted();
            mol.atoms[c].set_chiral_tag(tag);
        }
    }
}

/// SMILES parser state machine.
struct SmilesParser<'a> {
    input: &'a str,
    cursor: &'a [u8],
    mol: Molecule,
    // Stack for branch points
    branch_stack: SmallVec<[usize; 16]>,
    // Ring closures: ring_number -> (atom_idx, explicit bond if provided at
    // opening side, closure id, `/` (true) or `\\` (false) at the opening side)
    rings: Vec<Option<(AtomIdx, Option<BondType>, usize, Option<bool>)>>,
    // Ring-closure bonds, added after all chain bonds in closing order (RDKit
    // bond ordering): (begin = closing atom, end = opening atom, type).
    closure_bonds: Vec<(AtomIdx, AtomIdx, BondType)>,
    // `<-` dative bond: next bond is directed from the following atom to the previous one.
    pending_dative_reverse: bool,
    // `/` (true) or `\\` (false) seen for the pending bond.
    pending_direction: Option<bool>,
    // Direction of each ring-closure bond relative to its closing atom.
    closure_directions: Vec<Option<bool>>,
    // Stereo bookkeeping (RDKit `GetBondOrdering` inputs): SMILES chirality
    // per atom as written, whether the atom starts a SMILES fragment, the
    // ring-closure digits on each atom in order, and each closure's atom pair.
    chiral_atoms: Vec<(AtomIdx, ChiralTag, bool, bool)>,
    atom_closures: Vec<(AtomIdx, usize)>,
    closure_pairs: Vec<(AtomIdx, AtomIdx)>,
    // Last atom added
    last_atom: Option<u16>,
    // Pending bond type (for next atom connection)
    pending_bond: Option<BondType>,
}

impl<'a> SmilesParser<'a> {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn new(input: &'a str) -> Self {
        let capacity = input.len() / 2 + 1;
        Self {
            input,
            cursor: input.as_bytes(),
            mol: Molecule::with_capacity(capacity),
            branch_stack: SmallVec::new(),
            rings: Vec::new(),
            closure_bonds: Vec::new(),
            pending_dative_reverse: false,
            pending_direction: None,
            closure_directions: Vec::new(),
            chiral_atoms: Vec::new(),
            atom_closures: Vec::new(),
            closure_pairs: Vec::new(),
            last_atom: None,
            pending_bond: None,
        }
    }

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    #[inline(always)]
    fn pos(&self) -> usize {
        self.input.len() - self.cursor.len()
    }

    fn parse(&mut self) -> Result<Molecule> {
        while !self.cursor.is_empty() {
            self.parse_token()?;
        }

        // Validate all ring closures and branches are matched
        for (i, ring) in self.rings.iter().enumerate() {
            if ring.is_some() {
                return Err(FerrumError::ParseError(format!(
                    "Unclosed ring closure: {}",
                    i
                )));
            }
        }
        if !self.branch_stack.is_empty() {
            return Err(FerrumError::ParseError("Unclosed branch".to_string()));
        }
        if self.pending_bond.is_some() {
            return Err(FerrumError::ParseError("Dangling bond at end of SMILES".to_string()));
        }

        // Ring-closure bonds come after all chain bonds (RDKit bond order).
        let directions = std::mem::take(&mut self.closure_directions);
        for (k, (begin, end, bond_type)) in std::mem::take(&mut self.closure_bonds).into_iter().enumerate() {
            self.push_bond(begin, end, bond_type)?;
            if let Some(up) = directions.get(k).copied().flatten() {
                crate::algo::stereo::set_bond_direction(self.mol.bonds.last_mut().unwrap(), up);
            }
        }
        self.assign_chiral_tags();

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
                    self.branch_stack.push(atom as usize);
                    self.cursor = &self.cursor[1..];
                } else {
                    return Err(FerrumError::ParseError(
                        "Branch without preceding atom".to_string(),
                    ));
                }
            }
            // Branch closing
            b')' => {
                self.last_atom = self.branch_stack.pop().map(|idx| idx as u16);
                if self.last_atom.is_none() {
                    return Err(FerrumError::ParseError(
                        "Unmatched closing parenthesis".to_string(),
                    ));
                }
                self.cursor = &self.cursor[1..];
            }
            // Bond symbols
            b'-' => {
                if self.cursor.get(1) == Some(&b'>') {
                    self.pending_bond = Some(BondType::Dative);
                    self.cursor = &self.cursor[2..];
                } else {
                    self.pending_bond = Some(BondType::Single);
                    self.cursor = &self.cursor[1..];
                }
            }
            b'<' if self.cursor.get(1) == Some(&b'-') => {
                self.pending_bond = Some(BondType::Dative);
                self.pending_dative_reverse = true;
                self.cursor = &self.cursor[2..];
            }
            b'*' => {
                self.cursor = &self.cursor[1..];
                self.add_atom(0, false, 0, 0, 0, false)?;
            }
            b'=' => {
                self.pending_bond = Some(BondType::Double);
                self.cursor = &self.cursor[1..];
            }
            b'#' => {
                self.pending_bond = Some(BondType::Triple);
                self.cursor = &self.cursor[1..];
            }
            b':' => {
                self.pending_bond = Some(BondType::Aromatic);
                self.cursor = &self.cursor[1..];
            }
            b'/' | b'\\' => {
                // Directional single bond (RDKit ENDUPRIGHT / ENDDOWNRIGHT).
                self.pending_bond = Some(BondType::Single);
                self.pending_direction = Some(c == b'/');
                self.cursor = &self.cursor[1..];
            }
            // Ring closures (digits)
            b'0'..=b'9' => {
                // Single digit ring closure (0-9)
                let ring_num = self.parse_single_digit_ring()?;
                self.handle_ring_closure(ring_num)?;
            }
            b'%' => {
                // Multi-digit ring number (10-99)
                self.cursor = &self.cursor[1..];
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
                        self.pos())));
                }
            }
            b'.' => {
                // Disconnected fragment - reset connection
                if self.pending_bond.is_some() {
                    return Err(FerrumError::ParseError("Bond before '.'".to_string()));
                }
                self.last_atom = None;
                self.pending_bond = None;
                self.cursor = &self.cursor[1..];
            }
            _ => {
                return Err(FerrumError::ParseError(format!(
                    "Unexpected character '{}' at position {}",
                    c as char, self.pos())));
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

        let atomic_num = if self.cursor.len() > 1 {
            match (first, self.cursor[1]) {
                (b'B', b'r') => {
                    self.cursor = &self.cursor[2..];
                    35
                }
                (b'C', b'l') => {
                    self.cursor = &self.cursor[2..];
                    17
                }
                _ => {
                    self.cursor = &self.cursor[1..];
                    Self::organic_atomic_number(first).ok_or_else(|| {
                        FerrumError::ParseError(format!(
                            "Unknown element symbol '{}' at position {}",
                            first as char,
                            self.pos() - 1
                        ))
                    })?
                }
            }
        } else {
            self.cursor = &self.cursor[1..];
            Self::organic_atomic_number(first).ok_or_else(|| {
                FerrumError::ParseError(format!(
                    "Unknown element symbol '{}' at position {}",
                    first as char,
                    self.pos() - 1
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
        self.cursor = &self.cursor[1..]; // skip '['

        let mut isotope: u16 = 0;
        let mut charge: i8 = 0;
        let mut h_count: u8 = 0;
        let mut is_aromatic = false;

        // Parse isotope (optional digits)
        while !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
            let digit = (self.cursor[0] - b'0') as u16;
            isotope = isotope
                .checked_mul(10)
                .and_then(|v| v.checked_add(digit))
                .ok_or_else(|| FerrumError::ParseError("Isotope out of range".to_string()))?;
            self.cursor = &self.cursor[1..];
        }

        // Parse element symbol
        let start = self.pos();
        let c = self.current_byte()?;

        if c == b'*' {
            self.cursor = &self.cursor[1..];
        } else if is_aromatic_symbol(c as char) {
            is_aromatic = true;
            self.cursor = &self.cursor[1..];
            // Aromatic two-letter bracket atoms supported by RDKit/OpenSMILES.
            // Examples: [se], [as], [te]
            if !self.cursor.is_empty() && self.cursor[0].is_ascii_lowercase() {
                let candidate = &self.input[start..self.pos() + 1];
                if matches!(candidate, "se" | "as" | "te") {
                    self.cursor = &self.cursor[1..];
                }
            }
        } else if c.is_ascii_uppercase() {
            self.cursor = &self.cursor[1..];
            // Check for second lowercase letter
            if !self.cursor.is_empty() && self.cursor[0].is_ascii_lowercase() {
                self.cursor = &self.cursor[1..];
            }
        } else {
            return Err(FerrumError::ParseError(format!(
                "Invalid atom symbol in bracket at position {}",
                self.pos())));
        }

        let symbol = &self.input[start..self.pos()];
        let atomic_num = if symbol == "*" {
            0
        } else {
            symbol_to_atomic_number(symbol).ok_or_else(|| {
                FerrumError::ParseError(format!("Unknown element symbol: {}", symbol))
            })?
        };

        // Parse @/@@ chirality. `@TH1`/`@TH2` are `@`/`@@`; the other
        // classes (@AL, @SP, @TB, @OH) are kept as `ChiralTag::Other`.
        let mut chiral = ChiralTag::Unspecified;
        if !self.cursor.is_empty() && self.current_byte()? == b'@' {
            self.cursor = &self.cursor[1..];
            chiral = ChiralTag::TetrahedralCCW;
            if !self.cursor.is_empty() && self.current_byte()? == b'@' {
                self.cursor = &self.cursor[1..];
                chiral = ChiralTag::TetrahedralCW;
            }
            if self.cursor.len() >= 2
                && matches!(&self.cursor[..2], b"TH" | b"AL" | b"SP" | b"TB" | b"OH")
            {
                let class = &self.cursor[..2];
                self.cursor = &self.cursor[2..];
                let start = self.pos();
                while !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
                    self.cursor = &self.cursor[1..];
                }
                chiral = match (class, &self.input[start..self.pos()]) {
                    (b"TH", "1") => ChiralTag::TetrahedralCCW,
                    (b"TH", "2") => ChiralTag::TetrahedralCW,
                    _ => ChiralTag::Other,
                };
            }
        }

        // Parse explicit H count
        if !self.cursor.is_empty() && self.current_byte()? == b'H' {
            self.cursor = &self.cursor[1..];
            if !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
                h_count = self.cursor[0] - b'0';
                self.cursor = &self.cursor[1..];
            } else {
                h_count = 1;
            }
        }

        // Parse charge
        if !self.cursor.is_empty() {
            let c = self.current_byte()?;
            if c == b'+' || c == b'-' {
                let sign = if c == b'+' { 1 } else { -1 };
                self.cursor = &self.cursor[1..];

                // Count repeated +/- or parse digits
                let mut magnitude: i32 = 0;
                if !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
                    while !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
                        magnitude = magnitude * 10 + (self.cursor[0] - b'0') as i32;
                        self.cursor = &self.cursor[1..];
                        if magnitude > 99 {
                            break;
                        }
                    }
                } else {
                    magnitude = 1;
                    // Count additional +/- symbols
                    while !self.cursor.is_empty() && self.current_byte()? == c {
                        magnitude += 1;
                        self.cursor = &self.cursor[1..];
                        if magnitude > 99 {
                            break;
                        }
                    }
                }
                if magnitude > 15 {
                    return Err(FerrumError::ParseError("Formal charge out of range".to_string()));
                }
                charge = (magnitude as i8) * sign;
            }
        }

        // Atom map number (ignored)
        if self.cursor.first() == Some(&b':') {
            self.cursor = &self.cursor[1..];
            while !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
                self.cursor = &self.cursor[1..];
            }
        }

        // Expect closing bracket
        if self.cursor.is_empty() || self.current_byte()? != b']' {
            return Err(FerrumError::ParseError(
                "Missing closing bracket for atom".to_string(),
            ));
        }
        self.cursor = &self.cursor[1..];

        let is_start = self.last_atom.is_none();
        self.add_atom(atomic_num, is_aromatic, isotope, charge, h_count, true)?;
        if chiral != ChiralTag::Unspecified {
            let idx = (self.mol.atoms.len() - 1) as AtomIdx;
            self.chiral_atoms.push((idx, chiral, is_start, false));
        }
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
        if self.mol.atoms.len() >= u16::MAX as usize {
            return Err(FerrumError::ParseError("Too many atoms (max 65535)".to_string()));
        }
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
            if std::mem::take(&mut self.pending_dative_reverse) {
                self.push_bond(new_idx, prev_idx, bond_type)?;
            } else {
                self.push_bond(prev_idx, new_idx, bond_type)?;
            }
            if let Some(up) = self.pending_direction.take() {
                crate::algo::stereo::set_bond_direction(self.mol.bonds.last_mut().unwrap(), up);
            }
        } else if self.pending_bond.is_some() {
            return Err(FerrumError::ParseError("Bond without preceding atom".to_string()));
        }

        self.last_atom = Some(new_idx);
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

        self.cursor = &self.cursor[1..];
        Ok((c - b'0') as usize)
    }

    /// Parses multi digit ring.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse_multi_digit_ring(&mut self) -> Result<usize> {
        // SMILES spec: multi-digit ring closures use % prefix
        // After %, read exactly 2 digits (%nn) or a parenthesised number (%(nnn)).
        if self.cursor.first() == Some(&b'(') {
            self.cursor = &self.cursor[1..];
            let start = self.pos();
            while !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
                self.cursor = &self.cursor[1..];
            }
            let num_str = &self.input[start..self.pos()];
            if self.cursor.first() != Some(&b')') || num_str.is_empty() {
                return Err(FerrumError::ParseError("Invalid %(...) ring closure".to_string()));
            }
            self.cursor = &self.cursor[1..];
            return num_str
                .parse::<usize>()
                .map_err(|_| FerrumError::ParseError(format!("Invalid ring number: {}", num_str)));
        }
        let start = self.pos();

        // Read up to 2 digits
        let mut count = 0;
        while count < 2 && !self.cursor.is_empty() && self.cursor[0].is_ascii_digit() {
            self.cursor = &self.cursor[1..];
            count += 1;
        }

        if count == 0 {
            return Err(FerrumError::ParseError(
                "Expected ring number after %".to_string(),
            ));
        }

        let num_str = &self.input[start..self.pos()];
        num_str
            .parse::<usize>()
            .map_err(|_| FerrumError::ParseError(format!("Invalid ring number: {}", num_str)))
    }

    /// Implements ring closure.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn handle_ring_closure(&mut self, ring_num: usize) -> Result<()> {
        if ring_num > 99_999 {
            return Err(FerrumError::ParseError(format!(
                "Ring number {} exceeds maximum",
                ring_num
            )));
        }
        if ring_num >= self.rings.len() {
            self.rings.resize(ring_num + 1, None);
        }

        let current_atom = self
            .last_atom
            .ok_or_else(|| FerrumError::ParseError("Ring closure without atom".to_string()))?;
        self.check_ring_closure_branch_status(current_atom);

        if let Some((ring_atom, opening_bond, closure_id, opening_dir)) = self.rings[ring_num] {
            // Close the ring
            let closing_bond = self.pending_bond.take();
            self.pending_dative_reverse = false;

            if let (Some(ob), Some(cb)) = (opening_bond, closing_bond) {
                if ob != cb {
                    return Err(FerrumError::ParseError(format!(
                        "Inconsistent ring closure bond types for ring {}",
                        ring_num
                    )));
                }
            }
            if ring_atom == current_atom {
                return Err(FerrumError::ParseError(format!(
                    "Ring closure {} bonds atom {} to itself",
                    ring_num, current_atom
                )));
            }
            if self.has_bond(ring_atom, current_atom) {
                return Err(FerrumError::ParseError(format!(
                    "Duplicate bond between atoms {} and {}",
                    ring_atom, current_atom
                )));
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
            self.closure_bonds.push((current_atom, ring_atom, bond_type));
            // RDKit keeps the closing-side direction, else the opening-side
            // one flipped (the bond begins at the closing atom here).
            let closing_dir = self.pending_direction.take();
            self.closure_directions.push(closing_dir.or(opening_dir.map(|up| !up)));
            self.closure_pairs[closure_id] = (ring_atom, current_atom);
            self.atom_closures.push((current_atom, closure_id));
            self.rings[ring_num] = None;
        } else {
            // Open the ring
            let closure_id = self.closure_pairs.len();
            self.closure_pairs.push((current_atom, current_atom));
            self.atom_closures.push((current_atom, closure_id));
            let opening_dir = self.pending_direction.take();
            self.rings[ring_num] = Some((current_atom, self.pending_bond.take(), closure_id, opening_dir));
            self.pending_dative_reverse = false;
        }

        Ok(())
    }

    /// Stores the chirality of each `@`/`@@` atom relative to the atom's
    /// bond order in `adjacency` (implicit H last), reproducing RDKit's
    /// `AdjustAtomChiralityFlags`: the SMILES neighbour order is the
    /// preceding atom, then ring-closure partners in digit order, then the
    /// following atoms; plus RDKit's `chiralAtomNeedsTagInversion` quirk.
    fn assign_chiral_tags(&mut self) {
        for k in 0..self.chiral_atoms.len() {
            let (atom, tag, is_start, flip) = self.chiral_atoms[k];
            let a = atom as usize;
            if !tag.is_tetrahedral() {
                self.mol.atoms[a].set_chiral_tag(tag);
                continue;
            }
            let built: SmallVec<[usize; 6]> = self.mol.adjacency[a]
                .iter()
                .map(|&bi| {
                    let b = &self.mol.bonds[bi as usize];
                    if b.source == atom { b.target as usize } else { b.source as usize }
                })
                .collect();
            let closures: SmallVec<[usize; 4]> = self
                .atom_closures
                .iter()
                .filter(|&&(x, _)| x == atom)
                .map(|&(_, id)| {
                    let (p, q) = self.closure_pairs[id];
                    if p == atom { q as usize } else { p as usize }
                })
                .collect();
            let mut chain: SmallVec<[usize; 6]> =
                built.iter().copied().filter(|n| !closures.contains(n)).collect();
            chain.sort_unstable();
            let mut smiles_order: SmallVec<[usize; 6]> = SmallVec::new();
            let self_pos = if chain.first().is_some_and(|&n| n < a) { 1 } else { 0 };
            smiles_order.extend(chain[..self_pos].iter().copied());
            smiles_order.extend(closures.iter().copied());
            smiles_order.extend(chain[self_pos..].iter().copied());
            let mut odd = crate::algo::stereo::permutation_is_odd(&smiles_order, &built) ^ flip;
            let degree = built.len();
            let bracket_h = self.mol.atoms[a].num_explicit_h;
            let unsaturated = self.mol.adjacency[a].iter().any(|&bi| {
                matches!(
                    self.mol.bonds[bi as usize].bond_type,
                    BondType::Double | BondType::Triple | BondType::Aromatic
                )
            });
            if degree == 3
                && ((is_start && bracket_h == 1)
                    || (bracket_h != 1 && closures.len() == 1 && !unsaturated))
            {
                odd = !odd;
            }
            self.mol.atoms[a].set_chiral_tag(if odd { tag.inverted() } else { tag });
        }
    }

    /// RDKit `CheckRingClosureBranchStatus`: a ring-closure digit written
    /// after a branch (e.g. `[C@@](F)1(C)CCO1`) inverts the atom's chirality.
    fn check_ring_closure_branch_status(&mut self, atom: AtomIdx) {
        let idx = atom as usize;
        if idx + 1 == self.mol.atoms.len() {
            return;
        }
        let degree = self.mol.adjacency[idx].len();
        if degree == 1 || (degree == 2 && idx != 0) || (degree == 3 && idx == 0) {
            if let Some(entry) = self.chiral_atoms.iter_mut().rev().find(|e| e.0 == atom) {
                if entry.1.is_tetrahedral() {
                    entry.3 = !entry.3;
                }
            }
        }
    }

    fn has_bond(&self, a: AtomIdx, b: AtomIdx) -> bool {
        self.mol.adjacency[a as usize].iter().any(|&bi| {
            let bond = &self.mol.bonds[bi as usize];
            (bond.source == b && bond.target == a) || (bond.source == a && bond.target == b)
        }) || self
            .closure_bonds
            .iter()
            .any(|&(x, y, _)| (x == a && y == b) || (x == b && y == a))
    }

    fn push_bond(&mut self, begin: AtomIdx, end: AtomIdx, bond_type: BondType) -> Result<()> {
        if self.mol.bonds.len() >= u16::MAX as usize {
            return Err(FerrumError::ParseError("Too many bonds (max 65535)".to_string()));
        }
        let bond_idx = self.mol.bonds.len() as u16;
        let mut bond = Bond::new(begin, end, bond_type);
        if bond_type == BondType::Aromatic {
            bond.set_aromatic(true);
        }
        self.mol.bonds.push(bond);
        self.mol.adjacency[begin as usize].push(bond_idx);
        self.mol.adjacency[end as usize].push(bond_idx);
        Ok(())
    }

    #[inline]
    /// Implements byte.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn current_byte(&self) -> Result<u8> {
        self.cursor.first().copied()
            .ok_or_else(|| FerrumError::ParseError("Unexpected end of input".to_string()))
    }

    /// Implements two letter element.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn peek_two_letter_element(&self) -> bool {
        self.cursor.len() > 1
            && ((self.cursor[0] == b'B' && self.cursor[1] == b'r')
                || (self.cursor[0] == b'C' && self.cursor[1] == b'l'))
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

        // RDKit rejects [as]1cccc1 (cannot be kekulized without an H on As)
        assert!(parse_smiles("[as]1cccc1").is_err());
        let mol = parse_smiles("[asH]1cccc1").unwrap();
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

    #[test]
    /// Runs regression test for kekulized aromatic nitrogens keep correct implicit h.
    ///
    /// Function context: `crates/ferrum_core/src/io/smiles.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_kekulized_aromatic_nitrogens_keep_correct_implicit_h() {
        // Kekulized pyridine: aromatic N should stay pyridine-like (no implicit H).
        let pyridine = parse_smiles("C1=CC=NC=C1").unwrap();
        let n_idx = pyridine
            .atoms
            .iter()
            .position(|atom| atom.atomic_number == 7)
            .expect("contains nitrogen");
        assert!(pyridine.atoms[n_idx].is_aromatic());
        assert_eq!(pyridine.atoms[n_idx].num_implicit_h, 0);

        // Kekulized pyrrole: aromatic N keeps its H (RDKit stores it as explicit H).
        let pyrrole = parse_smiles("C1=CC=CN1").unwrap();
        let n_idx = pyrrole
            .atoms
            .iter()
            .position(|atom| atom.atomic_number == 7)
            .expect("contains nitrogen");
        assert!(pyrrole.atoms[n_idx].is_aromatic());
        assert_eq!(pyrrole.atoms[n_idx].total_h(), 1);
    }
}

/// Parses a batch of SMILES strings in parallel using Rayon.
pub fn parse_smiles_batch<S: AsRef<str> + Sync>(smiles_list: &[S]) -> Vec<crate::error::Result<crate::mol::molecule::Molecule>> {
    use rayon::prelude::*;
    smiles_list
        .par_iter()
        .map(|s| parse_smiles(s.as_ref()))
        .collect()
}
