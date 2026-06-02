//! I/O module: Sdf.
//!
//! This module handles external data interchange (SMILES/SMARTS/CSV/SDF) and
//! conversion into FerrumChem graph types. It is the boundary between textual
//! chemical formats and internal, typed molecular structures.
//!
//! Learning focus:
//! - Parser/state-machine techniques in Rust.
//! - Error-aware ingestion pipelines for large datasets.

//! ## File guide: Sdf
//!
//! This file belongs to FerrumChem's **input/output parsing layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn parse_sdf`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Sdf` treats textual chemistry formats as untrusted input and prefers explicit parse errors over silent repair.
//! - `Sdf` must preserve deterministic tokenization so repeated reads produce identical molecule/query graphs.
//! - Format-specific edge cases in `Sdf` are validated against toolkit-compatible semantics where possible.
//! - Specific scope anchor: `crates/ferrum_core/src/io/sdf.rs` defines sdf behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - SMILES/SMARTS/CTfile family specifications relevant to `Sdf` parser or reader behavior.
//! - RDKit parsing behavior used as external compatibility reference for `Sdf`.
//! - Rust parser/state-machine patterns for reliable streaming and error reporting.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/io/sdf.rs`.

use crate::error::{FerrumError, Result};
use crate::mol::atom::Atom;
use crate::mol::bond::Bond;
use crate::mol::elements::symbol_to_atomic_number;
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::fs;
use std::path::Path;

/// Parses an SDF file and returns a vector of molecules.
pub fn parse_sdf(path: &Path) -> Result<Vec<Molecule>> {
    let content = fs::read_to_string(path)
        .map_err(FerrumError::Io)?;

    // Split by $$$$ to get blocks and process in parallel
    let blocks: Vec<&str> = content.split("$$$$").collect();
    let molecules: Result<Vec<Molecule>> = blocks
        .into_par_iter()
        .filter(|block: &&str| !block.trim().is_empty())
        .map(|block| parse_sdf_block(block))
        .collect();

    molecules
}

/// Implements parse sdf block.
///
/// Function context: `crates/ferrum_core/src/io/sdf.rs`.
/// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
fn parse_sdf_block(block: &str) -> Result<Molecule> {
    let mut lines = block.lines().map(|l| l.trim_end());
    // Line 1: Header
    let _ = lines.next();
    // Line 2: Program
    let _ = lines.next();
    // Line 3: Comment
    let _ = lines.next();

    // Line 4: Counts
    let counts_line = lines
        .next()
        .ok_or_else(|| FerrumError::ParseError("Missing SDF counts line".to_string()))?;

    if counts_line.len() < 6 {
        return Err(FerrumError::ParseError(
            "Invalid SDF counts line".to_string(),
        ));
    }

    let num_atoms: usize = counts_line[0..3].trim().parse().unwrap_or(0);
    let num_bonds: usize = counts_line[3..6].trim().parse().unwrap_or(0);

    let mut mol = Molecule::with_capacity(num_atoms);

    // Atoms
    for _ in 0..num_atoms {
        let line = lines
            .next()
            .ok_or_else(|| FerrumError::ParseError("Unexpected end of atom block".to_string()))?;
        if line.len() < 34 {
            return Err(FerrumError::ParseError(
                "Invalid SDF atom line".to_string(),
            ));
        }
        let symbol = line[31..34].trim();
        let atomic_num = symbol_to_atomic_number(symbol).ok_or_else(|| {
            FerrumError::ParseError(format!("Unknown element: {}", symbol))
        })?;

        let mut atom = Atom::new(atomic_num);

        // Parse charge (column 37-39 in V2000: 0=unspecified, 1=+3, 2=+2, 3=+1, 4=radical, 5=-1, 6=-2, 7=-3)
        if line.len() >= 39 {
            let charge_val_str = line[36..39].trim();
            if let Ok(charge_val) = charge_val_str.parse::<u8>() {
                atom.formal_charge = match charge_val {
                    1 => 3,
                    2 => 2,
                    3 => 1,
                    5 => -1,
                    6 => -2,
                    7 => -3,
                    _ => 0,
                };
            }
        }

        mol.atoms.push(atom);
        mol.adjacency.push(SmallVec::new());
    }

    // Bonds
    for _ in 0..num_bonds {
        let line = lines
            .next()
            .ok_or_else(|| FerrumError::ParseError("Unexpected end of bond block".to_string()))?;
        if line.len() < 9 {
            return Err(FerrumError::ParseError(
                "Invalid SDF bond line".to_string(),
            ));
        }

        let a1: usize = line[0..3].trim().parse().unwrap_or(0);
        let a2: usize = line[3..6].trim().parse().unwrap_or(0);
        let type_val: usize = line[6..9].trim().parse().unwrap_or(1);

        if a1 == 0 || a2 == 0 || a1 > num_atoms || a2 > num_atoms {
            return Err(FerrumError::ParseError(
                "Invalid atom index in bond".to_string(),
            ));
        }

        let bond_type = match type_val {
            1 => BondType::Single,
            2 => BondType::Double,
            3 => BondType::Triple,
            4 => BondType::Aromatic,
            _ => BondType::Single,
        };

        if bond_type == BondType::Aromatic {
            mol.atoms[a1 - 1].set_aromatic(true);
            mol.atoms[a2 - 1].set_aromatic(true);
        }

        let bond_idx = mol.bonds.len() as u16;
        mol.bonds
            .push(Bond::new((a1 - 1) as u16, (a2 - 1) as u16, bond_type));
        mol.adjacency[a1 - 1].push(bond_idx);
        mol.adjacency[a2 - 1].push(bond_idx);
    }

    // Implicit Hydrogens
    crate::algo::valency::assign_implicit_hydrogens(&mut mol)?;

    // Aromaticity and rings
    let needs_aromaticity = mol
        .bonds
        .iter()
        .any(|b| matches!(b.bond_type, BondType::Double | BondType::Aromatic));
    let saw_rings = mol.bonds.len() >= mol.atoms.len();
    
    if needs_aromaticity {
        crate::algo::aromaticity::perceive_aromaticity(&mut mol);
        crate::algo::valency::assign_implicit_hydrogens(&mut mol)?;
    } else if saw_rings {
        crate::algo::rings::find_sssr(&mut mol);
    }

    if saw_rings {
        crate::algo::rings::ensure_minimum_cycle_basis_bonds_cached(&mut mol);
    }

    Ok(mol)
}
