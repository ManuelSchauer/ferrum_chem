//! # SDF / MOL file reader
//!
//! Reads V2000 and V3000 molfiles (one or many records separated by `$$$$`)
//! following RDKit's `MolFileParser` semantics, then runs the shared
//! RDKit-compatible sanitization (explicit H removal, kekulization,
//! aromaticity, ...), so molecules from SDF and SMILES are typed identically.
//!
//! Supported: atom block charges/radicals/mass differences/valence field,
//! `D`/`T` hydrogen isotopes, `M  CHG`, `M  ISO`, `M  RAD`, bond types 1-4,
//! query bond types (no valence contribution), coordination bonds (dative),
//! 2D/3D coordinates (stored as a conformer).
//!
//! Each record yields its own result, so one bad record does not abort a file.

use crate::error::{FerrumError, Result};
use crate::mol::atom::Atom;
use crate::mol::bond::Bond;
use crate::mol::conformer::Conformer;
use crate::mol::elements::symbol_to_atomic_number;
use crate::mol::molecule::Molecule;
use crate::mol::rdkit_data::ELEMENTS;
use crate::types::BondType;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::path::Path;

/// One SDF record: its title line and the parsed molecule (or the error).
#[derive(Debug)]
pub struct SdfRecord {
    /// First header line of the record (molecule name).
    pub name: String,
    pub molecule: Result<Molecule>,
}

/// Reads all records of an SDF file in parallel.
pub fn read_sdf(path: &Path) -> Result<Vec<SdfRecord>> {
    let content = std::fs::read_to_string(path).map_err(FerrumError::Io)?;
    Ok(parse_sdf_str(&content))
}

/// Parses all records of an SDF string.
pub fn parse_sdf_str(content: &str) -> Vec<SdfRecord> {
    split_records(content)
        .into_par_iter()
        .map(|lines| SdfRecord {
            name: lines.first().map(|l| l.trim().to_string()).unwrap_or_default(),
            molecule: parse_mol_block_lines(&lines),
        })
        .collect()
}

/// Parses a single MOL block.
pub fn parse_mol_block(block: &str) -> Result<Molecule> {
    let lines: Vec<&str> = block.lines().collect();
    parse_mol_block_lines(&lines)
}

/// Backwards-compatible helper: all successfully parsed molecules of a file.
pub fn parse_sdf(path: &Path) -> Result<Vec<Molecule>> {
    Ok(read_sdf(path)?.into_iter().filter_map(|r| r.molecule.ok()).collect())
}

/// Splits on `$$$$` lines; drops trailing empty records.
fn split_records(content: &str) -> Vec<Vec<&str>> {
    let mut records = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in content.lines() {
        if line.trim_end() == "$$$$" {
            records.push(std::mem::take(&mut current));
        } else {
            current.push(line.trim_end_matches('\r'));
        }
    }
    if current.iter().any(|l| !l.trim().is_empty()) {
        records.push(current);
    }
    records
}

fn err(msg: impl Into<String>) -> FerrumError {
    FerrumError::ParseError(msg.into())
}

/// Fixed-width field `[start, start+len)`, trimmed; empty if the line is short.
fn field(line: &str, start: usize, len: usize) -> &str {
    let end = (start + len).min(line.len());
    if start >= end {
        return "";
    }
    line.get(start..end).unwrap_or("").trim()
}

fn int_field(line: &str, start: usize, len: usize) -> Result<i32> {
    let f = field(line, start, len);
    if f.is_empty() {
        return Ok(0);
    }
    f.parse::<i32>().map_err(|_| err(format!("Cannot convert '{}' to int", f)))
}

/// Per-atom data that is resolved after the bond block is known.
#[derive(Default, Clone, Copy)]
struct AtomExtra {
    total_valence: i32,
}

struct Builder {
    mol: Molecule,
    extras: Vec<AtomExtra>,
    positions: Vec<[f64; 3]>,
}

impl Builder {
    fn new(n: usize) -> Self {
        Self { mol: Molecule::with_capacity(n), extras: Vec::with_capacity(n), positions: Vec::with_capacity(n) }
    }

    fn add_atom(&mut self, symbol: &str, pos: [f64; 3]) -> Result<usize> {
        let mut atom = match symbol {
            "D" => {
                let mut a = Atom::new(1);
                a.isotope = 2;
                a
            }
            "T" => {
                let mut a = Atom::new(1);
                a.isotope = 3;
                a
            }
            "*" | "R" | "A" | "Q" | "L" | "LP" | "R#" | "Pol" | "Mod" | "AH" | "QH" | "X" | "XH" | "M" | "MH" => {
                let mut a = Atom::new(0);
                if matches!(symbol, "*" | "R" | "A" | "Q" | "AH" | "QH" | "X" | "XH" | "M" | "MH") {
                    a.set_no_implicit_h(true);
                }
                a
            }
            s if s.starts_with('R') && s[1..].chars().all(|c| c.is_ascii_digit()) => Atom::new(0),
            s => Atom::new(
                symbol_to_atomic_number(s).ok_or_else(|| err(format!("Unknown element symbol '{}'", s)))?,
            ),
        };
        atom.hybridization = crate::types::Hybridization::Unspecified;
        if self.mol.atoms.len() >= u16::MAX as usize {
            return Err(err("Too many atoms (max 65535)"));
        }
        self.mol.atoms.push(atom);
        self.mol.adjacency.push(SmallVec::new());
        self.extras.push(AtomExtra::default());
        self.positions.push(pos);
        Ok(self.mol.atoms.len() - 1)
    }

    fn add_bond(&mut self, a1: usize, a2: usize, code: i32) -> Result<()> {
        let n = self.mol.atoms.len();
        if a1 == 0 || a2 == 0 || a1 > n || a2 > n || a1 == a2 {
            return Err(err(format!("Invalid atom index in bond {} {}", a1, a2)));
        }
        let (a, b) = (a1 - 1, a2 - 1);
        let bond_type = match code {
            1 => BondType::Single,
            2 => BondType::Double,
            3 => BondType::Triple,
            4 => BondType::Aromatic,
            9 => BondType::Dative,
            // 5-8: query bond types, 10: hydrogen bond -> no valence contribution
            _ => BondType::Unknown,
        };
        if self.mol.bonds.len() >= u16::MAX as usize {
            return Err(err("Too many bonds (max 65535)"));
        }
        let idx = self.mol.bonds.len() as u16;
        let mut bond = Bond::new(a as u16, b as u16, bond_type);
        if bond_type == BondType::Aromatic {
            bond.set_aromatic(true);
            self.mol.atoms[a].set_aromatic(true);
            self.mol.atoms[b].set_aromatic(true);
        }
        self.mol.bonds.push(bond);
        self.mol.adjacency[a].push(idx);
        self.mol.adjacency[b].push(idx);
        Ok(())
    }

    /// Applies the molfile valence field (RDKit `molTotValence`), then sanitizes.
    fn finish(mut self) -> Result<Molecule> {
        for i in 0..self.mol.atoms.len() {
            let tv = self.extras[i].total_valence;
            if tv == 0 {
                continue;
            }
            let atom_ev: f64 = self.mol.adjacency[i]
                .iter()
                .map(|&bi| match self.mol.bonds[bi as usize].bond_type {
                    BondType::Single => 1.0,
                    BondType::Double => 2.0,
                    BondType::Triple => 3.0,
                    BondType::Aromatic => 1.5,
                    BondType::Dative if self.mol.bonds[bi as usize].target as usize == i => 1.0,
                    _ => 0.0,
                })
                .sum();
            let ev = (atom_ev + 0.1).round() as i32;
            let atom = &mut self.mol.atoms[i];
            atom.set_no_implicit_h(true);
            atom.num_explicit_h = if tv == 15 || tv == -1 || ev > tv { 0 } else { (tv - ev) as u8 };
        }
        let mut mol = self.mol;
        mol.conformers.push(Conformer::new(self.positions));
        crate::algo::sanitize::sanitize(&mut mol)?;
        Ok(mol)
    }
}

fn parse_mol_block_lines(lines: &[&str]) -> Result<Molecule> {
    if lines.len() < 4 {
        return Err(err("MOL block too short"));
    }
    let counts = lines[3];
    if counts.contains("V3000") {
        return parse_v3000(&lines[4..]);
    }
    let num_atoms = int_field(counts, 0, 3)? as usize;
    let num_bonds = int_field(counts, 3, 3)? as usize;
    let mut b = Builder::new(num_atoms);
    let mut idx = 4;
    for _ in 0..num_atoms {
        let line = *lines.get(idx).ok_or_else(|| err("Unexpected end of atom block"))?;
        idx += 1;
        if line.len() < 32 {
            return Err(err(format!("Atom line too short: '{}'", line)));
        }
        let coord = |s: usize| field(line, s, 10).parse::<f64>().map_err(|_| err("Cannot process coordinates"));
        let pos = [coord(0)?, coord(10)?, coord(20)?];
        let symbol = field(line, 31, 3);
        let a = b.add_atom(symbol, pos)?;
        let mass_diff = int_field(line, 34, 2)?;
        let chg = int_field(line, 36, 3)?;
        if chg != 0 {
            // 1..3 -> +3..+1, 4 -> doublet radical (no charge), 5..7 -> -1..-3
            b.mol.atoms[a].formal_charge = (4 - chg) as i8;
        }
        if mass_diff != 0 {
            let z = b.mol.atoms[a].atomic_number as usize;
            let iso = ELEMENTS[z.min(118)].common_isotope as i32 + mass_diff;
            b.mol.atoms[a].isotope = iso.max(0) as u16;
        }
        b.extras[a].total_valence = int_field(line, 48, 3)?;
    }
    for _ in 0..num_bonds {
        let line = *lines.get(idx).ok_or_else(|| err("Unexpected end of bond block"))?;
        idx += 1;
        let a1 = int_field(line, 0, 3)? as usize;
        let a2 = int_field(line, 3, 3)? as usize;
        let code = int_field(line, 6, 3)?;
        b.add_bond(a1, a2, code)?;
    }
    // Properties block
    let mut first_charge_line = true;
    while idx < lines.len() {
        let line = lines[idx];
        idx += 1;
        if line.starts_with("M  END") {
            break;
        }
        let is_chg = line.starts_with("M  CHG");
        let is_rad = line.starts_with("M  RAD");
        let is_iso = line.starts_with("M  ISO");
        if !(is_chg || is_rad || is_iso) {
            continue;
        }
        // The first CHG/RAD line supersedes all atom-block charges/radicals.
        if (is_chg || is_rad) && first_charge_line {
            for atom in b.mol.atoms.iter_mut() {
                atom.formal_charge = 0;
                atom.set_num_radicals(0);
            }
            first_charge_line = false;
        }
        let nent = int_field(line, 6, 3)? as usize;
        let mut pos = 9;
        for _ in 0..nent {
            let aid = int_field(line, pos, 4)? as usize;
            let val = int_field(line, pos + 4, 4)?;
            pos += 8;
            if aid == 0 || aid > b.mol.atoms.len() {
                return Err(err(format!("Invalid atom index {} in property line", aid)));
            }
            let atom = &mut b.mol.atoms[aid - 1];
            if is_chg {
                atom.formal_charge = val.clamp(-15, 15) as i8;
            } else if is_iso {
                atom.isotope = val.max(0) as u16;
            } else {
                // RAD: 1 singlet / 3 triplet -> 2 radical electrons, 2 doublet -> 1
                atom.set_num_radicals(match val {
                    1 | 3 => 2,
                    2 => 1,
                    _ => 0,
                });
                atom.set_no_implicit_h(val != 0);
            }
        }
    }
    b.finish()
}

/// Joins V3000 continuation lines (`-` at end of line) and strips `M  V30 `.
fn v30_lines(lines: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut acc = String::new();
    for &l in lines {
        if l.starts_with("M  END") {
            break;
        }
        let Some(body) = l.strip_prefix("M  V30 ") else { continue };
        if let Some(stripped) = body.strip_suffix('-') {
            acc.push_str(stripped);
            continue;
        }
        acc.push_str(body);
        out.push(std::mem::take(&mut acc));
    }
    out
}

fn parse_v3000(lines: &[&str]) -> Result<Molecule> {
    let body = v30_lines(lines);
    let mut b = Builder::new(0);
    let mut section = "";
    let mut ids: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for line in &body {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("BEGIN ") {
            section = match rest.trim() {
                "ATOM" => "atom",
                "BOND" => "bond",
                _ => "",
            };
            continue;
        }
        if line.starts_with("END ") {
            section = "";
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        match section {
            "atom" => {
                if toks.len() < 5 {
                    return Err(err(format!("Bad V3000 atom line '{}'", line)));
                }
                let id: i64 = toks[0].parse().map_err(|_| err("Bad V3000 atom index"))?;
                let num = |s: &str| s.parse::<f64>().map_err(|_| err("Cannot process coordinates"));
                let pos = [num(toks[2])?, num(toks[3])?, num(toks[4])?];
                let a = b.add_atom(toks[1], pos)?;
                ids.insert(id, a);
                for kv in &toks[6.min(toks.len())..] {
                    let Some((k, v)) = kv.split_once('=') else { continue };
                    let v: i32 = v.parse().unwrap_or(0);
                    match k {
                        "CHG" => b.mol.atoms[a].formal_charge = v.clamp(-15, 15) as i8,
                        "MASS" => b.mol.atoms[a].isotope = v.max(0) as u16,
                        "RAD" => {
                            b.mol.atoms[a].set_num_radicals(match v {
                                1 | 3 => 2,
                                2 => 1,
                                _ => 0,
                            });
                            b.mol.atoms[a].set_no_implicit_h(v != 0);
                        }
                        "VAL" => b.extras[a].total_valence = v,
                        _ => {}
                    }
                }
            }
            "bond" => {
                if toks.len() < 4 {
                    return Err(err(format!("Bad V3000 bond line '{}'", line)));
                }
                let code: i32 = toks[1].parse().map_err(|_| err("Bad V3000 bond type"))?;
                let a1: i64 = toks[2].parse().map_err(|_| err("Bad V3000 bond atom"))?;
                let a2: i64 = toks[3].parse().map_err(|_| err("Bad V3000 bond atom"))?;
                let a1 = *ids.get(&a1).ok_or_else(|| err("Unknown atom in V3000 bond"))?;
                let a2 = *ids.get(&a2).ok_or_else(|| err("Unknown atom in V3000 bond"))?;
                b.add_bond(a1 + 1, a2 + 1, code)?;
            }
            _ => {}
        }
    }
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ETHANOL_V2000: &str = "ethanol
  RDKit          2D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
";

    #[test]
    fn multiple_records() {
        let sdf = format!("{ETHANOL_V2000}$$$$\n{ETHANOL_V2000}> <prop>\n1\n\n$$$$\n");
        let records = parse_sdf_str(&sdf);
        assert_eq!(records.len(), 2);
        for r in records {
            assert_eq!(r.name, "ethanol");
            let mol = r.molecule.unwrap();
            assert_eq!(mol.num_atoms(), 3);
            assert_eq!(mol.atoms[2].total_h(), 1);
        }
    }

    #[test]
    fn charge_lines_supersede_atom_block() {
        let block = "methylammonium
  RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  CHG  1   2   1
M  END
";
        let mol = parse_mol_block(block).unwrap();
        assert_eq!(mol.atoms[1].formal_charge, 1);
        assert_eq!(mol.atoms[1].total_h(), 3);
    }

    #[test]
    fn explicit_hydrogens_are_removed() {
        let block = "water
  RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.9600    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2400    0.9300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END
";
        let mol = parse_mol_block(block).unwrap();
        assert_eq!(mol.num_atoms(), 1);
        assert_eq!(mol.atoms[0].total_h(), 2);
        assert_eq!(mol.conformers[0].positions.len(), 1);
    }

    #[test]
    fn v3000_block() {
        let block = "ethanol
  RDKit          2D

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 3 2 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C 0 0 0 0
M  V30 2 C 1.299 0.75 0 0
M  V30 3 O 2.598 0 0 0 CHG=-1
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 END BOND
M  V30 END CTAB
M  END
";
        let mol = parse_mol_block(block).unwrap();
        assert_eq!(mol.num_atoms(), 3);
        assert_eq!(mol.atoms[2].formal_charge, -1);
        assert_eq!(mol.atoms[2].total_h(), 0);
    }
}
