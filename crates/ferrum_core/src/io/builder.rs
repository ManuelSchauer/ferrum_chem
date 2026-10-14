#![allow(dead_code)]
//! I/O module: Builder.
//!
//! This module handles external data interchange (SMILES/SMARTS/CSV/SDF) and
//! conversion into FerrumChem graph types. It is the boundary between textual
//! chemical formats and internal, typed molecular structures.
//!
//! Learning focus:
//! - Parser/state-machine techniques in Rust.
//! - Error-aware ingestion pipelines for large datasets.

use crate::mol::atom::Atom;
use crate::mol::bond::Bond;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use smallvec::SmallVec;

/// Abstraktion für den Aufbau eines Moleküls.
pub trait MoleculeBuilder {
    fn add_atom(&mut self, element: u8) -> AtomIdx;
    fn add_bond(&mut self, u: AtomIdx, v: AtomIdx, order: BondType);
    fn set_charge(&mut self, atom: AtomIdx, charge: i8);
    fn set_isotope(&mut self, atom: AtomIdx, iso: u16);
    fn set_explicit_h(&mut self, atom: AtomIdx, count: u8);
    fn set_aromatic(&mut self, atom: AtomIdx, aromatic: bool);
    fn finish(self) -> Molecule;
}

// Implementierung direkt auf dem Molecule Struct
impl MoleculeBuilder for Molecule {
    #[inline]
    fn add_atom(&mut self, element: u8) -> AtomIdx {
        self.invalidate_cached_properties();
        let idx = self.atoms.len() as AtomIdx;
        self.atoms.push(Atom::new(element));
        self.adjacency.push(SmallVec::new());
        idx
    }

    #[inline]
    fn add_bond(&mut self, u: AtomIdx, v: AtomIdx, order: BondType) {
        self.invalidate_cached_properties();
        let bond_idx = self.bonds.len() as u16;
        self.bonds.push(Bond::new(u, v, order));

        // Add to adjacency lists
        self.adjacency[u as usize].push(bond_idx);
        self.adjacency[v as usize].push(bond_idx);
    }

    #[inline]
    fn set_charge(&mut self, atom: AtomIdx, charge: i8) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].formal_charge = charge;
    }

    #[inline]
    fn set_isotope(&mut self, atom: AtomIdx, iso: u16) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].isotope = iso;
    }

    #[inline]
    fn set_explicit_h(&mut self, atom: AtomIdx, count: u8) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].num_explicit_h = count;
    }

    #[inline]
    fn set_aromatic(&mut self, atom: AtomIdx, aromatic: bool) {
        self.invalidate_cached_properties();
        self.atoms[atom as usize].set_aromatic(aromatic);
    }

    fn finish(self) -> Molecule {
        self
    }
}

impl Molecule {
    /// Creates a fluent builder for ergonomic programmatic molecule construction.
    pub fn builder() -> FluentMoleculeBuilder {
        FluentMoleculeBuilder::new()
    }
}

/// A fluent builder for programmatic molecule construction that supports method chaining.
/// Atom indices are implicitly sequential (0 for the first added atom, 1 for the second, etc.).
pub struct FluentMoleculeBuilder {
    mol: Molecule,
}

impl Default for FluentMoleculeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentMoleculeBuilder {
    pub fn new() -> Self {
        Self { mol: Molecule::default() }
    }

    pub fn add_atom(mut self, atomic_number: u8) -> Self {
        self.mol.add_atom(atomic_number);
        self
    }

    pub fn add_bond(mut self, u: AtomIdx, v: AtomIdx, order: BondType) -> Self {
        self.mol.add_bond(u, v, order);
        self
    }

    pub fn set_charge(mut self, atom: AtomIdx, charge: i8) -> Self {
        self.mol.set_charge(atom, charge);
        self
    }

    pub fn set_isotope(mut self, atom: AtomIdx, iso: u16) -> Self {
        self.mol.set_isotope(atom, iso);
        self
    }

    pub fn set_explicit_h(mut self, atom: AtomIdx, count: u8) -> Self {
        self.mol.set_explicit_h(atom, count);
        self
    }

    pub fn set_aromatic(mut self, atom: AtomIdx, aromatic: bool) -> Self {
        self.mol.set_aromatic(atom, aromatic);
        self
    }

    pub fn build(self) -> Molecule {
        self.mol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluent_builder() {
        let mol = Molecule::builder()
            .add_atom(6) // 0: Carbon
            .add_atom(8) // 1: Oxygen
            .add_bond(0, 1, BondType::Double)
            .set_charge(1, -1)
            .build();

        assert_eq!(mol.atoms.len(), 2);
        assert_eq!(mol.bonds.len(), 1);
        assert_eq!(mol.atoms[0].atomic_number, 6);
        assert_eq!(mol.atoms[1].atomic_number, 8);
        assert_eq!(mol.atoms[1].formal_charge, -1);
    }
}

