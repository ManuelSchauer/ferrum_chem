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
        let idx = self.atoms.len() as AtomIdx;
        self.atoms.push(Atom::new(element));
        self.adjacency.push(SmallVec::new());
        idx
    }

    #[inline]
    fn add_bond(&mut self, u: AtomIdx, v: AtomIdx, order: BondType) {
        let bond_idx = self.bonds.len() as u16;
        self.bonds.push(Bond::new(u, v, order));

        // Add to adjacency lists
        self.adjacency[u as usize].push(bond_idx);
        self.adjacency[v as usize].push(bond_idx);
    }

    #[inline]
    fn set_charge(&mut self, atom: AtomIdx, charge: i8) {
        self.atoms[atom as usize].formal_charge = charge;
    }

    #[inline]
    fn set_isotope(&mut self, atom: AtomIdx, iso: u16) {
        self.atoms[atom as usize].isotope = iso;
    }

    #[inline]
    fn set_explicit_h(&mut self, atom: AtomIdx, count: u8) {
        self.atoms[atom as usize].num_explicit_h = count;
    }

    #[inline]
    fn set_aromatic(&mut self, atom: AtomIdx, aromatic: bool) {
        self.atoms[atom as usize].set_aromatic(aromatic);
    }

    fn finish(self) -> Molecule {
        self
    }
}
