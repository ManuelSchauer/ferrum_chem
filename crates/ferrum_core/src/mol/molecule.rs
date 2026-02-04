use smallvec::SmallVec;
use crate::mol::{atom::Atom, bond::Bond};
use crate::types::{AtomIdx, BondIdx};

/// Molecular graph representation.
/// 
/// # Thread Safety
/// `Molecule` is `Send + Sync`, making it safe to share across threads.
/// This is critical for parallel batch processing with Rayon.
/// All internal data structures (Vec, SmallVec) are thread-safe.
#[derive(Debug, Default)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    // Adjazenzliste: Für jedes Atom eine Liste von Bindungs-Indizes.
    // SmallVec optimiert: Keine Heap-Allocation bei < 4 Bindungen.
    pub adjacency: Vec<SmallVec<[BondIdx; 4]>>,
}

// Static assertion: Molecule must be Send + Sync for parallel processing
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Molecule>;
};

impl Molecule {
    pub fn new() -> Self { Self::default() }

    pub fn num_atoms(&self) -> usize { self.atoms.len() }
    pub fn num_bonds(&self) -> usize { self.bonds.len() }

    pub fn get_degree(&self, atom: AtomIdx) -> usize {
        self.adjacency[atom as usize].len()
    }
}
