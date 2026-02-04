use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;

/// Morgan Algorithmus zur Priorisierung.
pub fn rank_atoms(mol: &Molecule) -> Vec<usize> {
    todo!()
}

/// Baut Molekül basierend auf neuem Rank neu auf.
pub fn reorder_molecule(mol: Molecule, new_order: &[usize]) -> Molecule {
    todo!()
}
