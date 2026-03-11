use super::{aromaticity, canon, components, rings, valency};
use crate::error::Result;
use crate::mol::molecule::Molecule;

/// Die Pipeline of Truth.
pub fn process(mut mol: Molecule) -> Result<Molecule> {
    // 1. Cleaning
    components::keep_largest_fragment(&mut mol);

    // 2. Topology
    valency::assign_implicit_hydrogens(&mut mol)?;

    // 3. Perception
    rings::find_sssr(&mut mol);
    aromaticity::apply_hueckel(&mut mol);

    // 4. Canonicalization
    let order = canon::rank_atoms(&mol);
    let sorted = canon::reorder_molecule(mol, &order);

    Ok(sorted)
}
