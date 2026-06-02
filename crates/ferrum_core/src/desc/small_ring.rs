//! Descriptor module: Small Ring.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Small Ring
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct SmallRingDescriptor`
//! - `fn name`
//! - `fn calculate`
//! - `fn compute_total_ring_count`
//! - `fn find_small_rings`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Small Ring` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Small Ring` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/small_ring.rs` defines small ring behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Small Ring` workflows.
//! - Repository unit and parity tests covering `Small Ring` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/small_ring.rs`.

use crate::algo::rings::{
    cached_symmetrized_sssr_bonds_ref, ensure_symmetrized_sssr_bonds_cached,
    find_symmetrized_sssr_bonds,
};
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::HashSet;

/// Small ring descriptor - analyzes ring structure in molecules.
///
/// Returns IntU16Vec with 11 values:
/// - [0]: Total small rings (3-9 members)
/// - [1]: Aromatic rings
/// - [2]: Ring blocks (distinct ring systems)
/// - [3]: Aromatic blocks
/// - [4-10]: Individual ring sizes (3, 4, 5, 6, 7, 8, 9)
///
/// Based on CDK's SmallRingDescriptor implementation.
///
/// # Thread Safety
/// Thread-safe: stateless and uses only immutable references.
pub struct SmallRingDescriptor;

impl Descriptor for SmallRingDescriptor {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "SmallRingDescriptor"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut counts = vec![0u16; 11];

        // RDKit ring counts are based on a symmetrized SSSR-like set, so bridged
        // cages can contain more rings than the raw cycle rank.
        let total_ring_count = compute_total_ring_count(mol);
        counts[0] = total_ring_count as u16;

        // Find all small rings (3-9 members) for detailed size breakdown
        let rings = find_small_rings(mol, 3, 9);

        // Count rings by size
        let mut size_counts = [0u16; 7]; // indices 0-6 for sizes 3-9
        for ring in &rings {
            let size = ring.len();
            if size >= 3 && size <= 9 {
                size_counts[size - 3] += 1;
            }
        }

        // Set individual size counts (indices 4-10)
        for i in 0..7 {
            counts[4 + i] = size_counts[i];
        }

        // Count aromatic rings
        let arom_ring_count = count_aromatic_rings(mol, &rings);
        counts[1] = arom_ring_count as u16;

        // Count ring blocks (distinct ring systems)
        let ring_blocks = count_ring_blocks(mol);
        counts[2] = ring_blocks as u16;

        // Count aromatically connected components
        let arom_blocks = count_aromatic_components(mol);
        counts[3] = arom_blocks as u16;

        DescriptorValue::IntU16Vec(counts)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<SmallRingDescriptor>;
};

/// Computes total ring count.
///
/// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_total_ring_count(mol: &mut Molecule) -> usize {
    ensure_symmetrized_sssr_bonds_cached(mol);
    cached_symmetrized_sssr_bonds_ref(mol)
        .expect("symmetrized SSSR cache initialized")
        .len()
}

/// Find all small rings in a molecule (sizes min_size to max_size).
fn find_small_rings(mol: &Molecule, min_size: usize, max_size: usize) -> Vec<Vec<AtomIdx>> {
    ring_atom_cycles_from_symmetrized(mol)
        .into_iter()
        .filter(|ring| ring.len() >= min_size && ring.len() <= max_size)
        .collect()
}

/// Convert symmetrized ring bond cycles to per-ring atom cycles.
fn ring_atom_cycles_from_symmetrized(mol: &Molecule) -> Vec<Vec<AtomIdx>> {
    let mut out = Vec::new();
    for ring_bonds in find_symmetrized_sssr_bonds(mol) {
        let mut ring_atoms = Vec::with_capacity(ring_bonds.len());
        for &bond_idx in &ring_bonds {
            let bond = &mol.bonds[bond_idx as usize];
            if !ring_atoms.contains(&bond.source) {
                ring_atoms.push(bond.source);
            }
            if !ring_atoms.contains(&bond.target) {
                ring_atoms.push(bond.target);
            }
        }
        if ring_atoms.len() >= 3 {
            out.push(ring_atoms);
        }
    }
    deduplicate_rings(out)
}

/// Identify which atoms belong to which ring blocks (ring systems).
/// Returns a vector where 0 = not in ring, 1..N = ring block ID.
fn identify_ring_blocks(mol: &Molecule) -> Vec<usize> {
    let n_atoms = mol.num_atoms();
    if n_atoms == 0 {
        return Vec::new();
    }

    let ring_cycles = find_symmetrized_sssr_bonds(mol);
    if ring_cycles.is_empty() {
        return vec![0; n_atoms];
    }

    let mut ring_bond_mask = vec![false; mol.num_bonds()];
    for ring in &ring_cycles {
        for &bond_idx in ring {
            ring_bond_mask[bond_idx as usize] = true;
        }
    }

    let mut ring_blocks = vec![0usize; n_atoms];
    let mut visited = vec![false; n_atoms];
    let mut block_id = 1usize;
    let mut stack = Vec::new();

    for start in 0..n_atoms {
        if visited[start] {
            continue;
        }
        let has_ring_bond = mol.adjacency[start]
            .iter()
            .any(|&bond_idx| ring_bond_mask[bond_idx as usize]);
        if !has_ring_bond {
            continue;
        }

        stack.clear();
        stack.push(start as AtomIdx);
        visited[start] = true;
        ring_blocks[start] = block_id;

        while let Some(current) = stack.pop() {
            for &bond_idx in &mol.adjacency[current as usize] {
                if !ring_bond_mask[bond_idx as usize] {
                    continue;
                }
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                };
                if !visited[neighbor as usize] {
                    visited[neighbor as usize] = true;
                    ring_blocks[neighbor as usize] = block_id;
                    stack.push(neighbor);
                }
            }
        }

        block_id += 1;
    }

    ring_blocks
}

/// Count the number of distinct ring blocks (ring systems).
fn count_ring_blocks(mol: &Molecule) -> usize {
    let ring_blocks = identify_ring_blocks(mol);
    let unique_blocks: HashSet<usize> = ring_blocks.iter().filter(|&&id| id > 0).copied().collect();
    unique_blocks.len()
}

/// Count aromatic rings.
fn count_aromatic_rings(mol: &Molecule, rings: &[Vec<AtomIdx>]) -> usize {
    rings
        .iter()
        .filter(|ring| is_ring_aromatic(mol, ring))
        .count()
}

/// Check if a ring is aromatic (all bonds in the ring are aromatic).
fn is_ring_aromatic(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    if ring.is_empty() {
        return false;
    }

    for i in 0..ring.len() {
        let atom1 = ring[i];
        let atom2 = ring[(i + 1) % ring.len()];

        // Find bond between these atoms
        let mut bond_found = false;
        let mut bond_aromatic = false;

        for &bond_idx in &mol.adjacency[atom1 as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            if (bond.source == atom1 && bond.target == atom2)
                || (bond.source == atom2 && bond.target == atom1)
            {
                bond_found = true;
                bond_aromatic = bond.bond_type == crate::types::BondType::Aromatic;
                break;
            }
        }

        if !bond_found || !bond_aromatic {
            return false;
        }
    }

    true
}

/// Count aromatically connected components.
fn count_aromatic_components(mol: &Molecule) -> usize {
    let n_atoms = mol.num_atoms();
    let mut visited = vec![false; n_atoms];
    let mut component_count = 0;

    for start in 0..n_atoms {
        if visited[start] || !mol.atoms[start].is_aromatic() {
            continue;
        }

        // Found a new aromatic component
        component_count += 1;

        // BFS to mark all atoms in this aromatic component
        let mut queue = vec![start as AtomIdx];
        visited[start] = true;

        while let Some(current) = queue.pop() {
            for &bond_idx in &mol.adjacency[current as usize] {
                let bond = &mol.bonds[bond_idx as usize];

                // Only follow aromatic bonds
                if bond.bond_type != crate::types::BondType::Aromatic {
                    continue;
                }

                let neighbor = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                };

                if !visited[neighbor as usize] && mol.atoms[neighbor as usize].is_aromatic() {
                    visited[neighbor as usize] = true;
                    queue.push(neighbor);
                }
            }
        }
    }

    component_count
}

/// Deduplicate rings (rings with same atoms but different starting points).
fn deduplicate_rings(mut rings: Vec<Vec<AtomIdx>>) -> Vec<Vec<AtomIdx>> {
    let mut unique = Vec::new();
    let mut seen_sets: Vec<HashSet<AtomIdx>> = Vec::new();

    for ring in rings.drain(..) {
        let ring_set: HashSet<AtomIdx> = ring.iter().copied().collect();

        // Check if we've seen this set of atoms before
        if !seen_sets.iter().any(|s| s == &ring_set) {
            seen_sets.push(ring_set);
            unique.push(ring);
        }
    }

    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::algo::rings::find_sssr;
    use crate::algo::valency::assign_implicit_hydrogens;
    use crate::io::smiles::parse_smiles;

    /// Implements molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn prepare_molecule(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        find_sssr(&mut mol);
        perceive_aromaticity(&mut mol);
        mol
    }

    /// Implements small ring.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_small_ring(smiles: &str) -> Vec<u16> {
        let mut mol = prepare_molecule(smiles);
        match SmallRingDescriptor.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::IntU16Vec(v) => v,
            other => panic!("expected IntU16Vec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_benzene() {
        let output = calc_small_ring("c1ccccc1");

        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[1], 1, "Should have 1 aromatic ring");
        assert_eq!(output[2], 1, "Should have 1 ring block");
        assert!(output[3] >= 1, "Should have at least 1 aromatic component");
        assert_eq!(
            output[4 + 3],
            1,
            "Should have 1 six-membered ring (index 7)"
        );
    }

    #[test]
    /// Runs regression test for naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_naphthalene() {
        let output = calc_small_ring("c1ccc2ccccc2c1");

        assert!(output[0] >= 2, "Should have at least 2 small rings");
        assert!(output[1] >= 2, "Should have at least 2 aromatic rings");
        assert_eq!(output[2], 1, "Should have 1 ring block (fused system)");
        assert_eq!(output[3], 1, "Should have 1 aromatic component");
    }

    #[test]
    /// Runs regression test for cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_cyclohexane() {
        let output = calc_small_ring("C1CCCCC1");

        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[1], 0, "Should have 0 aromatic rings");
        assert_eq!(output[2], 1, "Should have 1 ring block");
        assert_eq!(output[3], 0, "Should have 0 aromatic components");
        assert_eq!(output[4 + 3], 1, "Should have 1 six-membered ring");
    }

    #[test]
    /// Runs regression test for cyclopropane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_cyclopropane() {
        let output = calc_small_ring("C1CC1");

        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[4], 1, "Should have 1 three-membered ring (index 4)");
    }

    #[test]
    /// Runs regression test for total rings uses symmetrized ring count for cubane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_total_rings_uses_symmetrized_ring_count_for_cubane() {
        let output = calc_small_ring("C12C3C4C1C5C2C3C45");
        assert_eq!(
            output[0], 6,
            "Cubane should expose all 6 symmetrized face rings"
        );
    }

    #[test]
    /// Runs regression test for total rings uses symmetrized ring count for adamantane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_total_rings_uses_symmetrized_ring_count_for_adamantane() {
        let output = calc_small_ring("C1C2CC3CC1CC(C2)C3");
        assert_eq!(
            output[0], 4,
            "Adamantane should expose 4 symmetrized face rings"
        );
    }

    #[test]
    /// Runs regression test for no rings.
    ///
    /// Function context: `crates/ferrum_core/src/desc/small_ring.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_no_rings() {
        let output = calc_small_ring("CCCC");

        for i in 0..11 {
            assert_eq!(
                output[i], 0,
                "All counts should be 0 for non-cyclic molecule"
            );
        }
    }
}
