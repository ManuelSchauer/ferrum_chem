//! Descriptor module: Aromatic Ring Count.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Aromatic Ring Count
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn compute_aromatic_ring_count`
//! - `fn compute_aromatic_ring_count_cached`
//! - `fn ring_bonds_all_aromatic`
//! - `struct AromaticRingCount`
//! - `fn name`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Aromatic Ring Count` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Aromatic Ring Count` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/aromatic_ring_count.rs` defines aromatic ring count behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Aromatic Ring Count` workflows.
//! - Repository unit and parity tests covering `Aromatic Ring Count` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::rings::{ensure_minimum_cycle_basis_bonds_cached, find_minimum_cycle_basis_bonds};
use crate::mol::molecule::Molecule;

// ---------------------------------------------------------------------------
// Public computation function
// ---------------------------------------------------------------------------

/// Compute aromatic ring count.
///
/// RDKit's Lipinski aromatic ring descriptor counts aromatic SSSR rings.
/// A ring is aromatic only when every bond around that ring is aromatic;
/// aromatic atoms alone are not sufficient in fused/Kekulized systems.
pub fn compute_aromatic_ring_count(mol: &Molecule) -> u32 {
    find_minimum_cycle_basis_bonds(mol)
        .iter()
        .filter(|ring| ring_bonds_all_aromatic(mol, ring))
        .count() as u32
}

/// Computes aromatic ring count cached.
///
/// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub fn compute_aromatic_ring_count_cached(mol: &mut Molecule) -> u32 {
    ensure_minimum_cycle_basis_bonds_cached(mol);
    let cycles = mol
        .cached_min_cycle_basis_bonds()
        .expect("minimum cycle basis cache initialized");
    let aromatic_bond_mask: Vec<bool> = mol.bonds.iter().map(|bond| bond.is_aromatic()).collect();
    count_aromatic_cycles(cycles, &aromatic_bond_mask)
}

/// Implements bonds all aromatic.
///
/// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn ring_bonds_all_aromatic(mol: &Molecule, ring: &[u16]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx as usize].is_aromatic())
}

/// Counts aromatic cycles using a precomputed aromatic-bond mask.
///
/// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn count_aromatic_cycles(
    cycles: &[smallvec::SmallVec<[u16; 8]>],
    aromatic_bond_mask: &[bool],
) -> u32 {
    cycles
        .iter()
        .filter(|ring| ring.iter().all(|&bond_idx| aromatic_bond_mask[bond_idx as usize]))
        .count() as u32
}

// ---------------------------------------------------------------------------
// Typed Descriptor
// ---------------------------------------------------------------------------

/// Counts aromatic rings in the molecule.
///
/// Returns Int(u32) with number of aromatic rings.
/// Requires aromaticity perception to be run first.
pub struct AromaticRingCount;

impl Descriptor for AromaticRingCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "AromaticRingCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_aromatic_ring_count_cached(mol))
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<AromaticRingCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for aromatic ring count benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_ring_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for aromatic ring count naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_ring_count_naphthalene() {
        let mut mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            2
        );
    }

    #[test]
    /// Runs regression test for aromatic ring count cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_ring_count_cyclohexane() {
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for aromatic ring count pyridine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_ring_count_pyridine() {
        let mut mol = parse_smiles("c1ccncc1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for aromatic ring count pyrrole.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_ring_count_pyrrole() {
        let mut mol = parse_smiles("c1cc[nH]c1").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for aromatic ring count no rings.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_ring_count_no_rings() {
        let mut mol = parse_smiles("CCCC").unwrap();
        assert_eq!(
            AromaticRingCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for compute aromatic ring count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_compute_aromatic_ring_count() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(compute_aromatic_ring_count(&mol), 1);
    }

    #[test]
    /// Runs regression test for aromatic ring count requires aromatic ring bonds.
    ///
    /// Function context: `crates/ferrum_core/src/desc/aromatic_ring_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_ring_count_requires_aromatic_ring_bonds() {
        let mol = parse_smiles("CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2").unwrap();
        assert_eq!(compute_aromatic_ring_count(&mol), 3);
    }
}
