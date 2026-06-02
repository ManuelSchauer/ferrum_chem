//! Ring type counting descriptors
//!
//! Classifies rings by saturation and aromaticity:
//! - Saturated rings: no double/triple bonds
//! - Aliphatic rings: non-aromatic
//! - Aromatic rings: marked as aromatic

//! ## File guide: Ring Types
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn ring_bonds_all_single`
//! - `fn ring_bonds_all_aromatic`
//! - `fn num_saturated_rings`
//! - `fn num_aliphatic_rings`
//! - `fn num_aromatic_rings`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Ring Types` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Ring Types` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/ring_types.rs` defines ring types behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Ring Types` workflows.
//! - Repository unit and parity tests covering `Ring Types` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/ring_types.rs`.

use crate::algo::rings::{cached_symmetrized_sssr_bonds_ref, ensure_symmetrized_sssr_bonds_cached};
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// Implements bonds all single.
///
/// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn ring_bonds_all_single(mol: &Molecule, ring: &[u16]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx as usize].bond_type == BondType::Single)
}

/// Implements bonds all aromatic.
///
/// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn ring_bonds_all_aromatic(mol: &Molecule, ring: &[u16]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx as usize].is_aromatic())
}

/// Count number of saturated rings (no double/triple bonds)
///
/// A saturated ring contains only single bonds.
///
/// # Examples
/// - Cyclohexane: 1 saturated ring
/// - Benzene: 0 saturated rings (aromatic double bonds)
/// - Cyclohexene: 0 saturated rings (one double bond)
pub fn num_saturated_rings(mol: &mut Molecule) -> u32 {
    ensure_symmetrized_sssr_bonds_cached(mol);
    let rings = cached_symmetrized_sssr_bonds_ref(mol).expect("symmetrized SSSR cache initialized");

    rings
        .iter()
        .filter(|ring| ring_bonds_all_single(mol, ring))
        .count() as u32
}

/// Count number of aliphatic rings (non-aromatic)
///
/// RDKit defines aliphatic rings as rings containing at least one non-aromatic
/// bond. This includes both saturated rings and rings with double bonds that
/// do not qualify as fully aromatic rings.
///
/// # Examples
/// - Cyclohexane: 1 aliphatic ring
/// - Cyclohexene: 1 aliphatic ring
/// - Benzene: 0 aliphatic rings (aromatic)
/// - Naphthalene: 0 aliphatic rings (both rings aromatic)
pub fn num_aliphatic_rings(mol: &mut Molecule) -> u32 {
    ensure_symmetrized_sssr_bonds_cached(mol);
    let rings = cached_symmetrized_sssr_bonds_ref(mol).expect("symmetrized SSSR cache initialized");

    rings
        .iter()
        .filter(|ring| !ring_bonds_all_aromatic(mol, ring))
        .count() as u32
}

/// Count number of aromatic rings
///
/// RDKit counts aromatic SSSR rings based on aromatic bonds, not just aromatic
/// atoms. Fused systems can contain rings where every atom is aromatic while
/// one or more ring bonds remain non-aromatic/Kekulized.
///
/// # Examples
/// - Benzene: 1 aromatic ring
/// - Naphthalene: 2 aromatic rings
/// - Pyridine: 1 aromatic ring
/// - Cyclohexane: 0 aromatic rings
pub fn num_aromatic_rings(mol: &mut Molecule) -> u32 {
    ensure_symmetrized_sssr_bonds_cached(mol);
    let rings = cached_symmetrized_sssr_bonds_ref(mol).expect("symmetrized SSSR cache initialized");

    rings
        .iter()
        .filter(|ring| ring_bonds_all_aromatic(mol, ring))
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    /// Runs regression test for saturated cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_saturated_cyclohexane() {
        let mut mol = parse("C1CCCCC1");
        assert_eq!(num_saturated_rings(&mut mol), 1);
        assert_eq!(num_aliphatic_rings(&mut mol), 1);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for aromatic benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_benzene() {
        let mut mol = parse("c1ccccc1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 1);
    }

    #[test]
    /// Runs regression test for cyclohexene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_cyclohexene() {
        let mut mol = parse("C1=CCCCC1");
        assert_eq!(num_saturated_rings(&mut mol), 0); // Has double bond
        assert_eq!(num_aliphatic_rings(&mut mol), 1); // Not aromatic
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_naphthalene() {
        let mut mol = parse("c1ccc2ccccc2c1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 2); // Two fused aromatic rings
    }

    #[test]
    /// Runs regression test for pyridine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pyridine() {
        let mut mol = parse("c1ccncc1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 1);
    }

    #[test]
    /// Runs regression test for decalin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_decalin() {
        // Two fused saturated rings
        let mut mol = parse("C1CCC2CCCCC2C1");
        assert_eq!(num_saturated_rings(&mut mol), 2);
        assert_eq!(num_aliphatic_rings(&mut mol), 2);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for cubane uses symmetrized ring count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_cubane_uses_symmetrized_ring_count() {
        let mut mol = parse("C12C3C4C1C5C2C3C45");
        assert_eq!(num_saturated_rings(&mut mol), 6);
        assert_eq!(num_aliphatic_rings(&mut mol), 6);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for adamantane uses symmetrized ring count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_adamantane_uses_symmetrized_ring_count() {
        let mut mol = parse("C1C2CC3CC1CC(C2)C3");
        assert_eq!(num_saturated_rings(&mut mol), 4);
        assert_eq!(num_aliphatic_rings(&mut mol), 4);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for acyclic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acyclic() {
        let mut mol = parse("CCCCCC"); // Hexane
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    /// Runs regression test for rdkit like ring counts for fused heterocycle.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rdkit_like_ring_counts_for_fused_heterocycle() {
        let mut mol = parse(
            "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2",
        );
        assert_eq!(num_saturated_rings(&mut mol), 1);
        assert_eq!(num_aliphatic_rings(&mut mol), 3);
        assert_eq!(num_aromatic_rings(&mut mol), 3);
    }
}

// ===========================================================================
// Typed API: Ring Types
// ===========================================================================

/// Ring type counts descriptor.
///
/// Returns IntU16Vec with 3 values:
/// [saturated_rings, aliphatic_rings, aromatic_rings]
///
/// Uses u16 since molecules rarely have >65k rings of any type.
pub struct RingTypeCounts;

impl Descriptor for RingTypeCounts {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "RingTypeCounts"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        ensure_symmetrized_sssr_bonds_cached(mol);
        let rings =
            cached_symmetrized_sssr_bonds_ref(mol).expect("symmetrized SSSR cache initialized");

        let mut saturated = 0u16;
        let mut aliphatic = 0u16;
        let mut aromatic = 0u16;

        for ring in rings {
            let is_aromatic = ring_bonds_all_aromatic(mol, ring);

            if is_aromatic {
                aromatic += 1;
            } else {
                aliphatic += 1;

                if ring_bonds_all_single(mol, ring) {
                    saturated += 1;
                }
            }
        }

        DescriptorValue::IntU16Vec(vec![saturated, aliphatic, aromatic])
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/ring_types.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<RingTypeCounts>;
};
