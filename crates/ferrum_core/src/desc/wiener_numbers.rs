//! Descriptor module: Wiener Numbers.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.


//! ## File guide: Wiener Numbers
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct WienerIndex`
//! - `fn _compute_distance_matrix_unused`
//! - `fn wiener_from_distances`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Wiener Numbers` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Wiener Numbers` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/wiener_numbers.rs` defines wiener numbers behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Wiener Numbers` workflows.
//! - Repository unit and parity tests covering `Wiener Numbers` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/wiener_numbers.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::VecDeque;

/// Wiener Index: Sum of all shortest path distances between atom pairs.
///
/// The Wiener Index (W) is a fundamental topological descriptor that measures
/// the compactness of a molecule. It's defined as the sum of all shortest
/// topological distances (in bonds) between all pairs of non-hydrogen atoms.
///
/// # Formula
/// W = (1/2) × Σᵢ Σⱼ d(i,j)
///
/// where d(i,j) is the shortest path distance between atoms i and j.
///
/// # Properties
/// - Linear molecules have higher W than branched molecules of same size
/// - Correlates with boiling point, molar volume, and surface area
/// - Larger molecules have higher W
///
/// # Algorithm
/// Uses BFS (Breadth-First Search) from each atom to compute shortest paths.
/// Time complexity: O(n² + nm) where n = atoms, m = bonds
///
/// # Examples
/// - Methane (CH₄): 0 (only C-H bonds, no heavy-heavy paths)
/// - Ethane (C₂H₆): 1 (one C-C bond)
/// - Propane (C₃H₈): 4 (C0-C1=1, C1-C2=1, C0-C2=2)
/// - Benzene (C₆H₆): 27
///
/// # Thread Safety
/// This descriptor is stateless and thread-safe (Send + Sync).
pub struct WienerIndex;

impl WienerIndex {
    /// Compute all-pairs shortest path distances using BFS.
    #[allow(dead_code)]
fn _compute_distance_matrix_unused(mol: &Molecule) -> Vec<u32> {
        let n = mol.num_atoms();
        let mut distances = vec![u32::MAX; n * n];

        for start in 0..n {
            let start_idx = start as AtomIdx;
            let mut queue = VecDeque::new();
            queue.push_back(start_idx);
            distances[start * n + start] = 0;

            while let Some(current) = queue.pop_front() {
                let current_dist = distances[start * n + current as usize];

                for &bond_idx in &mol.adjacency[current as usize] {
                    let bond = &mol.bonds[bond_idx as usize];
                    let neighbor = if bond.source == current {
                        bond.target
                    } else {
                        bond.source
                    };

                    let neighbor_dist = &mut distances[start * n + neighbor as usize];
                    if *neighbor_dist == u32::MAX {
                        *neighbor_dist = current_dist + 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances
    }

    /// Calculate Wiener Index from distance matrix.
    fn wiener_from_distances(distances: &[u8], n: usize) -> f32 {
        let mut sum: u64 = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = distances[i * n + j];
                if dist != 255 {
                    sum += dist as u64;
                }
            }
        }

        sum as f32
    }
}

impl Descriptor for WienerIndex {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "WienerIndex"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let n = mol.num_atoms();

        if n <= 1 {
            return DescriptorValue::Float(0.0);
        }

        let distances = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
        let wiener = Self::wiener_from_distances(&distances, n);

        DescriptorValue::Float(wiener)
    }
}

// Static assertion: WienerIndex must be Send + Sync for thread safety
const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<WienerIndex>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Calculates wiener.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate_wiener(smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        WienerIndex.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_float()
    }

    #[test]
    /// Runs regression test for wiener methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_methane() {
        assert_eq!(calculate_wiener("C"), 0.0);
    }

    #[test]
    /// Runs regression test for wiener ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_ethane() {
        assert_eq!(calculate_wiener("CC"), 1.0);
    }

    #[test]
    /// Runs regression test for wiener propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_propane() {
        assert_eq!(calculate_wiener("CCC"), 4.0);
    }

    #[test]
    /// Runs regression test for wiener butane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_butane() {
        assert_eq!(calculate_wiener("CCCC"), 10.0);
    }

    #[test]
    /// Runs regression test for wiener isobutane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_isobutane() {
        assert_eq!(calculate_wiener("CC(C)C"), 9.0);
    }

    #[test]
    /// Runs regression test for wiener benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_benzene() {
        assert_eq!(calculate_wiener("c1ccccc1"), 27.0);
    }

    #[test]
    /// Runs regression test for wiener cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_cyclohexane() {
        assert_eq!(calculate_wiener("C1CCCCC1"), 27.0);
    }

    #[test]
    /// Runs regression test for wiener naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_naphthalene() {
        let wiener = calculate_wiener("c1ccc2ccccc2c1");
        assert!(
            wiener > 27.0,
            "Naphthalene should have higher W than benzene"
        );
    }

    #[test]
    /// Runs regression test for wiener linear vs branched.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_linear_vs_branched() {
        let linear = calculate_wiener("CCCC");
        let branched = calculate_wiener("CC(C)C");
        assert!(linear > branched);
    }

    #[test]
    /// Runs regression test for wiener comprehensive.
    ///
    /// Function context: `crates/ferrum_core/src/desc/wiener_numbers.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wiener_comprehensive() {
        let test_cases = vec![
            ("C", 0.0, "methane"),
            ("CC", 1.0, "ethane"),
            ("CCC", 4.0, "propane"),
            ("CCCC", 10.0, "butane"),
            ("CC(C)C", 9.0, "isobutane"),
            ("CCCCC", 20.0, "pentane"),
            ("c1ccccc1", 27.0, "benzene"),
            ("C1CCCCC1", 27.0, "cyclohexane"),
        ];

        for (smiles, expected, name) in test_cases {
            let actual = calculate_wiener(smiles);
            assert_eq!(
                actual, expected,
                "{} ({}): expected W={}, got W={}",
                name, smiles, expected, actual
            );
        }
    }
}
