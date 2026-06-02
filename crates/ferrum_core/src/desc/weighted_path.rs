//! Weighted Path descriptors.
//!
//! Molecular path descriptors weighted by atomic properties.
//! For each bond-path of length k (k=1..5), compute:
//!   WP(k) = Σ_paths (∏ w_i for atoms in path) / k
//!
//! Property weights: mass, electronegativity.
//!
//! Reference: Randić, M. (1993) "Novel graph theoretical approach to
//!            heteroatoms in QSAR", Chemometrics & Intell. Lab. Systems.

//! ## File guide: Weighted Path
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn weighted_paths`
//! - `fn dfs_wp`
//! - `struct WeightedPath`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Weighted Path` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Weighted Path` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/weighted_path.rs` defines weighted path behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Weighted Path` workflows.
//! - Repository unit and parity tests covering `Weighted Path` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/weighted_path.rs`.

use super::atomic_props::{atomic_mass, pauling_electronegativity};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;

const MAX_PATH_LEN: usize = 5;

/// Compute weighted path sums for given atom weights.
/// Returns [WP1, WP2, WP3, WP4, WP5].
fn weighted_paths(mol: &Molecule, weights: &[f64]) -> [f64; MAX_PATH_LEN] {
    let n = mol.num_atoms();
    let mut wp = [0.0f64; MAX_PATH_LEN];

    if n < 2 {
        return wp;
    }

    let mut path = Vec::with_capacity(MAX_PATH_LEN + 1);

    for start in 0..n {
        path.clear();
        path.push(start as AtomIdx);
        dfs_wp(mol, weights, &mut path, &mut wp);
    }

    // Each path found twice (forward + reverse)
    for v in wp.iter_mut() {
        *v /= 2.0;
    }

    wp
}

/// Implements wp.
///
/// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn dfs_wp(mol: &Molecule, weights: &[f64], path: &mut Vec<AtomIdx>, wp: &mut [f64; MAX_PATH_LEN]) {
    let depth = path.len();
    if depth >= 2 {
        let k = depth - 1; // path length in bonds
        if k <= MAX_PATH_LEN {
            let prod: f64 = path.iter().map(|&idx| weights[idx as usize]).product();
            wp[k - 1] += prod / (k as f64);
        }
    }
    if depth > MAX_PATH_LEN {
        return;
    }

    let current = *path.last().unwrap();
    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let nb = if bond.source == current {
            bond.target
        } else {
            bond.source
        };
        if !path.contains(&nb) {
            path.push(nb);
            dfs_wp(mol, weights, path, wp);
            path.pop();
        }
    }
}

/// Weighted Path descriptors.
///
/// Returns FloatVec with 10 values:
/// [WPm1..WPm5, WPe1..WPe5]
/// m = mass-weighted, e = electronegativity-weighted.
pub struct WeightedPath;

impl Descriptor for WeightedPath {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "WeightedPath"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = vec![0.0f32; MAX_PATH_LEN * 2];

        let n = mol.num_atoms();
        if n < 2 {
            return DescriptorValue::FloatVec(values);
        }

        let w_mass: Vec<f64> = (0..n)
            .map(|i| atomic_mass(mol.atoms[i].atomic_number))
            .collect();
        let w_en: Vec<f64> = (0..n)
            .map(|i| pauling_electronegativity(mol.atoms[i].atomic_number))
            .collect();

        let wp_mass = weighted_paths(mol, &w_mass);
        let wp_en = weighted_paths(mol, &w_en);

        for i in 0..MAX_PATH_LEN {
            values[i] = wp_mass[i] as f32;
            values[MAX_PATH_LEN + i] = wp_en[i] as f32;
        }

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<WeightedPath>;
};

#[cfg(test)]
mod tests {
    use super::super::typed::Descriptor;
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements wp.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_wp(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match WeightedPath.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for wp ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wp_ethane() {
        let wp = calc_wp("CC");
        assert!(wp[0] > 100.0, "WPm1 = {}", wp[0]);
        assert!(wp[1].abs() < 0.001, "WPm2 should be 0");
    }

    #[test]
    /// Runs regression test for wp propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wp_propane() {
        let wp = calc_wp("CCC");
        assert!(wp[0] > 0.0, "WPm1 > 0");
        assert!(wp[1] > 0.0, "WPm2 > 0");
    }

    #[test]
    /// Runs regression test for wp ethanol vs ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wp_ethanol_vs_ethane() {
        let ethane = calc_wp("CC");
        let ethanol = calc_wp("CCO");
        assert_ne!(ethane[0], ethanol[0]);
    }

    #[test]
    /// Runs regression test for wp symmetry.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wp_symmetry() {
        let a = calc_wp("CCC");
        let b = calc_wp("CCC");
        assert_eq!(a, b);
    }

    #[test]
    /// Runs regression test for wp single atom.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wp_single_atom() {
        let wp = calc_wp("C");
        for v in &wp {
            assert!(v.abs() < 0.001);
        }
    }

    #[test]
    /// Runs regression test for wp benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/weighted_path.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_wp_benzene() {
        let wp = calc_wp("c1ccccc1");
        assert!(wp[0] > 0.0);
        assert!(wp[1] > 0.0);
        assert!(wp[2] > 0.0);
    }
}
