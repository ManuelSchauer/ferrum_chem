#![allow(dead_code)]
//! Balaban J Index descriptor
//!
//! The Balaban J Index is a topological index that measures the degree of branching
//! in a molecular graph. It's based on the distance matrix and vertex degrees.
//!
//! Formula: J = m/(μ+1) × Σ[(d_i × d_j)^(-0.5)] over all edges
//! where:
//! - m = number of edges (bonds)
//! - μ = cyclomatic number (m - n + 1 for connected graphs)
//! - d_i = sum of distances from vertex i to all other vertices
//!
//! Reference: Balaban, A.T. (1982) Chemical Applications of Graph Theory

//! ## File guide: Balaban
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `type WeightedNeighbors`
//! - `fn compute_distance_matrix`
//! - `fn bond_weight`
//! - `fn build_weighted_neighbors`
//! - `fn compute_distance_matrix_small`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Balaban` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Balaban` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/balaban.rs` defines balaban behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Balaban` workflows.
//! - Repository unit and parity tests covering `Balaban` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/balaban.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::distance_matrix::compute_distance_matrix;
use crate::mol::molecule::Molecule;



/// Computes distance matrix cached.
///
/// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn compute_distance_matrix_cached(mol: &mut Molecule, use_bo: bool) -> Vec<f64> {
    if !use_bo {
        return compute_distance_matrix(mol, false);
    }
    if let Some(cached) = mol.cached_balaban_bo_dist() {
        return cached.clone();
    }
    let dist = compute_distance_matrix(mol, true);
    mol.store_balaban_bo_dist(dist.clone());
    dist
}

/// Implements bo distance matrix cached.
///
/// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn ensure_bo_distance_matrix_cached(mol: &mut Molecule) {
    if mol.cached_balaban_bo_dist().is_none() {
        let dist = compute_distance_matrix(mol, true);
        mol.store_balaban_bo_dist(dist);
    }
}

/// Implements bo distance matrix ref.
///
/// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn cached_bo_distance_matrix_ref(mol: &Molecule) -> Option<&[f64]> {
    mol.cached_balaban_bo_dist().map(|dist| dist.as_slice())
}

/// Calculates with distance matrix.
///
/// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn calculate_with_distance_matrix(mol: &Molecule, dist: &[f64]) -> f64 {
    if crate::algo::components::num_fragments(mol) > 1 {
        return 0.0;
    }

    let n = mol.num_atoms();
    let m = mol.num_bonds();

    if n == 0 || m == 0 {
        return 0.0;
    }

    if n == 1 {
        return 0.0;
    }

    // Connected molecules have finite distances for every pair, so this is a tight contiguous
    // row sum with no per-entry branching. This is kept explicit for readability and to make
    // future SIMD/blocked-sum experiments straightforward.
    let mut s = vec![0.0f64; n];
    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        s[i] = row.iter().sum();
    }

    // Cyclomatic number μ = m - n + 1
    let mu = m as i32 - n as i32 + 1;

    // Sum over all edges (using simple adjacency, not bond orders)
    let mut sum = 0.0;
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        let si = s[i];
        let sj = s[j];
        if si <= 0.0 || sj <= 0.0 {
            return f64::NAN;
        }
        sum += 1.0 / (si * sj).sqrt();
    }

    // J = m/(μ+1) × sum
    if mu + 1 != 0 {
        (m as f64 / (mu + 1) as f64) * sum
    } else {
        0.0
    }
}

/// Calculate Balaban J Index
///
/// # Returns
/// The Balaban J index value (typically 1.5-3.0 for most organic molecules)
///
/// # Examples
/// - Benzene: J ≈ 2.00 (highly symmetric)
/// - Adamantane: J ≈ 2.31
/// - Linear alkanes: J decreases with length
///
/// # Panics
/// Returns NaN for molecules with no bonds
pub fn calculate(mol: &Molecule) -> f64 {
    let dist = compute_distance_matrix(mol, true);
    calculate_with_distance_matrix(mol, &dist)
}

/// Calculates cached.
///
/// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn calculate_cached(mol: &mut Molecule) -> f64 {
    ensure_bo_distance_matrix_cached(mol);
    let dist = cached_bo_distance_matrix_ref(mol).expect("Balaban BO distance cache initialized");
    calculate_with_distance_matrix(mol, dist)
}

/// Balaban J Index descriptor.
///
/// Returns Float with the Balaban J index.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct BalabanIndex;

impl Descriptor for BalabanIndex {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "BalabanJ"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Float(calculate_cached(mol) as f32)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BalabanIndex>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    /// Runs regression test for balaban methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_methane() {
        let mol = parse("C");
        let j = calculate(&mol);
        println!("Methane J = {}", j);
        assert!(j.is_nan() || j == 0.0); // Single atom, no bonds
    }

    #[test]
    /// Runs regression test for balaban ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_ethane() {
        let mol = parse("CC");
        let j = calculate(&mol);
        // n=2, m=1, μ=0, d_sums=[1,1]
        // sum = 1/sqrt(1*1) = 1.0
        // J = 1/(0+1) * 1.0 = 1.0
        assert!((j - 1.0).abs() < 0.01);
    }

    #[test]
    /// Runs regression test for balaban propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_propane() {
        let mol = parse("CCC");
        let j = calculate(&mol);
        // Linear chain should have J around 1.63
        assert!(j > 1.5 && j < 1.8);
    }

    #[test]
    /// Runs regression test for balaban isobutane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_isobutane() {
        let mol = parse("CC(C)C");
        let j = calculate(&mol);
        // Branched should have higher J than linear
        // Isobutane J ≈ 2.32
        assert!(j > 2.2 && j < 2.4, "J = {}", j);
    }

    #[test]
    /// Runs regression test for balaban benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_benzene() {
        let mol = parse("c1ccccc1");
        let j = calculate(&mol);
        // Benzene: J = 3.0 (with bond-order-weighted distances)
        assert!((j - 3.0).abs() < 0.1, "J = {}", j);
    }

    #[test]
    /// Runs regression test for balaban cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_cyclohexane() {
        let mol = parse("C1CCCCC1");
        let j = calculate(&mol);
        // Cyclohexane: all single bonds, J ≈ 2.0
        assert!((j - 2.0).abs() < 0.1, "J = {}", j);
    }

    #[test]
    /// Runs regression test for balaban naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_naphthalene() {
        let mol = parse("c1ccc2ccccc2c1");
        let j = calculate(&mol);
        // Naphthalene: J ≈ 2.89 (with BO-weighted distances)
        assert!(j > 2.7 && j < 3.0, "J = {}", j);
    }

    #[test]
    /// Runs regression test for balaban disconnected.
    ///
    /// Function context: `crates/ferrum_core/src/desc/balaban.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_balaban_disconnected() {
        // Two separate methane molecules
        let mol = parse("C.C");
        let j = calculate(&mol);
        // Should handle disconnected graphs
        assert!(j.is_nan() || j >= 0.0);
    }
}
