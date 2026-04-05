//! Gravitational Index descriptor
//!
//! The Gravitational Index is a molecular descriptor that combines atomic masses
//! and topological distances to characterize molecular size and mass distribution.
//!
//! Formula: G = Σᵢ Σⱼ>ᵢ (mᵢ × mⱼ) / dᵢⱼ²
//!
//! where:
//! - mᵢ, mⱼ = atomic masses
//! - dᵢⱼ = topological distance (number of bonds)
//!
//! Reference: Katritzky et al. (1996)

//! ## File guide: Gravitational
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn _compute_distance_matrix_unused`
//! - `fn calculate`
//! - `mod tests`
//! - `fn parse`
//! - `fn test_gravitational_methane`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Gravitational` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Gravitational` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/gravitational.rs` defines gravitational behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Gravitational` workflows.
//! - Repository unit and parity tests covering `Gravitational` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/gravitational.rs`.

use crate::mol::elements::ATOMIC_MASSES;
use crate::mol::molecule::Molecule;
use std::collections::VecDeque;

/// Compute the distance matrix using BFS from each atom
#[allow(dead_code)]
fn _compute_distance_matrix_unused(mol: &Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    let mut dist = vec![u32::MAX; n * n];

    for start in 0..n {
        let mut queue = VecDeque::new();
        queue.push_back(start);
        dist[start * n + start] = 0;

        while let Some(current) = queue.pop_front() {
            let current_dist = dist[start * n + current];

            for &bond_idx in &mol.adjacency[current] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == current as u16 {
                    bond.target as usize
                } else {
                    bond.source as usize
                };

                if dist[start * n + neighbor] == u32::MAX {
                    dist[start * n + neighbor] = current_dist + 1;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    dist
}

/// Calculate Gravitational Index
///
/// # Returns
/// The gravitational index value (higher for larger/heavier molecules)
///
/// # Examples
/// - Larger molecules have higher G values
/// - Heavier atoms contribute more to G
/// - More compact structures have higher G (shorter distances)
pub fn calculate(mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> f64 {
    let n = mol.num_atoms();

    if n <= 1 {
        return 0.0;
    }

    // Compute distance matrix
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);

    // Calculate gravitational index
    let mut g = 0.0;
    for i in 0..n {
        let mass_i = ATOMIC_MASSES[mol.atoms[i].atomic_number as usize] as f64;

        for j in (i + 1)..n {
            let mass_j = ATOMIC_MASSES[mol.atoms[j].atomic_number as usize] as f64;
            let d = dist[i * n + j];

            if d != 255 && d > 0 {
                let inv_d2 = 1.0 / (d * d) as f64;
                g = mass_i.mul_add(mass_j * inv_d2, g);
            }
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    /// Runs regression test for gravitational methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_gravitational_methane() {
        let mut mol = parse("C");
        let g = calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        assert_eq!(g, 0.0); // Single atom
    }

    #[test]
    /// Runs regression test for gravitational ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_gravitational_ethane() {
        let mut mol = parse("CC");
        let g = calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        // Two carbons at distance 1: (12*12)/1² = 144
        assert!((g - 144.0).abs() < 1.0);
    }

    #[test]
    /// Runs regression test for gravitational propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_gravitational_propane() {
        let mut mol = parse("CCC");
        let g = calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        // C1-C2: d=1, C2-C3: d=1, C1-C3: d=2
        // G = 12*12/1 + 12*12/1 + 12*12/4 = 144 + 144 + 36 = 324
        assert!((g - 324.0).abs() < 1.0);
    }

    #[test]
    /// Runs regression test for gravitational size ordering.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_gravitational_size_ordering() {
        let mut ethane = parse("CC");
        let mut propane = parse("CCC");
        let mut butane = parse("CCCC");

        let g_eth = calculate(
            &mut ethane,
            &mut crate::desc::context::CalculationContext::new(),
        );
        let g_prop = calculate(
            &mut propane,
            &mut crate::desc::context::CalculationContext::new(),
        );
        let g_but = calculate(
            &mut butane,
            &mut crate::desc::context::CalculationContext::new(),
        );

        // Larger molecules should have higher G
        assert!(g_prop > g_eth);
        assert!(g_but > g_prop);
    }

    #[test]
    /// Runs regression test for gravitational heteroatoms.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_gravitational_heteroatoms() {
        let mut ethane = parse("CC");
        let mut ethanol = parse("CCO");

        let g_eth = calculate(
            &mut ethane,
            &mut crate::desc::context::CalculationContext::new(),
        );
        let g_etoh = calculate(
            &mut ethanol,
            &mut crate::desc::context::CalculationContext::new(),
        );

        // Oxygen is heavier than carbon, but slightly
        // The difference depends on the structure
        assert!(g_etoh > g_eth - 50.0); // Should be in similar range
    }

    #[test]
    /// Runs regression test for gravitational benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_gravitational_benzene() {
        let mut mol = parse("c1ccccc1");
        let g = calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        // Cyclic structure should have reasonable G value
        assert!(g > 500.0); // 6 carbons with various distances
    }

    #[test]
    /// Runs regression test for gravitational branched vs linear.
    ///
    /// Function context: `crates/ferrum_core/src/desc/gravitational.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_gravitational_branched_vs_linear() {
        let mut butane = parse("CCCC");
        let mut isobutane = parse("CC(C)C");

        let g_but = calculate(
            &mut butane,
            &mut crate::desc::context::CalculationContext::new(),
        );
        let g_iso = calculate(
            &mut isobutane,
            &mut crate::desc::context::CalculationContext::new(),
        );

        // Linear should have higher G (longer max distance)
        // Actually, branched has shorter average distances, so higher G
        assert!(g_iso > g_but * 0.8); // Close but branched might be higher
    }
}
