//! Eccentric Connectivity Index
//!
//! The Eccentric Connectivity Index is a topological descriptor that combines
//! vertex eccentricity with vertex degree.
//!
//! Eccentricity ε(i) = maximum distance from vertex i to any other vertex
//! ECI = Σ [degree(i) × ε(i)] for all vertices
//!
//! Higher values indicate more extended structures.
//!
//! Reference: Sharma et al. (1997) Eccentric Connectivity Index

//! ## File guide: Eccentric
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
//! - `fn test_eccentric_methane`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Eccentric` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Eccentric` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/eccentric.rs` defines eccentric behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Eccentric` workflows.
//! - Repository unit and parity tests covering `Eccentric` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/eccentric.rs`.

use super::cdk_graph::cdk_eccentricities;
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

            // adjacency[i] stores bond indices, not atom indices!
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

/// Calculate Eccentric Connectivity Index
///
/// # Returns
/// The eccentric connectivity index as a u32 (sum of degree × eccentricity)
///
/// # Examples
/// - Path graphs: increasing with length
/// - Cyclic: lower than linear of same size
/// - Branched: varies with structure
pub fn calculate(mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> i32 {
    // CDK computes on `removeHydrogens(container)`
    if let Some(mut suppressed) = crate::algo::sanitize::cdk_remove_hydrogens(mol) {
        return calculate(&mut suppressed, ctx);
    }
    let n = mol.num_atoms();
    if n <= 1 {
        return 0; // a lone atom has eccentricity 0 and degree 0
    }

    // CDK EccentricConnectivityIndexDescriptor: sum of eccentricity x degree
    // with Java `int` arithmetic. Unreachable pairs have CDK distance
    // 999999999, so disconnected molecules overflow exactly as in CDK.
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
    let ecc = cdk_eccentricities(dist, n);
    let mut eci = 0i32;
    for (i, &e) in ecc.iter().enumerate() {
        eci = eci.wrapping_add(e.wrapping_mul(mol.adjacency[i].len() as i32));
    }
    eci
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Parses the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    /// Runs regression test for eccentric methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_methane() {
        let mut mol = parse("C");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        assert_eq!(eci, 0); // Single atom
    }

    #[test]
    /// Runs regression test for eccentric ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_ethane() {
        let mut mol = parse("CC");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // n=2, both atoms have degree=1, eccentricity=1
        // ECI = 1*1 + 1*1 = 2
        assert_eq!(eci, 2);
    }

    #[test]
    /// Runs regression test for eccentric propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_propane() {
        let mut mol = parse("CCC");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // C1: degree=1, ecc=2; C2: degree=2, ecc=1; C3: degree=1, ecc=2
        // ECI = 1*2 + 2*1 + 1*2 = 6
        assert_eq!(eci, 6);
    }

    #[test]
    /// Runs regression test for eccentric butane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_butane() {
        let mut mol = parse("CCCC");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // Linear C4: end atoms ecc=3, middle atoms ecc=2
        // ECI = 1*3 + 2*2 + 2*2 + 1*3 = 14
        assert_eq!(eci, 14);
    }

    #[test]
    /// Runs regression test for eccentric isobutane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_isobutane() {
        let mut mol = parse("CC(C)C");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // Central C: degree=3, ecc=1; others: degree=1, ecc=2
        // ECI = 1*2 + 3*1 + 1*2 + 1*2 = 9
        assert_eq!(eci, 9);
    }

    #[test]
    /// Runs regression test for eccentric benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_benzene() {
        let mut mol = parse("c1ccccc1");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // All atoms: degree=2, eccentricity=3 (opposite side of ring)
        // ECI = 6 * (2*3) = 36
        assert_eq!(eci, 36);
    }

    #[test]
    /// Runs regression test for eccentric cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_cyclohexane() {
        let mut mol = parse("C1CCCCC1");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // Same as benzene structure-wise
        assert_eq!(eci, 36);
    }

    #[test]
    /// Runs regression test for eccentric naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_naphthalene() {
        let mut mol = parse("c1ccc2ccccc2c1");
        let mut ctx = crate::desc::context::CalculationContext::new();
        let eci = calculate(&mut mol, &mut ctx);
        // 10 atoms, varies by position
        assert!(eci > 40); // More extended than benzene
    }

    #[test]
    /// Runs regression test for eccentric path vs branched.
    ///
    /// Function context: `crates/ferrum_core/src/desc/eccentric.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eccentric_path_vs_branched() {
        let mut linear = parse("CCCCC"); // Pentane
        let mut branched = parse("CC(C)CC"); // Isopentane
        let lin_eci = calculate(
            &mut linear,
            &mut crate::desc::context::CalculationContext::new(),
        );
        let bra_eci = calculate(
            &mut branched,
            &mut crate::desc::context::CalculationContext::new(),
        );
        // Linear should have higher ECI (more extended)
        assert!(lin_eci > bra_eci);
    }

    #[test]
    fn test_eccentric_disconnected_overflows_like_cdk() {
        // CC.O: three atoms of eccentricity 999999999, degrees 1, 1, 0
        let mut mol = parse("CC.O");
        let mut ctx = crate::desc::context::CalculationContext::new();
        assert_eq!(calculate(&mut mol, &mut ctx), 1_999_999_998);
        // ethane x2: 4 x 999999999 wraps around in Java int arithmetic
        let mut mol = parse("CC.CC");
        assert_eq!(calculate(&mut mol, &mut ctx), 3_999_999_996u32 as i32);
    }
}

/// CDK `EccentricConnectivityIndexDescriptor` (`ECCEN`).
pub struct EccentricConnectivityIndex;

impl crate::desc::typed::Descriptor for EccentricConnectivityIndex {
    fn name(&self) -> &str {
        "EccentricConnectivityIndex"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> crate::desc::typed::DescriptorValue {
        crate::desc::typed::DescriptorValue::Float(calculate(mol, ctx) as f64)
    }
}
