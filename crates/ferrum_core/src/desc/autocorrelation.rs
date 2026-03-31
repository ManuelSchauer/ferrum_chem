//! Moreau-Broto Autocorrelation (ATS) descriptors.
//!
//! ATS(d, w) = Σ w_i × w_j  for all atom pairs (i, j) at topological distance d.
//!
//! Property weights:
//! - Mass (atomic mass)
//! - Polarizability (estimated from atomic number)
//! - Electronegativity (Pauling scale)
//! - VdW volume (van der Waals radius based)
//!
//! For each weight: ATS at distances 1..8 → 8 values per weight → 32 total.
//!
//! Reference: Moreau, G. & Broto, P. (1980) Nouv. J. Chim. 4, 359-360.


//! ## File guide: Autocorrelation
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
//! - `fn compute_ats`
//! - `struct Autocorrelation`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Autocorrelation` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Autocorrelation` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/autocorrelation.rs` defines autocorrelation behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Autocorrelation` workflows.
//! - Repository unit and parity tests covering `Autocorrelation` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/autocorrelation.rs`.

use super::atomic_props::{
    atomic_mass, burden_polarizability, burden_vdw_volume, pauling_electronegativity,
};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Atomic properties
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Distance matrix
// ---------------------------------------------------------------------------

#[allow(dead_code)]
/// Computes distance matrix unused.
///
/// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn _compute_distance_matrix_unused(mol: &Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    let mut distances = vec![u32::MAX; n * n];

    for start in 0..n {
        distances[start * n + start] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start as AtomIdx);

        while let Some(current) = queue.pop_front() {
            let current_dist = distances[start * n + current as usize];
            for &bond_idx in &mol.adjacency[current as usize] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                };
                let nd = &mut distances[start * n + neighbor as usize];
                if *nd == u32::MAX {
                    *nd = current_dist + 1;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    distances
}

// ---------------------------------------------------------------------------
// ATS computation
// ---------------------------------------------------------------------------

const MAX_LAG: usize = 8;

/// Compute ATS values for a given weight function.
/// Returns [ATS1, ATS2, ..., ATS8] for distances 1..8.
fn compute_ats(mol: &Molecule, dist: &[u8], weights: &[f64]) -> [f64; MAX_LAG] {
    let n = mol.num_atoms();
    let mut ats = [0.0f64; MAX_LAG];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist[i * n + j];
            if d == 0 || d == 255 {
                continue;
            }
            let lag = d as usize;
            if lag >= 1 && lag <= MAX_LAG {
                ats[lag - 1] += weights[i] * weights[j];
            }
        }
    }

    // Multiply by 2 since we only counted i<j
    for v in ats.iter_mut() {
        *v *= 2.0;
    }

    ats
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

// (Legacy descriptor removed — use Autocorrelation below)

// ===========================================================================
// Typed API: Autocorrelation
// ===========================================================================

/// Moreau-Broto Autocorrelation descriptors.
///
/// Returns FloatVec with 32 values:
/// [ATSm1..ATSm8, ATSp1..ATSp8, ATSe1..ATSe8, ATSv1..ATSv8]
pub struct Autocorrelation;

impl Descriptor for Autocorrelation {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "Autocorrelation"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut values = vec![0.0f32; MAX_LAG * 4];

        let n = mol.num_atoms();
        if n < 2 {
            return DescriptorValue::FloatVec(values);
        }

        let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);

        // Compute weight vectors
        let w_mass: Vec<f64> = (0..n)
            .map(|i| atomic_mass(mol.atoms[i].atomic_number))
            .collect();
        let w_pol: Vec<f64> = (0..n)
            .map(|i| burden_polarizability(mol.atoms[i].atomic_number))
            .collect();
        let w_en: Vec<f64> = (0..n)
            .map(|i| pauling_electronegativity(mol.atoms[i].atomic_number))
            .collect();
        let w_vdw: Vec<f64> = (0..n)
            .map(|i| burden_vdw_volume(mol.atoms[i].atomic_number))
            .collect();

        let ats_mass = compute_ats(mol, dist, &w_mass);
        let ats_pol = compute_ats(mol, dist, &w_pol);
        let ats_en = compute_ats(mol, dist, &w_en);
        let ats_vdw = compute_ats(mol, dist, &w_vdw);

        for i in 0..MAX_LAG {
            values[i] = ats_mass[i] as f32;
            values[MAX_LAG + i] = ats_pol[i] as f32;
            values[MAX_LAG * 2 + i] = ats_en[i] as f32;
            values[MAX_LAG * 3 + i] = ats_vdw[i] as f32;
        }

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Autocorrelation>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements ats.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_ats(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match Autocorrelation.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for ats size.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ats_size() {
        let a = calc_ats("CC");
        assert_eq!(a.len(), 32);
    }

    #[test]
    /// Runs regression test for ats ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ats_ethane() {
        let a = calc_ats("CC");
        assert!(a[0] > 200.0, "ATSm1 = {}", a[0]);
        assert!(a[1].abs() < 0.001, "ATSm2 = {}", a[1]);
    }

    #[test]
    /// Runs regression test for ats propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ats_propane() {
        let a = calc_ats("CCC");
        assert!(a[0] > 0.0, "ATSm1 > 0");
        assert!(a[1] > 0.0, "ATSm2 > 0");
        assert!(a[2].abs() < 0.001, "ATSm3 should be 0 for 3-atom chain");
    }

    #[test]
    /// Runs regression test for ats benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ats_benzene() {
        let a = calc_ats("c1ccccc1");
        assert!(a[0] > 0.0, "ATSm1");
        assert!(a[1] > 0.0, "ATSm2");
        assert!(a[2] > 0.0, "ATSm3");
        assert!(a[3].abs() < 0.001, "ATSm4 should be ~0 for benzene");
    }

    #[test]
    /// Runs regression test for ats ethanol vs ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ats_ethanol_vs_ethane() {
        let ethane = calc_ats("CC");
        let ethanol = calc_ats("CCO");
        assert!(ethanol[0] > ethane[0], "ethanol ATSm1 > ethane ATSm1");
    }

    #[test]
    /// Runs regression test for ats single atom.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ats_single_atom() {
        let a = calc_ats("C");
        for v in &a {
            assert!(v.abs() < 0.001);
        }
    }

    #[test]
    /// Runs regression test for ats long chain.
    ///
    /// Function context: `crates/ferrum_core/src/desc/autocorrelation.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ats_long_chain() {
        let a = calc_ats("CCCCCCCCCC");
        for i in 0..MAX_LAG {
            assert!(a[i] > 0.0, "ATSm{} should be > 0 for decane", i + 1);
        }
    }
}
