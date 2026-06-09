#![allow(dead_code)]
//! VSA (Van der Waals Surface Area) descriptors.
//!
//! Property-weighted surface area bins — among the most predictive 2D QSAR descriptors.
//!
//! - **SlogP_VSA**: 12 bins of VSA partitioned by Crippen logP contribution ranges
//! - **SMR_VSA**: 10 bins of VSA partitioned by Crippen MR contribution ranges
//! - **LabuteASA**: Labute's approximate surface area
//!
//! VSA per atom is computed using Labute's approximation:
//!   VSA_i = 4π × rᵢ² × (1 - r_correction)
//! where the correction accounts for bonded-atom overlap.
//!
//! References:
//! - Labute (2000) J. Mol. Graph. Model. 18, 464
//! - Wildman & Crippen (1999) JCICS 39(5), 868

//! ## File guide: Vsa
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn rb0_radius`
//! - `fn bond_scale_factor`
//! - `fn compute_labute_contribs_uncached`
//! - `fn compute_labute_contribs`
//! - `fn ensure_labute_contribs_cached`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Vsa` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Vsa` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/vsa.rs` defines vsa behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Vsa` workflows.
//! - Repository unit and parity tests covering `Vsa` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/vsa.rs`.

use super::logp::{
    cached_crippen_atom_contributions_ref, crippen_atom_contributions,
    ensure_crippen_atom_contributions_cached,
};
use super::typed::{Descriptor, DescriptorValue};
use crate::algo::electronic::{
    cached_gasteiger_charges_ref, compute_gasteiger_charges, ensure_gasteiger_charges_cached,
};
use crate::mol::molecule::Molecule;
use once_cell::sync::Lazy;
use std::f64::consts::PI;

static LABUTE_CACHE_ENABLED: Lazy<bool> =
    Lazy::new(|| std::env::var_os("FERRUM_DISABLE_LABUTE_CACHE").is_none());

// ---------------------------------------------------------------------------
// Covalent bond radii (Rb0, Å) — same as RDKit GetRb0()
// ---------------------------------------------------------------------------

/// Implements radius.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub fn rb0_radius(an: u8) -> f64 {
    match an {
        1 => 0.330,
        2 => 0.700,
        3 => 1.230,
        4 => 0.900,
        5 => 0.820,
        6 => 0.770,
        7 => 0.700,
        8 => 0.660,
        9 => 0.611,
        10 => 0.700,
        11 => 1.540,
        12 => 1.360,
        13 => 1.180,
        14 => 0.937,
        15 => 0.890,
        16 => 1.040,
        17 => 0.997,
        18 => 1.740,
        19 => 2.030,
        20 => 1.740,
        21 => 1.440,
        22 => 1.320,
        23 => 1.220,
        24 => 1.180,
        25 => 1.170,
        26 => 1.170,
        27 => 1.160,
        28 => 1.150,
        29 => 1.170,
        30 => 1.250,
        31 => 1.260,
        32 => 1.188,
        33 => 1.200,
        34 => 1.170,
        35 => 1.167,
        36 => 1.910,
        37 => 2.160,
        38 => 1.910,
        39 => 1.620,
        40 => 1.450,
        41 => 1.340,
        42 => 1.300,
        43 => 1.270,
        44 => 1.250,
        45 => 1.250,
        46 => 1.280,
        47 => 1.340,
        48 => 1.480,
        49 => 1.440,
        50 => 1.385,
        51 => 1.400,
        52 => 1.378,
        53 => 1.387,
        54 => 1.980,
        55 => 2.350,
        56 => 1.980,
        // Lanthanides (57-71) and 3rd-row transition metals (72-80)
        57 => 1.690,
        58 => 1.650,
        59 => 1.650,
        60 => 1.640,
        61 => 1.630,
        62 => 1.620,
        63 => 1.850,
        64 => 1.610,
        65 => 1.590,
        66 => 1.590,
        67 => 1.580,
        68 => 1.570,
        69 => 1.560,
        70 => 1.700,
        71 => 1.560,
        72 => 1.440,
        73 => 1.340,
        74 => 1.300,
        75 => 1.280,
        76 => 1.260,
        77 => 1.270,
        78 => 1.300,
        79 => 1.340,
        80 => 1.490,
        81 => 1.480,
        82 => 1.470,
        83 => 1.460,
        _ => 0.770,
    }
}

// Bond-type scaling factors: aromatic=0.1, single=0.0, double=0.2, triple=0.3
/// Implements scale factor.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn bond_scale_factor(bond: &crate::mol::bond::Bond) -> f64 {
    if bond.is_aromatic() {
        return 0.1;
    }
    match bond.bond_type {
        crate::types::BondType::Single => 0.0,
        crate::types::BondType::Double => 0.2,
        crate::types::BondType::Triple => 0.3,
        crate::types::BondType::Aromatic => 0.1,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Labute approximate surface area per atom
// ---------------------------------------------------------------------------

/// Compute Labute ASA contributions: per-heavy-atom values + total H contribution.
///
/// Uses Labute's bond-based formula (Labute 2000, JMGM 18:464):
///   For each bond (i,j): overlap accumulation using Rb0 covalent radii
///   Vi[i] = 4π·Ri² − π·Ri·Σoverlap
///
/// Returns (heavy_atom_vsa, h_contribution).
fn compute_labute_contribs_uncached<'a>(
    mol: &Molecule,
    ctx: &'a mut crate::desc::context::CalculationContext,
) -> (&'a [f64], f64) {
    let n = mol.atoms.len();
    ctx.vec_a.clear();
    ctx.vec_a.resize(n, 0.0);
    ctx.vec_b.clear();
    ctx.vec_b.resize(n, 0.0);

    let overlap = &mut ctx.vec_a;
    let mut h_overlap = 0.0f64;

    let r_h = rb0_radius(1);

    // Explicit bonds
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        let ai = mol.atoms[i].atomic_number;
        let aj = mol.atoms[j].atomic_number;
        if ai <= 1 || aj <= 1 {
            continue;
        }

        let ri = rb0_radius(ai);
        let rj = rb0_radius(aj);
        let scale = bond_scale_factor(bond);
        let bij = ri + rj - scale;
        let dij = (ri + rj).min((ri - rj).abs().max(bij));

        overlap[i] += rj * rj - (ri - dij) * (ri - dij) / dij;
        overlap[j] += ri * ri - (rj - dij) * (rj - dij) / dij;
    }

    // Implicit H bonds: one correction per heavy atom (Labute approximation)
    for i in 0..n {
        let ai = mol.atoms[i].atomic_number;
        if ai <= 1 {
            continue;
        }
        let ri = rb0_radius(ai);
        let bij = ri + r_h;
        let dij = (ri + r_h).min((ri - r_h).abs().max(bij));

        overlap[i] += r_h * r_h - (ri - dij) * (ri - dij) / dij;
        h_overlap += ri * ri - (r_h - dij) * (r_h - dij) / dij;
    }

    // Convert overlap sums to surface area contributions
    let vsa = &mut ctx.vec_b;
    for i in 0..n {
        let ai = mol.atoms[i].atomic_number;
        if ai <= 1 {
            continue;
        }
        let ri = rb0_radius(ai);
        vsa[i] = 4.0 * PI * ri * ri - PI * ri * overlap[i];
    }
    let h_contrib = 4.0 * PI * r_h * r_h - PI * r_h * h_overlap;

    (vsa, h_contrib)
}

/// Computes labute contribs.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_labute_contribs(
    mol: &Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
) -> (Vec<f64>, f64) {
    let (vsa, h) = compute_labute_contribs_uncached(mol, ctx);
    (vsa.to_vec(), h)
}

/// Implements labute contribs cached.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn ensure_labute_contribs_cached(
    mol: &mut Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
) {
    if !*LABUTE_CACHE_ENABLED || mol.cached_labute_contribs().is_some() {
        return;
    }
    let (atom_vsa, h_contrib) = compute_labute_contribs_uncached(mol, ctx);
    mol.store_labute_contribs(atom_vsa.to_vec(), h_contrib);
}

// ---------------------------------------------------------------------------
// SlogP_VSA — 12 bins
// ---------------------------------------------------------------------------

/// SlogP_VSA: VSA binned by Crippen logP atom contributions (12 values).
///
/// Returns FloatVec with 12 logP-weighted VSA bins.
pub struct SlogpVSA;

const SLOGP_BINS: [f64; 12] = [
    -0.40,
    -0.20,
    0.00,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    f64::INFINITY,
];

/// Implements binned vsa.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn fill_binned_vsa(
    mol: &Molecule,
    atom_property: &[f64],
    atom_vsa: &[f64],
    bins: &[f64],
    out: &mut [f32],
) {
    out.fill(0.0);
    for i in 0..mol.atoms.len() {
        if mol.atoms[i].atomic_number <= 1 {
            continue;
        }
        let bin = find_bin(atom_property[i], bins);
        out[bin] += atom_vsa[i] as f32;
    }
}

/// Implements slogp vsa into.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn fill_slogp_vsa_into(
    mol: &mut Molecule,
    out: &mut [f32],
    ctx: &mut crate::desc::context::CalculationContext,
) {
    assert_eq!(
        out.len(),
        SLOGP_BINS.len(),
        "SlogP_VSA output length mismatch"
    );

    ensure_labute_contribs_cached(mol, ctx);
    ensure_crippen_atom_contributions_cached(mol);
    if let (Some((atom_vsa, _)), Some((logp_contrib, _))) = (
        mol.cached_labute_contribs(),
        cached_crippen_atom_contributions_ref(mol),
    ) {
        fill_binned_vsa(mol, logp_contrib, atom_vsa, &SLOGP_BINS, out);
        return;
    }

    let (logp_contrib, _) = crippen_atom_contributions(mol);
    let (atom_vsa, _) = compute_labute_contribs_uncached(mol, ctx);
    fill_binned_vsa(mol, &logp_contrib, atom_vsa, &SLOGP_BINS, out);
}

impl Descriptor for SlogpVSA {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "SlogP_VSA"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = vec![0.0f32; 12];
        fill_slogp_vsa_into(mol, &mut values, ctx);
        DescriptorValue::FloatVec(values)
    }
}

// ---------------------------------------------------------------------------
// SMR_VSA — 10 bins
// ---------------------------------------------------------------------------

/// SMR_VSA: VSA binned by Crippen MR atom contributions (10 values).
///
/// Returns FloatVec with 10 MR-weighted VSA bins.
pub struct SmrVSA;

const SMR_BINS: [f64; 10] = [
    1.29,
    1.82,
    2.24,
    2.45,
    2.75,
    3.05,
    3.63,
    3.80,
    4.00,
    f64::INFINITY,
];

/// Implements smr vsa into.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn fill_smr_vsa_into(
    mol: &mut Molecule,
    out: &mut [f32],
    ctx: &mut crate::desc::context::CalculationContext,
) {
    assert_eq!(out.len(), SMR_BINS.len(), "SMR_VSA output length mismatch");

    ensure_labute_contribs_cached(mol, ctx);
    ensure_crippen_atom_contributions_cached(mol);
    if let (Some((atom_vsa, _)), Some((_, mr_contrib))) = (
        mol.cached_labute_contribs(),
        cached_crippen_atom_contributions_ref(mol),
    ) {
        fill_binned_vsa(mol, mr_contrib, atom_vsa, &SMR_BINS, out);
        return;
    }

    let (_, mr_contrib) = crippen_atom_contributions(mol);
    let (atom_vsa, _) = compute_labute_contribs_uncached(mol, ctx);
    fill_binned_vsa(mol, &mr_contrib, atom_vsa, &SMR_BINS, out);
}

impl Descriptor for SmrVSA {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "SMR_VSA"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = vec![0.0f32; 10];
        fill_smr_vsa_into(mol, &mut values, ctx);
        DescriptorValue::FloatVec(values)
    }
}

// ---------------------------------------------------------------------------
// LabuteASA
// ---------------------------------------------------------------------------

/// Labute's Approximate Surface Area: sum of all atom VSA contributions.
///
/// Returns Float with total approximate surface area.
pub struct LabuteASA;

impl Descriptor for LabuteASA {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "LabuteASA"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        ensure_labute_contribs_cached(mol, ctx);
        let (total, h_contrib) = if let Some((vsa, h_contrib)) = mol.cached_labute_contribs() {
            (vsa.iter().sum::<f64>(), h_contrib)
        } else {
            let (vsa, h_contrib) = compute_labute_contribs_uncached(mol, ctx);
            (vsa.iter().sum::<f64>(), h_contrib)
        };
        DescriptorValue::Float((total + h_contrib) as f32)
    }
}

// ---------------------------------------------------------------------------
// PEOE_VSA — 14 bins
// ---------------------------------------------------------------------------

/// PEOE_VSA: VSA binned by Gasteiger partial charge ranges (14 values).
///
/// Returns FloatVec with 14 charge-weighted VSA bins.
pub struct PeoeVsa;

const PEOE_BINS: [f64; 14] = [
    -0.30,
    -0.25,
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    f64::INFINITY,
];

/// Implements peoe vsa into.
///
/// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn fill_peoe_vsa_into(
    mol: &mut Molecule,
    out: &mut [f32],
    ctx: &mut crate::desc::context::CalculationContext,
) {
    assert_eq!(
        out.len(),
        PEOE_BINS.len(),
        "PEOE_VSA output length mismatch"
    );

    ensure_labute_contribs_cached(mol, ctx);
    ensure_gasteiger_charges_cached(mol);
    if let (Some((atom_vsa, _)), Some(charges)) = (
        mol.cached_labute_contribs(),
        cached_gasteiger_charges_ref(mol),
    ) {
        fill_binned_vsa(mol, charges, atom_vsa, &PEOE_BINS, out);
        return;
    }

    let charges = compute_gasteiger_charges(mol);
    let (atom_vsa, _) = compute_labute_contribs_uncached(mol, ctx);
    fill_binned_vsa(mol, &charges, atom_vsa, &PEOE_BINS, out);
}

impl Descriptor for PeoeVsa {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "PEOE_VSA"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = vec![0.0f32; 14];
        fill_peoe_vsa_into(mol, &mut values, ctx);
        DescriptorValue::FloatVec(values)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find bin index: returns i where bins[i-1] <= val < bins[i], with bins[-1] = -∞.
fn find_bin(val: f64, bins: &[f64]) -> usize {
    for (i, &upper) in bins.iter().enumerate() {
        if val < upper {
            return i;
        }
    }
    bins.len() - 1
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------
const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<SlogpVSA>;
    let _ = assert_send_sync::<SmrVSA>;
    let _ = assert_send_sync::<LabuteASA>;
    let _ = assert_send_sync::<PeoeVsa>;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::super::typed::Descriptor;
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements float vec.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_float_vec(d: &dyn Descriptor, smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match d.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    /// Implements float.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_float(d: &dyn Descriptor, smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match d.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Float(v) => v,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    // -- compute_labute_contribs --

    #[test]
    /// Runs regression test for vsa positive.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_vsa_positive() {
        let mut mol = parse_smiles("CCO").unwrap();
        let mut ctx = crate::desc::context::CalculationContext::new();
        let (vsa, _) = compute_labute_contribs(&mut mol, &mut ctx);
        for (i, &v) in vsa.iter().enumerate() {
            if mol.atoms[i].atomic_number > 1 {
                assert!(v > 0.0, "atom {} VSA = {}", i, v);
            }
        }
    }

    #[test]
    /// Runs regression test for vsa benzene symmetry.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_vsa_benzene_symmetry() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        let mut ctx = crate::desc::context::CalculationContext::new();
        let (vsa, _) = compute_labute_contribs(&mut mol, &mut ctx);
        let c_vsas: Vec<f64> = vsa
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| mol.atoms[*i].atomic_number == 6)
            .map(|(_, v)| v)
            .collect();
        for &v in &c_vsas {
            assert!((v - c_vsas[0]).abs() < 0.01, "asymmetric VSA: {:?}", c_vsas);
        }
    }

    // -- SlogP_VSA --

    #[test]
    /// Runs regression test for slogp vsa nonzero.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_slogp_vsa_nonzero() {
        let v = calc_float_vec(&SlogpVSA, "c1ccccc1");
        let total: f32 = v.iter().sum();
        assert!(total > 0.0, "total SlogP_VSA = {}", total);
    }

    #[test]
    /// Runs regression test for slogp vsa bins sum approx labute.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_slogp_vsa_bins_sum_approx_labute() {
        let slogp = calc_float_vec(&SlogpVSA, "CCO");
        let labute = calc_float(&LabuteASA, "CCO");
        let bin_sum: f32 = slogp.iter().sum();
        // Bin sum = heavy atom contributions only; LabuteASA includes H contribution
        assert!(
            bin_sum > 0.0 && bin_sum < labute + 1.0,
            "bin sum {} vs LabuteASA {}",
            bin_sum,
            labute
        );
    }

    #[test]
    /// Runs regression test for slogp vsa larger molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_slogp_vsa_larger_molecule() {
        let v = calc_float_vec(&SlogpVSA, "CC(=O)Oc1ccccc1C(=O)O");
        let total: f32 = v.iter().sum();
        assert!(total > 50.0, "aspirin total VSA = {}", total);
    }

    // -- SMR_VSA --

    #[test]
    /// Runs regression test for smr vsa nonzero.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_smr_vsa_nonzero() {
        let v = calc_float_vec(&SmrVSA, "c1ccccc1");
        let total: f32 = v.iter().sum();
        assert!(total > 0.0, "total SMR_VSA = {}", total);
    }

    #[test]
    /// Runs regression test for smr vsa bins sum approx labute.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_smr_vsa_bins_sum_approx_labute() {
        let smr = calc_float_vec(&SmrVSA, "CCO");
        let labute = calc_float(&LabuteASA, "CCO");
        let bin_sum: f32 = smr.iter().sum();
        // Bin sum = heavy atom contributions only; LabuteASA includes H contribution
        assert!(
            bin_sum > 0.0 && bin_sum < labute + 1.0,
            "bin sum {} vs LabuteASA {}",
            bin_sum,
            labute
        );
    }

    // -- LabuteASA --

    #[test]
    /// Runs regression test for labute positive.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_labute_positive() {
        let v = calc_float(&LabuteASA, "CCCCCC");
        assert!(v > 0.0, "hexane ASA = {}", v);
    }

    #[test]
    /// Runs regression test for labute larger more.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_labute_larger_more() {
        let small = calc_float(&LabuteASA, "CC");
        let large = calc_float(&LabuteASA, "CCCCCCCCCC");
        assert!(large > small, "decane {} > ethane {}", large, small);
    }

    #[test]
    /// Runs regression test for labute benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_labute_benzene() {
        let v = calc_float(&LabuteASA, "c1ccccc1");
        // RDKit: 37.43
        assert!(v > 30.0 && v < 45.0, "benzene ASA = {}", v);
    }

    #[test]
    /// Runs regression test for labute methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_labute_methane() {
        let v = calc_float(&LabuteASA, "C");
        // RDKit: 8.74
        assert!((v - 8.74).abs() < 1.0, "methane ASA = {}", v);
    }

    // -- PEOE_VSA --

    #[test]
    /// Runs regression test for peoe vsa benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_peoe_vsa_benzene() {
        let v = calc_float_vec(&PeoeVsa, "c1ccccc1");
        let nonzero: Vec<_> = v.iter().filter(|&&x| x > 0.0).collect();
        assert!(!nonzero.is_empty(), "should have at least one nonzero bin");
        let total: f32 = v.iter().sum();
        // Total should be positive and reasonable
        assert!(total > 0.0, "PEOE_VSA sum {} should be positive", total);
    }

    #[test]
    /// Runs regression test for peoe vsa length.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_peoe_vsa_length() {
        let v = calc_float_vec(&PeoeVsa, "CCO");
        assert_eq!(v.len(), 14, "PEOE_VSA should have 14 bins");
        let v2 = calc_float_vec(&PeoeVsa, "c1ccccc1");
        assert_eq!(v2.len(), 14);
    }

    #[test]
    /// Runs regression test for peoe vsa formaldehyde.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_peoe_vsa_formaldehyde() {
        // C=O: O is more electronegative → negative charge, C → positive
        let v = calc_float_vec(&PeoeVsa, "C=O");
        // Bins 0-6 are negative charges, bins 7-13 are non-negative
        let neg_sum: f32 = v[..7].iter().sum();
        let pos_sum: f32 = v[7..].iter().sum();
        assert!(
            neg_sum > 0.0,
            "O should contribute to negative bins: {:?}",
            v
        );
        assert!(
            pos_sum > 0.0,
            "C should contribute to positive bins: {:?}",
            v
        );
    }

    #[test]
    /// Runs regression test for peoe vsa sum.
    ///
    /// Function context: `crates/ferrum_core/src/desc/vsa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_peoe_vsa_sum() {
        for smi in &["CCO", "CC(=O)O", "c1ccccc1", "CCCCCC"] {
            let peoe = calc_float_vec(&PeoeVsa, smi);
            let bin_sum: f32 = peoe.iter().sum();
            // Bin sums only heavy atoms; should be positive and reasonable
            assert!(
                bin_sum > 0.0,
                "{}: PEOE_VSA sum {} should be positive",
                smi,
                bin_sum
            );
        }
    }
}
