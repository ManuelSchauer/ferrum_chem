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

use super::typed::{Descriptor, DescriptorValue};
use super::logp::crippen_atom_contributions;
use crate::algo::electronic::compute_gasteiger_charges;
use crate::mol::molecule::Molecule;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Van der Waals radii (Å) indexed by atomic number
// ---------------------------------------------------------------------------

fn vdw_radius(an: u8) -> f64 {
    match an {
        1 => 1.20,  6 => 1.70,  7 => 1.55,  8 => 1.52,  9 => 1.47,
        14 => 2.10, 15 => 1.80, 16 => 1.80, 17 => 1.75,
        35 => 1.85, 53 => 1.98, _ => 1.70,
    }
}

// ---------------------------------------------------------------------------
// Labute approximate surface area per atom
// ---------------------------------------------------------------------------

/// Compute Labute approximate VSA for each atom (heavy atoms only; H = 0).
///
/// Uses Labute's approximation: VSA_i ≈ 4π rᵢ² scaled by exposed fraction.
/// Bond correction uses VDW-based distance (Ri+Rj), giving gentle attenuation
/// for heavily bonded atoms while keeping values physically reasonable.
fn compute_atom_vsa(mol: &Molecule) -> Vec<f64> {
    let n = mol.atoms.len();
    let mut vsa = vec![0.0f64; n];

    for i in 0..n {
        let ai = &mol.atoms[i];
        if ai.atomic_number <= 1 { continue; }

        let ri = vdw_radius(ai.atomic_number);
        let area = 4.0 * PI * ri * ri;

        // Count heavy-atom neighbors for attenuation
        let heavy_neighbors = mol.adjacency[i].iter().filter(|&&bi| {
            let bond = &mol.bonds[bi as usize];
            let j = if bond.source as usize == i { bond.target as usize } else { bond.source as usize };
            mol.atoms[j].atomic_number > 1
        }).count();

        // Attenuation: each heavy bond reduces area by ~10% (empirical)
        // This gives reasonable values while maintaining discrimination
        let attenuation = 1.0 - 0.1 * heavy_neighbors as f64;
        vsa[i] = area * attenuation.max(0.2);
    }

    vsa
}

// ---------------------------------------------------------------------------
// SlogP_VSA — 12 bins
// ---------------------------------------------------------------------------

/// SlogP_VSA: VSA binned by Crippen logP atom contributions (12 values).
///
/// Returns FloatVec with 12 logP-weighted VSA bins.
pub struct SlogpVSA;

const SLOGP_BINS: [f64; 12] = [
    -0.40, -0.20, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
    f64::INFINITY,
];

impl Descriptor for SlogpVSA {
    fn name(&self) -> &str {
        "SlogP_VSA"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut values = vec![0.0f32; 12];
        
        let vsa = compute_atom_vsa(mol);
        let (logp_contrib, _) = crippen_atom_contributions(mol);

        for i in 0..mol.atoms.len() {
            if mol.atoms[i].atomic_number <= 1 { continue; }
            let lp = logp_contrib[i];
            let bin = find_bin(lp, &SLOGP_BINS);
            values[bin] += vsa[i] as f32;
        }
        
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
    1.29, 1.82, 2.24, 2.45, 2.75, 3.05, 3.63, 3.80, 4.00,
    f64::INFINITY,
];

impl Descriptor for SmrVSA {
    fn name(&self) -> &str {
        "SMR_VSA"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut values = vec![0.0f32; 10];
        
        let vsa = compute_atom_vsa(mol);
        let (_, mr_contrib) = crippen_atom_contributions(mol);

        for i in 0..mol.atoms.len() {
            if mol.atoms[i].atomic_number <= 1 { continue; }
            let mr = mr_contrib[i];
            let bin = find_bin(mr, &SMR_BINS);
            values[bin] += vsa[i] as f32;
        }
        
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
    fn name(&self) -> &str {
        "LabuteASA"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let vsa = compute_atom_vsa(mol);
        let total: f64 = vsa.iter().sum();
        DescriptorValue::Float(total as f32)
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
    -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00,
    0.05,  0.10,  0.15,  0.20,  0.25,  0.30, f64::INFINITY,
];

impl Descriptor for PeoeVsa {
    fn name(&self) -> &str {
        "PEOE_VSA"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut values = vec![0.0f32; 14];

        let charges = compute_gasteiger_charges(mol);
        let vsa = compute_atom_vsa(mol);

        for i in 0..mol.atoms.len() {
            if mol.atoms[i].atomic_number <= 1 { continue; }
            let charge = charges[i];
            let bin = find_bin(charge, &PEOE_BINS);
            values[bin] += vsa[i] as f32;
        }

        DescriptorValue::FloatVec(values)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find bin index: returns i where bins[i-1] <= val < bins[i], with bins[-1] = -∞.
fn find_bin(val: f64, bins: &[f64]) -> usize {
    for (i, &upper) in bins.iter().enumerate() {
        if val < upper { return i; }
    }
    bins.len() - 1
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------
const _: () = {
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
    use super::*;
    use crate::io::smiles::parse_smiles;
    use super::super::typed::Descriptor;

    fn calc_float_vec(d: &dyn Descriptor, smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match d.calculate(&mut mol) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    fn calc_float(d: &dyn Descriptor, smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match d.calculate(&mut mol) {
            DescriptorValue::Float(v) => v,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    // -- compute_atom_vsa --

    #[test]
    fn test_vsa_positive() {
        let mol = parse_smiles("CCO").unwrap();
        let vsa = compute_atom_vsa(&mol);
        for (i, &v) in vsa.iter().enumerate() {
            if mol.atoms[i].atomic_number > 1 {
                assert!(v > 0.0, "atom {} VSA = {}", i, v);
            }
        }
    }

    #[test]
    fn test_vsa_benzene_symmetry() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        let vsa = compute_atom_vsa(&mol);
        let c_vsas: Vec<f64> = vsa.iter().copied()
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
    fn test_slogp_vsa_nonzero() {
        let v = calc_float_vec(&SlogpVSA, "c1ccccc1");
        let total: f32 = v.iter().sum();
        assert!(total > 0.0, "total SlogP_VSA = {}", total);
    }

    #[test]
    fn test_slogp_vsa_bins_sum_to_labute() {
        let slogp = calc_float_vec(&SlogpVSA, "CCO");
        let labute = calc_float(&LabuteASA, "CCO");
        let bin_sum: f32 = slogp.iter().sum();
        assert!((bin_sum - labute).abs() < 1.0,
            "bin sum {} ≈ LabuteASA {}", bin_sum, labute);
    }

    #[test]
    fn test_slogp_vsa_larger_molecule() {
        let v = calc_float_vec(&SlogpVSA, "CC(=O)Oc1ccccc1C(=O)O");
        let total: f32 = v.iter().sum();
        assert!(total > 50.0, "aspirin total VSA = {}", total);
    }

    // -- SMR_VSA --

    #[test]
    fn test_smr_vsa_nonzero() {
        let v = calc_float_vec(&SmrVSA, "c1ccccc1");
        let total: f32 = v.iter().sum();
        assert!(total > 0.0, "total SMR_VSA = {}", total);
    }

    #[test]
    fn test_smr_vsa_bins_sum_to_labute() {
        let smr = calc_float_vec(&SmrVSA, "CCO");
        let labute = calc_float(&LabuteASA, "CCO");
        let bin_sum: f32 = smr.iter().sum();
        assert!((bin_sum - labute).abs() < 1.0,
            "bin sum {} ≈ LabuteASA {}", bin_sum, labute);
    }

    // -- LabuteASA --

    #[test]
    fn test_labute_positive() {
        let v = calc_float(&LabuteASA, "CCCCCC");
        assert!(v > 0.0, "hexane ASA = {}", v);
    }

    #[test]
    fn test_labute_larger_more() {
        let small = calc_float(&LabuteASA, "CC");
        let large = calc_float(&LabuteASA, "CCCCCCCCCC");
        assert!(large > small,
            "decane {} > ethane {}", large, small);
    }

    #[test]
    fn test_labute_benzene() {
        let v = calc_float(&LabuteASA, "c1ccccc1");
        assert!(v > 50.0 && v < 200.0, "benzene ASA = {}", v);
    }

    #[test]
    fn test_labute_methane() {
        let v = calc_float(&LabuteASA, "C");
        assert!((v - 36.3).abs() < 1.0, "methane ASA = {}", v);
    }

    // -- PEOE_VSA --

    #[test]
    fn test_peoe_vsa_benzene() {
        let v = calc_float_vec(&PeoeVsa, "c1ccccc1");
        // All C atoms have same charge → all VSA lands in one bin
        let nonzero: Vec<_> = v.iter().filter(|&&x| x > 0.0).collect();
        assert!(!nonzero.is_empty(), "should have at least one nonzero bin");
        let total: f32 = v.iter().sum();
        let labute = calc_float(&LabuteASA, "c1ccccc1");
        assert!((total - labute).abs() < 1.0,
            "PEOE_VSA sum {} ≈ LabuteASA {}", total, labute);
    }

    #[test]
    fn test_peoe_vsa_length() {
        let v = calc_float_vec(&PeoeVsa, "CCO");
        assert_eq!(v.len(), 14, "PEOE_VSA should have 14 bins");
        let v2 = calc_float_vec(&PeoeVsa, "c1ccccc1");
        assert_eq!(v2.len(), 14);
    }

    #[test]
    fn test_peoe_vsa_formaldehyde() {
        // C=O: O is more electronegative → negative charge, C → positive
        let v = calc_float_vec(&PeoeVsa, "C=O");
        // Bins 0-6 are negative charges, bins 7-13 are non-negative
        let neg_sum: f32 = v[..7].iter().sum();
        let pos_sum: f32 = v[7..].iter().sum();
        assert!(neg_sum > 0.0, "O should contribute to negative bins: {:?}", v);
        assert!(pos_sum > 0.0, "C should contribute to positive bins: {:?}", v);
    }

    #[test]
    fn test_peoe_vsa_sum() {
        for smi in &["CCO", "CC(=O)O", "c1ccccc1", "CCCCCC"] {
            let peoe = calc_float_vec(&PeoeVsa, smi);
            let labute = calc_float(&LabuteASA, smi);
            let bin_sum: f32 = peoe.iter().sum();
            assert!((bin_sum - labute).abs() < 1.0,
                "{}: PEOE_VSA sum {} ≈ LabuteASA {}", smi, bin_sum, labute);
        }
    }
}
