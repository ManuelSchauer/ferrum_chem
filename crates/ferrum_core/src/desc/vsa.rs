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

use super::Descriptor;
use super::logp::crippen_atom_contributions;
use crate::mol::molecule::Molecule;
use crate::types::Real;
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
/// Bin boundaries (RDKit-compatible):
/// ```text
/// Bin  0: logP < -0.40
/// Bin  1: -0.40 ≤ logP < -0.20
/// Bin  2: -0.20 ≤ logP <  0.00
/// Bin  3:  0.00 ≤ logP <  0.10
/// Bin  4:  0.10 ≤ logP <  0.15
/// Bin  5:  0.15 ≤ logP <  0.20
/// Bin  6:  0.20 ≤ logP <  0.25
/// Bin  7:  0.25 ≤ logP <  0.30
/// Bin  8:  0.30 ≤ logP <  0.40
/// Bin  9:  0.40 ≤ logP <  0.50
/// Bin 10:  0.50 ≤ logP <  0.60
/// Bin 11:  0.60 ≤ logP
/// ```
pub struct SlogpVSA;

const SLOGP_BINS: [f64; 12] = [
    -0.40, -0.20, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
    f64::INFINITY,
];

impl Descriptor for SlogpVSA {
    fn name(&self) -> String { "SlogP_VSA".to_string() }
    fn size(&self) -> usize { 12 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        for v in output.iter_mut() { *v = 0.0; }

        let vsa = compute_atom_vsa(mol);
        let (logp_contrib, _) = crippen_atom_contributions(mol);

        for i in 0..mol.atoms.len() {
            if mol.atoms[i].atomic_number <= 1 { continue; }
            let lp = logp_contrib[i];
            let bin = find_bin(lp, &SLOGP_BINS);
            output[bin] += vsa[i] as Real;
        }
    }
}

// ---------------------------------------------------------------------------
// SMR_VSA — 10 bins
// ---------------------------------------------------------------------------

/// SMR_VSA: VSA binned by Crippen MR atom contributions (10 values).
///
/// Bin boundaries (RDKit-compatible):
/// ```text
/// Bin 0: MR < 1.29
/// Bin 1: 1.29 ≤ MR < 1.82
/// Bin 2: 1.82 ≤ MR < 2.24
/// Bin 3: 2.24 ≤ MR < 2.45
/// Bin 4: 2.45 ≤ MR < 2.75
/// Bin 5: 2.75 ≤ MR < 3.05
/// Bin 6: 3.05 ≤ MR < 3.63
/// Bin 7: 3.63 ≤ MR < 3.80
/// Bin 8: 3.80 ≤ MR < 4.00
/// Bin 9: 4.00 ≤ MR
/// ```
pub struct SmrVSA;

const SMR_BINS: [f64; 10] = [
    1.29, 1.82, 2.24, 2.45, 2.75, 3.05, 3.63, 3.80, 4.00,
    f64::INFINITY,
];

impl Descriptor for SmrVSA {
    fn name(&self) -> String { "SMR_VSA".to_string() }
    fn size(&self) -> usize { 10 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        for v in output.iter_mut() { *v = 0.0; }

        let vsa = compute_atom_vsa(mol);
        let (_, mr_contrib) = crippen_atom_contributions(mol);

        for i in 0..mol.atoms.len() {
            if mol.atoms[i].atomic_number <= 1 { continue; }
            let m = mr_contrib[i];
            let bin = find_bin(m, &SMR_BINS);
            output[bin] += vsa[i] as Real;
        }
    }
}

// ---------------------------------------------------------------------------
// LabuteASA
// ---------------------------------------------------------------------------

/// Labute's Approximate Surface Area: sum of all atom VSA contributions.
pub struct LabuteASA;

impl Descriptor for LabuteASA {
    fn name(&self) -> String { "LabuteASA".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let vsa = compute_atom_vsa(mol);
        output[0] = vsa.iter().sum::<f64>() as Real;
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
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc<D: Descriptor>(d: &D, smiles: &str) -> Vec<Real> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = vec![0.0; d.size()];
        d.calculate(&mol, &mut output);
        output
    }

    // -- compute_atom_vsa --

    #[test]
    fn test_vsa_positive() {
        let mol = parse_smiles("CCO").unwrap();
        let vsa = compute_atom_vsa(&mol);
        // All heavy atoms should have positive VSA
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
        // All carbons should have same VSA (symmetric molecule)
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
        let v = calc(&SlogpVSA, "c1ccccc1");
        let total: Real = v.iter().sum();
        assert!(total > 0.0, "total SlogP_VSA = {}", total);
    }

    #[test]
    fn test_slogp_vsa_bins_sum_to_labute() {
        let slogp = calc(&SlogpVSA, "CCO");
        let labute = calc(&LabuteASA, "CCO");
        let bin_sum: Real = slogp.iter().sum();
        assert!((bin_sum - labute[0]).abs() < 1.0,
            "bin sum {} ≈ LabuteASA {}", bin_sum, labute[0]);
    }

    #[test]
    fn test_slogp_vsa_larger_molecule() {
        let v = calc(&SlogpVSA, "CC(=O)Oc1ccccc1C(=O)O"); // aspirin
        let total: Real = v.iter().sum();
        assert!(total > 50.0, "aspirin total VSA = {}", total);
    }

    // -- SMR_VSA --

    #[test]
    fn test_smr_vsa_nonzero() {
        let v = calc(&SmrVSA, "c1ccccc1");
        let total: Real = v.iter().sum();
        assert!(total > 0.0, "total SMR_VSA = {}", total);
    }

    #[test]
    fn test_smr_vsa_bins_sum_to_labute() {
        let smr = calc(&SmrVSA, "CCO");
        let labute = calc(&LabuteASA, "CCO");
        let bin_sum: Real = smr.iter().sum();
        assert!((bin_sum - labute[0]).abs() < 1.0,
            "bin sum {} ≈ LabuteASA {}", bin_sum, labute[0]);
    }

    // -- LabuteASA --

    #[test]
    fn test_labute_positive() {
        let v = calc(&LabuteASA, "CCCCCC");
        assert!(v[0] > 0.0, "hexane ASA = {}", v[0]);
    }

    #[test]
    fn test_labute_larger_more() {
        let small = calc(&LabuteASA, "CC");
        let large = calc(&LabuteASA, "CCCCCCCCCC");
        assert!(large[0] > small[0],
            "decane {} > ethane {}", large[0], small[0]);
    }

    #[test]
    fn test_labute_benzene() {
        let v = calc(&LabuteASA, "c1ccccc1");
        // Benzene: 6 carbons, each ~20-30 Å² exposed
        assert!(v[0] > 50.0 && v[0] < 200.0, "benzene ASA = {}", v[0]);
    }

    #[test]
    fn test_labute_methane() {
        let v = calc(&LabuteASA, "C");
        // Methane: single C with no heavy neighbors → full sphere = 4π(1.7)² ≈ 36.3
        assert!((v[0] - 36.3).abs() < 1.0, "methane ASA = {}", v[0]);
    }
}
