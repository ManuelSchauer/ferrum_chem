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

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Atomic properties
// ---------------------------------------------------------------------------

/// Pauling electronegativity for common elements.
fn electronegativity(atomic_number: u8) -> f64 {
    match atomic_number {
        1 => 2.20,  // H
        5 => 2.04,  // B
        6 => 2.55,  // C
        7 => 3.04,  // N
        8 => 3.44,  // O
        9 => 3.98,  // F
        14 => 1.90, // Si
        15 => 2.19, // P
        16 => 2.58, // S
        17 => 3.16, // Cl
        35 => 2.96, // Br
        53 => 2.66, // I
        _ => 2.00,  // default
    }
}

/// Van der Waals volume (Å³), approximated as (4/3)π r³.
fn vdw_volume(atomic_number: u8) -> f64 {
    let r: f64 = match atomic_number {
        1 => 1.20,
        5 => 1.92,
        6 => 1.70,
        7 => 1.55,
        8 => 1.52,
        9 => 1.47,
        14 => 2.10,
        15 => 1.80,
        16 => 1.80,
        17 => 1.75,
        35 => 1.85,
        53 => 1.98,
        _ => 1.70,
    };
    (4.0 / 3.0) * std::f64::consts::PI * r.powi(3)
}

/// Atomic polarizability (Å³, approximate).
fn polarizability(atomic_number: u8) -> f64 {
    match atomic_number {
        1 => 0.67,
        5 => 3.03,
        6 => 1.76,
        7 => 1.10,
        8 => 0.80,
        9 => 0.56,
        14 => 5.38,
        15 => 3.63,
        16 => 2.90,
        17 => 2.18,
        35 => 3.05,
        53 => 5.35,
        _ => 1.50,
    }
}

/// Standard atomic mass.
fn atomic_mass(atomic_number: u8) -> f64 {
    crate::mol::elements::ATOMIC_MASSES[atomic_number as usize] as f64
}

// ---------------------------------------------------------------------------
// Distance matrix
// ---------------------------------------------------------------------------

fn compute_distance_matrix(mol: &Molecule) -> Vec<u32> {
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
fn compute_ats(mol: &Molecule, dist: &[u32], weights: &[f64]) -> [f64; MAX_LAG] {
    let n = mol.num_atoms();
    let mut ats = [0.0f64; MAX_LAG];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist[i * n + j];
            if d == 0 || d == u32::MAX {
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
    fn name(&self) -> &str {
        "Autocorrelation"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut values = vec![0.0f32; MAX_LAG * 4];

        let n = mol.num_atoms();
        if n < 2 {
            return DescriptorValue::FloatVec(values);
        }

        let dist = compute_distance_matrix(mol);

        // Compute weight vectors
        let w_mass: Vec<f64> = (0..n)
            .map(|i| atomic_mass(mol.atoms[i].atomic_number))
            .collect();
        let w_pol: Vec<f64> = (0..n)
            .map(|i| polarizability(mol.atoms[i].atomic_number))
            .collect();
        let w_en: Vec<f64> = (0..n)
            .map(|i| electronegativity(mol.atoms[i].atomic_number))
            .collect();
        let w_vdw: Vec<f64> = (0..n)
            .map(|i| vdw_volume(mol.atoms[i].atomic_number))
            .collect();

        let ats_mass = compute_ats(mol, &dist, &w_mass);
        let ats_pol = compute_ats(mol, &dist, &w_pol);
        let ats_en = compute_ats(mol, &dist, &w_en);
        let ats_vdw = compute_ats(mol, &dist, &w_vdw);

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
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Autocorrelation>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_ats(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match Autocorrelation.calculate(&mut mol) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    fn test_ats_size() {
        let a = calc_ats("CC");
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn test_ats_ethane() {
        let a = calc_ats("CC");
        assert!(a[0] > 200.0, "ATSm1 = {}", a[0]);
        assert!(a[1].abs() < 0.001, "ATSm2 = {}", a[1]);
    }

    #[test]
    fn test_ats_propane() {
        let a = calc_ats("CCC");
        assert!(a[0] > 0.0, "ATSm1 > 0");
        assert!(a[1] > 0.0, "ATSm2 > 0");
        assert!(a[2].abs() < 0.001, "ATSm3 should be 0 for 3-atom chain");
    }

    #[test]
    fn test_ats_benzene() {
        let a = calc_ats("c1ccccc1");
        assert!(a[0] > 0.0, "ATSm1");
        assert!(a[1] > 0.0, "ATSm2");
        assert!(a[2] > 0.0, "ATSm3");
        assert!(a[3].abs() < 0.001, "ATSm4 should be ~0 for benzene");
    }

    #[test]
    fn test_ats_ethanol_vs_ethane() {
        let ethane = calc_ats("CC");
        let ethanol = calc_ats("CCO");
        assert!(ethanol[0] > ethane[0], "ethanol ATSm1 > ethane ATSm1");
    }

    #[test]
    fn test_ats_single_atom() {
        let a = calc_ats("C");
        for v in &a {
            assert!(v.abs() < 0.001);
        }
    }

    #[test]
    fn test_ats_long_chain() {
        let a = calc_ats("CCCCCCCCCC");
        for i in 0..MAX_LAG {
            assert!(a[i] > 0.0, "ATSm{} should be > 0 for decane", i + 1);
        }
    }
}
