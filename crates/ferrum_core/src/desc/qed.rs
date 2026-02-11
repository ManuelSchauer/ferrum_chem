//! QED (Quantitative Estimate of Drug-likeness) descriptor.
//!
//! Combines multiple molecular properties into a single drug-likeness score (0-1).
//!
//! Reference: Bickerton et al. (2012) "Quantifying the chemical beauty of drugs"
//! Nature Chemistry 4, 90-98.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::desc::molecular_weight::compute_molecular_weight;
use crate::desc::logp::compute_logp;
use crate::desc::hbond::{compute_hbond_acceptors, compute_hbond_donors};
use crate::desc::tpsa::compute_tpsa;
use crate::desc::bond_count::compute_rotatable_bond_count;
use crate::desc::aromatic_ring_count::compute_aromatic_ring_count;

/// QED parameters: (mean, sigma, weight) for each property.
/// From Bickerton et al. Table 1 (weighted version).
struct QedParam {
    ads: [f64; 7],
    weight: f64,
}

// ADS function parameters from RDKit's QED implementation
// Order: MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
static QED_PARAMS: &[QedParam] = &[
    // MW
    QedParam { ads: [2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561], weight: 0.66 },
    // ALOGP
    QedParam { ads: [3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604], weight: 0.46 },
    // HBA
    QedParam { ads: [2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046], weight: 0.05 },
    // HBD
    QedParam { ads: [1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616], weight: 0.26 },
    // PSA (TPSA)
    QedParam { ads: [1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51023704, 104.5686167], weight: 0.06 },
    // ROTB
    QedParam { ads: [0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403], weight: 0.65 },
    // AROM (aromatic rings)
    QedParam { ads: [3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610], weight: 0.48 },
];

/// Asymmetric Double Sigmoidal function used in QED calculation.
fn ads(x: f64, params: &[f64; 7]) -> f64 {
    let a = params[0];
    let b = params[1];
    let c = params[2];
    let d = params[3];
    let e = params[4];
    let f = params[5];
    let dmax = params[6];

    let numerator = a + b / (1.0 + (-1.0 * (x - c + d / 2.0) / e).exp())
        * (1.0 - 1.0 / (1.0 + (-1.0 * (x - c - d / 2.0) / f).exp()));
    numerator / dmax
}

/// Compute raw QED score from a molecule.
fn compute_qed(mol: &Molecule) -> f64 {
    let mw = compute_molecular_weight(mol) as f64;
    let logp = compute_logp(mol).0 as f64;
    let hba = compute_hbond_acceptors(mol) as f64;
    let hbd = compute_hbond_donors(mol) as f64;
    let psa = compute_tpsa(mol, false) as f64;
    let rotb = compute_rotatable_bond_count(mol) as f64;
    let arom = compute_aromatic_ring_count(mol) as f64;

    let values = [mw, logp, hba, hbd, psa, rotb, arom];

    let mut sum_weights = 0.0;
    let mut sum_log = 0.0;

    for (i, param) in QED_PARAMS.iter().enumerate() {
        let d = ads(values[i], &param.ads).max(1e-10);
        sum_log += param.weight * d.ln();
        sum_weights += param.weight;
    }

    if sum_weights > 0.0 {
        (sum_log / sum_weights).exp().clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// QED (Quantitative Estimate of Drug-likeness).
///
/// Computes a weighted geometric mean of desirability functions for 7 molecular properties.
/// Score ranges from 0 (least drug-like) to ~1 (most drug-like).
///
/// Properties used: MW, LogP, HBA, HBD, TPSA, RotatableBonds, AromaticRings.
///
/// Note: The original QED also uses structural alerts (PAINS-like patterns).
/// This implementation omits alerts (effectively treats alert count as 0).
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct Qed;

impl Descriptor for Qed {
    fn name(&self) -> &str {
        "QED"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_qed(mol) as f32)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Qed>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_qed(smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        Qed.calculate(&mut mol).as_float()
    }

    #[test]
    fn test_qed_range() {
        let qed = calc_qed("c1ccccc1");
        assert!(qed >= 0.0 && qed <= 1.0, "QED should be 0-1, got {}", qed);
    }

    #[test]
    fn test_qed_aspirin() {
        let qed = calc_qed("CC(=O)Oc1ccccc1C(=O)O");
        assert!(qed > 0.2 && qed < 0.9, "Aspirin QED={}", qed);
    }

    #[test]
    fn test_qed_caffeine() {
        let qed = calc_qed("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        assert!(qed > 0.1 && qed < 0.8, "Caffeine QED={}", qed);
    }

    #[test]
    fn test_qed_drug_like() {
        let qed = calc_qed("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
        assert!(qed > 0.2 && qed < 0.9);
    }

    #[test]
    fn test_qed_methane_low() {
        let qed = calc_qed("C");
        assert!(qed < 0.5, "Methane should have low QED: {}", qed);
    }

    #[test]
    fn test_qed_large_molecule_lower() {
        let qed_small = calc_qed("c1ccc(CC(=O)O)cc1");
        let qed_large = calc_qed("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC");
        assert!(qed_large < qed_small || qed_large < 0.5);
    }
}
