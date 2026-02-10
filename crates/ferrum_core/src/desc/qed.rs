//! QED (Quantitative Estimate of Drug-likeness) descriptor.
//!
//! Combines multiple molecular properties into a single drug-likeness score (0-1).
//!
//! Reference: Bickerton et al. (2012) "Quantifying the chemical beauty of drugs"
//! Nature Chemistry 4, 90-98.

use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;
use crate::desc::molecular_weight::MolecularWeight;
use crate::desc::logp::LogP;
use crate::desc::hbond::{HBondAcceptor, HBondDonor};
use crate::desc::tpsa::Tpsa;
use crate::desc::bond_count::RotatableBondCount;
use crate::desc::aromatic_ring_count::AromaticRingCount;

/// QED parameters: (mean, sigma, weight) for each property.
/// From Bickerton et al. Table 1 (weighted version).
struct QedParam {
    // Parameters for the asymmetric double sigmoidal desirability function
    // Each property has: [A, B, C, D, E, F, DMAX]
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

/// QED (Quantitative Estimate of Drug-likeness).
///
/// Computes a weighted geometric mean of desirability functions for 7 molecular properties.
/// Score ranges from 0 (least drug-like) to ~1 (most drug-like).
///
/// Properties used: MW, LogP, HBA, HBD, TPSA, RotatableBonds, AromaticRings.
///
/// Note: The original QED also uses structural alerts (PAINS-like patterns).
/// This implementation omits alerts (effectively treats alert count as 0).
pub struct Qed;

impl Descriptor for Qed {
    fn name(&self) -> String {
        "QED".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Calculate component descriptors
        let mut val = [0.0 as Real];

        MolecularWeight.calculate(mol, &mut val);
        let mw = val[0] as f64;

        LogP.calculate(mol, &mut val);
        let logp = val[0] as f64;

        HBondAcceptor.calculate(mol, &mut val);
        let hba = val[0] as f64;

        HBondDonor.calculate(mol, &mut val);
        let hbd = val[0] as f64;

        Tpsa.calculate(mol, &mut val);
        let psa = val[0] as f64;

        RotatableBondCount.calculate(mol, &mut val);
        let rotb = val[0] as f64;

        AromaticRingCount.calculate(mol, &mut val);
        let arom = val[0] as f64;

        let values = [mw, logp, hba, hbd, psa, rotb, arom];

        // Compute weighted geometric mean of ADS desirability scores
        let mut sum_weights = 0.0;
        let mut sum_log = 0.0;

        for (i, param) in QED_PARAMS.iter().enumerate() {
            let d = ads(values[i], &param.ads);
            let d = d.max(1e-10); // Prevent log(0)
            sum_log += param.weight * d.ln();
            sum_weights += param.weight;
        }

        let qed = if sum_weights > 0.0 {
            (sum_log / sum_weights).exp()
        } else {
            0.0
        };

        output[0] = qed.clamp(0.0, 1.0) as Real;
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

    fn calc_qed(smiles: &str) -> Real {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = [0.0];
        Qed.calculate(&mol, &mut output);
        output[0]
    }

    #[test]
    fn test_qed_range() {
        let qed = calc_qed("c1ccccc1"); // benzene
        assert!(qed >= 0.0 && qed <= 1.0, "QED should be 0-1, got {}", qed);
    }

    #[test]
    fn test_qed_aspirin() {
        // Aspirin: CC(=O)Oc1ccccc1C(=O)O
        // RDKit QED ~ 0.55
        let qed = calc_qed("CC(=O)Oc1ccccc1C(=O)O");
        println!("Aspirin QED: {}", qed);
        assert!(qed > 0.2 && qed < 0.9, "Aspirin QED={}", qed);
    }

    #[test]
    fn test_qed_caffeine() {
        // Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
        // RDKit QED ~ 0.40
        let qed = calc_qed("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        println!("Caffeine QED: {}", qed);
        assert!(qed > 0.1 && qed < 0.8, "Caffeine QED={}", qed);
    }

    #[test]
    fn test_qed_drug_like() {
        // Ibuprofen: CC(C)Cc1ccc(cc1)C(C)C(=O)O
        let qed = calc_qed("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
        println!("Ibuprofen QED: {}", qed);
        assert!(qed > 0.2 && qed < 0.9);
    }

    #[test]
    fn test_qed_methane_low() {
        // Methane is not drug-like
        let qed = calc_qed("C");
        println!("Methane QED: {}", qed);
        assert!(qed < 0.5, "Methane should have low QED: {}", qed);
    }

    #[test]
    fn test_qed_large_molecule_lower() {
        // Very large molecule should score lower
        let qed_small = calc_qed("c1ccc(CC(=O)O)cc1");
        let qed_large = calc_qed("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC");
        println!("Small drug-like QED: {}, Large chain QED: {}", qed_small, qed_large);
        // Large chain should be less drug-like
        assert!(qed_large < qed_small || qed_large < 0.5);
    }
}
