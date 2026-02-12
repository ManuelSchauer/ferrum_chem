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
use crate::io::smarts::parse_smarts;
use crate::algo::iso::count_matches;

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
    QedParam { ads: [1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616], weight: 0.61 },
    // PSA (TPSA)
    QedParam { ads: [1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167], weight: 0.06 },
    // ROTB
    QedParam { ads: [0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403], weight: 0.65 },
    // AROM (aromatic rings)
    QedParam { ads: [3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610], weight: 0.48 },
    // ALERTS (structural alerts — we assume 0 alerts since we don't implement alert detection)
    QedParam { ads: [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140], weight: 0.95 },
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
    let alerts = compute_structural_alerts(mol) as f64;

    let values = [mw, logp, hba, hbd, psa, rotb, arom, alerts];

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

/// RDKit QED structural alert SMARTS patterns.
/// Each entry is a list of SMARTS; if ANY matches, that alert fires (count += 1).
/// From Bickerton et al. (2012), Table S1.
static STRUCTURAL_ALERT_SMARTS: &[&[&str]] = &[
    &["*1[O,S,N]*1", "[S,C](=[O,S])[F,Br,Cl,I]", "[CX4][Cl,Br,I]", "[#6]S(=O)(=O)O[#6]"],
    &["[$([CH]),$(CC)]#CC(=O)[#6]", "[$([CH]),$(CC)]#CC(=O)O[#6]", "n[OH]"],
    &["[$([CH]),$(CC)]#CS(=O)(=O)[#6]", "C=C(C=O)C=O", "n1c([F,Cl,Br,I])cccc1", "[CH1](=O)", "[#8][#8]"],
    &["[C;!R]=[N;!R]", "[N!R]=[N!R]", "[#6](=O)[#6](=O)", "[#16][#16]", "[#7][NH2]", "C(=O)N[NH2]"],
    &["[#6]=S", "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]"],
    &["C1(=[O,N])C=CC(=[O,N])C=C1", "C1(=[O,N])C(=[O,N])C=CC=C1", "a21aa3a(aa1aaaa2)aaaa3"],
    &["a31a(a2a(aa1)aaaa2)aaaa3", "a1aa2a3a(a1)A=AA=A3=AA=A2", "c1cc([NH2])ccc1"],
    &["[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]"],
    &["I", "OS(=O)(=O)[O-]", "[N+](=O)[O-]", "C(=O)N[OH]"],
    &["C1NC(=O)NC(=O)1", "[SH]", "[S-]", "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]"],
    &["c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]", "[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1"],
    &["[CR1]1[CR1][CR1]cc[CR1][CR1]1", "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1"],
    &["[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1", "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1"],
    &["[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1", "C#C"],
    &["[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]", "[$([N+R]),$([n+R]),$([N+]=C)][O-]"],
    &["[#6]=N[OH]", "[#6]=NOC=O", "[#6](=O)[CX4,CR0X3,O][#6](=O)", "c1ccc2c(c1)ccc(=O)o2"],
    &["[O+,o+,S+,s+]", "N=C=O", "[NX3,NX4][F,Cl,Br,I]", "c1ccccc1OC(=O)[#6]", "[CR0]=[CR0][CR0]=[CR0]"],
    &["[C+,c+,C-,c-]", "N=[N+]=[N-]", "C12C(NC(N1)=O)CSC2", "c1c([OH])c([OH,NH2,NH])ccc1", "P"],
    &["[N,O,S]C#N", "C=C=O", "[Si][F,Cl,Br,I]", "[SX2]O", "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)"],
    &["O1CCCCC1OC2CCC3CCCCC3C2", "N=[CR0][N,n,O,S]"],
    &["[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2"],
    &["C=[C!r]C#N", "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1"],
    &["[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1"],
    &["[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])", "[OH]c1ccc([OH,NH2,NH])cc1"],
    &["c1ccccc1OC(=O)O", "[SX2H0][N]", "c12ccccc1(SC(S)=N2)", "c12ccccc1(SC(=S)N2)", "c1nnnn1C=O"],
    &["s1c(S)nnc1NC=O", "S1C=CSC1=S", "C(=O)Onnn", "OS(=O)(=O)C(F)(F)F", "N#CC[OH]", "N#CC(=O)"],
    &["S(=O)(=O)C#N", "N[CH2]C#N", "C1(=O)NCC1", "S(=O)(=O)[O-,OH]", "NC[F,Cl,Br,I]", "C=[C!r]O"],
    &["[NX2+0]=[O+0]", "[OR0,NR0][OR0,NR0]", "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]", "[CX2R0][NX3R0]"],
    &["c1ccccc1[C;!R]=[C;!R]c2ccccc2", "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]"],
    &["[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]"],
    &["[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]", "[*]=[N+]=[*]", "[SX3](=O)[O-,OH]"],
    &["N#N", "F.F.F.F", "[R0;D2][R0;D2][R0;D2][R0;D2]", "[cR,CR]~C(=O)NC(=O)~[cR,CR]", "C=!@CC=[O,S]"],
    &["[#6,#8,#16][#6](=O)O[#6]", "c[C;R0](=[O,S])[#6]", "c[SX2][C;!R]", "C=C=C"],
    &["c1nc([F,Cl,Br,I,S])ncc1", "c1ncnc([F,Cl,Br,I,S])c1", "c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])"],
    &["[#6]S(=O)(=O)c1ccc(cc1)F", "[15N]", "[13C]", "[18O]", "[34S]"],
];

/// Count the number of structural alert groups that match.
fn compute_structural_alerts(mol: &Molecule) -> u32 {
    let mut alert_count = 0u32;
    for alert_group in STRUCTURAL_ALERT_SMARTS {
        let mut fired = false;
        for smarts_str in *alert_group {
            if let Ok(query) = parse_smarts(smarts_str) {
                if count_matches(mol, &query) > 0 {
                    fired = true;
                    break;
                }
            }
        }
        if fired { alert_count += 1; }
    }
    alert_count
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
