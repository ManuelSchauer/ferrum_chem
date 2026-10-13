#![allow(dead_code)]
//! QED (Quantitative Estimate of Drug-likeness) descriptor.
//!
//! Combines multiple molecular properties into a single drug-likeness score (0-1).
//!
//! Reference: Bickerton et al. (2012) "Quantifying the chemical beauty of drugs"
//! Nature Chemistry 4, 90-98.

//! ## File guide: Qed
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct QedParam`
//! - `fn ads`
//! - `struct QueryScreen`
//! - `struct RequiredElementSet`
//! - `struct ScreenedQuery`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Qed` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Qed` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/qed.rs` defines qed behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Qed` workflows.
//! - Repository unit and parity tests covering `Qed` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/qed.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::iso::{count_matches_target, has_substruct_match_target, IsoTarget};
use crate::desc::bond_count::{compute_rotatable_bond_count, compute_rotatable_bond_count_cached};
use crate::desc::hbond::{compute_hbond_donors, compute_hbond_donors_cached};
use crate::desc::logp::{compute_logp, compute_logp_cached};
use crate::desc::molecular_weight::compute_molecular_weight;
use crate::desc::tpsa::{compute_tpsa, compute_tpsa_cached};
use crate::io::smarts::{QueryGraph, parse_smarts};
use crate::mol::molecule::Molecule;
use once_cell::sync::Lazy;

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
    QedParam {
        ads: [
            2.817065973,
            392.5754953,
            290.7489764,
            2.419764353,
            49.22325677,
            65.37051707,
            104.9805561,
        ],
        weight: 0.66,
    },
    // ALOGP
    QedParam {
        ads: [
            3.172690585,
            137.8624751,
            2.534937431,
            4.581497897,
            0.822739154,
            0.576295591,
            131.3186604,
        ],
        weight: 0.46,
    },
    // HBA
    QedParam {
        ads: [
            2.948620388,
            160.4605972,
            3.615294657,
            4.435986202,
            0.290141953,
            1.300669958,
            148.7763046,
        ],
        weight: 0.05,
    },
    // HBD
    QedParam {
        ads: [
            1.618662227,
            1010.051101,
            0.985094388,
            0.000000001,
            0.713820843,
            0.920922555,
            258.1632616,
        ],
        weight: 0.61,
    },
    // PSA (TPSA)
    QedParam {
        ads: [
            1.876861559,
            125.2232657,
            62.90773554,
            87.83366614,
            12.01999824,
            28.51324732,
            104.5686167,
        ],
        weight: 0.06,
    },
    // ROTB
    QedParam {
        ads: [
            0.010000000,
            272.4121427,
            2.558379970,
            1.565547684,
            1.271567166,
            2.758063707,
            105.4420403,
        ],
        weight: 0.65,
    },
    // AROM (aromatic rings)
    QedParam {
        ads: [
            3.217788970,
            957.7374108,
            2.274627939,
            0.000000001,
            1.317690384,
            0.375760881,
            312.3372610,
        ],
        weight: 0.48,
    },
    // ALERTS (structural alerts — we assume 0 alerts since we don't implement alert detection)
    QedParam {
        ads: [
            0.010000000,
            1199.094025,
            -0.09002883,
            0.000000001,
            0.185904477,
            0.875193782,
            417.7253140,
        ],
        weight: 0.95,
    },
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

/// RDKit QED uses its own HBA SMARTS set instead of the generic Lipinski-style HBA count.
static QED_ACCEPTOR_SMARTS: &[&str] = &[
    "[oH0;X2]",
    "[OH1;X2;v2]",
    "[OH0;X2;v2]",
    "[OH0;X1;v2]",
    "[O-;X1]",
    "[SH0;X2;v2]",
    "[SH0;X1;v2]",
    "[S-;X1]",
    "[nH0;X2]",
    "[NH0;X1;v3]",
    "[$([N;+0;X3;v3]);!$(N[C,S]=O)]",
];


struct ScreenedQuery {
    query: QueryGraph,
    screen: crate::algo::iso::QueryScreen,
}

static QED_ACCEPTOR_PATTERNS: Lazy<Vec<ScreenedQuery>> = Lazy::new(|| {
    QED_ACCEPTOR_SMARTS
        .iter()
        .filter_map(|smarts| parse_smarts(smarts).ok())
        .map(|query| {
            let screen = crate::algo::iso::QueryScreen::build(&query);
            ScreenedQuery { query, screen }
        })
        .collect()
});

/// Computes qed hbond acceptors.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_hbond_acceptors(mol: &Molecule) -> u32 {
    let iso_target = IsoTarget::new(mol);
    compute_qed_hbond_acceptors_target(&iso_target)
}

fn compute_qed_hbond_acceptors_target(target: &IsoTarget) -> u32 {
    QED_ACCEPTOR_PATTERNS
        .iter()
        .map(|entry| {
            if !entry.screen.passes_target(target) {
                return 0;
            }
            count_matches_target(target, &entry.query) as u32
        })
        .sum()
}

/// QED `AROM` exactly as RDKit computes it:
/// `len(Chem.GetSSSR(Chem.DeleteSubstructs(mol, '[$([A;R][!a])]')))`.
///
/// Atoms matching the aliphatic-ring pattern are deleted and the SSSR size of
/// what remains is counted (its cycle rank: bonds - atoms + fragments, dative
/// bonds excluded as in RDKit's ring finding). This is RDKit's definition, kept
/// verbatim for parity even though it can count non-aromatic rings.
fn compute_qed_aromatic_ring_count(mol: &Molecule) -> u32 {
    compute_qed_aromatic_ring_count_target(&IsoTarget::new(mol))
}

fn compute_qed_aromatic_ring_count_target(target: &IsoTarget) -> u32 {
    static ALIPHATIC_RINGS: Lazy<QueryGraph> =
        Lazy::new(|| parse_smarts("[$([A;R][!a])]").expect("valid QED AliphaticRings SMARTS"));
    let mol = target.target;
    let n = mol.num_atoms();
    let mut deleted = vec![false; n];
    crate::algo::iso::visit_start_atoms_target(target, &ALIPHATIC_RINGS, |_| true, |a| {
        deleted[a as usize] = true;
        true
    });
    // union-find over the remaining graph
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut n_bonds = 0i64;
    for bond in &mol.bonds {
        let (u, v) = (bond.source as usize, bond.target as usize);
        if deleted[u] || deleted[v] || bond.bond_type == crate::types::BondType::Dative {
            continue;
        }
        n_bonds += 1;
        let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
        if ru != rv {
            parent[ru] = rv;
        }
    }
    let n_atoms = deleted.iter().filter(|&&d| !d).count() as i64;
    let n_components = (0..n).filter(|&i| !deleted[i] && find(&mut parent, i) == i).count() as i64;
    (n_bonds - n_atoms + n_components).max(0) as u32
}

fn compute_qed_aromatic_ring_count_cached(mol: &mut Molecule) -> u32 {
    compute_qed_aromatic_ring_count(mol)
}

fn compute_qed(mol: &Molecule) -> f64 {
    let iso_target = IsoTarget::new(mol);
    let mw = compute_molecular_weight(mol);
    let logp = compute_logp(mol).0 as f64;
    let hba = compute_qed_hbond_acceptors_target(&iso_target) as f64;
    let hbd = compute_hbond_donors(mol) as f64;
    let psa = compute_tpsa(mol, false);
    let rotb = compute_rotatable_bond_count(mol) as f64;
    let arom = compute_qed_aromatic_ring_count(mol) as f64;
    let alerts = compute_structural_alerts_target(&iso_target) as f64;

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

/// Computes qed cached.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_cached(mol: &mut Molecule) -> f64 {
    let mw = compute_molecular_weight(mol);
    let logp = compute_logp_cached(mol).0 as f64;
    let hbd = compute_hbond_donors_cached(mol) as f64;
    let psa = compute_tpsa_cached(mol, false);
    let rotb = compute_rotatable_bond_count_cached(mol) as f64;

    let iso_target = IsoTarget::new(mol);
    let arom = compute_qed_aromatic_ring_count_target(&iso_target) as f64;
    let hba = compute_qed_hbond_acceptors_target(&iso_target) as f64;
    let alerts = compute_structural_alerts_target(&iso_target) as f64;

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
    &[
        "*1[O,S,N]*1",
        "[S,C](=[O,S])[F,Br,Cl,I]",
        "[CX4][Cl,Br,I]",
        "[#6]S(=O)(=O)O[#6]",
    ],
    &[
        "[$([CH]),$(CC)]#CC(=O)[#6]",
        "[$([CH]),$(CC)]#CC(=O)O[#6]",
        "n[OH]",
    ],
    &[
        "[$([CH]),$(CC)]#CS(=O)(=O)[#6]",
        "C=C(C=O)C=O",
        "n1c([F,Cl,Br,I])cccc1",
        "[CH1](=O)",
        "[#8][#8]",
    ],
    &[
        "[C;!R]=[N;!R]",
        "[N!R]=[N!R]",
        "[#6](=O)[#6](=O)",
        "[#16][#16]",
        "[#7][NH2]",
        "C(=O)N[NH2]",
    ],
    &[
        "[#6]=S",
        "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]",
    ],
    &[
        "C1(=[O,N])C=CC(=[O,N])C=C1",
        "C1(=[O,N])C(=[O,N])C=CC=C1",
        "a21aa3a(aa1aaaa2)aaaa3",
    ],
    &[
        "a31a(a2a(aa1)aaaa2)aaaa3",
        "a1aa2a3a(a1)A=AA=A3=AA=A2",
        "c1cc([NH2])ccc1",
    ],
    &[
        "[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]",
    ],
    &["I", "OS(=O)(=O)[O-]", "[N+](=O)[O-]", "C(=O)N[OH]"],
    &[
        "C1NC(=O)NC(=O)1",
        "[SH]",
        "[S-]",
        "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]",
    ],
    &[
        "c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]",
        "[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1",
    ],
    &[
        "[CR1]1[CR1][CR1]cc[CR1][CR1]1",
        "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1",
    ],
    &[
        "[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1",
        "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    ],
    &[
        "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
        "C#C",
    ],
    &[
        "[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]",
        "[$([N+R]),$([n+R]),$([N+]=C)][O-]",
    ],
    &[
        "[#6]=N[OH]",
        "[#6]=NOC=O",
        "[#6](=O)[CX4,CR0X3,O][#6](=O)",
        "c1ccc2c(c1)ccc(=O)o2",
    ],
    &[
        "[O+,o+,S+,s+]",
        "N=C=O",
        "[NX3,NX4][F,Cl,Br,I]",
        "c1ccccc1OC(=O)[#6]",
        "[CR0]=[CR0][CR0]=[CR0]",
    ],
    &[
        "[C+,c+,C-,c-]",
        "N=[N+]=[N-]",
        "C12C(NC(N1)=O)CSC2",
        "c1c([OH])c([OH,NH2,NH])ccc1",
        "P",
    ],
    &[
        "[N,O,S]C#N",
        "C=C=O",
        "[Si][F,Cl,Br,I]",
        "[SX2]O",
        "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
    ],
    &["O1CCCCC1OC2CCC3CCCCC3C2", "N=[CR0][N,n,O,S]"],
    &[
        "[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2",
    ],
    &[
        "C=[C!r]C#N",
        "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1",
    ],
    &["[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1"],
    &[
        "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])",
        "[OH]c1ccc([OH,NH2,NH])cc1",
    ],
    &[
        "c1ccccc1OC(=O)O",
        "[SX2H0][N]",
        "c12ccccc1(SC(S)=N2)",
        "c12ccccc1(SC(=S)N2)",
        "c1nnnn1C=O",
    ],
    &[
        "s1c(S)nnc1NC=O",
        "S1C=CSC1=S",
        "C(=O)Onnn",
        "OS(=O)(=O)C(F)(F)F",
        "N#CC[OH]",
        "N#CC(=O)",
    ],
    &[
        "S(=O)(=O)C#N",
        "N[CH2]C#N",
        "C1(=O)NCC1",
        "S(=O)(=O)[O-,OH]",
        "NC[F,Cl,Br,I]",
        "C=[C!r]O",
    ],
    &[
        "[NX2+0]=[O+0]",
        "[OR0,NR0][OR0,NR0]",
        "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]",
        "[CX2R0][NX3R0]",
    ],
    &[
        "c1ccccc1[C;!R]=[C;!R]c2ccccc2",
        "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]",
    ],
    &["[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]"],
    &[
        "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
        "[*]=[N+]=[*]",
        "[SX3](=O)[O-,OH]",
    ],
    &[
        "N#N",
        "F.F.F.F",
        "[R0;D2][R0;D2][R0;D2][R0;D2]",
        "[cR,CR]~C(=O)NC(=O)~[cR,CR]",
        "C=!@CC=[O,S]",
    ],
    &[
        "[#6,#8,#16][#6](=O)O[#6]",
        "c[C;R0](=[O,S])[#6]",
        "c[SX2][C;!R]",
        "C=C=C",
    ],
    &[
        "c1nc([F,Cl,Br,I,S])ncc1",
        "c1ncnc([F,Cl,Br,I,S])c1",
        "c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])",
    ],
    &[
        "[#6]S(=O)(=O)c1ccc(cc1)F",
        "[15N]",
        "[13C]",
        "[18O]",
        "[34S]",
    ],
];

/// Pre-parsed structural alert SMARTS.
static STRUCTURAL_ALERT_PATTERNS: Lazy<Vec<ScreenedQuery>> = Lazy::new(|| {
    STRUCTURAL_ALERT_SMARTS
        .iter()
        .flat_map(|group| group.iter())
        .filter_map(|smarts| parse_smarts(smarts).ok())
        .map(|query| {
            let screen = crate::algo::iso::QueryScreen::build(&query);
            ScreenedQuery { query, screen }
        })
        .collect()
});

/// Count the number of structural alert patterns that match.
fn compute_structural_alerts(mol: &Molecule) -> u32 {
    let iso_target = IsoTarget::new(mol);
    compute_structural_alerts_target(&iso_target)
}

fn compute_structural_alerts_target(target: &IsoTarget) -> u32 {
    STRUCTURAL_ALERT_PATTERNS
        .iter()
        .filter(|entry| entry.screen.passes_target(target) && has_substruct_match_target(target, &entry.query))
        .count() as u32
}

/// QED (Quantitative Estimate of Drug-likeness).
///
/// Computes a weighted geometric mean of desirability functions for 7 molecular properties.
/// Score ranges from 0 (least drug-like) to ~1 (most drug-like).
///
/// Properties used: MW, LogP, HBA, HBD, TPSA, RotatableBonds, AromaticRings, Alerts.
///
/// This follows RDKit's QED implementation, including its dedicated acceptor
/// SMARTS set, aromatic ring counting method, and structural alerts.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct Qed;

impl Descriptor for Qed {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "QED"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Float(compute_qed_cached(mol))
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Qed>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements qed.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_qed(smiles: &str) -> f64 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        Qed.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_float()
    }

    #[test]
    /// Runs regression test for qed range.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_range() {
        let qed = calc_qed("c1ccccc1");
        assert!(qed >= 0.0 && qed <= 1.0, "QED should be 0-1, got {}", qed);
    }

    #[test]
    /// Runs regression test for qed aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_aspirin() {
        let qed = calc_qed("CC(=O)Oc1ccccc1C(=O)O");
        assert!(qed > 0.2 && qed < 0.9, "Aspirin QED={}", qed);
    }

    #[test]
    /// Runs regression test for qed caffeine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_caffeine() {
        let qed = calc_qed("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        assert!(qed > 0.1 && qed < 0.8, "Caffeine QED={}", qed);
    }

    #[test]
    /// Runs regression test for qed drug like.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_drug_like() {
        let qed = calc_qed("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
        assert!(qed > 0.2 && qed < 0.9);
    }

    #[test]
    /// Runs regression test for qed methane low.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_methane_low() {
        let qed = calc_qed("C");
        assert!(qed < 0.5, "Methane should have low QED: {}", qed);
    }

    #[test]
    /// Runs regression test for qed large molecule lower.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_large_molecule_lower() {
        let qed_small = calc_qed("c1ccc(CC(=O)O)cc1");
        let qed_large = calc_qed("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC");
        assert!(qed_large < qed_small || qed_large < 0.5);
    }

    #[test]
    /// Runs regression test for qed aromatic ring count matches rdkit style.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_aromatic_ring_count_matches_rdkit_style() {
        let benzene = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(compute_qed_aromatic_ring_count(&benzene), 1);

        let cyclohexane = parse_smiles("C1CCCCC1").unwrap();
        assert_eq!(compute_qed_aromatic_ring_count(&cyclohexane), 0);
    }

    #[test]
    /// Runs regression test for qed matches rdkit cnp0298191.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_matches_rdkit_cnp0298191() {
        let mol =
            parse_smiles("CCOCC1=C2/C(=C\\[C@@]3(C)CC[C@](O)(O3)[C@@](C)(O)C[C@H]2OC(C)=O)OC1=O")
                .unwrap();
        assert_eq!(compute_qed_hbond_acceptors(&mol), 8);
        assert_eq!(compute_qed_aromatic_ring_count(&mol), 0);
        assert_eq!(compute_structural_alerts(&mol), 1);
        assert!((compute_qed(&mol) - 0.6956330228610187).abs() < 1e-3);
    }

    #[test]
    /// Runs regression test for qed matches rdkit cnp0554779.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_matches_rdkit_cnp0554779() {
        let mol = parse_smiles("CC12C=CC(=O)C=C1CCC1C2CCC2(C)C(OC(=O)CCC3CCCC3)CCC12").unwrap();
        assert_eq!(compute_structural_alerts(&mol), 1);
        assert!((compute_qed(&mol) - 0.5200628922737999).abs() < 1e-3);
    }

    #[test]
    /// Runs regression test for qed structural alerts match rdkit cluster cases.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_structural_alerts_match_rdkit_cluster_cases() {
        // These molecules were scalar QED parity outliers caused by alert
        // undercounting in pre-screened SMARTS matching.
        let cnp0454917 = parse_smiles("CCCCCC1=CC(O)=C(CCCC)C(OS(=O)(=O)O)=C1").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0454917), 2);

        let cnp0479681 = parse_smiles("NC(N)=NCCC[C@H](N)C(=O)O").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0479681), 3);

        let cnp0538427 = parse_smiles("CCCNC(=N)NCCC[C@H](N)C(=O)O").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0538427), 3);

        let cnp0334248 =
            parse_smiles("O=C1CCC[C@@H]2[C@H]3CCC[N@@+]4([O-])CCC[C@@H](CN12)[C@@H]34").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0334248), 2);
    }

    #[test]
    fn test_qed_cnp0138771() {
        let mol = parse_smiles("COC1=CC(OC)=C2C(=O)C3=C(OC)C4=C(C=C3C(=O)C2=C1)O[C@@H](O)[C@@H]4CCOC(C)=O").unwrap();
        let iso_target = IsoTarget::new(&mol);
        let qed_arom = compute_qed_aromatic_ring_count(&mol);
        let alerts = compute_structural_alerts(&mol);
        println!("QED AROM: {}", qed_arom);
        println!("ALERTS: {}", alerts);
        
        let query = parse_smarts("[#6,#8,#16][#6](=O)O[#6]").unwrap();
        let screen = crate::algo::iso::QueryScreen::build(&query);
        println!("Passes screen: {}", screen.passes_target(&iso_target));
        println!("Has substruct match: {}", crate::algo::iso::has_substruct_match(&mol, &query));
    }

    #[test]
    fn test_anthraquinone() {
        use crate::algo::rings::find_sssr;
        let mut mol = parse_smiles("O=C1C2=CC=CC=C2C(=O)C3=CC=CC=C13").unwrap();
        let sssr = find_sssr(&mut mol);
        println!("Anthraquinone SSSR length: {}", sssr.len());
    }

    #[test]
    fn all_qed_patterns_parse() {
        let failed: Vec<_> = QED_ACCEPTOR_SMARTS
            .iter()
            .chain(STRUCTURAL_ALERT_SMARTS.iter().flat_map(|g| g.iter()))
            .filter(|s| crate::io::smarts::parse_smarts(s).is_err())
            .collect();
        assert!(failed.is_empty(), "unparseable QED SMARTS: {:?}", failed);
    }
}
