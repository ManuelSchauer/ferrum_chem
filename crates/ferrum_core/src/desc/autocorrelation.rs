//! Moreau-Broto Autocorrelation (ATS) descriptors.
//!
//! ATS(d, w) = Σ w_i × w_j  for all atom pairs (i, j) at topological distance d.
//!
//! Output (33 values, `BatchDescriptor::feature_names` order):
//! - `ATSm1-5`, `ATSp1-5`, `ATSc1-5`: CDK 2.13 `AutocorrelationDescriptorMass`,
//!   `...Polarizability` and `...Charge` (lags 0..4; lag 0 is Σ w_i², lags > 0
//!   count every unordered pair once). Weights: natural mass / carbon mass,
//!   CDK Gasteiger-Hückel effective polarizability (explicit H, cdkLegacy
//!   aromaticity) and CDK Gasteiger-Marsili charges on the H-suppressed graph.
//! - `ATS0pe..ATS8pe`, `ATS0v..ATS8v`: Mordred 1.2.0 `ATS(order, prop)` with
//!   `pe` = Pauling electronegativity and `v` = van der Waals volume (Mordred
//!   atomic property tables, `mordred_data.rs`). Mordred works on the graph
//!   with every hydrogen explicit (`Chem.AddHs`); lag 0 = Σ w_i², lag k =
//!   Σ_{i<j, d(i,j)=k} w_i w_j. NaN for a property when an atom has no value.
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
//! - `fn cdk_ats` / `fn mordred_ats`
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

use super::cdk_perception::{CdkView, NATURAL_MASS};
use super::mordred_data::{MORDRED_PAULING_EN, MORDRED_VDW_VOLUME};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// CDK `AutocorrelationDescriptorMass` carbon mass.
const CARBON_MASS: f64 = 12.010735896788;
/// CDK lags (descriptor names 1..5 = topological distances 0..4).
const CDK_LAGS: usize = 5;
/// Mordred lags 0..=8 (`MAX_DISTANCE = 8`).
const MORDRED_LAGS: usize = 9;
const N_OUT: usize = 3 * CDK_LAGS + 2 * MORDRED_LAGS;

/// CDK autocorrelation: lag 0 = Σ_i w_i², lag k = Σ_{i<j, d(i,j)=k} w_i w_j.
/// Atoms with `skip[i]` (hydrogens in the polarizability variant) are ignored.
fn cdk_ats(n: usize, dist: &[u16], w: &[f64], skip: Option<&[bool]>) -> [f64; CDK_LAGS] {
    let mut ats = [0.0f64; CDK_LAGS];
    let skipped = |i: usize| skip.is_some_and(|s| s[i]);
    for i in 0..n {
        if skipped(i) {
            continue;
        }
        ats[0] += w[i] * w[i];
        let row = &dist[i * n..(i + 1) * n];
        for j in (i + 1)..n {
            let d = row[j] as usize;
            if d < CDK_LAGS && !skipped(j) {
                ats[d] += w[i] * w[j];
            }
        }
    }
    ats
}

/// Mordred ATS (lags 0..8) for `pe` and `v` on the hydrogen-explicit graph
/// (`Chem.AddHs`), written to `out[0..9]` (pe) and `out[9..18]` (v).
///
/// The `h_i` implicit/explicit hydrogens Mordred adds to graph atom `i` are
/// folded in analytically instead of being materialised: an added H on `j`
/// lies at distance d(i,j) + 1 from atom `i` and at d(i,j) + 2 from an added
/// H on `i`; two hydrogens on the same atom are 2 bonds apart.
fn mordred_ats(mol: &Molecule, dist: &[u16], out: &mut [f64]) {
    const L: usize = MORDRED_LAGS;
    let n = mol.num_atoms();
    let tables = [&MORDRED_PAULING_EN, &MORDRED_VDW_VOLUME];
    // Per atom: [w_pe, w_v, h].
    let mut atoms = Vec::with_capacity(n);
    let mut nan = [false; 2];
    let mut n_h = 0.0f64;
    let mut sq = [0.0f64; 2];
    let mut self_h = [0.0f64; 2]; // Σ w_i h_i (added H bonded to its own atom)
    let mut same_atom_hh = 0.0f64; // Σ C(h_i, 2)
    for a in &mol.atoms {
        let h = a.total_h() as f64;
        let mut w = [0.0f64; 2];
        for p in 0..2 {
            w[p] = tables[p].get(a.atomic_number as usize).copied().unwrap_or(f64::NAN);
            nan[p] |= w[p].is_nan();
            sq[p] += w[p] * w[p];
            self_h[p] += w[p] * h;
        }
        n_h += h;
        same_atom_hh += h * (h - 1.0) * 0.5;
        atoms.push((w, h));
    }

    // heavy-heavy products, Σ w_i h_j (heavy-H) and Σ h_i h_j (H-H) per lag.
    let mut hh = [[0.0f64; L]; 2];
    let mut wh = [[0.0f64; L]; 2];
    let mut hcount = [0.0f64; L];
    for i in 0..n {
        let (wi, hi) = atoms[i];
        let row = &dist[i * n..(i + 1) * n];
        for j in (i + 1)..n {
            let d = row[j] as usize;
            if d >= L {
                continue; // includes unreachable pairs
            }
            let (wj, hj) = atoms[j];
            for p in 0..2 {
                hh[p][d] += wi[p] * wj[p];
            }
            if d + 1 < L && (hi > 0.0 || hj > 0.0) {
                for p in 0..2 {
                    wh[p][d + 1] += wi[p] * hj + wj[p] * hi;
                }
                if d + 2 < L {
                    hcount[d + 2] += hi * hj;
                }
            }
        }
    }
    wh[0][1] += self_h[0];
    wh[1][1] += self_h[1];
    hcount[2] += same_atom_hh;

    for p in 0..2 {
        let o = &mut out[p * L..(p + 1) * L];
        if nan[p] {
            o.fill(f64::NAN);
            continue;
        }
        let w_h = tables[p][1];
        o[0] = sq[p] + n_h * w_h * w_h;
        for k in 1..L {
            o[k] = hh[p][k] + w_h * wh[p][k] + w_h * w_h * hcount[k];
        }
    }
}

// ===========================================================================
// Typed API: Autocorrelation
// ===========================================================================

pub struct Autocorrelation;

impl Descriptor for Autocorrelation {
    fn name(&self) -> &str {
        "Autocorrelation"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = vec![0.0f64; N_OUT];
        let n = mol.num_atoms();
        if n == 0 {
            return DescriptorValue::FloatVec(values);
        }
        CdkView::with_legacy_cached(mol, ctx, |view, ctx| autocorrelation_values(mol, view, ctx, &mut values));
        // CDK's ATSm/ATSc run on `removeHydrogens(container)` (ATSp and the
        // Mordred values keep the graph hydrogens)
        if let Some(suppressed) = crate::algo::sanitize::cdk_remove_hydrogens(mol) {
            let mut cdk = vec![0.0f64; N_OUT];
            CdkView::with_legacy_cached(&suppressed, ctx, |view, ctx| {
                autocorrelation_values(&suppressed, view, ctx, &mut cdk)
            });
            values[..CDK_LAGS].copy_from_slice(&cdk[..CDK_LAGS]);
            values[2 * CDK_LAGS..3 * CDK_LAGS].copy_from_slice(&cdk[2 * CDK_LAGS..3 * CDK_LAGS]);
        }
        DescriptorValue::FloatVec(values)
    }
}

/// All `N_OUT` autocorrelation values; `view`: CDK view with cdkLegacy aromaticity.
fn autocorrelation_values(mol: &Molecule, view: &CdkView, ctx: &mut crate::desc::context::CalculationContext, values: &mut [f64]) {
    let n = mol.num_atoms();
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);

    // ATSm: natural mass scaled by carbon.
    let w_mass: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| {
            NATURAL_MASS
                .get(a.atomic_number as usize)
                .copied()
                .unwrap_or(f64::NAN)
                / CARBON_MASS
        })
        .collect();
    let ats_m = cdk_ats(n, dist, &w_mass, None);

    // ATSp: effective polarizability (explicit H, cdkLegacy aromaticity), heavy atoms only.
    let w_pol = super::cdk_polarizability::effective_polarizabilities(mol, view, dist, None);
    let is_h: Vec<bool> = mol.atoms.iter().map(|a| a.atomic_number == 1).collect();
    let ats_p = cdk_ats(n, dist, &w_pol, Some(&is_h));

    // ATSc: Gasteiger-Marsili charges on the hydrogen-suppressed graph.
    let ats_c = match super::cdk_charges::charges_implicit_h(mol, view) {
        Some(q) => cdk_ats(n, dist, &q, None),
        None => [f64::NAN; CDK_LAGS],
    };

    for k in 0..CDK_LAGS {
        values[k] = ats_m[k] as f64;
        values[CDK_LAGS + k] = ats_p[k] as f64;
        values[2 * CDK_LAGS + k] = ats_c[k] as f64;
    }

    // Mordred ATS{0..8}pe / ATS{0..8}v.
    mordred_ats(mol, dist, &mut values[3 * CDK_LAGS..]);
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Autocorrelation>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_ats(smiles: &str) -> Vec<f64> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match Autocorrelation.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a as f64 - b).abs() <= 1e-4 + 1e-5 * b.abs()
    }

    #[test]
    fn test_ats_size() {
        assert_eq!(calc_ats("CC").len(), N_OUT);
        assert_eq!(N_OUT, 33);
    }

    // Expected values: CDK 2.13 (tests/data/cdk_parity_reference.csv).

    #[test]
    fn test_ats_benzene_cdk() {
        let a = calc_ats("c1ccccc1");
        // ATSm1..5 = 6, 6, 6, 3, 0
        for (k, e) in [6.0, 6.0, 6.0, 3.0, 0.0].into_iter().enumerate() {
            assert!(close(a[k], e), "ATSm{} = {}", k + 1, a[k]);
        }
        // ATSp1 = 6 * 6.243375^2 (CDK effective polarizability of an aromatic CH)
        assert!(close(a[5], 6.0 * 6.243375f64.powi(2)), "ATSp1 = {}", a[5]);
        // ATSc: all-carbon ring without hydrogens in the graph -> zero charges
        for k in 10..15 {
            assert!(a[k].abs() < 1e-6, "ATSc{} = {}", k - 9, a[k]);
        }
    }

    #[test]
    fn test_ats_salt_cdk() {
        let a = calc_ats("[Na+].[Cl-]");
        assert!(close(a[0], 12.376734996848878), "ATSm1 = {}", a[0]);
        assert!(close(a[5], 5.489649), "ATSp1 = {}", a[5]);
        assert!(
            a[10].is_nan(),
            "ATSc1 for Na must be NaN (no CDK parameters)"
        );
    }

    #[test]
    fn test_ats_hydrogen_molecule_cdk() {
        let a = calc_ats("[H][H]");
        assert!(close(a[0], 0.014085127136756317), "ATSm1 = {}", a[0]);
        assert!(close(a[1], 0.007042563568378159), "ATSm2 = {}", a[1]);
        assert_eq!(a[5], 0.0, "ATSp skips hydrogens");
    }

    #[test]
    fn test_ats_single_atom() {
        let a = calc_ats("C");
        assert!(close(a[0], 1.0));
    }

    fn assert_mordred(smiles: &str, pe: [f64; 9], v: [f64; 9]) {
        let a = calc_ats(smiles);
        for (k, e) in pe.into_iter().chain(v).enumerate() {
            let got = a[15 + k];
            if e.is_nan() {
                assert!(got.is_nan(), "{smiles} ATS[{}] = {got}, expected NaN", 15 + k);
            } else {
                assert!((got - e).abs() <= 1e-12 * e.abs().max(1.0), "{smiles} ATS[{}] = {got} expected {e}", 15 + k);
            }
        }
    }

    // Expected values: Mordred 1.2.0 (tests/data/mordred_parity_reference.csv).

    #[test]
    fn test_ats_mordred_ethanol() {
        assert_mordred(
            "CCO",
            [53.878600000000006, 50.8925, 76.92800000000001, 67.03400000000002, 14.520000000000001, 0.0, 0.0, 0.0, 0.0],
            [
                1249.927047048598, 1381.9431163032111, 1279.5114392735877, 609.4474636943611, 93.25123329279079,
                0.0, 0.0, 0.0, 0.0,
            ],
        );
    }

    #[test]
    fn test_ats_mordred_benzene() {
        assert_mordred(
            "c1ccccc1",
            [68.055, 72.675, 106.33500000000001, 115.8675, 62.7, 14.520000000000003, 0.0, 0.0, 0.0],
            [
                2727.6038770815603, 3229.52110871909, 3917.9408069422016, 2833.8925682797944, 874.9221648086933,
                93.25123329279079, 0.0, 0.0, 0.0,
            ],
        );
    }

    #[test]
    fn test_ats_mordred_kept_hydrogens_and_fragments() {
        assert_mordred(
            "[2H]C([2H])([2H])Cl",
            [31.008100000000002, 24.887999999999998, 35.376000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1020.7390934552113, 806.2057569708708, 468.7345754630015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_mordred(
            "C(=O)[O-].[Na+]",
            [35.8746, 23.153999999999996, 26.969600000000003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3288.0516275955556, 720.1956205221612, 380.4180382152906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn test_ats_mordred_missing_property_is_nan() {
        // He has no Pauling electronegativity in Mordred's table
        let mut v = [0.0; 9];
        v[0] = 132.11296292224125;
        assert_mordred("[He]", [f64::NAN; 9], v);
    }
}
