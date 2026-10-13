//! Moreau-Broto Autocorrelation (ATS) descriptors.
//!
//! ATS(d, w) = Σ w_i × w_j  for all atom pairs (i, j) at topological distance d.
//!
//! Output (621 values, `BatchDescriptor::feature_names` order):
//! - `ATSm1-5`, `ATSp1-5`, `ATSc1-5`: CDK 2.13 `AutocorrelationDescriptorMass`,
//!   `...Polarizability` and `...Charge` (lags 0..4; lag 0 is Σ w_i², lags > 0
//!   count every unordered pair once). Weights: natural mass / carbon mass,
//!   CDK Gasteiger-Hückel effective polarizability (explicit H, cdkLegacy
//!   aromaticity) and CDK Gasteiger-Marsili charges on the H-suppressed graph.
//! - The 606 columns of the Mordred (mordred-community 2.0.7)
//!   `Autocorrelation` module, in Mordred's order: `ATS{k}{p}`, `AATS{k}{p}`
//!   (k = 0..8, the 11 atomic properties without the charge `c`), then
//!   `ATSC{k}{p}`, `AATSC{k}{p}` (k = 0..8) and `MATS{k}{p}`, `GATS{k}{p}`
//!   (k = 1..8) for all 12 Mordred atomic properties (`mordred_atomic.rs`).
//!   Mordred works on the graph with every hydrogen explicit (`Chem.AddHs`),
//!   `A` atoms, `Δ_k` pairs at distance k (`Δ_0 = A`), `w̄` the mean property:
//!   `ATS_0 = Σ w_i²`, `ATS_k = Σ_{i<j, d(i,j)=k} w_i w_j`,
//!   `AATS_k = ATS_k / Δ_k`; `ATSC` / `AATSC` the same with `w - w̄`;
//!   `MATS_k = A · AATSC_k / ATSC_0`;
//!   `GATS_k = (Σ_{i<j, d=k} (w_i - w_j)² / (2 Δ_k)) / (ATSC_0 / (A - 1))`.
//!   NaN when an atom has no property value, for a division by zero, and
//!   (`GATS`) for a single atom.
//!
//! Reference: Moreau, G. & Broto, P. (1980) Nouv. J. Chim. 4, 359-360.

use super::cdk_perception::{CdkView, NATURAL_MASS};
use super::mordred_atomic::{mordred_atom_props, N_PROPS, PROP_C, PROP_NAMES};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// CDK `AutocorrelationDescriptorMass` carbon mass.
const CARBON_MASS: f64 = 12.010735896788;
/// CDK lags (descriptor names 1..5 = topological distances 0..4).
const CDK_LAGS: usize = 5;
/// Mordred lags 0..=8 (`MAX_DISTANCE = 8`).
const MORDRED_LAGS: usize = 9;
/// Mordred columns: ATS / AATS for 11 properties (no charge), ATSC / AATSC
/// for 12 (lags 0..8), MATS / GATS for 12 (lags 1..8).
pub(crate) const MORDRED_N: usize = 2 * (N_PROPS - 1) * MORDRED_LAGS + 2 * N_PROPS * MORDRED_LAGS + 2 * N_PROPS * (MORDRED_LAGS - 1);
/// CDK columns + Mordred columns.
pub(crate) const N_OUT: usize = 3 * CDK_LAGS + MORDRED_N;

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

/// Mordred lag-`k` sums over the atom pairs of the hydrogen-explicit graph.
struct LagSums {
    /// pairs at distance k (lag 0: atoms)
    count: [f64; MORDRED_LAGS],
    /// Σ w_a w_b (lag 0: Σ w_a²)
    raw: [[f64; N_PROPS]; MORDRED_LAGS],
    /// Σ (w_a - w̄)(w_b - w̄) (lag 0: Σ (w_a - w̄)²)
    centered: [[f64; N_PROPS]; MORDRED_LAGS],
    /// Σ (w_a - w_b)²
    diff: [[f64; N_PROPS]; MORDRED_LAGS],
}

/// `acc[p] += a[p] * b[p]`
#[inline(always)]
fn add_prod(acc: &mut [f64; N_PROPS], a: &[f64; N_PROPS], b: &[f64; N_PROPS]) {
    for p in 0..N_PROPS {
        acc[p] += a[p] * b[p];
    }
}

/// The two halves `(wc, huc)` of a per-atom `[wc | huc]` vector.
#[inline(always)]
fn parts(v: &[f64; 2 * N_PROPS]) -> (&[f64; N_PROPS], &[f64; N_PROPS]) {
    let (a, b) = v.split_first_chunk::<N_PROPS>().unwrap();
    (a, b.first_chunk::<N_PROPS>().unwrap())
}

/// `acc[p] += a[p] * s`
#[inline(always)]
fn add_scaled(acc: &mut [f64; N_PROPS], a: &[f64; N_PROPS], s: f64) {
    for p in 0..N_PROPS {
        acc[p] += a[p] * s;
    }
}

/// Mordred `ATS`, `AATS`, `ATSC`, `AATSC`, `MATS`, `GATS` (lags 0..8, `MATS`
/// / `GATS` 1..8) for all atomic properties, in Mordred's column order
/// (see [`mordred_feature_names`]), written to `out[..MORDRED_N]`.
///
/// Mordred works on the graph with every hydrogen explicit (`Chem.AddHs`).
/// The `h_i` hydrogens added to graph atom `i` are folded in analytically
/// instead of being materialised: an added H on `j` lies at distance
/// d(i,j) + 1 from atom `i` and at d(i,j) + 2 from an added H on `i`; it is
/// 1 bond from its own atom and 2 bonds from the other hydrogens on it.
///
/// Only the centered pair products `Σ x'_a x'_b` (`x' = x - x̄`) need a sum
/// over atom pairs per property; with `deg_k(a)` the number of atoms at
/// distance k from atom `a`, the other lag sums follow per atom:
/// `Σ_pairs x_a x_b = Σ x'_a x'_b + x̄ Σ_a x'_a deg_k(a) + x̄² Δ_k` and
/// `Σ_pairs (x_a - x_b)² = Σ_a x'_a² deg_k(a) - 2 Σ x'_a x'_b`.
fn mordred_autocorrelation(mol: &Molecule, dist: &[u16], out: &mut [f64]) {
    const L: usize = MORDRED_LAGS;
    const P: usize = N_PROPS;
    let n = mol.num_atoms();
    let props = mordred_atom_props(mol, true);
    let (w, u, h) = (&props.w, &props.u, &props.h);

    // mean over all atoms (graph atoms + added hydrogens)
    let mut total = [0.0f64; P];
    let mut n_atoms = 0.0f64;
    for i in 0..n {
        for p in 0..P {
            total[p] += w[i][p] + h[i] * u[i][p];
        }
        n_atoms += 1.0 + h[i];
    }
    let mean = total.map(|t| t / n_atoms);

    let mut s = LagSums {
        count: [0.0; L],
        raw: [[0.0; P]; L],
        centered: [[0.0; P]; L],
        diff: [[0.0; P]; L],
    };
    // centered values: wc = w - w̄ (graph atom), uc = u - w̄ (each added H),
    // huc = h · uc (all added hydrogens of the atom); `x[i] = [wc | huc]`
    let mut x: Vec<[f64; 2 * P]> = Vec::with_capacity(n);
    let mut uc = Vec::with_capacity(n);
    for i in 0..n {
        let hi = h[i];
        let mut xi = [0.0f64; 2 * P];
        let mut b = [0.0f64; P];
        for p in 0..P {
            let a = w[i][p] - mean[p];
            b[p] = u[i][p] - mean[p];
            let c = hi * b[p];
            xi[p] = a;
            xi[P + p] = c;
            // lag 0: Σ x², Σ x'²
            s.raw[0][p] += w[i][p] * w[i][p] + hi * u[i][p] * u[i][p];
            s.centered[0][p] += a * a + c * b[p];
            // each atom with its own hydrogens (lag 1) and pairs of them (lag 2)
            s.centered[1][p] += a * c;
            s.centered[2][p] += 0.5 * (hi - 1.0).max(0.0) * c * b[p];
        }
        x.push(xi);
        uc.push(b);
    }

    // `cnt[i * L + k]` / `nh[i * L + k]`: graph atoms / added hydrogens of the
    // graph atoms at distance k from atom i (both over j != i)
    let mut cnt = vec![0.0f64; n * L];
    let mut nh = vec![0.0f64; n * L];

    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        let hi = h[i];
        let mut g = [[0.0f64; 2 * P]; L];
        let mut lags = 0u32;
        // neighbour sums over j > i per distance
        for j in (i + 1)..n {
            let d = row[j] as usize;
            if d >= L {
                continue; // includes unreachable pairs
            }
            lags |= 1 << d;
            for (gp, xp) in g[d].iter_mut().zip(&x[j]) {
                *gp += xp;
            }
            cnt[i * L + d] += 1.0;
            cnt[j * L + d] += 1.0;
            nh[i * L + d] += h[j];
            nh[j * L + d] += hi;
        }
        // graph-graph pairs at d, graph-H at d + 1, H-H at d + 2
        let (wci, huci) = parts(&x[i]);
        for d in 1..L {
            if lags & (1 << d) == 0 {
                continue;
            }
            let (gw, gu) = parts(&g[d]);
            add_prod(&mut s.centered[d], wci, gw);
            if d + 1 < L {
                add_prod(&mut s.centered[d + 1], wci, gu);
                add_prod(&mut s.centered[d + 1], huci, gw);
            }
            if d + 2 < L && hi > 0.0 {
                add_prod(&mut s.centered[d + 2], huci, gu);
            }
        }
    }

    // per atom: Σ x' deg_k and Σ x'² deg_k; Δ_k = Σ deg_k / 2
    let mut lin = [[0.0f64; P]; L];
    let mut sq = [[0.0f64; P]; L];
    let mut deg_sum = [0.0f64; L];
    for i in 0..n {
        let hi = h[i];
        let at = |v: &[f64], k: usize| if k < L { v[i * L + k] } else { 0.0 };
        let (wci, huci) = parts(&x[i]);
        let mut wc2 = [0.0f64; P];
        let mut huc2 = [0.0f64; P];
        for p in 0..P {
            wc2[p] = wci[p] * wci[p];
            huc2[p] = huci[p] * uc[i][p];
        }
        for k in 1..L {
            // the graph atom: graph atoms at k, hydrogens of graph atoms at k - 1, own hydrogens
            let deg_g = at(&cnt, k) + at(&nh, k - 1) + if k == 1 { hi } else { 0.0 };
            // each added hydrogen: graph atoms at k - 1, their hydrogens at k - 2, parent, siblings
            let deg_h = at(&cnt, k - 1)
                + if k >= 2 { at(&nh, k - 2) } else { 0.0 }
                + match k {
                    1 => 1.0,
                    2 => hi - 1.0,
                    _ => 0.0,
                };
            if hi == 0.0 {
                if deg_g == 0.0 {
                    continue;
                }
                deg_sum[k] += deg_g;
                add_scaled(&mut lin[k], wci, deg_g);
                add_scaled(&mut sq[k], &wc2, deg_g);
            } else {
                deg_sum[k] += deg_g + hi * deg_h;
                for p in 0..P {
                    lin[k][p] += wci[p] * deg_g + huci[p] * deg_h;
                    sq[k][p] += wc2[p] * deg_g + huc2[p] * deg_h;
                }
            }
        }
    }
    s.count[0] = n_atoms;
    for k in 1..L {
        s.count[k] = 0.5 * deg_sum[k];
        for p in 0..P {
            let m = mean[p];
            s.raw[k][p] = s.centered[k][p] + m * lin[k][p] + m * m * s.count[k];
            // clamped: cancellation must not turn an exact 0 negative
            s.diff[k][p] = (sq[k][p] - 2.0 * s.centered[k][p]).max(0.0);
        }
    }

    // ATS / AATS (no charge), ATSC / AATSC (lags 0..8), MATS / GATS (lags 1..8)
    let div = |a: f64, b: f64| if b == 0.0 { f64::NAN } else { a / b };
    let ats_props = (0..P).filter(|&p| p != PROP_C);
    let mut o = 0;
    let mut put = |v: f64| {
        out[o] = v;
        o += 1;
    };
    for p in ats_props.clone() {
        (0..L).for_each(|k| put(if props.valid[p] { s.raw[k][p] } else { f64::NAN }));
    }
    for p in ats_props {
        (0..L).for_each(|k| put(if props.valid[p] { div(s.raw[k][p], s.count[k]) } else { f64::NAN }));
    }
    for p in 0..P {
        (0..L).for_each(|k| put(if props.valid[p] { s.centered[k][p] } else { f64::NAN }));
    }
    for p in 0..P {
        (0..L).for_each(|k| put(if props.valid[p] { div(s.centered[k][p], s.count[k]) } else { f64::NAN }));
    }
    // MATS_k = A · AATSC_k / Σ (w - w̄)²
    for p in 0..P {
        for k in 1..L {
            let aatsc = div(s.centered[k][p], s.count[k]);
            put(if props.valid[p] { div(n_atoms * aatsc, s.centered[0][p]) } else { f64::NAN });
        }
    }
    // GATS_k = (Σ_{a,b} B_ab (w_a - w_b)² / (4 Δ_k)) / (Σ (w - w̄)² / (A - 1)),
    // the double sum running over ordered pairs (2× the unordered sum)
    for p in 0..P {
        for k in 1..L {
            put(if props.valid[p] && n_atoms > 1.0 {
                let num = div(2.0 * s.diff[k][p], 4.0 * s.count[k]);
                div(num, s.centered[0][p] / (n_atoms - 1.0))
            } else {
                f64::NAN
            });
        }
    }
    debug_assert_eq!(o, MORDRED_N);
}

/// Mordred autocorrelation column names, in Mordred's order.
pub(crate) fn mordred_feature_names() -> Vec<String> {
    let mut v = Vec::with_capacity(MORDRED_N);
    let ats_props = || PROP_NAMES.iter().filter(|&&p| p != PROP_NAMES[PROP_C]);
    for kind in ["ATS", "AATS"] {
        for p in ats_props() {
            v.extend((0..MORDRED_LAGS).map(|k| format!("{kind}{k}{p}")));
        }
    }
    for kind in ["ATSC", "AATSC"] {
        for p in PROP_NAMES {
            v.extend((0..MORDRED_LAGS).map(|k| format!("{kind}{k}{p}")));
        }
    }
    for kind in ["MATS", "GATS"] {
        for p in PROP_NAMES {
            v.extend((1..MORDRED_LAGS).map(|k| format!("{kind}{k}{p}")));
        }
    }
    v
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
        if mol.num_atoms() == 0 {
            mordred_autocorrelation(mol, &[], &mut values[3 * CDK_LAGS..]);
            return DescriptorValue::FloatVec(values);
        }
        // Mordred: graph as parsed, hydrogens added analytically
        let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
        mordred_autocorrelation(mol, dist, &mut values[3 * CDK_LAGS..]);
        CdkView::with_legacy_cached(mol, ctx, |view, ctx| cdk_autocorrelation(mol, view, ctx, &mut values));
        // CDK's ATSm/ATSc run on `removeHydrogens(container)` (ATSp keeps the
        // graph hydrogens)
        if let Some(suppressed) = crate::algo::sanitize::cdk_remove_hydrogens(mol) {
            let mut cdk = [0.0f64; 3 * CDK_LAGS];
            CdkView::with_legacy_cached(&suppressed, ctx, |view, ctx| {
                cdk_autocorrelation(&suppressed, view, ctx, &mut cdk)
            });
            values[..CDK_LAGS].copy_from_slice(&cdk[..CDK_LAGS]);
            values[2 * CDK_LAGS..3 * CDK_LAGS].copy_from_slice(&cdk[2 * CDK_LAGS..3 * CDK_LAGS]);
        }
        DescriptorValue::FloatVec(values)
    }
}

/// The CDK values `ATSm1-5, ATSp1-5, ATSc1-5` (`values[..15]`); `view`: CDK
/// view with cdkLegacy aromaticity.
fn cdk_autocorrelation(mol: &Molecule, view: &CdkView, ctx: &mut crate::desc::context::CalculationContext, values: &mut [f64]) {
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
        values[k] = ats_m[k];
        values[CDK_LAGS + k] = ats_p[k];
        values[2 * CDK_LAGS + k] = ats_c[k];
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

    fn calc_ats(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        match Autocorrelation.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-4 + 1e-5 * b.abs()
    }

    #[test]
    fn test_ats_size() {
        assert_eq!(calc_ats("CC").len(), N_OUT);
        assert_eq!(MORDRED_N, 606);
        assert_eq!(mordred_feature_names().len(), MORDRED_N);
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

    /// Mordred column `name` of `values` (as returned by [`calc_ats`]).
    fn mordred_value(values: &[f64], name: &str) -> f64 {
        let k = mordred_feature_names().iter().position(|n| n == name).unwrap_or_else(|| panic!("no column {name}"));
        values[3 * CDK_LAGS + k]
    }

    fn assert_close_or_nan(smiles: &str, name: &str, got: f64, e: f64) {
        if e.is_nan() {
            assert!(got.is_nan(), "{smiles} {name} = {got}, expected NaN");
        } else {
            assert!((got - e).abs() <= 1e-10 * e.abs().max(1.0), "{smiles} {name} = {got} expected {e}");
        }
    }

    fn assert_mordred(smiles: &str, pe: [f64; 9], v: [f64; 9]) {
        let a = calc_ats(smiles);
        for (k, (e_pe, e_v)) in pe.into_iter().zip(v).enumerate() {
            for (name, e) in [(format!("ATS{k}pe"), e_pe), (format!("ATS{k}v"), e_v)] {
                assert_close_or_nan(smiles, &name, mordred_value(&a, &name), e);
            }
        }
    }

    /// `(column, Mordred value)` pairs for one molecule.
    fn assert_mordred_columns(smiles: &str, expected: &[(&str, f64)]) {
        let a = calc_ats(smiles);
        for &(name, e) in expected {
            assert_close_or_nan(smiles, name, mordred_value(&a, name), e);
        }
    }

    // Expected values: Mordred (mordred-community 2.0.7) (tests/data/mordred_parity_reference.csv).

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

    #[test]
    fn test_mordred_averaged_centered_moran_geary() {
        assert_mordred_columns(
            "CCO",
            &[
                ("ATS1dv", 12.0),
                ("AATS0Z", 15.777777777777779),
                ("ATSC0c", 0.21303866453856574),
                ("ATSC1c", -0.09961946882998138),
                ("AATSC2m", -8.989525386514718),
                ("MATS1dv", -0.060679611650485445),
                ("GATS2s", 0.9849325931800158),
                ("GATS1i", 1.929307462277428),
            ],
        );
        // hydrogen atoms kept in the graph by RDKit
        assert_mordred_columns(
            "[2H]C([2H])([2H])Cl",
            &[("ATS0s", 8.810852004267643), ("MATS1s", -0.8816615681854771), ("GATS1s", 2.2633231363709547)],
        );
    }

    #[test]
    fn test_mordred_missing_values() {
        // methane: carbon without heavy neighbours has no intrinsic state
        assert_mordred_columns("C", &[("ATS0s", f64::NAN), ("MATS1s", f64::NAN), ("MATS1Z", -1.0)]);
        // single atom: no pairs (AATS / MATS NaN) and GATS undefined
        assert_mordred_columns(
            "[Na+]",
            &[("ATSC0m", 0.0), ("AATS1m", f64::NAN), ("MATS1m", f64::NAN), ("GATS1m", f64::NAN)],
        );
    }
}
