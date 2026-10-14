//! BCUT (Burden-CAS-UT) descriptors.
//!
//! Modified Burden adjacency matrix with atomic property weights on the
//! diagonal. The highest and lowest eigenvalues capture molecular shape
//! and property extremes.
//!
//! Output (30 values, `BatchDescriptor::feature_names` order):
//! - `BCUTw-1l/h`, `BCUTc-1l/h`, `BCUTp-1l/h`: CDK 2.13 `BCUTDescriptor`
//!   (nhigh = nlow = 1, checkAromaticity). Hydrogen-suppressed Burden matrix:
//!   diagonal = major-isotope mass / Gasteiger-Marsili charge (computed with
//!   explicit H) / Gasteiger-Hückel effective polarizability; off-diagonal
//!   0.15 for cdkLegacy-aromatic bonds, else 0.1/0.2/0.3 by Kekulé order,
//!   +0.01 if a bond atom is terminal, 0.001 for non-bonded pairs.
//! - The 24 columns of the Mordred (mordred-community 2.0.7) `BCUT` module,
//!   `BCUT{p}-1h`, `BCUT{p}-1l` (highest / lowest eigenvalue) for the 12
//!   Mordred atomic properties `p` in Mordred's order (`mordred_atomic.rs`;
//!   `BCUTc-1h/l` and `BCUTp-1h/l` are output as `Mordred_BCUTc-1h` etc.,
//!   the plain names being the CDK columns). Burden matrix of the
//!   hydrogen-suppressed RDKit graph (hydrogen atoms kept by RDKit stay):
//!   off-diagonal `GetBondTypeAsDouble() / 10` (aromatic 0.15), `+ 0.01` if
//!   either bond atom has degree 1, 0.001 for non-bonded pairs. Mordred
//!   requires a connected molecule (NaN otherwise) and reports NaN when an
//!   atom has no property value.
//!
//! Reference: Pearlman, R.S. & Smith, K.M. (1999) "Metric Validation and
//!            the Receptor-Relevant Subspace Concept", JCICS 39, 28-35.
//!
//! References:
//! - Burden, F. (1989). Molecular identification number for substructure searches (Burden eigenvalues).
//! - Pearlman et al. BCUT descriptor literature for weighted adjacency eigenvalue descriptors.
//! - Moriwaki et al. (2018) Mordred: a molecular descriptor calculator, J. Cheminform. 10, 4.

use super::cdk_perception::{CdkView, MAJOR_ISOTOPE_MASS};
use super::mordred_atomic::{mordred_atom_props, N_PROPS, PROP_NAMES};
use super::typed::{Descriptor, DescriptorValue};
use crate::algo::linalg::BurdenMatrix;
use crate::algo::paths::DIST_INF;
use crate::mol::molecule::Molecule;
use crate::types::BondType;

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

/// Off-diagonal entries `(i, j, weight)` of the bonded pairs (0.001 elsewhere).
type BondEntries = Vec<(u32, u32, f64)>;
/// CDK: bond entries and the mass / charge / polarizability diagonals.
type CdkMatrices = (BondEntries, [Vec<f64>; 3]);
/// Mordred: bond entries and the diagonal of every atomic property (`None`
/// where Mordred reports the property as missing).
type MordredMatrices = (BondEntries, Vec<Option<Vec<f64>>>);
/// CDK columns (`BCUTw/c/p-1l/h`).
const CDK_N: usize = 6;
/// Mordred columns (`BCUT{p}-1h/l` for the 12 atomic properties).
const MORDRED_N: usize = 2 * N_PROPS;
/// All BCUT columns.
pub(crate) const N_OUT: usize = CDK_N + MORDRED_N;

/// CDK `BCUTDescriptor` Burden matrices: off-diagonal template and the
/// mass / charge / polarizability diagonals, or the final values where CDK
/// reports an error (NaN).
fn cdk_bcut_matrices(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> Result<CdkMatrices, [f64; 6]> {
    let n = mol.num_atoms();
    // Explicit hydrogen atoms: CDK fails (size mismatch) or has no heavy atoms.
    if n == 0 || mol.atoms.iter().any(|a| a.atomic_number == 1) {
        return Err([f64::NAN; 6]);
    }
    CdkView::with_legacy_cached(mol, ctx, |view, ctx| cdk_bcut_matrices_with(mol, view, ctx))
}

/// [`cdk_bcut_matrices`] given the CDK view (cdkLegacy aromaticity perceived;
/// the charges do not depend on the aromaticity flags).
fn cdk_bcut_matrices_with(
    mol: &Molecule,
    view: &CdkView,
    ctx: &mut crate::desc::context::CalculationContext,
) -> Result<CdkMatrices, [f64; 6]> {
    const NAN6: [f64; 6] = [f64::NAN; 6];
    // LonePairElectronChecker needs a typed atom with a lone pair count.
    if view.lp_typed.iter().any(|&t| !t) {
        return Err(NAN6);
    }
    let Some((charges, _)) = super::cdk_charges::charges_explicit_h(mol, view) else {
        return Err(NAN6);
    };
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
    let pol = super::cdk_polarizability::effective_polarizabilities(mol, view, dist, Some(&charges));
    let mass: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| MAJOR_ISOTOPE_MASS.get(a.atomic_number as usize).copied().unwrap_or(f64::NAN))
        .collect();
    if [&mass, &charges, &pol].iter().any(|diag| diag.iter().any(|v| !v.is_finite())) {
        return Err(NAN6);
    }

    // Off-diagonal entries of bonded pairs (0.001 elsewhere).
    let mut entries = Vec::with_capacity(mol.bonds.len());
    for (bi, bond) in mol.bonds.iter().enumerate() {
        let (i, j) = (bond.source as usize, bond.target as usize);
        let mut v = if view.arom_bond[bi] {
            0.15
        } else {
            match view.order[bi] {
                1 => 0.1,
                2 => 0.2,
                3 => 0.3,
                _ => 0.001,
            }
        };
        if mol.adjacency[i].len() == 1 || mol.adjacency[j].len() == 1 {
            v += 0.01;
        }
        entries.push((i as u32, j as u32, v));
    }
    Ok((entries, [mass, charges, pol]))
}

/// Mordred `BCUT` Burden matrix template and the atomic property diagonals
/// (`None` where Mordred reports a missing value), or `None` overall when
/// Mordred fails for the molecule.
fn mordred_bcut_matrices(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> Option<MordredMatrices> {
    let n = mol.num_atoms();
    // require_connected: Mordred counts RDKit fragments (an empty molecule has 0).
    if n == 0 {
        return None;
    }
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
    if dist[..n].contains(&DIST_INF) {
        return None;
    }

    // Mordred `Burden`: bond weight = GetBondTypeAsDouble() / 10, + 0.01 when
    // either atom is terminal.
    let mut entries = Vec::with_capacity(mol.bonds.len());
    for bond in &mol.bonds {
        let (i, j) = (bond.source as usize, bond.target as usize);
        let bo = match bond.bond_type {
            BondType::Single | BondType::Dative => 1.0,
            BondType::Double => 2.0,
            BondType::Triple => 3.0,
            BondType::Aromatic => 1.5,
            // RDKit raises for bond types without an order -> Mordred error.
            BondType::Unknown => return None,
        };
        let mut w = bo / 10.0;
        if mol.adjacency[i].len() == 1 || mol.adjacency[j].len() == 1 {
            w += 0.01;
        }
        entries.push((i as u32, j as u32, w));
    }
    let props = mordred_atom_props(mol, false);
    let diags = (0..N_PROPS)
        .map(|p| props.valid[p].then(|| props.w.iter().map(|w| w[p]).collect()))
        .collect();
    Some((entries, diags))
}

/// The BCUT values: CDK `[w_lo, w_hi, c_lo, c_hi, p_lo, p_hi]` (`cdk`) then
/// Mordred `[c_hi, c_lo, dv_hi, dv_lo, ...]` (`mordred`; NaN where not
/// requested). All Burden matrices share one batched eigen-solve.
fn bcut_values(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext, cdk: bool, mordred: bool) -> [f64; N_OUT] {
    let n = mol.num_atoms();
    let mut out = [f64::NAN; N_OUT];
    let cdk = if cdk { Some(cdk_bcut_matrices(mol, ctx)) } else { None };
    let mordred = if mordred { mordred_bcut_matrices(mol, ctx) } else { None };
    let mut jobs: Vec<BurdenMatrix> = Vec::with_capacity(3 + N_PROPS);
    // (low, high) output slots of each job
    let mut slots: Vec<(usize, usize)> = Vec::with_capacity(3 + N_PROPS);
    match &cdk {
        None => {}
        Some(Err(v)) => out[..CDK_N].copy_from_slice(v),
        Some(Ok((entries, diags))) => {
            for (k, diag) in diags.iter().enumerate() {
                jobs.push(BurdenMatrix { fill: 0.001, entries, diag });
                slots.push((2 * k, 2 * k + 1));
            }
        }
    }
    if let Some((entries, diags)) = &mordred {
        for (k, diag) in diags.iter().enumerate() {
            if let Some(diag) = diag {
                jobs.push(BurdenMatrix { fill: 0.001, entries, diag });
                slots.push((CDK_N + 2 * k + 1, CDK_N + 2 * k));
            }
        }
    }
    for (&(lo_slot, hi_slot), (lo, hi)) in slots.iter().zip(crate::algo::linalg::burden_extremes(n, &jobs)) {
        out[lo_slot] = lo;
        out[hi_slot] = hi;
    }
    out
}

/// BCUT column names (CDK, then Mordred in Mordred's order with the
/// [`super::mordred_names`] prefix for names CDK already uses).
pub(crate) fn feature_names() -> Vec<String> {
    let mut v: Vec<String> =
        ["BCUTw-1l", "BCUTw-1h", "BCUTc-1l", "BCUTc-1h", "BCUTp-1l", "BCUTp-1h"].map(String::from).to_vec();
    for p in PROP_NAMES {
        for end in ["h", "l"] {
            v.push(super::mordred_names::ferrum_column(&format!("BCUT{p}-1{end}")));
        }
    }
    v
}

/// BCUT descriptors.
///
/// Returns FloatVec with 30 values (see [`feature_names`]): CDK
/// `[BCUTw-1l, BCUTw-1h, BCUTc-1l, BCUTc-1h, BCUTp-1l, BCUTp-1h]`, then the
/// Mordred `BCUT{p}-1h`, `BCUT{p}-1l` for the 12 Mordred atomic properties.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct BCUT;

impl Descriptor for BCUT {
    fn name(&self) -> &str {
        "BCUT"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        if mol.num_atoms() == 0 {
            return DescriptorValue::FloatVec(vec![0.0f64; N_OUT]);
        }
        // CDK computes on `removeHydrogens(container)`; Mordred keeps the
        // hydrogens RDKit keeps
        let suppressed = crate::algo::sanitize::cdk_remove_hydrogens(mol);
        let mut values = bcut_values(mol, ctx, suppressed.is_none(), true);
        if let Some(suppressed) = suppressed {
            values[..CDK_N].copy_from_slice(&bcut_values(&suppressed, ctx, true, false)[..CDK_N]);
        }
        DescriptorValue::FloatVec(values.to_vec())
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BCUT>;
};

#[cfg(test)]
mod tests {
    use super::super::typed::Descriptor;
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_bcut(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        match BCUT.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    fn test_bcut_len() {
        assert_eq!(calc_bcut("CC").len(), N_OUT);
        assert_eq!(feature_names().len(), N_OUT);
    }

    #[test]
    fn test_bcut_ethanol_different_from_ethane() {
        assert_ne!(calc_bcut("CC"), calc_bcut("CCO"));
    }

    #[test]
    fn test_bcut_high_ge_low() {
        let b = calc_bcut("c1ccccc1");
        for i in 0..3 {
            assert!(b[i * 2 + 1] >= b[i * 2], "prop {i}: high {} >= low {}", b[i * 2 + 1], b[i * 2]);
        }
        for i in 0..N_PROPS {
            let (hi, lo) = (b[CDK_N + 2 * i], b[CDK_N + 2 * i + 1]);
            assert!(hi >= lo, "Mordred prop {i}: high {hi} >= low {lo}");
        }
    }

    /// CDK 2.13 retypes with explicit hydrogens: the untyped `[SH](=O)` becomes
    /// `S.sp3d1`, `[Cl+3]` is `Cl.perchlorate.charged`, `[I+]` is `I.plus.sp3`,
    /// the alkyne carbons of the dehydroannulene are `C.sp`; `[PH](C)(C)C`
    /// stays untyped and `LonePairElectronChecker` fails (NaN).
    #[test]
    fn test_bcut_cdk_retyping() {
        let cases: [(&str, [f64; 6]); 5] = [
            (
                "C1=CC=C2C(=C1)[I+]C1=CC=CC=C12",
                [11.67781017337252, 126.90464715606052, -0.3542054271098595, 0.323929598100913, 7.622228139047258, 13.78925233674102],
            ),
            (
                "C1#CC2=C(C#CC=C2)C=C1",
                [11.628379356686397, 12.37898649199995, -0.41234744585073657, 0.3585545104557385, 6.630346592244038, 9.372139439965316],
            ),
            (
                "CC(=O)O[C@H]1[C@@H](O[S@H](=O)C2=CC=C(C)C=C2)CCN1OC(=O)CC1=CC=CC=C1",
                [11.693562735220056, 31.97595890668967, -0.38923223465672974, 0.4012065402736136, 4.12930941812046, 11.304577612430801],
            ),
            (
                "C1CCN2CC3CC(CN4CCCC[C@@H]34)[C@H]2C1.[O-][Cl+3]([O-])([O-])O",
                [11.78661433848233, 34.97140439987911, -0.3895100127841371, 0.35985502321940255, 4.068524152374238, 10.555400600855673],
            ),
            ("C1=CC=C([PH](C2CCCCC2)C2CCCCC2)C=C1", [f64::NAN; 6]),
        ];
        for (smi, expected) in cases {
            let b = calc_bcut(smi);
            for (i, (&g, &e)) in b[..6].iter().zip(&expected).enumerate() {
                let ok = if e.is_nan() { g.is_nan() } else { (g - e).abs() <= 1e-9 + 1e-9 * e.abs() };
                assert!(ok, "{smi}[{i}]: got {g}, expected {e}");
            }
        }
    }

    #[test]
    fn test_bcut_single_atom() {
        let b = calc_bcut("C");
        // CDK: hydrogen-suppressed 1x1 matrix -> both eigenvalues = 12.0 (C-12 mass)
        assert_eq!(b[0], 12.0);
        assert_eq!(b[1], 12.0);
        // Mordred: 1x1 matrix -> the property itself
        assert_eq!(mordred_value(&b, "BCUTpe-1h"), 2.55);
        assert_eq!(mordred_value(&b, "BCUTpe-1l"), 2.55);
    }

    /// Column `name` of `values` (as returned by [`calc_bcut`]).
    fn mordred_value(values: &[f64], name: &str) -> f64 {
        values[feature_names().iter().position(|n| n == name).unwrap_or_else(|| panic!("no column {name}"))]
    }

    /// `expected`: Mordred `[BCUTpe-1l, BCUTpe-1h, BCUTv-1l, BCUTv-1h]`.
    fn assert_mordred(smiles: &str, expected: [f64; 4]) {
        let b = calc_bcut(smiles);
        for (name, e) in ["BCUTpe-1l", "BCUTpe-1h", "BCUTv-1l", "BCUTv-1h"].into_iter().zip(expected) {
            assert_mordred_column(smiles, name, mordred_value(&b, name), e);
        }
    }

    fn assert_mordred_column(smiles: &str, name: &str, got: f64, e: f64) {
        if e.is_nan() {
            assert!(got.is_nan(), "{smiles} {name} = {got}, expected NaN");
        } else {
            assert!((got - e).abs() < 1e-10 * e.abs().max(1.0), "{smiles} {name} = {got} expected {e}");
        }
    }

    // Expected values: Mordred (mordred-community 2.0.7) (tests/data/mordred_parity_reference.csv).

    #[test]
    fn test_bcut_mordred_values() {
        assert_mordred("c1ccccc1", [2.2510000000000003, 2.853000000000001, 20.280526276115527, 20.88252627611552]);
        assert_mordred("CCO", [2.4339309485564105, 3.453623159584941, 14.708165908556165, 20.690561052837154]);
        assert_mordred("C#N", [2.399873437997393, 3.190126562002607, 15.579311947692068, 20.59874545227239]);
        // hydrogen atoms kept by RDKit stay in Mordred's hydrogen-suppressed graph
        assert_mordred(
            "[2H]C([2H])([2H])Cl",
            [2.1161876812644715, 3.1805185092936377, 5.574279762570687, 22.455756479817776],
        );
    }

    #[test]
    fn test_bcut_mordred_missing_values() {
        // multiple fragments: Mordred requires a connected molecule
        assert_mordred("C(=O)[O-].[Na+]", [f64::NAN; 4]);
        // He has no Pauling electronegativity in Mordred's table
        assert_mordred("[He]", [f64::NAN, f64::NAN, 11.494040321933852, 11.494040321933852]);
        // methane: no intrinsic state for an atom without heavy neighbours
        assert!(mordred_value(&calc_bcut("C"), "BCUTs-1h").is_nan());
    }

    #[test]
    fn test_bcut_mordred_all_properties() {
        let b = calc_bcut("c1ccccc1O");
        for (name, e) in [
            // Gasteiger charge incl. implicit-hydrogen charge (`Mordred_` prefix: CDK has BCUTc)
            ("Mordred_BCUTc-1h", 0.3567978162306468),
            ("Mordred_BCUTc-1l", -0.2910752886763698),
            ("BCUTdv-1h", 5.012264812820403),
            ("BCUTs-1l", 1.548548182537098),
            ("Mordred_BCUTp-1h", 1.9749407620173023),
            ("BCUTi-1l", 10.960543795382984),
        ] {
            assert_mordred_column("c1ccccc1O", name, mordred_value(&b, name), e);
        }
    }

    fn assert_cdk(smiles: &str, expected: [f64; 6]) {
        let b = calc_bcut(smiles);
        for (k, e) in expected.into_iter().enumerate() {
            assert!((b[k] - e).abs() < 1e-4, "{smiles} BCUT[{k}] = {} expected {e}", b[k]);
        }
    }

    // Expected values: CDK 2.13 BCUTDescriptor (tests/data/cdk_parity_reference.csv).

    #[test]
    fn test_bcut_cdk_benzene() {
        assert_cdk("c1ccccc1", [11.701, 12.303, -0.3607581520692432, 0.24124184793075693, 5.944375, 6.546375]);
    }

    #[test]
    fn test_bcut_cdk_pyrrole() {
        assert_cdk(
            "c1cc[nH]c1",
            [11.75791293567627, 14.025461505956631, -0.4026774477676637, 0.23795405940206094, 5.5100092018641, 6.082807208436651],
        );
    }

    #[test]
    fn test_bcut_cdk_nitromethane_keeps_pentavalent_nitro() {
        assert_cdk(
            "CN(=O)=O",
            [11.993921878140798, 16.039305493709886, -0.37260147315819003, 0.2895468107331407, 2.464581354352091, 3.8029433877027814],
        );
    }

    #[test]
    fn test_bcut_cdk_errors_are_nan() {
        // explicit hydrogen atoms / unsupported elements: CDK reports an error
        for smi in ["[H][H]", "[Na+].[Cl-]", "c1cc[se]c1"] {
            let b = calc_bcut(smi);
            assert!(b[..6].iter().all(|v| v.is_nan()), "{smi}: {b:?}");
        }
    }

}
