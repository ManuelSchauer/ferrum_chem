//! BCUT (Burden-CAS-UT) descriptors.
//!
//! Modified Burden adjacency matrix with atomic property weights on the
//! diagonal. The highest and lowest eigenvalues capture molecular shape
//! and property extremes.
//!
//! Output (10 values, `BatchDescriptor::feature_names` order):
//! - `BCUTw-1l/h`, `BCUTc-1l/h`, `BCUTp-1l/h`: CDK 2.13 `BCUTDescriptor`
//!   (nhigh = nlow = 1, checkAromaticity). Hydrogen-suppressed Burden matrix:
//!   diagonal = major-isotope mass / Gasteiger-Marsili charge (computed with
//!   explicit H) / Gasteiger-Hückel effective polarizability; off-diagonal
//!   0.15 for cdkLegacy-aromatic bonds, else 0.1/0.2/0.3 by Kekulé order,
//!   +0.01 if a bond atom is terminal, 0.001 for non-bonded pairs.
//! - `BCUTpe-1l/h`, `BCUTv-1l/h`: Mordred 1.2.0 `BCUT(prop, nth)` with
//!   `pe` = Pauling electronegativity and `v` = van der Waals volume
//!   (Mordred atomic property tables, `mordred_data.rs`). Burden matrix of the
//!   hydrogen-suppressed RDKit graph (hydrogen atoms kept by RDKit stay):
//!   off-diagonal `GetBondTypeAsDouble() / 10` (aromatic 0.15), `+ 0.01` if
//!   either bond atom has degree 1, 0.001 for non-bonded pairs. Mordred
//!   requires a connected molecule (NaN otherwise) and reports NaN when an
//!   atom has no property value.
//!
//! Reference: Pearlman, R.S. & Smith, K.M. (1999) "Metric Validation and
//!            the Receptor-Relevant Subspace Concept", JCICS 39, 28-35.

//! ## File guide: Bcut
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! Key symbols in this file:
//! - `fn cdk_bcut_matrices` (CDK part) / `fn mordred_bcut_matrices` (Mordred part)
//! - `fn bcut_values` (batched eigen-solve of all Burden matrices)
//! - `struct BCUT`
//!
//! ### Algorithmic assumptions for this file
//! - Only the extreme eigenvalues are needed: `crate::algo::linalg::burden_extremes`
//!   (Householder tridiagonalization + certified Laguerre/Sturm iteration,
//!   backward stable like the dense solvers of CDK/JAMA and numpy, agreeing
//!   to ~1e-14), two matrices at a time in SIMD lanes.
//! - The off-diagonal Burden template is built once per flavour and reused
//!   for every atomic property.
//! ### Chemistry/ML references
//! - Burden, F. (1989). Molecular identification number for substructure searches (Burden eigenvalues).
//! - Pearlman et al. BCUT descriptor literature for weighted adjacency eigenvalue descriptors.
//! - Moriwaki et al. (2018) Mordred: a molecular descriptor calculator, J. Cheminform. 10, 4.

use super::cdk_perception::{CdkView, MAJOR_ISOTOPE_MASS};
use super::mordred_data::{MORDRED_PAULING_EN, MORDRED_VDW_VOLUME};
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
/// Mordred: bond entries and the electronegativity / vdW volume diagonals.
type MordredMatrices = (BondEntries, [Option<Vec<f64>>; 2]);

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

/// Mordred `BCUT` Burden matrix template and the Pauling electronegativity /
/// vdW volume diagonals (`None` where Mordred reports a missing value), or
/// `None` overall when Mordred fails for the molecule.
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
    let diags = [&MORDRED_PAULING_EN, &MORDRED_VDW_VOLUME].map(|table| {
        let diag: Vec<f64> = mol
            .atoms
            .iter()
            .map(|atom| table.get(atom.atomic_number as usize).copied().unwrap_or(f64::NAN))
            .collect();
        (!diag.iter().any(|p| p.is_nan())).then_some(diag)
    });
    Some((entries, diags))
}

/// All 10 BCUT values: CDK `[w_lo, w_hi, c_lo, c_hi, p_lo, p_hi]` then
/// Mordred `[pe_lo, pe_hi, v_lo, v_hi]`. The (up to five) Burden matrices of
/// both flavours share one batched eigen-solve.
fn bcut_values(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> [f64; 10] {
    let n = mol.num_atoms();
    let mut out = [f64::NAN; 10];
    let cdk = cdk_bcut_matrices(mol, ctx);
    let mordred = mordred_bcut_matrices(mol, ctx);
    let mut jobs: Vec<BurdenMatrix> = Vec::with_capacity(5);
    let mut slots: Vec<usize> = Vec::with_capacity(5);
    match &cdk {
        Err(v) => out[..6].copy_from_slice(v),
        Ok((entries, diags)) => {
            for (k, diag) in diags.iter().enumerate() {
                jobs.push(BurdenMatrix { fill: 0.001, entries, diag });
                slots.push(2 * k);
            }
        }
    }
    if let Some((entries, diags)) = &mordred {
        for (k, diag) in diags.iter().enumerate() {
            if let Some(diag) = diag {
                jobs.push(BurdenMatrix { fill: 0.001, entries, diag });
                slots.push(6 + 2 * k);
            }
        }
    }
    for (&slot, (lo, hi)) in slots.iter().zip(crate::algo::linalg::burden_extremes(n, &jobs)) {
        out[slot] = lo;
        out[slot + 1] = hi;
    }
    out
}

/// BCUT descriptors.
///
/// Returns FloatVec with 10 values:
/// `[BCUTw-1l, BCUTw-1h, BCUTc-1l, BCUTc-1h, BCUTp-1l, BCUTp-1h,
///   BCUTpe-1l, BCUTpe-1h, BCUTv-1l, BCUTv-1h]`
/// (CDK BCUT, then Mordred Pauling electronegativity / vdW volume BCUT).
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
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        if mol.num_atoms() == 0 {
            return DescriptorValue::FloatVec(vec![0.0f64; 10]);
        }
        let mut values = bcut_values(mol, ctx);
        // CDK computes on `removeHydrogens(container)`; Mordred keeps the
        // hydrogens RDKit keeps
        if let Some(suppressed) = crate::algo::sanitize::cdk_remove_hydrogens(mol) {
            values[..6].copy_from_slice(&bcut_values(&suppressed, ctx)[..6]);
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
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match BCUT.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    fn test_bcut_len() {
        assert_eq!(calc_bcut("CC").len(), 10);
    }

    #[test]
    fn test_bcut_ethanol_different_from_ethane() {
        assert_ne!(calc_bcut("CC"), calc_bcut("CCO"));
    }

    #[test]
    fn test_bcut_high_ge_low() {
        let b = calc_bcut("c1ccccc1");
        for i in 0..5 {
            assert!(b[i * 2 + 1] >= b[i * 2], "prop {i}: high {} >= low {}", b[i * 2 + 1], b[i * 2]);
        }
    }

    #[test]
    fn test_bcut_single_atom() {
        let b = calc_bcut("C");
        // CDK: hydrogen-suppressed 1x1 matrix -> both eigenvalues = 12.0 (C-12 mass)
        assert_eq!(b[0], 12.0);
        assert_eq!(b[1], 12.0);
        // Mordred: 1x1 matrix -> the property itself
        assert_eq!(b[6], 2.55);
        assert_eq!(b[7], 2.55);
    }

    fn assert_mordred(smiles: &str, expected: [f64; 4]) {
        let b = calc_bcut(smiles);
        for (k, e) in expected.into_iter().enumerate() {
            let got = b[6 + k];
            if e.is_nan() {
                assert!(got.is_nan(), "{smiles} BCUT[{}] = {got}, expected NaN", 6 + k);
            } else {
                assert!((got - e).abs() < 1e-12 * e.abs().max(1.0), "{smiles} BCUT[{}] = {got} expected {e}", 6 + k);
            }
        }
    }

    // Expected values: Mordred 1.2.0 (tests/data/mordred_parity_reference.csv).

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
    }

    fn assert_cdk(smiles: &str, expected: [f64; 6]) {
        let b = calc_bcut(smiles);
        for (k, e) in expected.into_iter().enumerate() {
            assert!((b[k] as f64 - e).abs() < 1e-4, "{smiles} BCUT[{k}] = {} expected {e}", b[k]);
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
