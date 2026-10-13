//! Regression tests for previously found bugs.

use ferrum_core::algo::paths::{DIST_INF, compute_unweighted_distance_matrix_cached};
use ferrum_core::desc::batch::BatchDescriptor;
use ferrum_core::desc::calculator::DescriptorCalculator;
use ferrum_core::desc::context::CalculationContext;
use ferrum_core::io::smiles::parse_smiles;

/// Distances >= 255 used to saturate (u8 matrix) and the atom-pair BFS looped forever.
#[test]
fn distance_matrix_handles_long_chains() {
    let n = 400;
    let mol = parse_smiles(&"C".repeat(n)).unwrap();
    let mut ctx = CalculationContext::new();
    let dist = compute_unweighted_distance_matrix_cached(&mol, &mut ctx);
    assert_eq!(dist[n - 1], (n - 1) as u16);
    assert_eq!(dist[(n - 1) * n], (n - 1) as u16);
}

#[test]
fn distance_matrix_marks_disconnected_atoms() {
    let mol = parse_smiles("CC.O").unwrap();
    let mut ctx = CalculationContext::new();
    let dist = compute_unweighted_distance_matrix_cached(&mol, &mut ctx);
    assert_eq!(dist[1], 1);
    assert_eq!(dist[2], DIST_INF);
}

/// Every descriptor in the default bundle must finish on a long chain, and the
/// distance-based ones must stay finite (MDE used to overflow `d * d` in u8).
#[test]
fn full_bundle_on_long_chain_is_finite_for_topological_descriptors() {
    let mols = vec![parse_smiles(&"C".repeat(300)).ok()];
    let descriptors = [
        BatchDescriptor::Mde,
        BatchDescriptor::WienerIndex,
        BatchDescriptor::PetitjeanIndex,
        BatchDescriptor::Autocorrelation,
        BatchDescriptor::EStateIndices,
        BatchDescriptor::AtomPairFingerprint,
    ];
    let table = DescriptorCalculator::new(&descriptors).calculate(&mols);
    assert!(table.scalars.iter().all(|v| v.is_finite()));

    // Wiener index of a linear chain: sum_{d=1}^{n-1} d * (n - d) = (n^3 - n) / 6.
    let n = 300f64;
    let wiener_col = BatchDescriptor::Mde.feature_len();
    assert_eq!(table.scalars[wiener_col], (n * n * n - n) / 6.0);
}

/// RDKit 2026.03 (NumRotatableBonds 3.2.0) excludes bonds to methyl groups also
/// when their hydrogens are explicit isotopes (CD3); values from RDKit 2026.03.2.
#[test]
fn rotatable_bonds_ignore_deuterated_methyl_groups() {
    let cases = [
        ("[2H]C([2H])([2H])C(=O)C1=CC=C(OC)C=C1O", 2.0),
        ("[2H]C([2H])([2H])CC", 0.0),
        ("[2H]C([2H])CC", 0.0),
        ("[2H]C(C)CC", 1.0),
        ("[2H]OCC", 1.0),
        ("[3H]C([3H])([3H])C(=O)c1ccccc1", 1.0),
        ("FC(F)(F)C(=O)c1ccccc1", 1.0),
        ("[2H]C([2H])([2H])C([2H])([2H])c1ccccc1", 1.0),
    ];
    let calc = DescriptorCalculator::new(&[BatchDescriptor::RotatableBondCount]);
    for (smiles, expected) in cases {
        let table = calc.calculate(&[parse_smiles(smiles).ok()]);
        assert_eq!(table.scalars[0], expected, "{smiles}");
    }
}
