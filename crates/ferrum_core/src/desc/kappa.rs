//! Kier-Hall Kappa shape indices.
//!
//! Three shape descriptors based on path counts:
//! - κ₁: based on 1-bond paths (edges)
//! - κ₂: based on 2-bond paths
//! - κ₃: based on 3-bond paths
//!
//! Reference: Kier, L.B. (1985) "A Shape Index from Molecular Graphs"
//!            Quant. Struct.-Act. Relat. 4, 109-116.

//! ## File guide: Kappa
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct KappaIndices`
//! - `fn name`
//! - `fn calculate`
//! - `mod tests`
//! - `fn calc_kappa`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Kappa` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Kappa` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/kappa.rs` defines kappa behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Kappa` workflows.
//! - Repository unit and parity tests covering `Kappa` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/kappa.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

// (Legacy descriptor removed — use KappaIndices below)

// ===========================================================================
// Typed API: Kappa Indices
// ===========================================================================

/// Kier-Hall Kappa Shape Indices (κ₁, κ₂, κ₃).
///
/// Measures molecular shape/branching:
/// - κ₁ = n(n-1)² / p₁²  where p₁ = number of bonds
/// - κ₂ = (n-1)(n-2)² / p₂²  where p₂ = number of 2-bond paths
/// - κ₃ depends on parity of n, using p₃ = number of 3-bond paths
///
/// Returns FloatVec with 3 values: [κ₁, κ₂, κ₃]
pub struct KappaIndices;

impl Descriptor for KappaIndices {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "KappaIndices"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        // RDKit calcKappa1-3: A = heavy atoms, P1 = all graph bonds,
        // P2/P3 = bond paths that avoid hydrogen atoms (useHs = false).
        let n = mol.atoms.iter().filter(|a| a.atomic_number > 1).count();

        if mol.num_atoms() == 0 {
            return DescriptorValue::FloatVec(vec![0.0, 0.0, 0.0]);
        }

        let a = n as f64;

        // Hall-Kier alpha correction
        let alpha = crate::desc::extra_counts::compute_hall_kier_alpha_f64(mol);
        let a_alpha = a + alpha;

        let p1 = mol.num_bonds() as f64;
        let mut heavy_paths = |len: usize| {
            let mut count = 0.0f64;
            for path in crate::algo::paths::cached_paths_of_length(mol, ctx, len).chunks_exact(len + 1) {
                if path.iter().all(|&i| mol.atoms[i as usize].atomic_number != 1) {
                    count += 1.0;
                }
            }
            count
        };
        let p2 = heavy_paths(2);
        let p3 = heavy_paths(3);

        // κ₁ = (A + α)(A + α - 1)² / (P1 + α)²
        let denom1 = p1 + alpha;
        let kappa1 = if denom1.abs() > 1e-10 {
            a_alpha * (a_alpha - 1.0).powi(2) / denom1.powi(2)
        } else {
            0.0
        };

        // κ₂ = (A + α - 1)(A + α - 2)² / (P2 + α)²
        let denom2 = p2 + alpha;
        let kappa2 = if denom2.abs() > 1e-10 {
            (a_alpha - 1.0) * (a_alpha - 2.0).powi(2) / denom2.powi(2)
        } else {
            0.0
        };

        // κ₃ depends on parity of A
        let denom3 = p3 + alpha;
        let kappa3 = if denom3.abs() > 1e-10 {
            if n % 2 == 1 {
                (a_alpha - 1.0) * (a_alpha - 3.0).powi(2) / denom3.powi(2)
            } else {
                (a_alpha - 2.0) * (a_alpha - 3.0).powi(2) / denom3.powi(2)
            }
        } else {
            0.0
        };

        DescriptorValue::FloatVec(vec![kappa1, kappa2, kappa3])
    }
}

/// CDK `KappaShapeIndicesDescriptor` (`Kier1`..`Kier3`).
///
/// Unlike RDKit's Kappa there is no alpha correction, and CDK's 3-path set
/// also contains every 3-membered ring (three distinct bonds reachable by a
/// 3-step walk). Hydrogens are suppressed with CDK's `removeHydrogens` rule,
/// so explicit H atoms RDKit keeps (`[2H]`, `[H][H]`, `[H+]`) are counted.
/// CDK's float semantics are kept: no bonds gives `inf`, an empty path set NaN.
pub struct CdkKappaShapeIndices;

impl Descriptor for CdkKappaShapeIndices {
    fn name(&self) -> &str {
        "CdkKappaShapeIndices"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(cdk_kier(mol).to_vec())
    }
}

fn cdk_kier(mol: &Molecule) -> [f64; 3] {
    let n_all = mol.num_atoms();
    let other = |bi: usize, a: usize| {
        let b = &mol.bonds[bi];
        if b.source as usize == a { b.target as usize } else { b.source as usize }
    };
    // Hydrogens.isSafeToSuppress
    let suppressed = |a: usize| {
        let at = &mol.atoms[a];
        at.atomic_number == 1
            && at.formal_charge == 0
            && at.isotope == 0
            && at.total_h() == 0
            && mol.adjacency[a].len() == 1
            && !matches!(mol.atoms[other(mol.adjacency[a][0] as usize, a)].atomic_number, 0 | 1)
    };
    let keep: Vec<bool> = (0..n_all).map(|a| !suppressed(a)).collect();
    let nbrs = |a: usize| {
        mol.adjacency[a].iter().map(move |&bi| other(bi as usize, a)).filter(|&b| keep[b])
    };
    let deg: Vec<u64> = (0..n_all).map(|a| if keep[a] { nbrs(a).count() as u64 } else { 0 }).collect();

    let n = keep.iter().filter(|&&k| k).count() as f64;
    let mut p1 = 0u64;
    let mut p3 = 0u64;
    let mut triangles = 0u64;
    for b in &mol.bonds {
        let (u, v) = (b.source as usize, b.target as usize);
        if !keep[u] || !keep[v] {
            continue;
        }
        p1 += 1;
        p3 += (deg[u] - 1) * (deg[v] - 1);
        triangles += nbrs(u).filter(|&w| w != v && nbrs(w).any(|x| x == v)).count() as u64;
    }
    // every triangle was seen from each of its 3 bonds (`triangles` = 3T) and
    // counted 3 times as a closed walk in `p3`; CDK keeps it once
    let p3 = (p3 - triangles / 3 * 2) as f64;
    let p2 = deg.iter().map(|&d| d * d.saturating_sub(1) / 2).sum::<u64>() as f64;
    let p1 = p1 as f64;

    let ratio = |num: f64, p: f64| if p == 0.0 { f64::NAN } else { num / (p * p) };
    if n == 1.0 {
        return [0.0; 3];
    }
    let kier1 = n * (n - 1.0) * (n - 1.0) / (p1 * p1);
    if n == 2.0 {
        return [kier1, 0.0, 0.0];
    }
    let kier2 = ratio((n - 1.0) * (n - 2.0) * (n - 2.0), p2);
    let kier3 = if n == 3.0 {
        0.0
    } else if n as u64 % 2 != 0 {
        ratio((n - 1.0) * (n - 3.0) * (n - 3.0), p3)
    } else {
        ratio((n - 3.0) * (n - 2.0) * (n - 2.0), p3)
    };
    [kier1, kier2, kier3]
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<KappaIndices>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements kappa.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_kappa(smiles: &str) -> Vec<f64> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match KappaIndices.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for kappa methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa_methane() {
        let k = calc_kappa("C");
        assert_eq!(k[0], 0.0);
        assert_eq!(k[1], 0.0);
        assert_eq!(k[2], 0.0);
    }

    #[test]
    /// Runs regression test for kappa ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa_ethane() {
        let k = calc_kappa("CC");
        assert!((k[0] - 2.0).abs() < 0.01, "κ₁ of ethane = {}", k[0]);
        assert_eq!(k[1], 0.0);
        assert_eq!(k[2], 0.0);
    }

    #[test]
    /// Runs regression test for kappa propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa_propane() {
        let k = calc_kappa("CCC");
        assert!((k[0] - 3.0).abs() < 0.01, "κ₁ = {}", k[0]);
        assert!((k[1] - 2.0).abs() < 0.01, "κ₂ = {}", k[1]);
    }

    #[test]
    /// Runs regression test for kappa butane vs isobutane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa_butane_vs_isobutane() {
        let k_linear = calc_kappa("CCCC");
        let k_branched = calc_kappa("CC(C)C");
        assert!((k_linear[0] - k_branched[0]).abs() < 0.01);
        assert!(
            k_linear[1] > k_branched[1],
            "linear κ₂={} should > branched κ₂={}",
            k_linear[1],
            k_branched[1]
        );
    }

    #[test]
    /// Runs regression test for kappa pentane linear.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa_pentane_linear() {
        let k = calc_kappa("CCCCC");
        assert!((k[0] - 5.0).abs() < 0.01, "κ₁ = {}", k[0]);
    }

    #[test]
    /// Runs regression test for kappa benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa_benzene() {
        let k = calc_kappa("c1ccccc1");
        assert!(k[0] > 3.0 && k[0] < 6.0, "κ₁ = {}", k[0]);
        assert!(k[1] > 0.0);
        assert!(k[2] > 0.0);
    }

    #[test]
    /// Runs regression test for kappa3 epoxide regression matches rdkit.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa3_epoxide_regression_matches_rdkit() {
        let k = calc_kappa("CC1OC1");
        assert!((k[2] - 0.20616508).abs() < 1e-5, "κ₃ = {}", k[2]);
    }

    #[test]
    /// Runs regression test for kappa size.
    ///
    /// Function context: `crates/ferrum_core/src/desc/kappa.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_kappa_size() {
        let k = calc_kappa("C");
        assert_eq!(k.len(), 3);
    }

    #[test]
    /// CDK Kier values (CDK 2.13): no alpha correction, triangles count as 3-paths.
    fn cdk_kier_values() {
        let v = calc_cdk_kier("CCCC");
        assert!((v[0] - 4.0).abs() < 1e-5 && (v[1] - 3.0).abs() < 1e-5 && (v[2] - 4.0).abs() < 1e-5, "{v:?}");
        // cyclopropane: P1=3, P2=3, n=3 -> Kier3 = 0
        let v = calc_cdk_kier("C1CC1");
        assert!((v[0] - 4.0 / 3.0).abs() < 1e-5 && (v[1] - 2.0 / 9.0).abs() < 1e-5 && v[2] == 0.0, "{v:?}");
        // methylcyclopropane: P3 = 2 paths + 1 triangle
        let v = calc_cdk_kier("CC1CC1");
        assert!((v[2] - 1.0 * 4.0 / 9.0).abs() < 1e-5, "{v:?}");
    }

    fn calc_cdk_kier(smiles: &str) -> Vec<f64> {
        let mut mol = crate::io::smiles::parse_smiles(smiles).unwrap();
        let mut ctx = crate::desc::context::CalculationContext::new();
        match CdkKappaShapeIndices.calculate(&mut mol, &mut ctx) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("{other:?}"),
        }
    }
}
