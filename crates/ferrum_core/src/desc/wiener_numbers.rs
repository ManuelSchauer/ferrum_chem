//! Descriptor module: Wiener Numbers.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Wiener Numbers
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct WienerIndex`
//! - `fn wiener_numbers`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Wiener Numbers` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Wiener Numbers` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/wiener_numbers.rs` defines wiener numbers behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Wiener Numbers` workflows.
//! - Repository unit and parity tests covering `Wiener Numbers` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/wiener_numbers.rs`.

use super::cdk_graph::cdk_distance;
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// CDK `WienerNumbersDescriptor`: `[WienerPath, WienerPolarity]`.
///
/// - WienerPath: half the sum of all topological distances between
///   (non-hydrogen) atoms. As in CDK, atom pairs in different components count
///   with CDK's unreachable distance 999 999 999.
/// - WienerPolarity: number of atom pairs at distance 3.
pub struct WienerIndex;

impl WienerIndex {
    /// `(WienerPath, WienerPolarity)` from an `n x n` distance matrix.
    fn wiener_numbers(distances: &[u16], n: usize) -> (f64, f64) {
        let mut path: u64 = 0;
        let mut polarity: u64 = 0;
        for i in 0..n {
            for &d in &distances[i * n + i + 1..(i + 1) * n] {
                path += cdk_distance(d) as u64;
                polarity += (d == 3) as u64;
            }
        }
        (path as f64, polarity as f64)
    }
}

impl Descriptor for WienerIndex {
    fn name(&self) -> &str {
        "WienerIndex"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        // CDK computes on `removeHydrogens(container)`
        if let Some(mut suppressed) = crate::algo::sanitize::cdk_remove_hydrogens(mol) {
            return self.calculate(&mut suppressed, ctx);
        }
        let n = mol.num_atoms();
        if n <= 1 {
            return DescriptorValue::FloatVec(vec![0.0, 0.0]);
        }
        let distances = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
        let (path, polarity) = Self::wiener_numbers(distances, n);
        DescriptorValue::FloatVec(vec![path, polarity])
    }
}

// Static assertion: WienerIndex must be Send + Sync for thread safety
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<WienerIndex>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn wiener_numbers(smiles: &str) -> (f64, f64) {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        let v = WienerIndex
            .calculate(&mut mol, &mut crate::desc::context::CalculationContext::new())
            .as_float_vec()
            .to_vec();
        assert_eq!(v.len(), 2);
        (v[0], v[1])
    }

    fn calculate_wiener(smiles: &str) -> f64 {
        wiener_numbers(smiles).0
    }

    #[test]
    fn test_wiener_methane() {
        assert_eq!(calculate_wiener("C"), 0.0);
    }

    #[test]
    fn test_wiener_ethane() {
        assert_eq!(calculate_wiener("CC"), 1.0);
    }

    #[test]
    fn test_wiener_propane() {
        assert_eq!(calculate_wiener("CCC"), 4.0);
    }

    #[test]
    fn test_wiener_butane() {
        assert_eq!(calculate_wiener("CCCC"), 10.0);
    }

    #[test]
    fn test_wiener_isobutane() {
        assert_eq!(calculate_wiener("CC(C)C"), 9.0);
    }

    #[test]
    fn test_wiener_benzene() {
        assert_eq!(calculate_wiener("c1ccccc1"), 27.0);
    }

    #[test]
    fn test_wiener_cyclohexane() {
        assert_eq!(calculate_wiener("C1CCCCC1"), 27.0);
    }

    #[test]
    fn test_wiener_naphthalene() {
        let wiener = calculate_wiener("c1ccc2ccccc2c1");
        assert!(wiener > 27.0, "Naphthalene should have higher W than benzene");
    }

    #[test]
    fn test_wiener_linear_vs_branched() {
        let linear = calculate_wiener("CCCC");
        let branched = calculate_wiener("CC(C)C");
        assert!(linear > branched);
    }

    // CDK 2.13 WienerNumbersDescriptor values.
    #[test]
    fn test_wiener_polarity() {
        assert_eq!(wiener_numbers("CCCC"), (10.0, 1.0));
        assert_eq!(wiener_numbers("CC(C)C"), (9.0, 0.0));
        assert_eq!(wiener_numbers("c1ccccc1"), (27.0, 3.0));
    }

    #[test]
    fn test_wiener_disconnected_uses_cdk_sentinel() {
        // CDK counts unreachable pairs with distance 999999999
        let (path, polarity) = wiener_numbers("CC.O");
        assert_eq!(path as f64, (1.0 + 2.0 * 999_999_999.0f64) as f64 as f64);
        assert_eq!(polarity, 0.0);
    }

    #[test]
    fn test_wiener_comprehensive() {
        let test_cases = vec![
            ("C", 0.0, "methane"),
            ("CC", 1.0, "ethane"),
            ("CCC", 4.0, "propane"),
            ("CCCC", 10.0, "butane"),
            ("CC(C)C", 9.0, "isobutane"),
            ("CCCCC", 20.0, "pentane"),
            ("c1ccccc1", 27.0, "benzene"),
            ("C1CCCCC1", 27.0, "cyclohexane"),
        ];

        for (smiles, expected, name) in test_cases {
            let actual = calculate_wiener(smiles);
            assert_eq!(actual, expected, "{} ({}): expected W={}, got W={}", name, smiles, expected, actual);
        }
    }
}
