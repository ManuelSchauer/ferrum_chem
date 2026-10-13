//! Descriptor module: Zagreb.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Zagreb Indices: Degree-based topological descriptors.
///
/// The Zagreb indices are among the oldest and most studied molecular descriptors,
/// introduced by Gutman and Trinajstić in 1972. They are based on vertex degrees
/// in the molecular graph.
///
/// # Indices
/// - **M1 (First Zagreb Index)**: Sum of squared degrees — M1 = Σᵢ deg(i)²
///   (CDK 2.13 `ZagrebIndexDescriptor`)
/// - **M2 (Second Zagreb Index)**: Sum of degree products over edges — M2 = Σ(i,j)∈E deg(i) × deg(j)
///   (Mordred 1.2.0 `Zagreb2`)
///
/// # Properties
/// - Correlate with molecular branching
/// - Higher values indicate more branched structures
///
/// Returns FloatVec with 2 values: [M1, M2]
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct ZagrebIndices;

impl Descriptor for ZagrebIndices {
    fn name(&self) -> &str {
        "ZagrebIndices"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let n = mol.num_atoms();

        if n == 0 {
            return DescriptorValue::FloatVec(vec![0.0, 0.0]);
        }

        // Zagreb1 follows CDK's ZagrebIndexDescriptor: hydrogen atoms (e.g. the
        // isotopic ones kept as atoms) are skipped and not counted as neighbours.
        let is_h = |a: usize| mol.atoms[a].atomic_number == 1;
        let mut m1: u64 = 0;
        for atom_idx in 0..n {
            if is_h(atom_idx) {
                continue;
            }
            let degree = mol.adjacency[atom_idx]
                .iter()
                .filter(|&&bi| {
                    let b = &mol.bonds[bi as usize];
                    let other = if b.source as usize == atom_idx { b.target } else { b.source };
                    !is_h(other as usize)
                })
                .count();
            m1 += (degree * degree) as u64;
        }

        // Zagreb2 follows Mordred 1.2.0 `ZagrebIndex(2, 1)`: Σ_bonds d_i·d_j with
        // d = RDKit adjacency-matrix degree of the hydrogen-suppressed graph,
        // so hydrogen atoms kept by RDKit (e.g. [2H]) do count.
        let mut m2: u64 = 0;
        for bond in &mol.bonds {
            let deg_u = mol.get_degree(bond.source);
            let deg_v = mol.get_degree(bond.target);
            m2 += (deg_u * deg_v) as u64;
        }

        DescriptorValue::FloatVec(vec![m1 as f64, m2 as f64])
    }
}

// Static assertion: ZagrebIndices must be Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ZagrebIndices>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calculate_zagreb(smiles: &str) -> (f64, f64) {
        let mol = parse_smiles(smiles).expect("Invalid SMILES");
        let v = ZagrebIndices
            .calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new(),
            )
            .as_float_vec()
            .to_vec();
        (v[0], v[1])
    }

    #[test]
    fn test_zagreb_methane() {
        let (m1, m2) = calculate_zagreb("C");
        assert_eq!(m1, 0.0);
        assert_eq!(m2, 0.0);
    }

    #[test]
    fn test_zagreb_ethane() {
        let (m1, m2) = calculate_zagreb("CC");
        assert_eq!(m1, 2.0);
        assert_eq!(m2, 1.0);
    }

    #[test]
    fn test_zagreb_propane() {
        let (m1, m2) = calculate_zagreb("CCC");
        assert_eq!(m1, 6.0);
        assert_eq!(m2, 4.0);
    }

    #[test]
    fn test_zagreb_isobutane() {
        let (m1, m2) = calculate_zagreb("CC(C)C");
        assert_eq!(m1, 12.0);
        assert_eq!(m2, 9.0);
    }

    #[test]
    fn test_zagreb_butane() {
        let (m1, m2) = calculate_zagreb("CCCC");
        assert_eq!(m1, 10.0);
        assert_eq!(m2, 8.0);
    }

    #[test]
    fn test_zagreb_benzene() {
        let (m1, m2) = calculate_zagreb("c1ccccc1");
        assert_eq!(m1, 24.0);
        assert_eq!(m2, 24.0);
    }

    #[test]
    fn test_zagreb_cyclohexane() {
        let (m1, m2) = calculate_zagreb("C1CCCCC1");
        assert_eq!(m1, 24.0);
        assert_eq!(m2, 24.0);
    }

    #[test]
    fn test_zagreb_naphthalene() {
        let (m1, m2) = calculate_zagreb("c1ccc2ccccc2c1");
        assert!(m1 > 24.0, "Naphthalene M1 should be > benzene");
        assert!(m2 > 24.0, "Naphthalene M2 should be > benzene");
    }

    #[test]
    fn test_zagreb_branched_vs_linear() {
        let (m1_linear, _) = calculate_zagreb("CCCC");
        let (m1_branched, _) = calculate_zagreb("CC(C)C");
        assert!(m1_branched > m1_linear);
    }

    #[test]
    fn test_zagreb1_ignores_hydrogen_atoms() {
        // CDK: [2H] atoms are neither counted nor neighbours
        let (m1, m2) = calculate_zagreb("[2H]C([2H])([2H])Cl");
        assert_eq!(m1, 2.0);
        // Mordred: kept hydrogen atoms are graph atoms (3 * 4*1 + 4*1)
        assert_eq!(m2, 16.0);
    }
}
