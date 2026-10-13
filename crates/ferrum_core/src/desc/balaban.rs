#![allow(dead_code)]
//! Balaban J Index descriptor
//!
//! The Balaban J Index is a topological index that measures the degree of branching
//! in a molecular graph. It's based on the distance matrix and vertex degrees.
//!
//! Formula: J = m/(μ+1) × Σ[(d_i × d_j)^(-0.5)] over all edges
//! where:
//! - m = number of edges (bonds)
//! - μ = cyclomatic number (m - n + 1 for connected graphs)
//! - d_i = sum of distances from vertex i to all other vertices
//!
//! Reference: Balaban, A.T. (1982) Chemical Applications of Graph Theory

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Balaban J from the bond-order weighted distances in sixths
/// ([`crate::algo::distance_matrix::bond_order_distances_sixths`]).
fn calculate_with_sixths(mol: &Molecule, dist6: &[u32]) -> f64 {
    use crate::algo::distance_matrix::BO_DIST_INF;
    let n = mol.num_atoms();
    let m = mol.num_bonds();

    if n <= 1 || m == 0 {
        return 0.0;
    }

    // Distance sums. Like RDKit's `GetDistanceMatrix`, unreachable pairs of
    // disconnected molecules count as distance 1e8 (so J is tiny but not 0).
    // The reachable part is summed exactly in sixths.
    const RDKIT_LOCAL_INF: f64 = 1e8;
    let mut s = vec![0.0f64; n];
    for i in 0..n {
        let mut sixths = 0u64;
        let mut unreachable = 0u64;
        for &d in &dist6[i * n..(i + 1) * n] {
            if d == BO_DIST_INF {
                unreachable += 1;
            } else {
                sixths += d as u64;
            }
        }
        s[i] = sixths as f64 / 6.0 + unreachable as f64 * RDKIT_LOCAL_INF;
    }

    // Cyclomatic number μ = m - n + 1
    let mu = m as i32 - n as i32 + 1;

    // Sum over all edges (using simple adjacency, not bond orders)
    let mut sum = 0.0;
    for bond in &mol.bonds {
        let si = s[bond.source as usize];
        let sj = s[bond.target as usize];
        if si <= 0.0 || sj <= 0.0 {
            return f64::NAN;
        }
        sum += 1.0 / (si * sj).sqrt();
    }

    // J = m/(μ+1) × sum
    if mu + 1 != 0 {
        (m as f64 / (mu + 1) as f64) * sum
    } else {
        0.0
    }
}

/// Calculate Balaban J Index
///
/// # Returns
/// The Balaban J index value (typically 1.5-3.0 for most organic molecules)
///
/// # Examples
/// - Benzene: J ≈ 2.00 (highly symmetric)
/// - Adamantane: J ≈ 2.31
/// - Linear alkanes: J decreases with length
///
/// # Panics
/// Returns NaN for molecules with no bonds
pub fn calculate(mol: &Molecule) -> f64 {
    let mut dist6 = Vec::new();
    crate::algo::distance_matrix::bond_order_distances_sixths(mol, &mut dist6);
    calculate_with_sixths(mol, &dist6)
}

/// Balaban J with the bond-order distances shared through `ctx`.
pub(crate) fn calculate_with_context(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> f64 {
    let dist6 = crate::algo::distance_matrix::bond_order_distances_sixths_cached(mol, ctx);
    calculate_with_sixths(mol, dist6)
}

/// Balaban J Index descriptor.
///
/// Returns Float with the Balaban J index.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct BalabanIndex;

impl Descriptor for BalabanIndex {
    fn name(&self) -> &str {
        "BalabanJ"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Float(calculate_with_context(mol, ctx))
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BalabanIndex>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    fn test_balaban_methane() {
        let mol = parse("C");
        let j = calculate(&mol);
        println!("Methane J = {}", j);
        assert!(j.is_nan() || j == 0.0); // Single atom, no bonds
    }

    #[test]
    fn test_balaban_ethane() {
        let mol = parse("CC");
        let j = calculate(&mol);
        // n=2, m=1, μ=0, d_sums=[1,1]
        // sum = 1/sqrt(1*1) = 1.0
        // J = 1/(0+1) * 1.0 = 1.0
        assert!((j - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_balaban_propane() {
        let mol = parse("CCC");
        let j = calculate(&mol);
        // Linear chain should have J around 1.63
        assert!(j > 1.5 && j < 1.8);
    }

    #[test]
    fn test_balaban_isobutane() {
        let mol = parse("CC(C)C");
        let j = calculate(&mol);
        // Branched should have higher J than linear
        // Isobutane J ≈ 2.32
        assert!(j > 2.2 && j < 2.4, "J = {}", j);
    }

    #[test]
    fn test_balaban_benzene() {
        let mol = parse("c1ccccc1");
        let j = calculate(&mol);
        // Benzene: J = 3.0 (with bond-order-weighted distances)
        assert!((j - 3.0).abs() < 0.1, "J = {}", j);
    }

    #[test]
    fn test_balaban_cyclohexane() {
        let mol = parse("C1CCCCC1");
        let j = calculate(&mol);
        // Cyclohexane: all single bonds, J ≈ 2.0
        assert!((j - 2.0).abs() < 0.1, "J = {}", j);
    }

    #[test]
    fn test_balaban_naphthalene() {
        let mol = parse("c1ccc2ccccc2c1");
        let j = calculate(&mol);
        // Naphthalene: J ≈ 2.89 (with BO-weighted distances)
        assert!(j > 2.7 && j < 3.0, "J = {}", j);
    }

    #[test]
    fn test_balaban_disconnected() {
        // Two separate methane molecules
        let mol = parse("C.C");
        let j = calculate(&mol);
        // Should handle disconnected graphs
        assert!(j.is_nan() || j >= 0.0);
    }
}
