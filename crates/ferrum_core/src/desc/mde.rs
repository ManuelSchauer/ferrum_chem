//! Molecular Distance Edge (MDE) descriptors.
//!
//! Distance edge vectors based on atom-pair topological distances,
//! partitioned by bond type. Encodes molecular shape and heteroatom
//! distribution.
//!
//! For each pair type (C-C, C-N, C-O, N-N, N-O, O-O, C-X, etc.):
//!   MDE(type) = Σ 1/d²  for all pairs of that type at distance d.
//!
//! Reference: Liu, S. et al. (1998) "Molecular Distance Edge (MDE)
//!            Vector", JCICS 38, 536-540.


//! ## File guide: Mde
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn _compute_distance_matrix_unused`
//! - `fn atom_class`
//! - `fn pair_index`
//! - `struct MDE`
//! - `fn name`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Mde` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Mde` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/mde.rs` defines mde behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Mde` workflows.
//! - Repository unit and parity tests covering `Mde` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/mde.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::VecDeque;

#[allow(dead_code)]
/// Computes distance matrix unused.
///
/// Function context: `crates/ferrum_core/src/desc/mde.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn _compute_distance_matrix_unused(mol: &Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    let mut distances = vec![u32::MAX; n * n];
    for start in 0..n {
        distances[start * n + start] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start as AtomIdx);
        while let Some(current) = queue.pop_front() {
            let current_dist = distances[start * n + current as usize];
            for &bond_idx in &mol.adjacency[current as usize] {
                let bond = &mol.bonds[bond_idx as usize];
                let nb = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                };
                let nd = &mut distances[start * n + nb as usize];
                if *nd == u32::MAX {
                    *nd = current_dist + 1;
                    queue.push_back(nb);
                }
            }
        }
    }
    distances
}

/// Classify an atom into a coarse type for MDE.
/// 0=C, 1=N, 2=O, 3=S, 4=halogen, 5=other
fn atom_class(atomic_number: u8) -> usize {
    match atomic_number {
        6 => 0,
        7 => 1,
        8 => 2,
        16 => 3,
        9 | 17 | 35 | 53 => 4,
        _ => 5,
    }
}

// Pair types: (0,0)=CC, (0,1)=CN, (0,2)=CO, (0,3)=CS, (0,4)=CX,
//             (1,1)=NN, (1,2)=NO, (1,3)=NS, (1,4)=NX,
//             (2,2)=OO, (2,3)=OS, (2,4)=OX,
//             (3,3)=SS, (3,4)=SX, (4,4)=XX
// Total: 15 pair types (upper triangle of 6×6, but we only count 5 classes: 0-4)
const NUM_PAIR_TYPES: usize = 15;

/// Implements index.
///
/// Function context: `crates/ferrum_core/src/desc/mde.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn pair_index(a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    if lo > 4 || hi > 4 {
        return None;
    }
    // Map (lo, hi) to index in upper triangle of 5×5
    // Row lo, col hi: offset = sum(5-i for i in 0..lo) + (hi - lo)
    let offset: usize = (0..lo).map(|i| 5 - i).sum();
    Some(offset + (hi - lo))
}

/// MDE descriptors.
///
/// Returns FloatVec with 15 values (one per atom-pair type).
/// Each value is the sum of 1/d² for all atom pairs of that type.
pub struct MDE;

impl Descriptor for MDE {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "MDE"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut values = vec![0.0f32; NUM_PAIR_TYPES];

        let n = mol.num_atoms();
        if n < 2 {
            return DescriptorValue::FloatVec(values);
        }

        let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);
        let classes: Vec<usize> = mol
            .atoms
            .iter()
            .map(|a| atom_class(a.atomic_number))
            .collect();

        for i in 0..n {
            for j in i + 1..n {
                let d = dist[i * n + j];
                if d > 0 && d < 255 {
                    if let Some(pidx) = pair_index(classes[i], classes[j]) {
                        values[pidx] += 1.0 / (d * d) as f32;
                    }
                }
            }
        }

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MDE>;
};

#[cfg(test)]
mod tests {
    use super::super::typed::Descriptor;
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements mde.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_mde(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match MDE.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for mde len.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_mde_len() {
        let m = calc_mde("CC");
        assert_eq!(m.len(), 15);
    }

    #[test]
    /// Runs regression test for mde ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_mde_ethane() {
        let m = calc_mde("CC");
        let cc_idx = pair_index(0, 0).unwrap();
        assert!((m[cc_idx] - 1.0).abs() < 0.01, "CC = {}", m[cc_idx]);
    }

    #[test]
    /// Runs regression test for mde ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_mde_ethanol() {
        let m = calc_mde("CCO");
        let cc = pair_index(0, 0).unwrap();
        let co = pair_index(0, 2).unwrap();
        assert!(m[cc] > 0.0, "should have C-C pairs");
        assert!(m[co] > 0.0, "should have C-O pairs");
    }

    #[test]
    /// Runs regression test for mde no nitrogen in ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_mde_no_nitrogen_in_ethane() {
        let m = calc_mde("CC");
        let cn = pair_index(0, 1).unwrap();
        assert!(m[cn].abs() < 0.001, "no C-N pairs in ethane");
    }

    #[test]
    /// Runs regression test for mde benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_mde_benzene() {
        let m = calc_mde("c1ccccc1");
        let cc = pair_index(0, 0).unwrap();
        assert!(m[cc] > 5.0, "many CC pairs in benzene: {}", m[cc]);
    }

    #[test]
    /// Runs regression test for mde deterministic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_mde_deterministic() {
        let a = calc_mde("CCCC");
        let b = calc_mde("CCCC");
        assert_eq!(a, b);
    }

    #[test]
    /// Runs regression test for mde halogen.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_mde_halogen() {
        let m = calc_mde("CCl");
        let cx = pair_index(0, 4).unwrap();
        assert!(m[cx] > 0.0, "should have C-X pair");
    }

    #[test]
    /// Runs regression test for pair index.
    ///
    /// Function context: `crates/ferrum_core/src/desc/mde.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_pair_index() {
        assert_eq!(pair_index(0, 0), Some(0));
        assert_eq!(pair_index(0, 1), Some(1));
        assert_eq!(pair_index(0, 2), Some(2));
        assert_eq!(pair_index(1, 1), Some(5));
    }
}
