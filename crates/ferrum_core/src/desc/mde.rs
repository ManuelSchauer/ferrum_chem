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

use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, Real};
use std::collections::VecDeque;

fn compute_distance_matrix(mol: &Molecule) -> Vec<u32> {
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
                let nb = if bond.source == current { bond.target } else { bond.source };
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

fn pair_index(a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    if lo > 4 || hi > 4 { return None; }
    // Map (lo, hi) to index in upper triangle of 5×5
    // Row lo, col hi: offset = sum(5-i for i in 0..lo) + (hi - lo)
    let offset: usize = (0..lo).map(|i| 5 - i).sum();
    Some(offset + (hi - lo))
}

/// MDE descriptors: 15 values (one per atom-pair type).
///
/// Each value is the sum of 1/d² for all atom pairs of that type.
pub struct MDE;

impl Descriptor for MDE {
    fn name(&self) -> String { "MDE".to_string() }
    fn size(&self) -> usize { NUM_PAIR_TYPES }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        for v in output.iter_mut() { *v = 0.0; }

        let n = mol.num_atoms();
        if n < 2 { return; }

        let dist = compute_distance_matrix(mol);
        let classes: Vec<usize> = mol.atoms.iter()
            .map(|a| atom_class(a.atomic_number))
            .collect();

        for i in 0..n {
            if classes[i] > 4 { continue; }
            for j in (i + 1)..n {
                if classes[j] > 4 { continue; }
                let d = dist[i * n + j];
                if d == 0 || d == u32::MAX { continue; }

                if let Some(idx) = pair_index(classes[i], classes[j]) {
                    output[idx] += (1.0 / (d as f64 * d as f64)) as Real;
                }
            }
        }
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MDE>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_mde(smiles: &str) -> Vec<Real> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = vec![0.0; NUM_PAIR_TYPES];
        MDE.calculate(&mol, &mut output);
        output
    }

    #[test]
    fn test_mde_size() {
        assert_eq!(MDE.size(), 15);
    }

    #[test]
    fn test_mde_ethane() {
        let m = calc_mde("CC");
        // One C-C pair at distance 1: 1/1² = 1.0
        let cc_idx = pair_index(0, 0).unwrap();
        assert!((m[cc_idx] - 1.0).abs() < 0.01, "CC = {}", m[cc_idx]);
    }

    #[test]
    fn test_mde_ethanol() {
        let m = calc_mde("CCO");
        let cc = pair_index(0, 0).unwrap();
        let co = pair_index(0, 2).unwrap();
        assert!(m[cc] > 0.0, "should have C-C pairs");
        assert!(m[co] > 0.0, "should have C-O pairs");
    }

    #[test]
    fn test_mde_no_nitrogen_in_ethane() {
        let m = calc_mde("CC");
        let cn = pair_index(0, 1).unwrap();
        assert!((m[cn]).abs() < 0.001, "no C-N pairs in ethane");
    }

    #[test]
    fn test_mde_benzene() {
        let m = calc_mde("c1ccccc1");
        let cc = pair_index(0, 0).unwrap();
        // 15 C-C pairs at various distances
        assert!(m[cc] > 5.0, "many CC pairs in benzene: {}", m[cc]);
    }

    #[test]
    fn test_mde_deterministic() {
        let a = calc_mde("CCCC");
        let b = calc_mde("CCCC");
        assert_eq!(a, b);
    }

    #[test]
    fn test_mde_halogen() {
        let m = calc_mde("CCl");
        let cx = pair_index(0, 4).unwrap();
        assert!(m[cx] > 0.0, "should have C-X pair");
    }

    #[test]
    fn test_pair_index() {
        assert_eq!(pair_index(0, 0), Some(0)); // CC
        assert_eq!(pair_index(0, 1), Some(1)); // CN
        assert_eq!(pair_index(0, 2), Some(2)); // CO
        assert_eq!(pair_index(1, 1), Some(5)); // NN
    }
}
