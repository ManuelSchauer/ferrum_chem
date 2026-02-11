//! Atom Pair and Topological Torsion fingerprints.
//!
//! Path-based fingerprints that encode distance relationships between atoms.
//!
//! - **Atom Pairs**: Encode (atom_type_i, distance, atom_type_j) for all pairs
//! - **Topological Torsion**: Encode 4-atom linear paths
//!
//! References:
//! - Carhart et al. (1985) "Atom pairs as molecular features" JCICS 25, 64-73
//! - Nilakantan et al. (1987) "Topological torsion" JCICS 27, 82-85

use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use super::typed::{DescriptorValue, Descriptor};
use bitvec::prelude::*;

/// Maximum topological distance for atom pair encoding.
const MAX_PAIR_DISTANCE: u32 = 7;

/// Compute a hash from a slice of u64 values.
fn combine_hash(values: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for v in values {
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// Atom type encoding for atom pair fingerprints.
///
/// Encodes: atomic_number, num_pi_electrons, total_degree, total_h
fn atom_pair_type(mol: &Molecule, idx: usize) -> u64 {
    let atom = &mol.atoms[idx];
    let degree = mol.get_degree(idx as u16) as u64;
    let total_h = atom.total_h() as u64;

    // Estimate pi electrons from bond types
    let pi_electrons: u64 = mol.adjacency[idx].iter()
        .map(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            match bond.bond_type {
                BondType::Double => 1,
                BondType::Triple => 2,
                BondType::Aromatic => 1, // approximate
                _ => 0,
            }
        })
        .sum();

    combine_hash(&[
        atom.atomic_number as u64,
        pi_electrons,
        degree,
        total_h,
    ])
}

/// Compute all-pairs shortest path distances using BFS.
fn compute_distances(mol: &Molecule) -> Vec<u32> {
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
                let neighbor = if bond.source == current { bond.target } else { bond.source };
                let nd = &mut distances[start * n + neighbor as usize];
                if *nd == u32::MAX {
                    *nd = current_dist + 1;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    distances
}

/// Fold hashes into a fixed-size bit vector.
fn fold_bitvec(hashes: &[u64], num_bits: usize) -> BitVec<u8, Lsb0> {
    let mut bits = bitvec![u8, Lsb0; 0; num_bits];
    for &hash in hashes {
        let bit = (hash as usize) % num_bits;
        bits.set(bit, true);
    }
    bits
}

/// Atom Pair Fingerprint descriptor.
///
/// Encodes all pairs of atoms at topological distances 1-7.
/// Each pair is hashed as: (atom_type_i, distance, atom_type_j) with canonical ordering.
/// Returns bit-packed fingerprint.
pub struct AtomPairFingerprint {
    pub num_bits: usize,
}

impl AtomPairFingerprint {
    pub fn new(num_bits: usize) -> Self {
        Self { num_bits }
    }

    /// Default: 2048 bits.
    pub fn default_size() -> Self {
        Self { num_bits: 2048 }
    }

    /// Get raw feature hashes.
    pub fn raw_features(&self, mol: &Molecule) -> Vec<u64> {
        atom_pair_features(mol)
    }
}

impl Descriptor for AtomPairFingerprint {
    fn name(&self) -> &str {
        "AtomPair"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let hashes = atom_pair_features(mol);
        let bits = fold_bitvec(&hashes, self.num_bits);
        DescriptorValue::Binary(bits)
    }
}

fn atom_pair_features(mol: &Molecule) -> Vec<u64> {
    let n = mol.num_atoms();
    if n < 2 {
        return vec![];
    }

    let distances = compute_distances(mol);
    let atom_types: Vec<u64> = (0..n).map(|i| atom_pair_type(mol, i)).collect();

    let mut features = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = distances[i * n + j];
            if dist == 0 || dist > MAX_PAIR_DISTANCE || dist == u32::MAX {
                continue;
            }
            // Canonical ordering: smaller type hash first
            let (t1, t2) = if atom_types[i] <= atom_types[j] {
                (atom_types[i], atom_types[j])
            } else {
                (atom_types[j], atom_types[i])
            };
            features.push(combine_hash(&[t1, dist as u64, t2]));
        }
    }

    features.sort();
    features.dedup();
    features
}

/// Topological Torsion Fingerprint descriptor.
///
/// Encodes all linear 4-atom paths in the molecule.
/// Each path is hashed as: (type1, type2, type3, type4) with canonical ordering.
/// Returns bit-packed fingerprint.
pub struct TopologicalTorsionFingerprint {
    pub num_bits: usize,
}

impl TopologicalTorsionFingerprint {
    pub fn new(num_bits: usize) -> Self {
        Self { num_bits }
    }

    pub fn default_size() -> Self {
        Self { num_bits: 2048 }
    }

    pub fn raw_features(&self, mol: &Molecule) -> Vec<u64> {
        torsion_features(mol)
    }
}

impl Descriptor for TopologicalTorsionFingerprint {
    fn name(&self) -> &str {
        "TopologicalTorsion"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let hashes = torsion_features(mol);
        let bits = fold_bitvec(&hashes, self.num_bits);
        DescriptorValue::Binary(bits)
    }
}

/// Atom type for topological torsion (similar to atom pair but includes ring info).
fn torsion_atom_type(mol: &Molecule, idx: usize) -> u64 {
    let atom = &mol.atoms[idx];
    let degree = mol.get_degree(idx as u16) as u64;
    let pi_electrons: u64 = mol.adjacency[idx].iter()
        .map(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            match bond.bond_type {
                BondType::Double => 1,
                BondType::Triple => 2,
                BondType::Aromatic => 1,
                _ => 0,
            }
        })
        .sum();

    combine_hash(&[
        atom.atomic_number as u64,
        pi_electrons,
        degree,
        atom.is_in_ring() as u64,
    ])
}

/// Find all linear 4-atom paths using DFS.
fn find_4_atom_paths(mol: &Molecule) -> Vec<[usize; 4]> {
    let n = mol.num_atoms();
    let mut paths = Vec::new();

    for start in 0..n {
        // DFS for paths of length 3 (4 atoms)
        for &b1 in &mol.adjacency[start] {
            let bond1 = &mol.bonds[b1 as usize];
            let a1 = if bond1.source as usize == start { bond1.target as usize } else { bond1.source as usize };

            for &b2 in &mol.adjacency[a1] {
                let bond2 = &mol.bonds[b2 as usize];
                let a2 = if bond2.source as usize == a1 { bond2.target as usize } else { bond2.source as usize };
                if a2 == start { continue; } // no backtracking

                for &b3 in &mol.adjacency[a2] {
                    let bond3 = &mol.bonds[b3 as usize];
                    let a3 = if bond3.source as usize == a2 { bond3.target as usize } else { bond3.source as usize };
                    if a3 == a1 || a3 == start { continue; } // no backtracking

                    // Only add each path once: use canonical ordering (start < end)
                    if start < a3 {
                        paths.push([start, a1, a2, a3]);
                    }
                }
            }
        }
    }

    paths
}

fn torsion_features(mol: &Molecule) -> Vec<u64> {
    let n = mol.num_atoms();
    if n < 4 {
        return vec![];
    }

    let atom_types: Vec<u64> = (0..n).map(|i| torsion_atom_type(mol, i)).collect();
    let paths = find_4_atom_paths(mol);

    let mut features: Vec<u64> = paths.iter()
        .map(|p| {
            let forward = combine_hash(&[atom_types[p[0]], atom_types[p[1]], atom_types[p[2]], atom_types[p[3]]]);
            let reverse = combine_hash(&[atom_types[p[3]], atom_types[p[2]], atom_types[p[1]], atom_types[p[0]]]);
            forward.min(reverse) // canonical
        })
        .collect();

    features.sort();
    features.dedup();
    features
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<AtomPairFingerprint>;
    let _ = assert_send_sync::<TopologicalTorsionFingerprint>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::desc::fingerprint::tanimoto_similarity;

    fn count_bits(fp: &BitVec<u8, Lsb0>) -> usize {
        fp.count_ones()
    }

    // ========== Atom Pair Tests ==========

    fn calc_ap(smiles: &str, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let ap = AtomPairFingerprint::new(bits);
        ap.calculate(&mut mol).as_binary().clone()
    }

    #[test]
    fn test_ap_methane_no_pairs() {
        let fp = calc_ap("C", 2048);
        assert_eq!(count_bits(&fp), 0);
    }

    #[test]
    fn test_ap_ethane() {
        let fp = calc_ap("CC", 2048);
        assert!(count_bits(&fp) > 0, "Ethane should have at least 1 atom pair");
    }

    #[test]
    fn test_ap_benzene() {
        let fp = calc_ap("c1ccccc1", 2048);
        let bits = count_bits(&fp);
        assert!(bits > 0 && bits < 100);
    }

    #[test]
    fn test_ap_identical() {
        let fp1 = calc_ap("CCO", 2048);
        let fp2 = calc_ap("CCO", 2048);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_ap_self_tanimoto() {
        let fp = calc_ap("c1ccc(CC(=O)O)cc1", 2048);
        let sim = tanimoto_similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ap_similar_molecules() {
        let fp1 = calc_ap("c1ccc(O)cc1", 2048);
        let fp2 = calc_ap("Cc1ccc(O)cc1", 2048);
        let sim = tanimoto_similarity(&fp1, &fp2);
        assert!(sim > 0.2, "Similar molecules should have some AP overlap: {}", sim);
    }

    #[test]
    fn test_ap_larger_molecules_more_bits() {
        let fp_small = calc_ap("CC", 2048);
        let fp_large = calc_ap("CCCCCCCCCC", 2048);
        assert!(count_bits(&fp_large) > count_bits(&fp_small));
    }

    #[test]
    fn test_ap_drug_molecule() {
        let fp = calc_ap("CC(=O)Oc1ccccc1C(=O)O", 2048);
        let bits = count_bits(&fp);
        assert!(bits > 5 && bits < 500);
    }

    // ========== Topological Torsion Tests ==========

    fn calc_tt(smiles: &str, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let tt = TopologicalTorsionFingerprint::new(bits);
        tt.calculate(&mut mol).as_binary().clone()
    }

    #[test]
    fn test_tt_small_molecules_no_paths() {
        assert_eq!(count_bits(&calc_tt("C", 2048)), 0);
        assert_eq!(count_bits(&calc_tt("CC", 2048)), 0);
        assert_eq!(count_bits(&calc_tt("CCC", 2048)), 0);
    }

    #[test]
    fn test_tt_butane() {
        let fp = calc_tt("CCCC", 2048);
        assert!(count_bits(&fp) > 0, "Butane has one 4-atom path");
    }

    #[test]
    fn test_tt_benzene() {
        let fp = calc_tt("c1ccccc1", 2048);
        assert!(count_bits(&fp) > 0);
    }

    #[test]
    fn test_tt_identical() {
        let fp1 = calc_tt("CCCCC", 2048);
        let fp2 = calc_tt("CCCCC", 2048);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_tt_self_tanimoto() {
        let fp = calc_tt("c1ccc(CC(=O)O)cc1", 2048);
        let sim = tanimoto_similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tt_different_molecules() {
        let fp1 = calc_tt("CCCCCC", 2048);
        let fp2 = calc_tt("c1ccccc1", 2048);
        assert_ne!(fp1, fp2, "Hexane and benzene should differ");
    }

    #[test]
    fn test_tt_drug_molecule() {
        let fp = calc_tt("CC(=O)Oc1ccccc1C(=O)O", 2048);
        let bits = count_bits(&fp);
        assert!(bits > 3 && bits < 500);
    }
}
